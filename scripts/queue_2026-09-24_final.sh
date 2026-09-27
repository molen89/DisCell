#!/bin/bash
# The final-configuration re-pin (devlog 2026-09-24 "Final configuration
# frozen; re-pin launched (author's decision)").
#
# Final configuration = each dataset's runs/best_s1/config.json (alpha_z at
# 1/2 x 1/mean-count: GSE 0.0018, ovarian 0.0035, lung 0.002, FF 0.00035),
# 500/40, --figures-every 100, plus --w-warmup-epochs 30. Seeds 0 1 2, run
# names final_s<seed>. A preflight re-derives every TrainConfig field from
# these flags and aborts the queue if anything but seed / run name / warm-up
# differs from best_s1/config.json.
#
# Phase 1, per dataset lane (GSE, ovarian, lung first; FF as a GPU allows):
#   fit -> degeneracy (the w-channel guard) -> dead? (metrics.json
#   dead_w_channel OR degeneracy.json w_channel.dead_context_channel): refit
#   the slot with the next unused seed (3, 4, ...) as final_s<seed>, recorded
#   in DEAD_RUNS.tsv -> validate morans,niche -> atlas (first pass) ->
#   transport both/hvg 1000 kmeans (+ tumour-band on ovarian) -> GSE
#   crossslide on the dual section; once the dataset's three live runs exist:
#   atlas --compare-runs <the other two> -> report.
# Phase 2, after every lane: runs/best -> the slot-0 run (final_s0) on each
# dataset (previous target in REPIN.tsv; best_s* untouched), envelope tables
# --at best, the DisCell baseline-battery columns (slot 0 last, so the cached
# context is runs/best's; GSE + dual only after the MintFlow queue exits, whose
# pending battery columns read the same cache), then every existing baseline
# column (resolVI, SIMVI, MintFlow incl. the full fit once its queue writes it)
# re-scored against that new cache under its own --method label, so it is
# replaced in place (coordinator decision 2026-09-24; latents only, no refits),
# external criteria, GO localisation, read-out (READOUT.md).
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_wcollapse_b.sh
# (FF gate 18500 MiB, others 9500, one job of ours per GPU, MintFlow charged at
# its peak) plus: the 8.9b queue's GPU locks and FF waiters are respected while
# it runs, and phase-2 reads fall back to the CPU after 30 min without a GPU.
# DONE markers are keyed by step AND dataset AND run; a relaunch resumes.
#
#   setsid nohup scripts/queue_2026-09-24_final.sh \
#     >> scripts/logs/final_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ALL="$GS $OV $LU $FF"

ROOT=scripts/logs/final_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting
SLOTS=$ROOT/slots; SEEDS=$ROOT/seeds; ACC=$ROOT/accepted; FAILED=$ROOT/failed
BACKUP=$ROOT/backup_pre_final
mkdir -p $QL $DONE $LOCK $WAIT $SLOTS $SEEDS $ACC $FAILED $BACKUP
GPUS="0 1"
NEED_FF=18500
NEED_OV=9500          # every non-FF dataset
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
MAX_SEED=8            # replacement seeds 3..8 per dataset
PHASE2_WAIT=1800      # phase-2 reads go to the CPU after this long without a GPU
MINTFLOW_WAIT_MAX=72000

COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002 --label-key graphclust"
  [$FF]="--alpha-z 0.00035 --label-key graphclust")

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
code_hash() { sha256sum discell/model/train.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

# -- GPU picker ---------------------------------------------------------------
# MintFlow (fit, then its battery columns) is charged at its peak on its card.
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
# the 8.9b grid (another queue) finishing on GPU 1: its locks and FF waiters
FQ_PATTERN=queue_2026-09-24_wcollapse_b.sh
FQ_LOCK=scripts/logs/wcollapse_b_2026-09-24/locks
FQ_WAIT=scripts/logs/wcollapse_b_2026-09-24/waiting

gpu_free() {    # free MiB on GPU $1, net of the guarded neighbour's headroom
  local g=$1 free pids pid mem held=0 found=0
  free=$(nvidia-smi --query-gpu=memory.total,memory.used \
         --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  pids=" $(pgrep -f "$GUARD_PATTERN" | tr '\n' ' ') "
  while IFS=', ' read -r pid mem; do
    [ -n "$pid" ] || continue
    case "$pids" in *" $pid "*) held=$((held + mem)); found=1 ;; esac
  done < <(nvidia-smi --query-compute-apps=pid,used_memory \
           --format=csv,noheader,nounits -i $g)
  if [ $found -eq 1 ] && [ $held -lt $GUARD_PEAK ]; then
    free=$((free - (GUARD_PEAK - held)))
  fi
  echo $free
}

foreign_busy() {  # foreign_busy <gpu> <free>: the 8.9b queue holds or wants it
  pgrep -f "$FQ_PATTERN" > /dev/null || return 1
  [ -d "$FQ_LOCK/gpu$1" ] && return 0
  [ "$2" -ge "$NEED_FF" ] && ls "$FQ_WAIT"/ff.* > /dev/null 2>&1 && return 0
  return 1
}

# Block until a GPU has >= $1 MiB free and holds no job of ours, take it and
# echo its index. Role "ff" announces itself in $WAIT while waiting; role "ov"
# skips any card an FF step could use while one waits. With $3 > 0, give up
# after $3 s and echo "cpu". Everything else goes to stderr.
acquire_gpu() {
  local need=$1 role=$2 maxw=${3:-0} g free waited=0 t0=$SECONDS
  local me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      foreign_busy $g $free && continue
      if [ "$role" = ov ] && [ "$free" -ge "$NEED_FF" ] \
         && ls "$WAIT"/ff.* > /dev/null 2>&1; then
        continue                                      # FF goes first
      fi
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! foreign_busy $g $free; then
        [ "$role" = ff ] && rm -f "$me"
        echo $g; return 0
      fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    if [ "$maxw" -gt 0 ] && [ $((SECONDS - t0)) -ge "$maxw" ]; then
      [ "$role" = ff ] && rm -f "$me"
      log "[$role] no GPU with ${need} MiB after ${maxw}s -- using the CPU" >&2
      echo cpu; return 0
    fi
    if [ $((waited % 10)) -eq 0 ]; then
      log "[$role] waiting for a GPU with ${need} MiB free (0: $(gpu_free 0), 1: $(gpu_free 1))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps --------------------------------------------------------------------
_step() {  # _step <max wait> <need> <role> <marker> <cmd...>
  local maxw=$1 need=$2 role=$3 name=$4; shift 4
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $need $role $maxw)
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
    t0=$SECONDS
    if [ "$gpu" = cpu ]; then
      log "CPU  start $name (try $try)"
      CUDA_VISIBLE_DEVICES= "$@" --device cpu > $QL/$name.log 2>&1; rc=$?
    else
      log "GPU$gpu start $name (try $try)"
      CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
      rmdir "$LOCK/gpu$gpu" 2>/dev/null
    fi
    log "$([ "$gpu" = cpu ] && echo CPU || echo GPU$gpu) done  $name (exit $rc, $((SECONDS - t0))s)"
    if [ $rc -eq 0 ]; then
      touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0
    fi
    cp $QL/$name.log $QL/$name.try$try.log
    if grep -qE "OutOfMemoryError|CUDA out of memory" $QL/$name.log; then
      log "FAIL $name: CUDA OOM (try $try/$TRIES); see $QL/$name.try$try.log"
    else
      other=$((other + 1))
      log "FAIL $name (exit $rc; attempt $other/$TRIES_OTHER); see $QL/$name.try$try.log"
      [ $other -ge $TRIES_OTHER ] && break
    fi
    sleep 300
  done
  log "GIVING UP on $name"; touch "$FAILED/$name"; return 1
}
step() { _step 0 "$@"; }                     # GPU, waits as long as it takes
step_or_cpu() { _step $PHASE2_WAIT "$@"; }   # GPU, else the CPU after 30 min

cpu_step() {  # cpu_step <marker> <cmd...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}

# -- phase 1: fits, the guard, the per-run battery ---------------------------
fit() {   # fit <dataset> <run> <seed>
  local ds=$1 run=$2 seed=$3 mk; mk=$(marker fit $1 $2)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  step $(need_of $ds) $(role_of $ds) "$mk" uv run python -m discell.model.train \
    --dataset $ds --run-name $run ${FLAGS[$ds]} $COMMON --seed $seed
}

dead_reason() {  # prints why <dataset>/<run> is a dead fit; nothing when live
  python3 - "$(runs $1)/$2" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); why = []
if json.loads((d / "metrics.json").read_text()).get("dead_w_channel"):
    why.append("metrics.json dead_w_channel")
g = d / "degeneracy.json"
w = (json.loads(g.read_text()).get("w_channel") or {}) if g.exists() else {}
if w.get("dead_context_channel"):
    why.append(f"guard: I(niche;w) excess {w.get('w_niche_mi_excess'):+.4f}")
print("; ".join(why))
EOF
}

next_seed() {  # next_seed <dataset>: claim the smallest unused seed >= 3
  local s
  for s in $(seq 3 $MAX_SEED); do
    mkdir $SEEDS/$1/s$s 2>/dev/null && { echo $s; return 0; }
  done
  return 1
}

battery() {  # battery <dataset> <run>: everything before the cross-seed atlas
  local ds=$1 run=$2 need role; need=$(need_of $1); role=$(role_of $1)
  step $need $role "$(marker validate $ds $run)" \
    uv run python -m discell.model.validate --dataset $ds --run $run \
    --analyses morans,niche
  step $need $role "$(marker atlas $ds $run)" \
    uv run python -m discell.model.atlas --dataset $ds --run $run
  step $need $role "$(marker transport $ds $run)" \
    uv run python -m discell.model.transport --dataset $ds --run $run \
    --read both --hvg 1000
  if [ "$ds" = "$OV" ]; then   # only the ovarian slide is tumour-annotated
    step $need $role "$(marker transport_band $ds $run)" \
      uv run python -m discell.model.transport --dataset $ds --run $run \
      --read both --hvg 1000 --niche-source tumour-band
  fi
  if [ "$ds" = "$GS" ]; then
    step $need $role "$(marker crossslide $ds $run)" \
      uv run python -m discell.model.crossslide --dataset $GS --run $run \
      --eval-dataset $GD
  fi
  return 0
}

slot() {  # slot <dataset> <k>: seed k; while the fit is dead, the next seed
  local ds=$1 k=$2 chain=$SLOTS/${1}_slot$2 seed run why need role
  need=$(need_of $ds); role=$(role_of $ds)
  [ -s $chain ] || echo $k > $chain
  seed=$(tail -1 $chain)
  while true; do
    run=final_s$seed
    if ! fit $ds $run $seed; then
      log "SLOT $ds/$k: fit $run failed -- slot left empty"; return 1
    fi
    step $need $role "$(marker degeneracy $ds $run)" \
      uv run python -m discell.model.degeneracy --dataset $ds --run $run \
      || log "SLOT $ds/$k: guard failed on $run -- deciding on metrics.json alone"
    if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then
      log "SLOT $ds/$k: $run has no metrics.json -- slot left empty"; return 1
    fi
    if ! why=$(dead_reason $ds $run); then
      log "SLOT $ds/$k: could not read the flags of $run -- slot left empty"; return 1
    fi
    if [ -z "$why" ]; then
      echo $run > $ACC/${ds}_slot$k; log "SLOT $ds/$k: $run live"; break
    fi
    grep -qP "\t$ds\t$run\t" $ROOT/DEAD_RUNS.tsv 2>/dev/null || \
      printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $ds $run "$why" \
      >> $ROOT/DEAD_RUNS.tsv
    if ! seed=$(next_seed $ds); then
      log "SLOT $ds/$k: $run DEAD ($why) and no unused seed <= $MAX_SEED left"
      return 1
    fi
    echo $seed >> $chain
    log "SLOT $ds/$k: $run DEAD ($why) -- refitting as final_s$seed"
  done
  battery $ds $run
}

accepted() {  # the live run of each slot, slot order (0 1 2)
  local k
  for k in 0 1 2; do [ -s $ACC/${1}_slot$k ] && cat $ACC/${1}_slot$k; done
}
accepted_s0_last() {  # slots 1 2 0: whatever runs last leaves the cache
  local k
  for k in 1 2 0; do [ -s $ACC/${1}_slot$k ] && cat $ACC/${1}_slot$k; done
}
slot0() { cat $ACC/${1}_slot0 2>/dev/null; }

compare_pass() {  # compare_pass <dataset>: cross-seed atlas, then the report
  local ds=$1 r others need role live; need=$(need_of $1); role=$(role_of $1)
  live=$(accepted $ds)
  for r in $live; do
    others=$(echo $live | tr ' ' '\n' | grep -vx $r | tr '\n' ' ')
    if [ -n "$others" ]; then
      step $need $role "$(marker atlas_cmp $ds $r)" \
        uv run python -m discell.model.atlas --dataset $ds --run $r \
        --compare-runs $others
    else
      log "$ds/$r: no other live seed -- no cross-seed atlas"
    fi
    step $need $role "$(marker report $ds $r)" \
      uv run python -m discell.model.report --dataset $ds --run $r
  done
}

lane() { local ds=$1 k; shift; for k in "$@"; do slot $ds $k; done; }
laneGS() { lane $GS 0 1 2; compare_pass $GS; log "GSE lane finished"; }
laneOV() { lane $OV 0 1 2; compare_pass $OV; log "ovarian lane finished"; }
laneLU() { lane $LU 0 1 2; compare_pass $LU; log "lung lane finished"; }
# two FF sub-lanes on disjoint slots: while only GPU 1 can host FF they take
# turns on it; once GPU 0 frees (MintFlow done) both run at once
laneFF_A() { lane $FF 0 2; log "FF sub-lane A finished"; }
laneFF_B() { lane $FF 1; log "FF sub-lane B finished"; }

# -- phase 2 ------------------------------------------------------------------
backup() {  # backup <file>...: keep the pre-final version once (never clobber)
  local f dst
  for f in "$@"; do
    [ -f "$f" ] || continue
    dst=$BACKUP/${f#data/datasets/}; mkdir -p "$(dirname $dst)"
    cp -n "$f" "$dst"
  done
}

repin() {  # repin <dataset>: runs/best -> the slot-0 run
  local ds=$1 d new old; d=$(runs $1); new=$(slot0 $ds)
  if [ -z "$new" ]; then log "$ds: slot 0 has no live run -- best NOT re-pinned"; return 0; fi
  if [ -L $d/best ] && [ "$(readlink $d/best)" = "$new" ]; then
    log "$ds: best already -> $new"; return 0
  fi
  if [ -L $d/best ]; then
    old=$(readlink $d/best)
  elif [ -d $d/best ]; then
    old="directory, moved to best_pre_final"; mv $d/best $d/best_pre_final
  else
    old=none
  fi
  ln -sfn $new $d/best.final_tmp && mv -T $d/best.final_tmp $d/best
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $ds "$old" $new >> $ROOT/REPIN.tsv
  log "$ds: best was -> $old; re-pinned to $new"
}

envelope() {
  local ds std=1 args
  for ds in $ALL; do
    backup data/datasets/$ds/experiments/envelope_table_at_best.md
    [ "$(accepted $ds | tr '\n' ' ')" = "final_s0 final_s1 final_s2 " ] || std=0
  done
  if [ $std -eq 1 ]; then
    cpu_step envelope_final_all uv run python scripts/envelope_tables.py \
      --datasets $ALL --runs final_s0 final_s1 final_s2 --at best \
      --combined scratchpad/envelope_tables_final.md
  else   # a refitted slot: each dataset with its own live triple
    for ds in $ALL; do
      args=$(accepted $ds)
      [ -n "$args" ] || { log "$ds: no live run -- no envelope table"; continue; }
      cpu_step "$(marker envelope $ds final)" uv run python scripts/envelope_tables.py \
        --datasets $ds --runs $args --at best \
        --combined scratchpad/envelope_tables_final_$ds.md
    done
  fi
}

battery_columns() {  # battery_columns <dataset> [--config-from <ds> <runs from>]
  local ds=$1 from=${2:-$1} r extra=()
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  backup data/datasets/$ds/experiments/baseline_battery.json \
         data/datasets/$ds/experiments/baseline_battery.md
  for r in $(accepted_s0_last $from); do
    step_or_cpu $(need_of $ds) $(role_of $ds) "$(marker battery_discell $ds $r)" \
      uv run python -m discell.experiments.baseline_battery --dataset $ds \
      --discell-run $r "${extra[@]}"
  done
}

# Baseline latents already on disk (DisCell-baselines results). Re-scored only
# once the dataset's slot-0 DisCell column has rewritten the cached context.
BRES=/home/rmolen/github/DisCell-baselines/results
rescore() {  # rescore <dataset> <context source dataset> <method label> <latents dir>
  local ds=$1 from=$2 method=$3 dir=$4 extra=() s0 tag
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  s0=$(slot0 $from)
  if [ -z "$s0" ] || [ ! -f "$DONE/$(marker battery_discell $ds $s0)" ]; then
    log "$ds '$method': the slot-0 DisCell column did not run -- context not refreshed, not re-scored"
    return 0
  fi
  if [ ! -f "$dir/latents.h5ad" ]; then
    log "$ds '$method': no latents at $dir -- not re-scored"; return 0
  fi
  tag=$(printf '%s' "$method" | tr -c 'A-Za-z0-9=.' '_')
  step_or_cpu $(need_of $ds) $(role_of $ds) "$(marker rescore_$tag $ds $s0)" \
    uv run python -m discell.experiments.baseline_battery --dataset $ds \
    --method "$method" --latents "$dir/latents.h5ad" "${extra[@]}"
}

ext() {  # ext <read> <dataset> <run>
  [ -n "$3" ] || { log "no live run for $1 on $2 -- skipped"; return 0; }
  step_or_cpu $(need_of $2) $(role_of $2) "$(marker ext_$1 $2 $3)" \
    uv run python -m discell.experiments.external_criteria $1 \
    --dataset $2 --run $3
}

go_loc() {
  local ds r args
  for ds in $ALL; do
    backup data/datasets/$ds/experiments/go_localisation.json \
           data/datasets/$ds/experiments/go_localisation.md \
           data/datasets/$ds/experiments/go_localisation.png
    args=(); for r in $(accepted $ds); do args+=(--run $r); done
    [ ${#args[@]} -gt 0 ] || { log "$ds: no live run -- no GO read"; continue; }
    cpu_step "$(marker go_localisation $ds final)" \
      uv run python -m discell.experiments.go_localisation --dataset $ds "${args[@]}"
  done
}

wait_mintflow() {  # the MintFlow queue's battery columns read the cached context
  local t0=$SECONDS n=0
  while pgrep -f queue_2026-09-23_mintflow_full.sh > /dev/null; do
    if [ $((SECONDS - t0)) -ge $MINTFLOW_WAIT_MAX ]; then
      log "MintFlow queue still alive after ${MINTFLOW_WAIT_MAX}s -- going anyway"
      return 0
    fi
    [ $((n % 6)) -eq 0 ] && log "waiting for the MintFlow queue to exit before the GSE/dual DisCell columns"
    n=$((n + 1)); sleep 600
  done
  log "MintFlow queue gone"
}

readout() {
  uv run python - $ROOT $ALL > $QL/readout.log 2>&1 <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); datasets = sys.argv[2:]
def load(p):
    try: return json.loads(p.read_text())
    except Exception: return None
def f(x, n=4):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
acc = {}
for p in (root / "accepted").glob("*_slot*"):
    ds, k = p.name.rsplit("_slot", 1); acc.setdefault(ds, {})[int(k)] = p.read_text().strip()
L = ["# Final re-pin read-out (scripts/queue_2026-09-24_final.sh)", "",
     "All reads at the accepted checkpoint (`best`); `final_epoch` present means the "
     "R26 closing-evaluation fix was in the trainer.", "",
     "| dataset | run | slot | best epoch | final_epoch | last epoch | recon_val | NMI | "
     "dead_w_channel | I(niche;w) excess | across-cell w var | I(z;t)/H(t) | minutes |",
     "|" + "---|" * 13]
for ds in datasets:
    slots = {v: k for k, v in acc.get(ds, {}).items()}
    rd = pathlib.Path(f"data/datasets/{ds}/runs")
    for d in sorted(rd.glob("final_s*"), key=lambda p: int(p.name[7:]) if p.name[7:].isdigit() else 99):
        m = load(d / "metrics.json") or {}; g = load(d / "degeneracy.json") or {}
        b = m.get("best") or {}; w = g.get("w_channel") or {}; dg = g.get("degeneracy") or {}
        L.append(f"| {ds} | {d.name} | {slots.get(d.name, 'dead/unused')} | {f(b.get('epoch'))} | "
                 f"{f(m.get('final_epoch'))} | {f(m.get('last_epoch'))} | {f(b.get('recon_val'))} | "
                 f"{f(b.get('nmi'))} | {f(m.get('dead_w_channel'))} | {f(w.get('w_niche_mi_excess'))} | "
                 f"{f(w.get('w_var_fraction_across_cells'))} | {f(dg.get('mi_ratio'), 3)} | "
                 f"{f(m.get('minutes'), 1)} |")
L += ["", "## Live triples (slot 0 is runs/best)", ""]
L += [f"* {ds}: " + ", ".join(acc.get(ds, {}).get(k, "EMPTY") for k in (0, 1, 2)) for ds in datasets]
for name, title in (("DEAD_RUNS.tsv", "Dead fits refitted"), ("REPIN.tsv", "runs/best re-pins (time, dataset, previous, new)")):
    p = root / name
    L += ["", f"## {title}", "", "```", p.read_text().strip() if p.exists() else "(none)", "```"]
rescored = sorted(p.name for p in (root / "done").iterdir() if p.name.startswith("rescore_"))
L += ["", "## Baseline columns re-scored against the final context", ""] + ([f"* {n}" for n in rescored] or ["(none)"])
failed = sorted(p.name for p in (root / "failed").iterdir())
L += ["", "## Failed steps (logs in logs/<step>.log)", ""] + ([f"* {n}" for n in failed] or ["(none)"])
hashes = root / "code_hashes.tsv"
if hashes.exists():
    rows = [r.split("\t") for r in hashes.read_text().strip().splitlines()]
    distinct = sorted({r[2].strip() for r in rows})
    L += ["", "## Trainer code (sha256[:12] of train.py, elbo.py) at each fit start", ""]
    L += [f"* {h}: {sum(r[2].strip() == h for r in rows)} fit launches" for h in distinct]
    if len(distinct) > 1:
        L.append("* **the trainer changed between fits** -- see code_hashes.tsv")
L += ["", "## Outputs", ""]
for p in [pathlib.Path("scratchpad/envelope_tables_final_at_best.md")] + \
         [pathlib.Path(f"data/datasets/{ds}/experiments/{n}") for ds in datasets
          for n in ("envelope_table_at_best.md", "baseline_battery.md", "go_localisation.md")] + \
         [pathlib.Path("data/datasets/gse315411_pdltma06_10_prime_dual/experiments/baseline_battery.md")]:
    L.append(f"* {'ok     ' if p.exists() else 'MISSING'} {p}")
out = root / "READOUT.md"; out.write_text("\n".join(L) + "\n"); print("\n".join(L))
EOF
  log "read-out written: $ROOT/READOUT.md (exit $?)"
}

# -- preflight: the flags reproduce best_s1/config.json but for the warm-up ---
preflight() {
  local specs=() ds
  for ds in $ALL; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  uv run python - "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
ALLOWED = {"run_name", "seed", "git", "w_warmup_epochs"}
default = asdict(TrainConfig(dataset=""))
bad = 0
for spec in sys.argv[1:]:
    ds, flags = spec.split("::", 1)
    args = vars(build_parser().parse_args(["--dataset", ds] + shlex.split(flags)))
    args.pop("quiet")
    new = asdict(TrainConfig(**args))
    ref = json.load(open(f"data/datasets/{ds}/runs/best_s1/config.json"))
    diffs = [f"{k}: {new.get(k)!r} vs best_s1 {ref.get(k, '<absent>')!r}"
             for k in sorted(set(new) | set(ref)) if k not in ALLOWED
             and ((k in ref and new.get(k) != ref[k])
                  or (k not in ref and new[k] != default[k]))]
    print(f"{ds}: alpha_z {new['alpha_z']} (best_s1 {ref['alpha_z']}), "
          f"w_warmup_epochs {new['w_warmup_epochs']}, {new['epochs']}/{new['patience']}: "
          + ("identical to best_s1 otherwise" if not diffs else "DIFFERS: " + "; ".join(diffs)))
    bad += bool(diffs) or new["w_warmup_epochs"] != 30
sys.exit(1 if bad else 0)
EOF
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, others $NEED_OV MiB, GPUs: $GPUS; trainer $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED: flags do not reproduce best_s1 -- stopping"; exit 1; fi
for ds in $ALL; do mkdir -p $SEEDS/$ds/s0 $SEEDS/$ds/s1 $SEEDS/$ds/s2; done

log "=== phase 1 ==="
laneFF_A & PFA=$!
(sleep 90; laneFF_B) & PFB=$!    # A takes the first FF slot (seed 0 first)
(sleep 10; laneGS) & PGS=$!      # after A has announced itself
(sleep 20; laneOV) & POV=$!
(sleep 30; laneLU) & PLU=$!
wait $PFA $PFB
compare_pass $FF; log "FF lane finished"
wait $PGS $POV $PLU
log "all lanes finished"

log "=== phase 2 ==="
for ds in $ALL; do repin $ds; done
envelope
battery_columns $OV; rescore $OV $OV resolVI $BRES/resolvi/$OV
battery_columns $LU; rescore $LU $LU resolVI $BRES/resolvi/$LU
battery_columns $FF; rescore $FF $FF resolVI $BRES/resolvi/$FF
rescore $FF $FF "SIMVI (100k-cell window)" $BRES/simvi/$FF
for r in $(accepted $OV); do
  ext signalling-share $OV $r; ext mi-quadrant $OV $r; ext axis-test $OV $r
done
ext signalling-share $FF "$(slot0 $FF)"; ext mi-quadrant $FF "$(slot0 $FF)"
ext mi-quadrant $GS "$(slot0 $GS)"
go_loc
wait_mintflow
battery_columns $GS
rescore $GS $GS resolVI $BRES/resolvi/$GS
rescore $GS $GS SIMVI $BRES/simvi/$GS
rescore $GS $GS MintFlow $BRES/mintflow/$GS
rescore $GS $GS "MintFlow (w=600, 5/50 epochs)" $BRES/mintflow/${GS}_w600
rescore $GS $GS "MintFlow (50 epochs, w=600)" $BRES/mintflow/${GS}_full
battery_columns $GD $GS
rescore $GD $GS resolVI $BRES/resolvi/$GD
rescore $GD $GS "SIMVI (fit on this section)" $BRES/simvi/${GD}_fit_on_target
rescore $GD $GS "MintFlow (transfer, 5/50 epochs)" $BRES/mintflow/${GD}_transfer
rescore $GD $GS "MintFlow (transfer, 50 epochs)" $BRES/mintflow/${GD}_transfer_full
readout
log "queue finished"
