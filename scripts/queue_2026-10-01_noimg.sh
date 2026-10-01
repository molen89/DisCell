#!/bin/bash
# The no-image ablation (devlog "No-image ablation at the final configuration
# (motivation, 2026-10-01; author)"): the programme-driver decomposition puts
# the unique signal on the image on FF and on composition on the TMA core, but
# the programmes were learned with both inputs. The direct test: fit without
# the image.
#
# Fits: noimgL_s{0,1,2} on the four trained sections, launched exactly as the
# finalL fits were (scripts/queue_2026-09-25_final_lineage.sh: every model
# flag explicit, per-dataset alpha_z, GSE's variant pdl018d / tiles 2048,
# --label-key lineage, 500/40, --w-warmup-epochs 30, --adv-comp-weight 3)
# plus --no-image (TrainConfig.no_image: Phi is dropped from the context c,
# so c = the GAT over neighbour composition + the isolated flag; the
# adversary's e_Phi target and everything else are unchanged). Same seeds,
# hence the same tile splits, as finalL_s{0,1,2}. A preflight re-derives every
# arm's TrainConfig and stops unless it equals finalL_s<seed>/config.json but
# for no_image (and run_name / git). A dead w channel is recorded in
# DEAD_RUNS.tsv and NOT refitted: the seed is the split.
#
# Reads after each fit (finalL's read set, under the Unassigned mask):
#   degeneracy (guard + the in-trainer battery at best.pt: NMI z, cycle q90,
#   held-out recon, I(niche; w)) -> validate morans,niche -> probe_regrade
#   --force vs uncontrolledL_s{0,1} (the battery's probe share) -> atlas ->
#   transport --read mean (published read: fold 0 scored, cell-split ceiling).
#   After a section's three fits: atlas --compare-runs over the noimgL seeds
#   (cross-seed stability, hallmark label recurrence).
# Control: GSE finalL_s0 refitted with today's code as noimgctl_finalL_s0
# (finalL's flags, no --no-image), compared with finalL_s0's metrics.json --
# fit only (as the omega = 0 queue's control).
# At the end: scripts/noimg_readout.py writes $ROOT/READOUT.md.
#
# Order: GSE, ovarian, lung, FF. Scheduling: waits (pgrep -f with the
# non-self-matching pattern '[q]ueue_2026-10-01_breakdown_gaps.sh', poll
# 10 min) until the breakdown-gaps queue has ended. Then GPU 0 ONLY (never
# GPU 1: the MintFlow refits), with the omega = 0 queue's memory gate and
# lock protocol. FF steps also wait for host MemAvailable >= FF_HOST_GB.
# Threads: 8 for fits and reads (finalL's), 2 for probe_regrade and the
# read-out. WANDB_MODE=disabled.
#
# Idempotent: done/<step> markers (a fit is also skipped when its
# metrics.json exists); failed/<step> records. A CUDA OOM is retried (up to
# TRIES); any other failure is recorded once and the queue carries on.
# Delete failed/<step> to retry it on a relaunch.
#
# Estimate (omega = 0 queue's fit times; the Unassigned queue's atlas and
# transport times): GSE ~15 min, ovarian ~1 h, lung ~50 min, FF ~4.5 h,
# settle ~1 h: ~7-8 h of GPU 0 after the breakdown-gaps queue ends.
#
#   mkdir -p scripts/logs/noimg_2026-10-01 && setsid nohup \
#     bash scripts/queue_2026-10-01_noimg.sh \
#     >> scripts/logs/noimg_2026-10-01/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ORDER="$GS $OV $LU $FF"                        # the author's order
SEEDS="0 1 2"
ARM=noimgL

ROOT=scripts/logs/noimg_2026-10-01
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
GPU=0                  # GPU 0 only; GPU 1 is never used
WAIT_PATTERN='[q]ueue_2026-10-01_breakdown_gaps.sh'
POLL=600
NEED=9500              # MiB, a non-FF step (the finalL queue's gate)
NEED_FF=18500          # MiB, an FF step
FF_HOST_GB=60          # host MemAvailable before an FF step
TRIES=6
SETTLE=45; FRESH=600
PY8="env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python"
PY2="env OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"

# finalL's launch, flag for flag (scripts/queue_2026-09-25_final_lineage.sh)
COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30
        --adv-comp-weight 3 --label-key lineage"
NOIMG="--no-image"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002"
  [$FF]="--alpha-z 0.00035")
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"
CTL=noimgctl_finalL_s0                         # the code-drift control (GSE)

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# -- GPU 0 picker (scripts/queue_2026-10-01_breakdown.sh's) -----------------------
GUARD_PATTERN='run_mintflow|baseline_battery.*MintFlow'
GUARD_PEAK=9000
gpu_free() {
  local free pids pid mem held=0 found=0
  free=$(nvidia-smi --query-gpu=memory.total,memory.used \
         --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $1-$2}')
  pids=" $(pgrep -f "$GUARD_PATTERN" | tr '\n' ' ') "
  while IFS=', ' read -r pid mem; do
    [ -n "$pid" ] || continue
    case "$pids" in *" $pid "*) held=$((held + mem)); found=1 ;; esac
  done < <(nvidia-smi --query-compute-apps=pid,used_memory \
           --format=csv,noheader,nounits -i $GPU)
  if [ $found -eq 1 ] && [ $held -lt $GUARD_PEAK ]; then
    free=$((free - (GUARD_PEAK - held)))
  fi
  echo $free
}
held_elsewhere() {  # another queue's fresh GPU-0 lock
  local dir now age
  now=$(date +%s)
  for dir in scripts/logs/*/locks/gpu$GPU; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}
acquire_gpu() {  # acquire_gpu <need MiB>
  local need=$1 free waited=0
  while true; do
    free=$(gpu_free)
    if [ "$free" -ge "$need" ] && ! held_elsewhere \
       && mkdir "$LOCK/gpu$GPU" 2>/dev/null; then
      sleep $SETTLE
      free=$(gpu_free)
      if [ "$free" -ge "$need" ] && ! held_elsewhere; then return 0; fi
      rmdir "$LOCK/gpu$GPU" 2>/dev/null
    fi
    [ $((waited % 10)) -eq 0 ] && log "waiting for GPU $GPU with ${need} MiB free ($(gpu_free) free)"
    waited=$((waited + 1)); sleep 60
  done
}
wait_host() {  # wait_host <GB>
  local n=0
  while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt "$1" ]; do
    [ $((n % 10)) -eq 0 ] && log "waiting for $1 GB host MemAvailable"
    n=$((n + 1)); sleep 60
  done
}

# gstep <threads 8|2> <dataset> <marker> <cmd...>: a GPU-0 step
gstep() {
  local threads=$1 ds=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  if [ -f "$FAILED/$name" ]; then log "skip $name (failed before; see $FAILED/$name)"; return 1; fi
  local py=$PY8; [ "$threads" = 2 ] && py=$PY2
  local need=$NEED; [ "$ds" = "$FF" ] && need=$NEED_FF
  local try=0 rc=1 t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    [ "$ds" = "$FF" ] && wait_host $FF_HOST_GB
    acquire_gpu $need
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
    t0=$SECONDS
    log "GPU$GPU start $name (try $try, ${threads}t)"
    CUDA_VISIBLE_DEVICES=$GPU $py "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$GPU" 2>/dev/null
    log "GPU$GPU done  $name (exit $rc, $((SECONDS - t0))s)"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
    cp $QL/$name.log $QL/$name.try$try.log
    if grep -qE "OutOfMemoryError|CUDA out of memory" $QL/$name.log; then
      log "FAIL $name: CUDA OOM (try $try/$TRIES)"; sleep 300; continue
    fi
    break
  done
  log "FAILED $name (exit $rc); see $QL/$name.log"
  { echo "exit $rc"; tail -5 $QL/$name.log; } > "$FAILED/$name"
  return 1
}
cstep() {  # cstep <marker> <cmd...>: a CPU step (re-run every time it is reached)
  local name=$1; shift
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= $PY2 "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAILED $name; see $QL/$name.log"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

# -- preflight -----------------------------------------------------------------------
preflight_config() {  # every noimgL arm's TrainConfig vs finalL_s<seed>/config.json
  local specs=() ds
  for ds in $ORDER; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  CUDA_VISIBLE_DEVICES= $PY2 - "$NOIMG" "$SEEDS" "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
noimg, seeds, bad = sys.argv[1], [int(s) for s in sys.argv[2].split()], 0
BASE = {"run_name", "git"}
default = asdict(TrainConfig(dataset=""))
def derive(ds, flags, seed):
    args = vars(build_parser().parse_args(["--dataset", ds, "--seed", str(seed)] + shlex.split(flags)))
    for k in ("quiet", "time_only"):
        args.pop(k, None)
    return asdict(TrainConfig(**args))
for spec in sys.argv[3:]:
    ds, flags = spec.split("::", 1)
    for seed in seeds:
        ref = json.load(open(f"data/datasets/{ds}/runs/finalL_s{seed}/config.json"))
        for name, arm_flags, extra, want in (
                (f"finalL_s{seed} (control)", flags, set(), {"no_image": False}),
                (f"noimgL_s{seed}", f"{flags} {noimg}", {"no_image"}, {"no_image": True})):
            new = derive(ds, arm_flags, seed)
            diffs = [f"{k}: {new.get(k)!r} vs finalL_s{seed} {ref.get(k, '<absent>')!r}"
                     for k in sorted(set(new) | set(ref)) if k not in BASE | extra
                     and ((k in ref and new.get(k) != ref[k])
                          or (k not in ref and new[k] != default[k]))]
            wrong = [f"{k}={new[k]!r} (want {v!r})" for k, v in
                     {**want, "seed": seed, "epochs": 500, "patience": 40,
                      "label_key": "lineage", "adv_comp_weight": 3.0,
                      "w_warmup_epochs": 30}.items() if new[k] != v]
            ok = not diffs and not wrong
            bad += not ok
            print(f"{ds} {name}: " + ("finalL's config but for the arm's own fields "
                                      f"(no_image={new['no_image']})"
                                      if ok else "DIFFERS: " + "; ".join(diffs + wrong)))
sys.exit(1 if bad else 0)
EOF
}

# -- fits and reads ---------------------------------------------------------------------
fit() {   # fit <dataset> <run> <seed> [extra flags...]
  local ds=$1 run=$2 seed=$3 mk; shift 3; mk=fit_${ds}_$run
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  gstep 8 $ds "$mk" -m discell.model.train --dataset $ds --run-name $run \
    ${FLAGS[$ds]} $COMMON --seed $seed "$@"
}

record_dead() {  # record_dead <dataset> <run>: DEAD_RUNS.tsv, once; never refitted
  local why
  why=$(python3 - "$(runs $1)/$2" <<'EOF'
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
) || return 0
  [ -n "$why" ] || return 0
  grep -qP "\t$1\t$2\t" $ROOT/DEAD_RUNS.tsv 2>/dev/null || \
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $1 $2 "$why" "recorded (ablation; not refitted)" >> $ROOT/DEAD_RUNS.tsv
  log "$1/$2 DEAD ($why) -- recorded, not refitted"
}

reads() {  # reads <dataset> <run>: finalL's read set
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  gstep 8 $ds "degeneracy_${ds}_$run" -m discell.model.degeneracy --dataset $ds --run $run \
    && record_dead $ds $run
  gstep 8 $ds "validate_${ds}_$run" -m discell.model.validate --dataset $ds --run $run \
    --analyses morans,niche
  # shellcheck disable=SC2086
  gstep 2 $ds "probe_regrade_${ds}_$run" -m discell.experiments.probe_regrade \
    --dataset $ds --run $run --force $REFS
  gstep 8 $ds "atlas_${ds}_$run" -m discell.model.atlas --dataset $ds --run $run
  gstep 8 $ds "transport_${ds}_$run" -m discell.model.transport --dataset $ds \
    --run $run --read mean
  return 0
}

compare_group() {  # compare_group <dataset>: cross-seed atlas over the arm's live seeds
  local ds=$1 s r others live=""
  for s in $SEEDS; do
    r=${ARM}_s$s
    [ -f "$(runs $ds)/$r/atlas/programs.npy" ] && live="$live $r"
  done
  for r in $live; do
    others=$(echo $live | tr ' ' '\n' | grep -vx $r | tr '\n' ' ')
    if [ -n "$others" ]; then
      # shellcheck disable=SC2086
      gstep 8 $ds "atlas_cmp_${ds}_$r" -m discell.model.atlas --dataset $ds \
        --run $r --compare-runs $others
    else
      log "$ds/$r: no other seed -- no cross-seed atlas"
    fi
  done
}

control() {  # GSE finalL_s0 refitted with today's code; compared with the original
  fit $GS $CTL 0 || return 1
  CUDA_VISIBLE_DEVICES= $PY2 - "$(runs $GS)" "$CTL" "$ROOT/control.json" <<'EOF' > $QL/control_compare.log 2>&1
import json, sys
from pathlib import Path
d, ctl, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
best = lambda r: json.loads((d / r / "metrics.json").read_text())["best"]
ref, new = best("finalL_s0"), best(ctl)
seeds = [best(f"finalL_s{s}") for s in (0, 1, 2)]
row = {k: {"finalL_s0": ref[k], ctl: new[k],
           "finalL_range": [min(s[k] for s in seeds), max(s[k] for s in seeds)]}
       for k in ("epoch", "recon_val", "nmi")}
same = all(ref[k] == new[k] for k in ("epoch", "recon_val", "nmi"))
inside = all(r["finalL_range"][0] <= r[ctl] <= r["finalL_range"][1] for r in row.values())
text = ("reproduces finalL_s0 exactly (best epoch, recon_val, NMI)" if same else
        "differs from finalL_s0: " + "; ".join(
            f"{k} {r['finalL_s0']:.6g} -> {r[ctl]:.6g}" for k, r in row.items())
        + (" -- inside finalL's seed range on all three" if inside
           else " -- OUTSIDE finalL's seed range on at least one"))
out.write_text(json.dumps({"run": ctl, "same": same, "inside_seed_range": inside,
                           "reads": row, "text": text}, indent=1))
print(text)
EOF
  log "control: $(cat $QL/control_compare.log)"
}

section() {  # section <dataset>: three fits, each followed by its reads; then the cross-seed atlas
  local ds=$1 s
  for s in $SEEDS; do
    # shellcheck disable=SC2086
    fit $ds ${ARM}_s$s $s $NOIMG && reads $ds ${ARM}_s$s
  done
  compare_group $ds
  log "$ds finished"
}

# -- main ----------------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only; waiting for: $WAIT_PATTERN; model code $(code_hash)"
if ! preflight_config > $QL/preflight.log 2>&1; then
  cat $QL/preflight.log; log "PREFLIGHT FAILED: an arm's config is not finalL's -- stopping"; exit 1
fi
cat $QL/preflight.log
n=0
while pgrep -f "$WAIT_PATTERN" > /dev/null; do
  [ $((n % 6)) -eq 0 ] && log "breakdown-gaps queue still running (pid $(pgrep -f "$WAIT_PATTERN" | tr '\n' ' ')) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the breakdown-gaps queue has ended; starting (GPU $GPU only)"
rmdir $LOCK/gpu$GPU 2>/dev/null      # one instance at a time: a lock of ours is stale

control
for ds in $ORDER; do section $ds; done
cstep readout scripts/noimg_readout.py $ROOT
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
