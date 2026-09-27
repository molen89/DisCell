#!/bin/bash
# Review R12 Test 2: the leak coefficient's form (devlog 2026-09-24, "Review
# R12: what the leakage term assumes, tested (motivation)").
#
# Ovarian core, final configuration (warm-up 30, alpha_z 1/2 = 0.0035, alpha_w
# 0.1, kappa 0.1, d_w 6, type_only, 200/20), seeds 0 1 2, tag r12:
#   global  CONTROL, reused: wfix_warmup30_aw0.1_s{0,1,2} are this exact
#           configuration (the preflight below re-derives it from these flags
#           and aborts otherwise). Their fits, validate, degeneracy and k-means
#           transport are reused as they are; the missing reads are added
#           (atlas, tumour-band transport, 6b.1 signalling share, GO).
#   depth   r12_depth_s<seed>  --kappa-mode depth
#   gene    r12_gene_s<seed>   --kappa-mode gene (s_g: the dataset's
#           experiments/gene_extranuclear_share.npy)
#   density r12_density_s<seed> --kappa-mode density (third arm, author
#           2026-09-24: depth on l/A; the run dir's kappa_density.json holds
#           the area rule and the per-type nucleus-expansion fraction)
#   Amendment (coordinator, before any fit): depth and density divide their
#   ratio by its mean over the connected training cells (computed once at
#   setup; config.json kappa_ratio_mean, run dir kappa_<mode>.json), so the
#   mean leak fraction is kappa, as in the gene arm. Amendment 2: that
#   normaliser is the clip-aware m (bisection), so the mean holds after the
#   0.5 cap too.
# Per run: fit -> degeneracy -> validate morans,niche -> atlas (programme 0
# for GO) -> transport both/hvg 1000 k-means -> the same on tumour bands ->
# external_criteria signalling-share -> GO localisation (CPU, own file stem).
# Then scripts/r12_table.py --at best -> experiments/r12_arms.{json,md}.
# A dead fit is recorded and read, not refitted (failed legs are findings).
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_wcollapse_b.sh
# (gate 9500 MiB, one job of ours per GPU, MintFlow charged at its peak, a
# re-read of free memory after taking the lock and a settle delay), plus two
# courtesies to the other queues, whose locks live in their own folders: their
# FF steps go first (a card that could host one is left alone while one waits),
# and a card one of them locked less than FRESH s ago is skipped, because that
# job may not have allocated yet. Markers are keyed by step AND dataset AND run;
# a relaunch resumes. Launch detached:
#   setsid nohup scripts/queue_2026-09-24_r12_arms.sh \
#     >> scripts/logs/r12_arms_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OV=xenium_prime_ovarian_cancer_ffpe
ROOT=scripts/logs/r12_arms_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
GPUS="0 1"
NEED_OV=9500          # an ovarian fit is ~6-7 GB resident; post-hoc reads reuse it
NEED_FF=18500         # an FF fit of another queue: such a card is theirs first
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=300
SEEDS="0 1 2"
CONTROL=wfix_warmup30_aw0.1_s   # + seed
OVA="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6 --gat-sources type_only
     --w-warmup-epochs 30 --epochs 200 --patience 20 --figures-every 200"

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$OV/runs; }
marker() { echo "${1}_${OV}_${2}"; }   # marker <step> <run>
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

# -- GPU picker ---------------------------------------------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
# other queues: <log root>:<process pattern>
FOREIGN="scripts/logs/final_2026-09-24:queue_2026-09-24_final.sh
scripts/logs/wcollapse_b_2026-09-24:queue_2026-09-24_wcollapse_b.sh"

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

foreign_hold() {  # foreign_hold <gpu> <free>: another queue has first call on it
  local g=$1 free=$2 entry root pat now age
  now=$(date +%s)
  while IFS=: read -r root pat; do
    [ -n "$root" ] || continue
    pgrep -f "$pat" > /dev/null || continue           # that queue is gone
    if [ "$free" -ge "$NEED_FF" ] && ls "$root"/waiting/ff.* > /dev/null 2>&1; then
      return 0                                        # its FF step goes first
    fi
    if [ -d "$root/locks/gpu$g" ]; then
      age=$((now - $(stat -c %Y "$root/locks/gpu$g")))
      [ $age -lt $FRESH ] && return 0                 # may not have allocated yet
    fi
  done <<< "$FOREIGN"
  return 1
}

# Block until a GPU has >= $1 MiB free, no job of ours and no foreign first
# call, take it and echo its index. Everything else goes to stderr.
acquire_gpu() {
  local need=$1 g free waited=0
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      foreign_hold $g $free && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! foreign_hold $g $free; then
        echo $g; return 0
      fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    if [ $((waited % 10)) -eq 0 ]; then
      log "waiting for a GPU with ${need} MiB free (0: $(gpu_free 0), 1: $(gpu_free 1))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps --------------------------------------------------------------------
step() {  # step <marker> <cmd...>: a GPU step, retried on OOM
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $NEED_OV)
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
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

reuse() {  # reuse <step> <run> <file>: an existing output counts as the step
  local name; name=$(marker $1 $2)
  if [ -f "$DONE/$name" ]; then return 0; fi
  if [ -f "$(runs)/$2/$3" ]; then
    touch "$DONE/$name"; log "reuse $name ($3 present)"
  fi
}

fit() {   # fit <arm> <seed>
  local run=r12_${1}_s$2 mk; mk=$(marker fit r12_${1}_s$2)
  if [ -f "$(runs)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  step "$mk" uv run python -m discell.model.train --dataset $OV --run-name $run \
    $OVA --seed $2 --kappa-mode $1
}

battery() {  # battery <run>: every read of one fit
  local run=$1
  if [ ! -f "$(runs)/$run/metrics.json" ]; then
    log "no fit for $run; no reads"; return 0
  fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$(runs)/$run/metrics.json')).get('dead_w_channel') else 1)"; then
    log "$run: dead context channel flagged by the trainer -- recorded, reads continue"
    grep -qx "$run" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo $run >> $ROOT/DEAD_RUNS.txt
  fi
  step "$(marker degeneracy $run)" uv run python -m discell.model.degeneracy \
    --dataset $OV --run $run
  step "$(marker validate $run)" uv run python -m discell.model.validate \
    --dataset $OV --run $run --analyses morans,niche
  step "$(marker atlas $run)" uv run python -m discell.model.atlas \
    --dataset $OV --run $run
  step "$(marker transport $run)" uv run python -m discell.model.transport \
    --dataset $OV --run $run --read both --hvg 1000
  step "$(marker transport_band $run)" uv run python -m discell.model.transport \
    --dataset $OV --run $run --read both --hvg 1000 --niche-source tumour-band
  step "$(marker ext_signalling-share $run)" uv run python -m \
    discell.experiments.external_criteria signalling-share --dataset $OV --run $run
  cpu_step "$(marker go_localisation $run)" uv run python -m \
    discell.experiments.go_localisation --dataset $OV --run $run \
    --out-stem go_localisation_$run
  return 0
}

lane_arm() {  # lane_arm <arm>
  local s
  for s in $SEEDS; do fit $1 $s; battery r12_${1}_s$s; done
  log "$1 lane finished"
}

lane_control() {
  local s run
  for s in $SEEDS; do
    run=$CONTROL$s
    # the reads the 8.9 queue already made on these runs, reused as they are
    reuse degeneracy $run degeneracy.json
    reuse validate $run validation/validation.json
    reuse transport $run transport/transport_distribution.json
    battery $run
  done
  log "control lane finished"
}

# -- preflight: the arms' flags reproduce the control's config.json -----------
preflight() {
  uv run python - "$OVA" "$(runs)" "$CONTROL" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
flags, runs, control = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ALLOWED = {"run_name", "seed", "git", "kappa_mode", "kappa_gene_source",
           "kl_warmup_epochs"}   # the last is absent from the control: default 0
default = asdict(TrainConfig(dataset=""))
bad = 0
for arm in ("depth", "gene", "density"):
    for seed in (0, 1, 2):
        args = vars(build_parser().parse_args(
            ["--dataset", "xenium_prime_ovarian_cancer_ffpe", "--seed", str(seed),
             "--kappa-mode", arm] + shlex.split(flags)))
        args.pop("quiet")
        new = asdict(TrainConfig(**args))
        ref = json.loads((runs / f"{control}{seed}" / "config.json").read_text())
        diffs = [f"{k}: {new.get(k)!r} vs control {ref.get(k, '<absent>')!r}"
                 for k in sorted(set(new) | set(ref)) if k not in ALLOWED
                 and ((k in ref and new.get(k) != ref[k])
                      or (k not in ref and new[k] != default[k]))]
        ok = not diffs and new["kl_warmup_epochs"] == 0
        if arm == "gene":
            ok = ok and Path(new["kappa_gene_source"]).exists()
        print(f"{arm} s{seed}: " + ("identical to the control but for the "
              f"kappa form ({new['kappa_mode']}; s_g {new['kappa_gene_source']})"
              if ok else "DIFFERS: " + "; ".join(diffs)))
        bad += not ok
sys.exit(1 if bad else 0)
EOF
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
trap 'rmdir $LOCK/gpu* 2>/dev/null' EXIT
log "queue start (gate $NEED_OV MiB, GPUs: $GPUS; code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi

lane_arm depth & PD=$!
(sleep 60; lane_arm gene) & PG=$!
(sleep 90; lane_arm density) & PN=$!
(sleep 120; lane_control) & PC=$!
wait $PD $PG $PN $PC
log "all lanes finished"

log "read-out"
uv run python scripts/r12_table.py --at best > $QL/r12_table.log 2>&1
rc=$?
log "read-out done (exit $rc); see $QL/r12_table.log"
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Read-out (exit $rc): data/datasets/$OV/experiments/r12_arms.{json,md}"
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (trainer flag): $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
