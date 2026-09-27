#!/bin/bash
# The w-collapse robustness grid (devlog 2026-09-23 pre-registration).
#
# Two lanes -- FF (32 fits) and ovarian (12) -- each strictly sequential, but
# neither is pinned to a GPU: every step takes whichever GPU has enough free
# memory at launch, so the two lanes run concurrently when both cards are free
# and the short ovarian fits fill the gaps while FF waits for a big one. One
# job per GPU at a time, enforced by a lock directory.
#
# Every step is idempotent: a DONE marker (and, for fits, metrics.json) skips
# it, so relaunching the script resumes. Launch detached:
#   setsid nohup scripts/queue_2026-09-23_wcollapse.sh \
#     >> scripts/logs/wcollapse_2026-09-23/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
# allocator-only setting: less fragmentation on a GPU shared with other agents
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FF=xenium_prime_human_ovary_ff
OV=xenium_prime_ovarian_cancer_ffpe
ROOT=scripts/logs/wcollapse_2026-09-23
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks
mkdir -p $QL $DONE $LOCK
GPUS="0 1"
# measured on this box: an FF 200/20 fit is ~17.2 GB resident (the whole slide
# goes to device), an ovarian fit over 6.1 GB. Asked for with headroom.
NEED_FF=18500
NEED_OV=9500
# validate/degeneracy/transport each run a full-slide forward pass, so they
# want about what a fit of that slide wants -- measured: validate OOMed at
# 6.36 GB on ovarian. They therefore reuse the per-dataset fit gate.
# the GPUs are shared with other agents, so an OOM is usually a neighbour's
# spike rather than a real failure
TRIES=10

# the shared operating point; alpha_z is each slide's pinned best_s1 value / 2
FFA="--label-key graphclust --alpha-z 0.00035 --kappa 0.1 --d-w 6
     --gat-sources type_only --epochs 200 --patience 20 --figures-every 200"
OVA="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6
     --gat-sources type_only --epochs 200 --patience 20 --figures-every 200"

log() { echo "[$(date '+%F %T')] $*"; }

arm_flags() {   # arm_flags <arm>
  case "$1" in
    control)            echo "" ;;
    warmup30)           echo "--w-warmup-epochs 30" ;;
    fb0.05)             echo "--w-free-bits 0.05" ;;
    warmup30+fb0.05)    echo "--w-warmup-epochs 30 --w-free-bits 0.05" ;;
    *) log "unknown arm $1"; exit 2 ;;
  esac
}

gpu_free() {    # free MiB on GPU $1
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
    -i $1 | awk -F', ' '{print $1-$2}'
}

# Block until some GPU has >= $1 MiB free AND is not already running one of our
# jobs, then take it. Echoes the index on stdout -- every other word this
# function says goes to stderr, or it would be read as the index.
acquire_gpu() {
  local need=$1 g free waited=0
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # someone of ours has it
      # re-read after taking the lock: a neighbour may have just grown
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ]; then echo $g; return 0; fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    if [ $((waited % 10)) -eq 0 ]; then
      log "waiting for a GPU with ${need} MiB free (0: $(gpu_free 0), 1: $(gpu_free 1))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

step() {  # step <need_MiB> <marker> <cmd...>
  local need=$1 name=$2; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try rc gpu
  for try in $(seq $TRIES); do
    gpu=$(acquire_gpu $need)
    log "GPU$gpu start $name (try $try/$TRIES)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1
    rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
    log "GPU$gpu FAIL $name (exit $rc); see $QL/$name.log"
    sleep 300
  done
  log "GIVING UP on $name after $TRIES tries"
  return 1
}

# DONE markers MUST be dataset-qualified: the FF alpha_w 0.1 block and the
# ovarian block use identical run names (wfix_<arm>_aw0.1_s<seed>) in different
# dataset directories, so an unqualified marker made the ovarian runs silently
# skip the 12 FF fits of seeds 0-2. Found 2026-09-23 when sub-lane B jumped
# straight to seed 3.
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>

fit() {   # fit <dataset> <run> <flags...>
  local ds=$1 run=$2; shift 2
  local mk=$(marker fit $ds $run)
  if [ -f "data/datasets/$ds/runs/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  local need=$NEED_FF
  [ "$ds" = "$OV" ] && need=$NEED_OV
  step $need "$mk" uv run python -m discell.model.train \
    --dataset $ds --run-name $run "$@"
}

post() {  # post <dataset> <run> [transport]
  local ds=$1 run=$2 tr=${3:-no}
  [ -f "$DONE/$(marker fit $ds $run)" ] || {
    log "no fit for $ds/$run; skipping post"; return 0; }
  local need=$NEED_FF
  [ "$ds" = "$OV" ] && need=$NEED_OV
  step $need "$(marker validate $ds $run)" uv run python -m discell.model.validate \
    --dataset $ds --run $run --analyses morans,niche
  step $need "$(marker degeneracy $ds $run)" uv run python -m discell.model.degeneracy \
    --dataset $ds --run $run
  [ "$tr" = "transport" ] && step $need "$(marker transport $ds $run)" \
    uv run python -m discell.model.transport --dataset $ds --run $run --read both
  return 0
}

ARMS="control warmup30 fb0.05 warmup30+fb0.05"

# Arms are interleaved within each seed, so a four-arm comparison exists after
# four fits instead of after sixteen (coordinator, 2026-09-23). alpha_w 0.3
# still comes first: it is the certain-death setting and the fastest verdict.
#
# The FF work is split over two sub-lanes on DISJOINT seeds so both GPUs are
# used once the ovarian lane finishes; a single sequential FF lane would leave
# one card idle for ~30 fits. Both sub-lanes start at alpha_w 0.3, so the 0.3
# block -- the one that decides the collapse question -- completes at roughly
# twice the rate. Disjoint seeds mean no fit can be attempted twice.
ff_block() {   # ff_block <alpha_w> <seed>... : four arms per seed, in order
  local aw=$1; shift
  local s arm run
  for s in "$@"; do
    for arm in $ARMS; do
      run="wfix_${arm}_aw${aw}_s${s}"
      fit $FF "$run" $FFA --alpha-w $aw --seed $s $(arm_flags $arm)
      post $FF "$run"
    done
  done
}

# Phase order (coordinator, 2026-09-23), no change to the grid itself:
#   1. alpha_w 0.3 seed 0  -- the stress block already in flight, finished first
#   2. the whole alpha_w 0.1 block, seeds 0-3 -- alpha_w 0.1 is the PINNED
#      setting, so its verdict decides the re-pin and should land first
#   3. the remaining alpha_w 0.3 stress seeds 1-3
# Two sub-lanes over disjoint (alpha_w, seed) blocks so both GPUs are used and
# no fit can be attempted twice.
# alpha_w 0.1 seeds are dealt alternately to the two sub-lanes so that seeds 0
# and 1 -- the pair the four-arm table needs -- run CONCURRENTLY rather than
# serially in one lane: the table lands after four fits per lane instead of
# eight in one. 16 fits per lane, disjoint.
laneFF_A() {
  ff_block 0.3 0      # phase 1: finish the seed-0 stress block
  ff_block 0.1 0      # phase 2: pinned setting, seed 0 (with B's seed 1)
  ff_block 0.1 2
  ff_block 0.3 1      # phase 3: remaining stress seeds
  log "FF sub-lane A finished"
}

laneFF_B() {
  ff_block 0.1 1      # phase 2: pinned setting, seed 1 (with A's seed 0)
  ff_block 0.1 3
  ff_block 0.3 2      # phase 3
  ff_block 0.3 3
  log "FF sub-lane B finished"
}

laneOV() {
  for s in 0 1 2; do
    for arm in $ARMS; do
      run="wfix_${arm}_aw0.1_s${s}"
      fit $OV "$run" $OVA --seed $s $(arm_flags $arm)
      post $OV "$run" transport
    done
  done
  log "ovarian lane finished"
}

# only one queue instance runs at a time, so any lock here is stale
rmdir $LOCK/gpu* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, ovarian $NEED_OV MiB, GPUs: $GPUS)"

laneFF_A & PFF_A=$!
laneFF_B & PFF_B=$!
laneOV & POV=$!
wait $PFF_A $PFF_B $POV
log "FF lane finished"

# the read-out, and the hand-off another queued job waits on. It withholds
# DECISION.json while any arm is still undecided, so a partial grid cannot
# start the downstream job on the wrong arm.
log "read-out"
uv run python scripts/wcollapse_table.py --tag wfix > $QL/wcollapse_table.log 2>&1
log "read-out done (exit $?); see $QL/wcollapse_table.log"
log "queue finished"
