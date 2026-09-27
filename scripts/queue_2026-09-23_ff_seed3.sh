#!/bin/bash
# The dead-context-channel follow-up of 2026-09-23 (devlog "FF MI follow-up on
# seeds 1 and 2"): FF best_s2's context channel never opened, so
#
#   stage guard    -- the new w-channel guard, post hoc on every `best_s*` fit
#                     of every dataset (it rewrites runs/<run>/degeneracy.json,
#                     adding the "w_channel" block; nothing else is touched)
#   stage battery  -- the lane-1 battery on the fresh FF seed 3
#   stage reads    -- the FF envelope table, the FF DisCell battery column and
#                     the two external reads, all for best_s3
#
# Idempotent: every step writes scripts/logs/ff_seed3_2026-09-23/<step>.log and
# is skipped when that log ends in DONE. A failing step is logged FAILED and
# the queue continues.
#
#   setsid nohup scripts/queue_2026-09-23_ff_seed3.sh guard battery \
#       > scripts/logs/ff_seed3_2026-09-23/queue.out 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
# GPU 0 is shared with the w-collapse lane; the guard stage is cheap enough on
# the CPU (4 min on the 407k-cell ovarian slide) that it does not queue behind
# it. Override with GUARD_DEVICE=cuda GUARD_CUDA=0 when the GPU is free.
GUARD_DEVICE=${GUARD_DEVICE:-cpu}
GUARD_CUDA=${GUARD_CUDA:-}
# Same story for the battery: the FF slide's resident tiles are 7.5 GB, the
# other lane holds 16 GB, and the card is 23.5 GB -- the two do not fit, and a
# step loses the race during its three-minute assemble. cpu is the default so
# the leg finishes; BATTERY_DEVICE=cuda BATTERY_CUDA=0 when GPU 0 is our own.
BATTERY_DEVICE=${BATTERY_DEVICE:-cpu}
BATTERY_CUDA=${BATTERY_CUDA:-}
DEV=(--device $BATTERY_DEVICE)
ENVP=(env CUDA_VISIBLE_DEVICES=$BATTERY_CUDA)

QL=scripts/logs/ff_seed3_2026-09-23; mkdir -p $QL
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff
LU=xenium_prime_human_lung_cancer_ffpe
GS=gse315411_pdltma06_11_prime_solo

log() { echo "[$(date '+%F %T')] $*" | tee -a $QL/queue.log; }

#: attempts per step. GPU 0 is shared, and a step that loses the race for the
#: resident tiles dies on an OOM in its first two minutes; the retry waits for
#: room again rather than abandoning the leg.
TRIES=${TRIES:-6}

step() {                      # step <name> <command...>
  local name=$1; shift
  if grep -q '^DONE' $QL/$name.log 2>/dev/null; then log "skip $name (DONE)"; return 0; fi
  local try code t0
  for try in $(seq 1 $TRIES); do
    [ $try -gt 1 ] && wait_gpu
    log "start $name (try $try/$TRIES): $*"
    t0=$SECONDS
    if "$@" > $QL/$name.log 2>&1; then
      echo DONE >> $QL/$name.log; log "done  $name ($((SECONDS - t0))s)"
      return 0
    fi
    code=$?
    log "FAILED $name (exit $code, $((SECONDS - t0))s, try $try/$TRIES) -- $QL/$name.log"
    grep -q "OutOfMemoryError" $QL/$name.log || break   # not contention: stop
  done
  echo "FAILED (exit ${code:-1})" >> $QL/$name.log
  return 0; }

runs() { echo data/datasets/$1/runs; }

# GPU 0 is shared with the w-collapse lane. The FF slide is 1.15M cells and
# its resident tiles need ~8 GB, so a step that starts while the other lane
# holds 16 GB dies on an OOM half an hour in. Wait for room instead.
NEED_MIB=${NEED_MIB:-11000}
WAIT_GPU_MAX=${WAIT_GPU_MAX:-14400}      # seconds; then run anyway and log it
wait_gpu() {
  [ "$BATTERY_DEVICE" = cpu ] && return 0      # nothing to wait for
  local t0=$SECONDS free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "${free:-0}" -ge "$NEED_MIB" ] && { log "GPU 0: ${free} MiB free, go"; return 0; }
    if [ $((SECONDS - t0)) -ge $WAIT_GPU_MAX ]; then
      log "GPU 0: still only ${free} MiB free after ${WAIT_GPU_MAX}s -- going anyway"
      return 0; fi
    sleep 60
  done; }

stage_guard() {
  # best_s3 first: it is the decision the rest of the night waits on
  for spec in "$FF:best_s3" \
              "$FF:best_s0" "$FF:best_s1" "$FF:best_s2" \
              "$OV:best_s0" "$OV:best_s1" "$OV:best_s2" \
              "$LU:best_s0" "$LU:best_s1" "$LU:best_s2" \
              "$GS:best_s0" "$GS:best_s1" "$GS:best_s2"; do
    ds=${spec%%:*}; run=${spec##*:}
    [ -e $(runs $ds)/$run/best.pt ] || { log "no $ds/$run -- guard skipped"; continue; }
    step guard_${ds}_${run} env CUDA_VISIBLE_DEVICES=$GUARD_CUDA \
      uv run python -m discell.model.degeneracy \
      --dataset $ds --run $run --device $GUARD_DEVICE
  done; }

stage_battery() {
  wait_gpu; step validate_${FF}_best_s3 "${ENVP[@]}" \
    uv run python -m discell.model.validate \
    --dataset $FF --run best_s3 --analyses morans,niche "${DEV[@]}"
  wait_gpu
  step atlas_${FF}_best_s3 "${ENVP[@]}" uv run python -m discell.model.atlas \
    --dataset $FF --run best_s3 --compare-runs best_s0 best_s1 "${DEV[@]}"
  wait_gpu
  step transport_${FF}_best_s3 "${ENVP[@]}" uv run python -m discell.model.transport \
    --dataset $FF --run best_s3 --read both --hvg 1000 "${DEV[@]}"
  wait_gpu
  step report_${FF}_best_s3 "${ENVP[@]}" uv run python -m discell.model.report \
    --dataset $FF --run best_s3 "${DEV[@]}"; }

stage_reads() {
  step envelope_${FF} uv run python scripts/envelope_tables.py \
    --datasets $FF --runs best_s0 best_s1 best_s3 \
    --combined scratchpad/envelope_table_ff_s3.md
  wait_gpu
  step battery_discell_${FF}_best_s3 "${ENVP[@]}" \
    uv run python -m discell.experiments.baseline_battery \
    --dataset $FF --discell-run best_s3
  wait_gpu
  step external_share_${FF}_best_s3 "${ENVP[@]}" \
    uv run python -m discell.experiments.external_criteria \
    signalling-share --dataset $FF --run best_s3 "${DEV[@]}"
  wait_gpu
  step external_mi_${FF}_best_s3 "${ENVP[@]}" \
    uv run python -m discell.experiments.external_criteria \
    mi-quadrant --dataset $FF --run best_s3 "${DEV[@]}"; }

for stage in "$@"; do
  log "=== stage $stage ==="
  stage_$stage
done
log "queue finished ($*)"
