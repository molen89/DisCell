#!/bin/bash
# Per-cell KL and posterior-uncertainty maps (devlog "Per-cell KL and
# posterior-uncertainty maps; within-type Moran's I table (motivation,
# 2026-09-29; author, run by the writer)"). discell/experiments/kl_maps.py on
# finalL_s0-s2 of GSE, lung, ovarian, then FF last; one run at a time, a
# dataset summary after its three seeds.
#
# GPU: per run, the card with the most free memory at launch, once it has
# NEED MiB free (FF: NEED_FF, and NEED_HOST_FF GB host MemAvailable first).
# A lock dir under $ROOT/locks marks the card for the other queues' pickers.
# CUDA OOM: wait RETRY_WAIT s, retry once, then run that step on the CPU.
# Never touches another process.
#
# Idempotent: a run whose kl_maps_<run>.json exists is skipped.
#
#   mkdir -p scripts/logs/kl_maps_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-09-29_kl_maps.sh \
#     >> scripts/logs/kl_maps_2026-09-29/queue.log 2>&1 < /dev/null &
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
LU=xenium_prime_human_lung_cancer_ffpe
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff
RUNS="finalL_s0 finalL_s1 finalL_s2"

ROOT=scripts/logs/kl_maps_2026-09-29
LOCK=$ROOT/locks
mkdir -p $LOCK
NEED=9500; NEED_FF=18500; NEED_HOST_FF=60; RETRY_WAIT=600
PY="uv run python"

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

best_gpu() {  # prints "<index> <free MiB>" of the card with the most free memory
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '{print $1, $2-$3}' | sort -k2 -n -r | head -1
}

wait_gpu() {  # wait_gpu <need MiB>: prints the chosen index
  local need=$1 idx free n=0
  while true; do
    read -r idx free < <(best_gpu)
    [ "$free" -ge "$need" ] && { echo $idx; return 0; }
    [ $((n % 10)) -eq 0 ] && log "waiting for a GPU with ${need} MiB free (best: GPU$idx ${free} MiB)" >&2
    n=$((n + 1)); sleep 60
  done
}

wait_host() {
  local need=$1 avail n=0
  while true; do
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && return 0
    [ $((n % 10)) -eq 0 ] && log "waiting for ${need} GB host memory (${avail} GB available)"
    n=$((n + 1)); sleep 60
  done
}

one_run() {  # one_run <dataset> <run>
  local ds=$1 run=$2 need=$NEED try gpu rc t0 lg
  local out=data/datasets/$ds/experiments/kl_maps_$run.json
  lg=scripts/logs/kl_maps_2026-09-29_${ds}_$run.log
  if [ -f "$out" ]; then log "skip $ds/$run (done)"; return 0; fi
  [ "$ds" = "$FF" ] && need=$NEED_FF
  for try in 1 2 cpu; do
    [ "$ds" = "$FF" ] && wait_host $NEED_HOST_FF
    t0=$SECONDS
    if [ "$try" = cpu ]; then
      log "start $ds/$run on CPU"
      CUDA_VISIBLE_DEVICES= $PY -m discell.experiments.kl_maps --dataset $ds --run $run \
        --device cpu > $lg 2>&1 < /dev/null; rc=$?
    else
      gpu=$(wait_gpu $need)
      mkdir -p $LOCK/gpu$gpu
      log "start $ds/$run on GPU$gpu (try $try; $(best_gpu | awk '{print $2}') MiB free on the best card)"
      CUDA_VISIBLE_DEVICES=$gpu $PY -m discell.experiments.kl_maps --dataset $ds --run $run \
        > $lg 2>&1 < /dev/null; rc=$?
      rmdir $LOCK/gpu$gpu 2>/dev/null
    fi
    log "done  $ds/$run (exit $rc, $((SECONDS - t0)) s)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$ds/$run" "$try" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
    [ $rc -eq 0 ] && return 0
    cp $lg $lg.try$try
    if [ "$try" != cpu ] && grep -qE "OutOfMemoryError|CUDA out of memory" $lg; then
      log "CUDA OOM on $ds/$run (try $try)"; [ "$try" = 1 ] && sleep $RETRY_WAIT
      continue
    fi
    log "FAILED $ds/$run (exit $rc, not OOM) -- see $lg.try$try"; return 1
  done
  return 1
}

log "queue start (pid $$)"
for ds in $GS $LU $OV $FF; do
  ok=1
  for run in $RUNS; do one_run $ds $run || ok=0; done
  if [ $ok -eq 1 ]; then
    CUDA_VISIBLE_DEVICES= $PY -m discell.experiments.kl_maps --dataset $ds --summary \
      > scripts/logs/kl_maps_2026-09-29_${ds}_summary.log 2>&1 < /dev/null
    log "summary $ds (exit $?)"
  else
    log "summary $ds skipped: a run failed"
  fi
done
log "queue finished"
