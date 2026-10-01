#!/bin/bash
# RECOMB additions 1-2 (devlog "RECOMB story map approved; three additions
# (motivation, 2026-10-01; author)", items 1 and 2).
#
#   (1) planted spill-over positive control: discell.experiments.planted_spillover,
#       kappa_true 0.1 and 0.2 first, then the kappa_true = 0 null; the breakdown
#       grid {0, 0.05, 0.1, 0.2, 0.3, 0.4} x model seeds 0-2 each (54 fits), then
#       the aggregate -> data/datasets/synthetic_smoke/experiments/planted_spillover.*
#   (2) plain-regression reference for relocation: transport --read regression on
#       finalL_s0-s2 of the four sections (CPU only: no model is loaded), then the
#       per-dataset comparison -> experiments/transport_regression_reference.*
#   (3) READOUT.md: scripts/recomb_additions_readout.py.
#
# Devices. GPU 1 is never used (MintFlow refits). A spill-over fit takes GPU 0 only
# when GPU 0 is free: no other queue's fresh scripts/logs/*/locks/gpu0 (the
# breakdown-gaps queue holds one while each of its steps runs) and >= NEED MiB free
# after the MintFlow guard (the gaps queue's gpu_free). It then holds our own
# locks/gpu0 for the fit, so the other queues wait for it. Otherwise it runs on
# CPU (a 30k-cell fit takes ~3 min on 5 threads). The regression steps are CPU-only;
# FF steps wait for MemAvailable >= FF_HOST_GB (the FF assembly peaks ~36 GB RSS).
# OMP_NUM_THREADS=8 for the regression steps (k-means niche labels, issue T-omp).
#
# Idempotent: done/<step> markers, failed/<step> records; delete failed/<step> to
# retry on a relaunch. The aggregate, comparisons and READOUT run every time.
#
#   mkdir -p scripts/logs/recomb_additions_2026-10-01 && setsid nohup \
#     bash scripts/queue_2026-10-01_recomb_additions.sh \
#     >> scripts/logs/recomb_additions_2026-10-01/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

ROOT=scripts/logs/recomb_additions_2026-10-01
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
export ROOT QL DONE LOCK FAILED
GPU=0; NEED=3000; SETTLE=20; FRESH=600
WORKERS=3; FIT_THREADS=5
FF_HOST_GB=60
export GPU NEED SETTLE FRESH FIT_THREADS
GS=gse315411_pdltma06_11_prime_solo
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff

log() { echo "[$(date '+%F %T')] $*"; }
export -f log

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# -- GPU 0: free only when no other queue holds it (the gaps queue's conventions) --
GUARD_PATTERN='run_mintflow|baseline_battery.*MintFlow'
GUARD_PEAK=9000
export GUARD_PATTERN GUARD_PEAK
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
held_elsewhere() {
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
try_gpu() {  # 0 = we hold GPU 0 now; 1 = run on CPU
  [ "$(gpu_free)" -ge $NEED ] && ! held_elsewhere || return 1
  mkdir "$LOCK/gpu$GPU" 2>/dev/null || return 1
  sleep $SETTLE
  if [ "$(gpu_free)" -ge $NEED ] && ! held_elsewhere; then return 0; fi
  rmdir "$LOCK/gpu$GPU" 2>/dev/null; return 1
}
export -f gpu_free held_elsewhere try_gpu

# spill_step <kappa_true> <assumed> <seed>
spill_step() {
  local kt=$1 a=$2 s=$3 name="spill_p${1}_a${2}_m${3}" rc t0 dev
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  if [ -f "$FAILED/$name" ]; then log "skip $name (failed before)"; return 0; fi
  t0=$SECONDS
  local th="OMP_NUM_THREADS=$FIT_THREADS MKL_NUM_THREADS=$FIT_THREADS OPENBLAS_NUM_THREADS=$FIT_THREADS NUMBA_NUM_THREADS=$FIT_THREADS"
  if try_gpu; then
    dev=cuda
    log "GPU$GPU start $name"
    env $th CUDA_VISIBLE_DEVICES=$GPU uv run python -m discell.experiments.planted_spillover \
      fit --kappa-true $kt --assumed $a --model-seed $s --device cuda > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$GPU" 2>/dev/null
  else
    dev=cpu
    log "CPU  start $name"
    env $th CUDA_VISIBLE_DEVICES= uv run python -m discell.experiments.planted_spillover \
      fit --kappa-true $kt --assumed $a --model-seed $s --device cpu > $QL/$name.log 2>&1; rc=$?
  fi
  log "done  $name on $dev (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$dev" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; else tail -5 $QL/$name.log > "$FAILED/$name"; fi
  return 0
}
export -f spill_step

# reg_step <dataset> <run>: CPU only
reg_step() {
  local ds=$1 r=$2 name="regression_${1}_${2}" rc t0 n=0
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  if [ -f "$FAILED/$name" ]; then log "skip $name (failed before)"; return 0; fi
  if [ "$ds" = "$FF" ]; then
    while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt $FF_HOST_GB ]; do
      [ $((n % 10)) -eq 0 ] && log "waiting for $FF_HOST_GB GB host MemAvailable"
      n=$((n + 1)); sleep 60
    done
  fi
  t0=$SECONDS
  log "CPU  start $name"
  env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES= uv run python -m discell.model.transport --dataset $ds --run $r \
    --read regression --device cpu > $QL/$name.log 2>&1; rc=$?
  log "done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "cpu" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; else tail -5 $QL/$name.log > "$FAILED/$name"; fi
}

spill_list() {
  local kt a s
  for kt in 0.1 0.2 0; do
    for a in 0 0.05 0.1 0.2 0.3 0.4; do
      for s in 0 1 2; do echo "$kt $a $s"; done
    done
  done
}

# -- main -------------------------------------------------------------------------
log "queue start (pid $$): spill-over fits on GPU $GPU when free else CPU ($WORKERS workers); regression on CPU"
rmdir $LOCK/gpu$GPU 2>/dev/null     # one instance at a time: a lock of ours is stale

( spill_list | xargs -P $WORKERS -L 1 bash -c 'spill_step "$@"' _ ) &
SPILL_PID=$!

for ds in $GS $OV $LU $FF; do
  for r in finalL_s0 finalL_s1 finalL_s2; do reg_step $ds $r; done
  log "CPU  compare $ds"
  env OMP_NUM_THREADS=2 uv run python -m discell.model.transport_regression compare \
    --dataset $ds > $QL/compare_$ds.log 2>&1 || log "FAILED compare $ds"
done

wait $SPILL_PID
log "CPU  aggregate spill-over"
env OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= uv run python -m discell.experiments.planted_spillover \
  aggregate > $QL/aggregate_spillover.log 2>&1 || log "FAILED aggregate (see $QL/aggregate_spillover.log)"
env OMP_NUM_THREADS=2 uv run python scripts/recomb_additions_readout.py > $QL/readout.log 2>&1 \
  || log "FAILED readout (see $QL/readout.log)"
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
