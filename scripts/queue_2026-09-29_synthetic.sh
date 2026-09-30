#!/bin/bash
# Synthetic recovery at the final configuration (devlog 2026-09-29, "Synthetic
# recovery at the final configuration (motivation, 2026-09-29; author)").
#
# Waits (poll every 10 min, pgrep -f) for the freeze only, i.e. until neither
#   queue_2026-09-28_unassigned.sh nor queue_2026-09-28_sens_final.sh
# is running (the detection of queue_2026-09-29_baselines_complete.sh), then
# runs strictly in sequence on GPU 0 only (CUDA_VISIBLE_DEVICES=0 on every GPU
# step; GPU 1 is never used), alongside the other GPU 0 queues, gated on
# >= NEED_MIB free on GPU 0 (twice, 30 s apart) before each fit:
#   (0) simulator checks (CPU)
#   (1) final arm, planted = assumed kappa in {0, 0.2} x worlds 0-2 x model seeds 0-2   18 fits
#   (2) uncontrolled arm (alpha_a = 0), the same 18 cells                                  18 fits
#   (3) misspecified kappa: planted 0.2, assumed {0, 0.1, 0.3, 0.4} x 3 x 3                36 fits
#   (4) aggregation -> data/datasets/synthetic_smoke/experiments/synthetic_recovery.{json,md}
# Every fit: python -m discell.experiments.synthetic_recovery fit (TrainConfig
# defaults = the final configuration; widths d_z 8 / d_w 2 / hidden 128 /
# attention 16, tiles 512, alpha_z = 0.5 / median count, kappa explicit).
# Smoke: 40 epochs in 0.3 min, peak 27 MiB allocated; est. ~3-5 min per full
# fit (500/40), 72 fits ~4-6 h on a shared GPU 0.
#
# Idempotent: a step with $ROOT/done/<step> is skipped on a relaunch; a failed
# step writes $ROOT/failed/<step> (exit code, log tail) and the queue goes on.
# A CUDA OOM is retried once.
#
#   mkdir -p scripts/logs/synthetic_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-09-29_synthetic.sh \
#     >> scripts/logs/synthetic_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

ROOT=scripts/logs/synthetic_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; FAILED=$ROOT/failed; FITS=$ROOT/fits
mkdir -p $QL $DONE $FAILED $FITS
GPU=0                      # GPU 0 only; GPU 1 is never used
NEED_MIB=3000              # a synthetic fit holds < 1 GiB incl. the CUDA context
WAIT_FOR="unassigned sens_final"
POLL=600
PY="uv run python"
MOD="-m discell.experiments.synthetic_recovery"

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

queues_running() {  # the names of the queues we wait for that are still running
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-28_${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}

gpu_free() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
    -i $GPU | awk -F', ' '{print $1-$2}'
}

wait_gpu() {  # >= NEED_MIB free on GPU 0, twice 30 s apart
  local n=0 f
  while true; do
    f=$(gpu_free)
    if [ "$f" -ge $NEED_MIB ]; then
      sleep 30; f=$(gpu_free)
      [ "$f" -ge $NEED_MIB ] && return 0
    fi
    [ $((n % 10)) -eq 0 ] && log "waiting for $NEED_MIB MiB free on GPU $GPU ($f free)"
    n=$((n + 1)); sleep 60
  done
}

step() {  # step <name> <gpu|cpu> <cmd...>
  local name=$1 kind=$2 try rc t0; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  for try in 1 2; do
    if [ "$kind" = gpu ]; then wait_gpu; dev=$GPU; else dev=""; fi
    log "start $name (try $try, GPU '${dev}'): $*"
    t0=$SECONDS
    env CUDA_VISIBLE_DEVICES=$dev "$@" > "$QL/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "ok $((SECONDS - t0)) s" > "$DONE/$name"; rm -f "$FAILED/$name"
      log "done $name ($((SECONDS - t0)) s)"; return 0
    fi
    if [ $try -eq 1 ] && grep -qE "CUDA out of memory|OutOfMemoryError" "$QL/$name.log"; then
      log "$name: CUDA OOM -- retrying once"; cp "$QL/$name.log" "$QL/$name.try1.log"; continue
    fi
    { echo "exit $rc after $((SECONDS - t0)) s (try $try), $(date '+%F %T')"
      tail -n 30 "$QL/$name.log"; } > "$FAILED/$name"
    log "FAILED $name (exit $rc); see $FAILED/$name"
    return 1
  done
}

fit() {  # fit <planted> <assumed> <world> <model> [--uncontrolled]
  local p=$1 a=$2 w=$3 m=$4 extra=${5:-}
  local name="fit_p${p}_a${a}_w${w}_m${m}${extra:+_unc}"
  # shellcheck disable=SC2086
  step "$name" gpu $PY $MOD fit --planted $p --assumed $a --world-seed $w \
    --model-seed $m --out $FITS $extra
}

# -- main -------------------------------------------------------------------------
log "queue start (pid $$, ppid $PPID): GPU $GPU only, >= $NEED_MIB MiB free per fit; waiting for: $WAIT_FOR"
n=0
while true; do
  running=$(queues_running)
  [ -z "$running" ] && break
  [ $((n % 6)) -eq 0 ] && log "waiting: still running:$running"
  n=$((n + 1)); sleep $POLL
done
log "no queue we wait for is running -- starting (GPU $GPU: $(gpu_free) MiB free)"

step checks cpu $PY $MOD checks --out $ROOT/simulator_checks.json

for p in 0 0.2; do for w in 0 1 2; do for m in 0 1 2; do
  fit $p $p $w $m
done; done; done
for p in 0 0.2; do for w in 0 1 2; do for m in 0 1 2; do
  fit $p $p $w $m --uncontrolled
done; done; done
for a in 0 0.1 0.3 0.4; do for w in 0 1 2; do for m in 0 1 2; do
  fit 0.2 $a $w $m
done; done; done

rm -f "$DONE/aggregate"          # always re-aggregate over what is present
step aggregate cpu $PY $MOD aggregate --fits $FITS --checks $ROOT/simulator_checks.json
log "fits: $(ls $FITS | wc -l) records; failed: $(ls $FAILED | wc -l)"
log "queue finished ($(ls $DONE | wc -l) markers)"
