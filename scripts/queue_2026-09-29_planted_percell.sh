#!/bin/bash
# Planted per-cell response (devlog 2026-09-29, "Planted per-cell response
# (motivation, 2026-09-29; author, from the writer and reviewer)").
#
# Waits (poll every 10 min, pgrep -f) until queue_2026-09-29_synthetic.sh (the
# synthetic recovery queue, same simulator code) has ended, then runs strictly
# in sequence on GPU 0 only (CUDA_VISIBLE_DEVICES=0 on every GPU step; GPU 1 is
# never used), alongside the other GPU 0 queues, gated on >= NEED_MIB free on
# GPU 0 (twice, 30 s apart) before each fit:
#   (0) subset predictability check (CPU)
#   (1) the rule: world 0, s in {0, 0.5, 1, 2} x model seeds 0-2            12 fits
#   (2) replication: worlds 1-2, the same                                      24 fits
#   (3) aggregation -> data/datasets/synthetic_smoke/experiments/planted_percell.{json,md}
# Every fit: python -m discell.experiments.planted_percell fit (planted and
# assumed kappa 0.2; synthetic_recovery.fit_config = the final configuration
# at the recovery widths). ~1 min per fit, 36 fits ~1 h.
#
# Idempotent: a step with $ROOT/done/<step> is skipped on a relaunch; a failed
# step writes $ROOT/failed/<step> (exit code, log tail) and the queue goes on.
# A CUDA OOM is retried once.
#
#   mkdir -p scripts/logs/planted_percell_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-09-29_planted_percell.sh \
#     >> scripts/logs/planted_percell_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

ROOT=scripts/logs/planted_percell_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; FAILED=$ROOT/failed; FITS=$ROOT/fits
mkdir -p $QL $DONE $FAILED $FITS
GPU=0                      # GPU 0 only; GPU 1 is never used
NEED_MIB=3000              # a synthetic fit holds < 1 GiB incl. the CUDA context
WAIT_FOR="synthetic"
POLL=600
PY="uv run python"
MOD="-m discell.experiments.planted_percell"

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

queues_running() {  # the names of the queues we wait for that are still running
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-29_${q}\.sh" > /dev/null && out="$out $q"
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

fit() {  # fit <s> <world> <model>
  local s=$1 w=$2 m=$3
  step "fit_s${s}_w${w}_m${m}" gpu $PY $MOD fit --s $s --world-seed $w \
    --model-seed $m --out $FITS
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

step predictability cpu $PY $MOD predictability --out $ROOT/predictability.json

for w in 0 1 2; do for s in 0 0.5 1 2; do for m in 0 1 2; do
  fit $s $w $m
done; done; done

rm -f "$DONE/aggregate"          # always re-aggregate over what is present
step aggregate cpu $PY $MOD aggregate --fits $FITS --predictability $ROOT/predictability.json
log "fits: $(ls $FITS/*.json 2>/dev/null | wc -l) records; failed: $(ls $FAILED | wc -l)"
log "queue finished ($(ls $DONE | wc -l) markers)"
