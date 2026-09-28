#!/bin/bash
# 6b.2 scoring: resolVI double-positive metric (discell/experiments/marker_pairs.py)
# on finalL_s{0,1,2} + sweepL_k{0,0.05,0.2,0.3,0.4}_s{0,1,2} of the four datasets
# (kappa 0.1 = finalL via the sweepL_k0.1 symlinks), then the kappa summaries.
# One job at a time, on a GPU with > 8 GB free. Idempotent (scored runs skip).
# Launch detached:
#   setsid nohup scripts/queue_2026-09-27_marker_pairs.sh \
#     >> scripts/logs/marker_pairs_score_2026-09-27/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=scripts/logs/marker_pairs_score_2026-09-27
mkdir -p $ROOT/logs
NEED=8192
RUNS="finalL_s0 finalL_s1 finalL_s2"
for k in 0 0.05 0.2 0.3 0.4; do for s in 0 1 2; do RUNS="$RUNS sweepL_k${k}_s${s}"; done; done

log() { echo "[$(date '+%F %T')] $*"; }
pick_gpu() {   # the card with the most free memory, if above NEED
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
    [ "$free" -gt "$NEED" ] && { echo "$idx"; return; }
    sleep 60
  done
}

for ds in gse315411_pdltma06_11_prime_solo xenium_prime_human_lung_cancer_ffpe \
          xenium_prime_ovarian_cancer_ffpe xenium_prime_human_ovary_ff; do
  gpu=$(pick_gpu)
  log "$ds on GPU $gpu"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python -m discell.experiments.marker_pairs \
       --dataset $ds --run $RUNS --kappa > $ROOT/logs/$ds.log 2>&1; then
    log "$ds done"
  else
    log "$ds FAILED (see $ROOT/logs/$ds.log)"
  fi
done
log "queue finished"
