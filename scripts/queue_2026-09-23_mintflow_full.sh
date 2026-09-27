#!/usr/bin/env bash
# Full MintFlow fit on the GSE core at MintFlow's default width (50 epochs, ~14 h),
# transfer to the held-out dual section, battery columns. Waits for the collapse
# grid to release a GPU (DECISION.json or the wcollapse queue gone) and >= 8 GB free.
set -u
BASE=/home/rmolen/github/DisCell-baselines; DISCELL=/home/rmolen/github/DisCell
DATA=$BASE/data; RES=$BASE/results
GSE=gse315411_pdltma06_11_prime_solo; DUAL=gse315411_pdltma06_10_prime_dual
L=$DISCELL/scripts/logs/mintflow_full_2026-09-23; mkdir -p $L
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L/queue.log; }
pick_gpu(){ for g in 0 1; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ "$free" -ge 8000 ] && { echo $g; return 0; }; done; return 1; }
log "waiting for the collapse grid to finish (DECISION.json) or exit"
until [ -f $DISCELL/scripts/logs/wcollapse_2026-09-23/DECISION.json ] || ! pgrep -f queue_2026-09-23_wcollapse.sh >/dev/null; do sleep 600; done
until GPU=$(pick_gpu); do sleep 300; done
log "GPU$GPU: start mintflow full fit (width 600, 50 epochs, cap 20 h)"
OUT=$RES/mintflow/${GSE}_full
if [ ! -f $OUT/latents.h5ad ]; then
  cd $BASE/mintflow && CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python run_mintflow.py --h5ad $DATA/$GSE.h5ad --out $OUT \
    --epochs 50 --width-window 600 --max-hours 20 --transfer-h5ad $DATA/$DUAL.h5ad --transfer-out $RES/mintflow/${DUAL}_transfer_full > $L/fit.log 2>&1
  log "fit exit $?"
fi
cd $DISCELL
CUDA_VISIBLE_DEVICES=$GPU uv run python -m discell.experiments.baseline_battery --dataset $GSE --method "MintFlow (50 epochs, w=600)" --latents $OUT/latents.h5ad > $L/battery_gse.log 2>&1; log "battery GSE exit $?"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m discell.experiments.baseline_battery --dataset $DUAL --method "MintFlow (transfer, 50 epochs)" --latents $RES/mintflow/${DUAL}_transfer_full/latents.h5ad --config-from $GSE > $L/battery_dual.log 2>&1; log "battery dual exit $?"
log "queue finished"
