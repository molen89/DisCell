#!/bin/bash
# Re-run the sweepL sweep report after the split fix in discell/model/sweep.py
# (devlog 2026-09-29 "Sweep report graded seeds 1-2 on seed 0's split"). The report
# assembled the data once with seed 0, while every sweepL fit (one train.py call per
# fit) drew its own split, so the I(niche;w) guard, per-type |w| and recon strata of
# seeds 1-2 were read on the wrong held-out cells. The old files are kept as
# kappa_sweep_sweepL_sharedsplit.json. GPU 0 only.
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
ROOT=scripts/logs/sweep_report_fix_2026-09-29; mkdir -p $ROOT/done
GS=gse315411_pdltma06_11_prime_solo; OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe; FF=xenium_prime_human_ovary_ff
declare -A F=([$GS]="--variant pdl018d --tile-cells 2048 --alpha-z 0.0018"
  [$OV]="--alpha-z 0.0035" [$LU]="--alpha-z 0.002" [$FF]="--alpha-z 0.00035")
log() { echo "[$(date '+%F %T')] $*"; }
for ds in $GS $OV $LU $FF; do
  [ -e $ROOT/done/$ds ] && { log "$ds done -- skipped"; continue; }
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) -lt 6000 ]; do sleep 60; done
  if [ $ds = $FF ]; then
    while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 60 ]; do sleep 120; done
  fi
  e=data/datasets/$ds/experiments
  cp -n $e/kappa_sweep_sweepL.json $e/kappa_sweep_sweepL_sharedsplit.json
  log "start $ds"
  # shellcheck disable=SC2086
  if uv run python -m discell.model.sweep --dataset $ds --report-only --tag sweepL \
      --param kappa --values 0 0.05 0.1 0.2 0.3 0.4 --seeds 0 1 2 --label-key lineage \
      ${F[$ds]} --epochs 500 --patience 40 > $ROOT/$ds.log 2>&1; then
    touch $ROOT/done/$ds; log "done $ds"
  else log "FAILED $ds (see $ROOT/$ds.log)"; fi
done
log "queue finished"
