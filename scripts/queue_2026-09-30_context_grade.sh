#!/bin/bash
# Context-latent grading for every method (devlog 2026-09-30, "Context-latent
# grading for every method"). CPU only: the GPUs are hidden. Per section, the
# k-means niches at OMP_NUM_THREADS=8 (the thread count the w guard was read at),
# then the grading at 2 threads. GSE core, serial section, ovarian, lung; FF last,
# gated on host MemAvailable. Idempotent: a finished section is skipped.
set -u
cd /home/rmolen/github/DisCell
export NUMBA_NUM_THREADS=2 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=""
ROOT=scripts/logs/context_grade_2026-09-30; mkdir -p $ROOT/done
GSE=gse315411_pdltma06_11_prime_solo; DUAL=gse315411_pdltma06_10_prime_dual
FF=xenium_prime_human_ovary_ff
FF_MIN_GB=60
log() { echo "[$(date '+%F %T')] $*"; }
threads() { export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1; }
section() {   # dataset [config-from]
  local ds=$1 from=${2:-}; local extra=""; [ -n "$from" ] && extra="--config-from $from"
  [ -e $ROOT/done/$ds ] && { log "skip $ds (done)"; return; }
  log "start $ds"
  threads 8
  if uv run python -m discell.experiments.context_grade niches --dataset $ds $extra > $ROOT/$ds.log 2>&1 \
     && threads 2 \
     && uv run python -m discell.experiments.context_grade grade --dataset $ds $extra >> $ROOT/$ds.log 2>&1; then
    touch $ROOT/done/$ds; log "done $ds"
  else log "FAILED $ds (see $ROOT/$ds.log)"; fi
}
section $GSE
section $DUAL $GSE
section xenium_prime_ovarian_cancer_ffpe
section xenium_prime_human_lung_cancer_ffpe
until [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge $FF_MIN_GB ]; do
  log "FF waits: MemAvailable below $FF_MIN_GB GB"; sleep 300
done
section $FF
log "queue finished"
