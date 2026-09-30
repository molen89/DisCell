#!/bin/bash
# Re-read the ovarian axis test and its breakdown draws after the gene-pairing fix
# (devlog 2026-09-30 "Axis test paired genes by position"). finalL_s0-2 + sweepL
# (k0, 0.05, 0.2, 0.3, 0.4 x s0-2). Old files are overwritten by the reads; the
# pre-fix axis files are copied to *_prefix.* first. CPU (2 threads); axis reads
# need a forward pass, on GPU 0 when free. Breakdown tables re-rendered at the end
# unless the breakdown queue is still running (its own final step renders them).
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
ROOT=scripts/logs/axis_fix_2026-09-30; mkdir -p $ROOT/done
OV=xenium_prime_ovarian_cancer_ffpe; E=data/datasets/$OV/experiments; R=data/datasets/$OV/runs
log() { echo "[$(date '+%F %T')] $*"; }
runs="finalL_s0 finalL_s1 finalL_s2"
for k in 0 0.05 0.2 0.3 0.4; do for s in 0 1 2; do runs="$runs sweepL_k${k}_s$s"; done; done
for r in $runs; do
  [ -e $ROOT/done/axis_$r ] && continue
  for f in $E/external_axis_test_$r.json $R/$r/breakdown/axis.npz; do
    [ -e $f ] && cp -n $f ${f%.*}_prefix.${f##*.}
  done
  log "start $r"
  if uv run python -m discell.experiments.external_criteria axis-test --dataset $OV --run $r > $ROOT/axis_$r.log 2>&1 \
     && uv run python -m discell.experiments.breakdown_draws --dataset $OV --run $r --groups axis >> $ROOT/axis_$r.log 2>&1; then
    touch $ROOT/done/axis_$r; log "done $r"
  else log "FAILED $r (see $ROOT/axis_$r.log)"; fi
done
if pgrep -f queue_2026-10-01_breakdown.sh > /dev/null; then
  log "breakdown queue still running: its final step renders the tables"
else
  uv run python -m discell.experiments.breakdown --all > $ROOT/tables.log 2>&1 && log "tables re-rendered"
fi
log "queue finished"
