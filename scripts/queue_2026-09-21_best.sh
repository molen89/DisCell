#!/usr/bin/env bash
# Per-dataset "best" runs (2026-09-21): the tuned operating point
# kappa 0.1, d_w 6, alpha_w 0.1, alpha_a 0.3, type_only sources, 500/40, seed 0
# (ovarian: seed 1, the run every 2026-09 experiment was read on).
# Three exist already and get a `best` symlink; lung is refit because its
# reference predates type_only. Idempotent: skips steps whose log says done.
set -u
cd /home/rmolen/github/DisCell
QL=scripts/logs/best_2026-09-21; mkdir -p $QL
log() { echo "$(date '+%F %T') $*" | tee -a $QL/queue.log; }
step() { local gpu=$1 name=$2; shift 2
  if grep -q "^DONE" $QL/$name.log 2>/dev/null; then log "skip $name"; return; fi
  log "GPU$gpu start $name"; CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1 && echo DONE >> $QL/$name.log
  log "GPU$gpu done  $name (exit $?)"; }
link() { local ds=$1 src=$2; local d=data/datasets/$ds/runs
  [ -e $d/best ] || ln -s $src $d/best; log "$ds/runs/best -> $(readlink $d/best)"; }

link xenium_prime_ovarian_cancer_ffpe ablation_gat_type_only_s1
link xenium_prime_human_ovary_ff      reference_graphclust
link gse315411_pdltma06_11_prime_solo reference

LU=xenium_prime_human_lung_cancer_ffpe
step 1 fit_lung_best uv run python -m discell.model.train --dataset $LU --run-name best \
  --label-key graphclust --alpha-z 0.004 --gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 \
  --epochs 500 --patience 40 --figures-every 100 --seed 0
step 1 validate_lung_best  uv run python -m discell.model.validate  --dataset $LU --run best --analyses morans,niche
step 1 atlas_lung_best     uv run python -m discell.model.atlas     --dataset $LU --run best
step 1 transport_lung_best uv run python -m discell.model.transport --dataset $LU --run best --read both
step 1 report_lung_best    uv run python -m discell.model.report    --dataset $LU --run best
log "queue finished"
