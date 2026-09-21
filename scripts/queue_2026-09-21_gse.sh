#!/bin/bash
# Follow-up to queue_2026-09-17: the GSE core legs (all six died on the
# type_degeneracy index bug, fixed 2026-09-21), the cross-slide leg, and a
# --report-only regeneration of every sweep report (the report now reads the
# pooled cycle R^2 instead of r2_mean_types). Idempotent; relaunch to resume.
set -u; cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
OV=xenium_prime_ovarian_cancer_ffpe; LU=xenium_prime_human_lung_cancer_ffpe; FF=xenium_prime_human_ovary_ff
GS=gse315411_pdltma06_11_prime_solo; GD=gse315411_pdltma06_10_prime_dual
QL=data/queue_logs; AW="0.02 0.03 0.05 0.07 0.1 0.2 0.3"; KA="0 0.05 0.1 0.2 0.3 0.4"; DW="2 3 6 8"
GSA="--variant pdl018d --alpha-z 0.0036 --tile-cells 2048 --epochs 200 --patience 20 --figures-every 200"
log() { echo "[$(date '+%F %T')] $*"; }
step() { local gpu=$1 name=$2; shift 2; log "GPU$gpu start $name"; CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; log "GPU$gpu done  $name (exit $?)"; }
sweep() { local gpu=$1 ds=$2 param=$3 vals=$4 seeds=$5; shift 5; step $gpu "${ds}_${param}_s${seeds// /}_v2" uv run python -m discell.model.sweep --dataset $ds --param $param --values $vals --seeds $seeds --tag sweep3 "$@"; }
laneA() { sweep 0 $GS alpha_w "$AW" "0 1" $GSA; sweep 0 $GS kappa "$KA" "0 1" $GSA; sweep 0 $GS d_w "$DW" "0 1" $GSA; touch $QL/laneA2.done; }
laneB() { sleep 120; sweep 1 $GS alpha_w "$AW" "2" $GSA; sweep 1 $GS kappa "$KA" "2" $GSA; sweep 1 $GS d_w "$DW" "2" $GSA
  # regenerate the finished datasets' reports with the pooled cycle key while lane A finishes GSE
  for p in alpha_w kappa; do v=$AW; [ $p = kappa ] && v=$KA; step 1 "report2_${OV}_$p" uv run python -m discell.model.sweep --dataset $OV --param $p --values $v --tag sweep3 --report-only; done
  for p in alpha_w kappa d_w; do v=$AW; [ $p = kappa ] && v=$KA; [ $p = d_w ] && v=$DW
    step 1 "report2_${LU}_$p" uv run python -m discell.model.sweep --dataset $LU --param $p --values $v --tag sweep3 --label-key graphclust --report-only
    step 1 "report2_${FF}_$p" uv run python -m discell.model.sweep --dataset $FF --param $p --values $v --tag sweep3 --label-key graphclust --report-only; done
  touch $QL/laneB2.done; }
laneA & laneB & wait
for p in alpha_w kappa d_w; do v=$AW; [ $p = kappa ] && v=$KA; [ $p = d_w ] && v=$DW
  step 0 "report2_${GS}_$p" uv run python -m discell.model.sweep --dataset $GS --param $p --values $v --tag sweep3 --variant pdl018d --tile-cells 2048 --report-only; done
for r in data/datasets/$GS/runs/sweep3_*; do [ -f $r/metrics.json ] && step 1 "crossslide2_$(basename $r)" uv run python -m discell.model.crossslide --dataset $GS --run $(basename $r) --eval-dataset $GD; done
log "queue2 complete"
