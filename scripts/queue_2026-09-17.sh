#!/bin/bash
# DisCell unattended queue, written 2026-09-17 for a 3-4 day window.
# Two lanes (GPU 0 / GPU 1), each strictly sequential; every fit is idempotent
# (a run with metrics.json is skipped), so relaunching this script resumes.
# Launch detached:
#   setsid nohup scripts/queue_2026-09-17.sh > data/queue_2026-09-17.log 2>&1 < /dev/null & disown
# Order per docs/sweep_programme.md 5 with the author's decisions of 2026-09-17
# (docs/todo.md 0): alpha_w legs first (robust to the pending alpha_w = 0.05
# verdict), then kappa, then d_w; GSE core -> lung -> FF (500/40 budget).
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
QL=data/queue_logs; mkdir -p $QL
AW="0.02 0.03 0.05 0.07 0.1 0.2 0.3"
KA="0 0.05 0.1 0.2 0.3 0.4"
DW="2 3 6 8"

log() { echo "[$(date '+%F %T')] $*"; }
step() { # step <gpu> <name> <cmd...>
  local gpu=$1 name=$2; shift 2
  log "GPU$gpu start $name"
  CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1
  log "GPU$gpu done  $name (exit $?)"
}
sweep() { # sweep <gpu> <dataset> <param> <values> <seeds> <extra args...>
  local gpu=$1 ds=$2 param=$3 vals=$4 seeds=$5; shift 5
  step $gpu "${ds}_${param}_s${seeds// /}" uv run python -m discell.model.sweep \
    --dataset $ds --param $param --values $vals --seeds $seeds --tag sweep3 "$@"
}
battery() { # battery <gpu> <dataset> <run> [analyses]
  local gpu=$1 ds=$2 run=$3 an=${4:-morans,niche}
  step $gpu "validate_${ds}_${run}" uv run python -m discell.model.validate --dataset $ds --run $run --analyses $an
  step $gpu "atlas_${ds}_${run}"    uv run python -m discell.model.atlas    --dataset $ds --run $run
  step $gpu "transport_${ds}_${run}" uv run python -m discell.model.transport --dataset $ds --run $run
  step $gpu "report_${ds}_${run}"   uv run python -m discell.model.report   --dataset $ds --run $run
}

# per-dataset knobs
GSA="--variant pdl018d --alpha-z 0.0036 --tile-cells 2048 --epochs 200 --patience 20 --figures-every 200"
LUA="--label-key graphclust --alpha-z 0.004 --epochs 200 --patience 20 --figures-every 200"
FFA="--label-key graphclust --alpha-z 0.0007 --epochs 500 --patience 40 --figures-every 500"
OVA="--epochs 200 --patience 20 --figures-every 200"

log "waiting for the alpha_w=0.05 seed fits to finish"
while pgrep -f "run-name alphaw0.05_type_only" > /dev/null; do sleep 300; done
log "alpha_w=0.05 fits finished"

# CPU: degeneracy diagnostics on the existing sweep3 kappa runs (todo 1.2)
( for r in data/datasets/$OV/runs/sweep3_k*_s*; do
    uv run python -m discell.model.degeneracy --dataset $OV --run $(basename $r) --device cpu > $QL/degeneracy_$(basename $r).log 2>&1
  done; log "degeneracy on sweep3 done" ) &

laneA() {  # GPU 0
  for s in 0 1 2; do battery 0 $OV alphaw0.05_type_only_s$s morans,niche,landmarks,matrix; done
  sweep 0 $OV alpha_w "$AW" "0 1" $OVA
  sweep 0 $GS alpha_w "$AW" "0 1" $GSA
  sweep 0 $GS kappa   "$KA" "0 1" $GSA
  sweep 0 $GS d_w     "$DW" "0 1" $GSA
  sweep 0 $LU alpha_w "$AW" "0 1" $LUA
  sweep 0 $LU kappa   "$KA" "0 1" $LUA
  sweep 0 $LU d_w     "$DW" "0 1" $LUA
  sweep 0 $FF alpha_w "$AW" "0" $FFA
  sweep 0 $FF kappa   "$KA" "0" $FFA
  sweep 0 $FF d_w     "$DW" "0" $FFA
  touch $QL/laneA.done
}
laneB() {  # GPU 1, staggered
  sleep 120
  battery 1 $FF reference_graphclust morans,niche
  sweep 1 $OV alpha_w "$AW" "2" $OVA
  sweep 1 $GS alpha_w "$AW" "2" $GSA
  sweep 1 $GS kappa   "$KA" "2" $GSA
  sweep 1 $GS d_w     "$DW" "2" $GSA
  sweep 1 $LU alpha_w "$AW" "2" $LUA
  sweep 1 $LU kappa   "$KA" "2" $LUA
  sweep 1 $LU d_w     "$DW" "2" $LUA
  sweep 1 $FF alpha_w "$AW" "1 2" $FFA
  sweep 1 $FF kappa   "$KA" "1 2" $FFA
  sweep 1 $FF d_w     "$DW" "1 2" $FFA
  touch $QL/laneB.done
}
laneA & laneB & wait
log "both lanes finished; reports and cross-slide"

# aggregate reports (need all seeds) and the GSE cross-slide leg
for p in alpha_w; do step 0 "report_${OV}_$p" uv run python -m discell.model.sweep --dataset $OV --param $p --values $AW --tag sweep3 --report-only; done
for p in alpha_w kappa d_w; do
  v=$AW; [ $p = kappa ] && v=$KA; [ $p = d_w ] && v=$DW
  step 0 "report_${GS}_$p" uv run python -m discell.model.sweep --dataset $GS --param $p --values $v --tag sweep3 --variant pdl018d --tile-cells 2048 --report-only
  step 1 "report_${LU}_$p" uv run python -m discell.model.sweep --dataset $LU --param $p --values $v --tag sweep3 --label-key graphclust --report-only
  step 0 "report_${FF}_$p" uv run python -m discell.model.sweep --dataset $FF --param $p --values $v --tag sweep3 --label-key graphclust --report-only
done
for r in data/datasets/$GS/runs/sweep3_*; do
  step 1 "crossslide_$(basename $r)" uv run python -m discell.model.crossslide --dataset $GS --run $(basename $r) --eval-dataset $GD
done
log "queue complete"
