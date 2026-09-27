#!/bin/bash
# Lane 1 of the overnight programme of 2026-09-22 (devlog "Phase change:
# model frozen"; todo 8.1, 8.3, 8.2, 8.6). GPU 0 only -- lane 2 owns GPU 1.
#
#   1. the alpha_z ladder on the ovarian core + the pre-registered decision
#   2. transport reads A and B with an honest target (6a.6) on ovarian + FF
#   3. `best` seed triples on every dataset, each with the full battery
#   4. the per-dataset envelope tables
#
# Idempotent: every step writes scripts/logs/lane1_2026-09-22/<step>.log and
# is skipped when that log ends in DONE. A failing step is logged FAILED and
# the queue continues -- a failed leg is a finding, not a reason to stop.
#
#   setsid nohup scripts/queue_2026-09-22_lane1.sh \
#       > scripts/logs/lane1_2026-09-22/queue.out 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8

QL=scripts/logs/lane1_2026-09-22; mkdir -p $QL
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff
LU=xenium_prime_human_lung_cancer_ffpe
GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
AZ_OV=0.007; AZ_FF=0.0007; AZ_LU=0.004; AZ_GS=0.0036   # 1/mean count per slide

log() { echo "[$(date '+%F %T')] $*" | tee -a $QL/queue.log; }

step() {                      # step <name> <command...>
  local name=$1; shift
  if grep -q '^DONE' $QL/$name.log 2>/dev/null; then log "skip $name (DONE)"; return 0; fi
  log "start $name: $*"
  local t0=$SECONDS
  if "$@" > $QL/$name.log 2>&1; then
    echo DONE >> $QL/$name.log; log "done  $name ($((SECONDS - t0))s)"
  else
    local code=$?
    echo "FAILED (exit $code)" >> $QL/$name.log
    log "FAILED $name (exit $code, $((SECONDS - t0))s) -- $QL/$name.log"
  fi
  return 0; }

skip_if() {                   # skip_if <file> <name> <command...>: skip when
  local f=$1; shift           # the artefact is already on disk
  if [ -e "$f" ]; then log "skip $1 ($f exists)"; return 0; fi
  step "$@"; }

runs() { echo data/datasets/$1/runs; }

# ---------------------------------------------------------------- step 1 ---
# The alpha_z ladder. The control rung (alpha_z = 0.007 = 1/mean count) is
# NOT refitted: sweep3_k0.1_s{0,1,2} are that fit at this exact operating
# point and budget (200/20, type_only, kappa 0.1, d_w 6, alpha_w 0.1), and
# the decision script reads them through --control-runs.
step ladder_az_fit uv run python -m discell.model.sweep \
  --dataset $OV --param alpha_z --values 0.00175 0.0035 0.014 --seeds 0 1 2 \
  --tag ladder_az --kappa 0.1 --d-w 6 --alpha-w 0.1 --gat-sources type_only \
  --epochs 200 --patience 20 --figures-every 200

step ladder_az_decision uv run python scripts/alpha_z_decision.py \
  --dataset $OV --tag ladder_az --base $AZ_OV --factors 0.25 0.5 1 2 \
  --seeds 0 1 2 --control-runs sweep3_k0.1_s0 sweep3_k0.1_s1 sweep3_k0.1_s2

DEC=data/datasets/$OV/experiments/alpha_z_decision.json
FACTOR=$(python3 -c "import json;print(json.load(open('$DEC'))['factor'])" 2>/dev/null || echo 1.0)
log "alpha_z factor adopted: $FACTOR (from $DEC)"

scaled() { python3 -c "print(repr(float($1) * float($FACTOR)))"; }

# ---------------------------------------------------------------- step 2 ---
# 6a.6: reads A and B again with every target cell decoded at its OWN
# posterior mu_w and real context/leak (transport --target-side both, the
# default), reported beside the group-w keys, which stay bit-identical.
step transport_own_${OV}_kmeans uv run python -m discell.model.transport \
  --dataset $OV --run best --read both --hvg 1000
step transport_own_${OV}_band uv run python -m discell.model.transport \
  --dataset $OV --run best --read both --hvg 1000 --niche-source tumour-band
step transport_own_${FF}_kmeans uv run python -m discell.model.transport \
  --dataset $FF --run best --read both --hvg 1000
step report_own_${OV} uv run python -m discell.model.report --dataset $OV --run best
step report_own_${FF} uv run python -m discell.model.report --dataset $FF --run best

# ---------------------------------------------------------------- step 3 ---
# `best` seed triples. If the ladder moved alpha_z, seed 0 is refitted on
# every dataset at the new value and `best` is re-pinned to it (the old
# target is recorded here and nothing is deleted).
repin() {                     # repin <dataset> <new run name>
  local ds=$1 new=$2 d; d=$(runs $ds)
  if [ -L $d/best ]; then
    log "$ds: best was -> $(readlink $d/best); re-pinning to $new"; rm $d/best
  elif [ -d $d/best ]; then
    log "$ds: best was a directory; moved to best_pre_az${FACTOR}"
    mv $d/best $d/best_pre_az${FACTOR}
  fi
  ln -s $new $d/best; }

link() {                      # link <dataset> <name> <target>
  local d; d=$(runs $1)
  if [ -e $d/$2 ]; then log "$1/$2 exists ($(readlink $d/$2 || echo dir))"
  else ln -s $3 $d/$2; log "$1/runs/$2 -> $3"; fi; }

fit() {                       # fit <name> <dataset> <alpha_z> <seed> <extra...>
  local name=$1 ds=$2 az=$3 seed=$4; shift 4
  skip_if $(runs $ds)/$name/metrics.json fit_${ds}_${name} \
    uv run python -m discell.model.train --dataset $ds --run-name $name \
    --alpha-z $az --gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 \
    --alpha-a 0.3 --epochs 500 --patience 40 --figures-every 100 \
    --seed $seed "$@"; }

if [ "$FACTOR" != "1.0" ] && [ "$FACTOR" != "1" ]; then
  S0=best_az${FACTOR}_s0
  fit $S0 $OV $(scaled $AZ_OV) 0
  fit $S0 $FF $(scaled $AZ_FF) 0 --label-key graphclust
  fit $S0 $LU $(scaled $AZ_LU) 0 --label-key graphclust
  fit $S0 $GS $(scaled $AZ_GS) 0 --variant pdl018d --tile-cells 2048
  for ds in $OV $FF $LU $GS; do
    [ -f $(runs $ds)/$S0/metrics.json ] && repin $ds $S0
    link $ds best_s0 $S0
  done
  fit best_s1 $OV $(scaled $AZ_OV) 1
  fit best_s2 $OV $(scaled $AZ_OV) 2
else
  # alpha_z stays: the existing seed-0 fits keep their place and only get a
  # seed-named alias. Ovarian already has its triple (2026-09-21 ablation).
  link $OV best_s0 ablation_gat_type_only
  link $OV best_s1 ablation_gat_type_only_s1
  link $OV best_s2 ablation_gat_type_only_s2
  link $FF best_s0 reference_graphclust
  link $GS best_s0 reference
  link $LU best_s0 best
fi

AZ_FF_USE=$(scaled $AZ_FF); AZ_LU_USE=$(scaled $AZ_LU); AZ_GS_USE=$(scaled $AZ_GS)
for seed in 1 2; do
  fit best_s$seed $GS $AZ_GS_USE $seed --variant pdl018d --tile-cells 2048
done
for seed in 1 2; do
  fit best_s$seed $FF $AZ_FF_USE $seed --label-key graphclust
done
for seed in 1 2; do
  fit best_s$seed $LU $AZ_LU_USE $seed --label-key graphclust
done

# -- the battery on every seed of every dataset ------------------------------
battery() {                   # battery <dataset> <run> <siblings...>
  local ds=$1 run=$2; shift 2
  local d; d=$(runs $ds)
  [ -e $d/$run/best.pt ] || { log "no $ds/$run -- battery skipped"; return 0; }
  skip_if $d/$run/validation/validation.json validate_${ds}_${run} \
    uv run python -m discell.model.validate --dataset $ds --run $run \
    --analyses morans,niche
  skip_if $d/$run/degeneracy.json degeneracy_${ds}_${run} \
    uv run python -m discell.model.degeneracy --dataset $ds --run $run
  skip_if $d/$run/atlas/atlas.json atlas_${ds}_${run} \
    uv run python -m discell.model.atlas --dataset $ds --run $run \
    --compare-runs "$@"
  if [ -f $d/$run/transport/transport_distribution.json ] && \
     grep -q '"summary_model_own"' $d/$run/transport/transport_distribution.json
  then log "skip transport_${ds}_${run} (own-target keys present)"
  else step transport_${ds}_${run} uv run python -m discell.model.transport \
    --dataset $ds --run $run --read both --hvg 1000; fi
  if [ "$ds" = "$OV" ]; then    # only the ovarian slide is tumour-annotated
    if [ -f $d/$run/transport/transport_tumour-band_distribution.json ] && \
       grep -q '"summary_model_own"' $d/$run/transport/transport_tumour-band_distribution.json
    then log "skip transport_band_${ds}_${run} (own-target keys present)"
    else step transport_band_${ds}_${run} uv run python -m discell.model.transport \
      --dataset $ds --run $run --read both --hvg 1000 --niche-source tumour-band; fi
  fi
  if [ "$ds" = "$GS" ]; then
    skip_if $d/$run/crossslide/$GD.json crossslide_${ds}_${run} \
      uv run python -m discell.model.crossslide --dataset $GS --run $run \
      --eval-dataset $GD
  fi
  step report_${ds}_${run} uv run python -m discell.model.report \
    --dataset $ds --run $run; }

for ds in $GS $FF $LU $OV; do
  battery $ds best_s0 best_s1 best_s2
  battery $ds best_s1 best_s0 best_s2
  battery $ds best_s2 best_s0 best_s1
done

# ---------------------------------------------------------------- step 4 ---
mkdir -p scratchpad
step envelope_tables uv run python scripts/envelope_tables.py \
  --runs best_s0 best_s1 best_s2

log "lane 1 finished (alpha_z factor $FACTOR)"
