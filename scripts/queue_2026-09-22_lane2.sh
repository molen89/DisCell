#!/usr/bin/env bash
# Lane 2 of the overnight programme (devlog "Phase change", 2026-09-22):
# external baselines on our datasets, then the shared battery (todo 8.4/8.5).
#
#   setsid nohup bash scripts/queue_2026-09-22_lane2.sh \
#       > scripts/logs/lane2_2026-09-22/queue.log 2>&1 < /dev/null & disown
#
# GPU 1 only (another agent owns GPU 0). Every step is idempotent: a step that
# finished leaves a DONE marker and is skipped on a rerun; a step that failed
# leaves a FAILED marker with the reason and the queue carries on. Delete a
# marker to force the step to run again.
#
# Order = priority (the devlog's lane 2): exports, the DisCell columns (they
# also cache the context vector every mirror read needs), resolVI on GSE and
# FF, SIMVI, MintFlow, then resolVI on the two secondary slides.

set -u
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

DISCELL=/home/rmolen/github/DisCell
BASE=/home/rmolen/github/DisCell-baselines
DATA=$BASE/data
RES=$BASE/results
LOGS=$DISCELL/scripts/logs/lane2_2026-09-22
mkdir -p "$DATA" "$RES" "$LOGS"

GSE=gse315411_pdltma06_11_prime_solo
DUAL=gse315411_pdltma06_10_prime_dual
FF=xenium_prime_human_ovary_ff
OVA=xenium_prime_ovarian_cancer_ffpe
LUNG=xenium_prime_human_lung_cancer_ffpe

step() {            # step <name> <command...>
  local name=$1; shift
  local marker=$LOGS/$name
  if [ -f "$marker.DONE" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (DONE)"; return 0
  fi
  rm -f "$marker.FAILED"
  echo "[$(date +%H:%M:%S)] START $name"
  local t0=$SECONDS
  if "$@" > "$LOGS/$name.log" 2>&1; then
    echo "$(date) $(( SECONDS - t0 ))s" > "$marker.DONE"
    echo "[$(date +%H:%M:%S)] DONE  $name ($(( SECONDS - t0 ))s)"
  else
    local code=$?
    echo "$(date) exit $code after $(( SECONDS - t0 ))s; see $name.log" \
      > "$marker.FAILED"
    echo "[$(date +%H:%M:%S)] FAILED $name (exit $code) -- see $LOGS/$name.log"
  fi
  return 0
}

# -- step 1: exports --------------------------------------------------------
# skip when the .h5ad is newer than the bundle it comes from

export_one() {      # export_one <dataset> <variant> <tile_cells> [extra args]
  local ds=$1 var=$2 tc=$3; shift 3
  local out=$DATA/$ds.h5ad
  local bundle=$DISCELL/data/datasets/$ds/bundle/$var.h5ad
  if [ -f "$out" ] && [ "$out" -nt "$bundle" ]; then
    echo "export up to date: $out"; return 0
  fi
  cd "$DISCELL" || return 1
  uv run python -m discell.experiments.export_for_baselines \
      --dataset "$ds" --variant "$var" --tile-cells "$tc" \
      --out-dir "$DATA" "$@" || return 1
  mv "$DATA/${var}_baselines.h5ad" "$out"
}

step export_$GSE   export_one $GSE   pdl018d 2048
step export_$DUAL  export_one $DUAL  pdl018d 2048
step export_$FF    export_one $FF    full 4096 --label-key graphclust
step export_$OVA   export_one $OVA   full 4096
step export_$LUNG  export_one $LUNG  full 4096 --label-key graphclust

# -- the battery ------------------------------------------------------------

battery() {         # battery <dataset> <args...>
  cd "$DISCELL" || return 1
  uv run python -m discell.experiments.baseline_battery --dataset "$@"
}

discell_col() {     # discell_col <dataset> <run> [--config-from <ds>]
  local ds=$1 run=$2; shift 2
  battery "$ds" --discell-run "$run" "$@"
}

baseline_col() {    # baseline_col <dataset> <method> <latents dir> [extra]
  local ds=$1 method=$2 dir=$3; shift 3
  [ -f "$dir/latents.h5ad" ] || { echo "no latents at $dir"; return 1; }
  battery "$ds" --method "$method" --latents "$dir/latents.h5ad" "$@"
}

# -- step 2: the DisCell columns (they cache the shared context c) ----------

step battery_discell_$GSE   discell_col $GSE  best
step battery_discell_$DUAL  discell_col $DUAL best --config-from $GSE
step battery_discell_$FF    discell_col $FF   best
step battery_discell_$OVA   discell_col $OVA  best
step battery_discell_$LUNG  discell_col $LUNG best

# -- step 3: the fits, in priority order ------------------------------------

resolvi() {         # resolvi <in dataset> <out dataset> [transfer dataset]
  cd "$DISCELL" || return 1
  local args=(--h5ad "$DATA/$1.h5ad" --out "$RES/resolvi/$2")
  if [ $# -ge 3 ]; then
    args+=(--transfer-h5ad "$DATA/$3.h5ad" --transfer-out "$RES/resolvi/$3")
  fi
  uv run python "$BASE/resolvi/run_resolvi.py" "${args[@]}"
}

simvi() {           # simvi <dataset> <out name> [extra args]
  local ds=$1 out=$2; shift 2
  cd "$BASE/simvi" || return 1
  ./.venv/bin/python run_simvi.py --h5ad "$DATA/$ds.h5ad" \
      --out "$RES/simvi/$out" "$@"
}

mintflow() {        # mintflow <dataset> <out name> <max hours>
  local ds=$1 out=$2 hours=$3
  cd "$BASE/mintflow" || return 1
  ./.venv/bin/python run_mintflow.py --h5ad "$DATA/$ds.h5ad" \
      --out "$RES/mintflow/$out" --max-hours "$hours"
}

# (1) resolVI on the GSE solo section, and the trained model applied to the
#     held-out dual section (it falls back to a fit on the dual section and
#     says so in its config.json if resolVI refuses the transfer)
step resolvi_$GSE          resolvi $GSE $GSE $DUAL
step battery_resolvi_$GSE  baseline_col $GSE  resolVI "$RES/resolvi/$GSE"
step battery_resolvi_$DUAL baseline_col $DUAL resolVI "$RES/resolvi/$DUAL" \
     --config-from $GSE

# (2) resolVI on the fresh-frozen ovary
step resolvi_$FF           resolvi $FF $FF
step battery_resolvi_$FF   baseline_col $FF resolVI "$RES/resolvi/$FF"

# (3) SIMVI: GSE solo, then the dual section (SIMVI cannot transfer a fit --
#     its `get_latent_representation` ignores the adata argument -- so the
#     dual column is a fit on the dual section, named accordingly), then FF
step simvi_$GSE            simvi $GSE $GSE
step battery_simvi_$GSE    baseline_col $GSE SIMVI "$RES/simvi/$GSE"
step simvi_$DUAL           simvi $DUAL ${DUAL}_fit_on_target
step battery_simvi_$DUAL   baseline_col $DUAL "SIMVI (fit on this section)" \
     "$RES/simvi/${DUAL}_fit_on_target" --config-from $GSE
step simvi_$FF             simvi $FF $FF
step battery_simvi_$FF     baseline_col $FF "SIMVI (200k-cell window)" \
     "$RES/simvi/$FF"

# (4) MintFlow. ~60x resolVI per epoch, so every leg carries a wall-clock cap;
#     a capped leg still writes its latents and records `truncated: true`.
step mintflow_$GSE         mintflow $GSE $GSE 5
step battery_mintflow_$GSE baseline_col $GSE MintFlow "$RES/mintflow/$GSE"
# the dual leg only if the solo leg was not itself truncated by its cap --
# a truncated fit on the training section makes a second truncated fit on the
# held-out section the least valuable hour in the queue
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$RES/mintflow/$GSE/config.json'))['truncated'] is False else 1)" 2>/dev/null; then
step mintflow_$DUAL        mintflow $DUAL ${DUAL}_fit_on_target 3
step battery_mintflow_$DUAL baseline_col $DUAL "MintFlow (fit on this section)" \
     "$RES/mintflow/${DUAL}_fit_on_target" --config-from $GSE
else
  echo "[$(date +%H:%M:%S)] SKIP mintflow_$DUAL (the solo leg did not finish "\
       "its 50 epochs inside the cap)"
fi
step mintflow_$FF          mintflow $FF $FF 6
step battery_mintflow_$FF  baseline_col $FF MintFlow "$RES/mintflow/$FF"

# (5) resolVI on the two secondary slides
step resolvi_$OVA          resolvi $OVA $OVA
step battery_resolvi_$OVA  baseline_col $OVA resolVI "$RES/resolvi/$OVA"
step resolvi_$LUNG         resolvi $LUNG $LUNG
step battery_resolvi_$LUNG baseline_col $LUNG resolVI "$RES/resolvi/$LUNG"

# -- step 4: the other DisCell seeds, if lane 1 has produced them -----------

for ds in $GSE $FF $OVA $LUNG; do
  for seed in best_s1 best_s2; do
    if [ -d "$DISCELL/data/datasets/$ds/runs/$seed" ]; then
      step battery_discell_${ds}_$seed discell_col "$ds" "$seed"
    else
      echo "[$(date +%H:%M:%S)] SKIP battery_discell_${ds}_$seed (no run yet)"
    fi
  done
done

echo "[$(date +%H:%M:%S)] QUEUE FINISHED"
grep -l . "$LOGS"/*.FAILED 2>/dev/null | sed 's/^/FAILED: /' || echo "no failures"
