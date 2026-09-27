#!/bin/bash
# The metrics package (devlog "Metrics package for the final tables
# (motivation, 2026-09-25 07:30)"), four items, one detached queue.
#
# Lanes (run in parallel; each step takes a GPU through the picker):
#   phiproj  item 3, ovarian: phiproj32_s{0,1,2} at the final configuration
#            (final_s1's flags, 200/20, --w-warmup-epochs 30, --phi-proj 32);
#            per fit validate morans,niche,probe -> degeneracy ->
#            probe_regrade --force -> recon_modes. Control
#            wfix_warmup30_aw0.1_s{0,1,2} (reused; the preflight checks the
#            arm reproduces its config but for phi_proj): recon_modes only.
#   recon    item 2: recon_modes on final_s{0,1,2} of all four datasets, and
#            the GSE fits on the dual section (--eval-dataset).
#   boot     item 1: discell.experiments.bootstrap (all reads, 1000 draws)
#            on final_s{0,1,2} of all four datasets.
# Then, alone: timing (item 4) -- scripts/timing_mode.py, 20 epochs, arms phi /
#   phi_zeroed / phi_dropped per dataset plus phiproj32 on ovarian. A timing
#   step waits for a GPU with NO compute process on it and keeps its lock
#   fresh while it runs, so no other queue co-schedules onto it.
# Read-out: envelope_tables --ci, phi_projection_table, timing tables, all
# appended to $ROOT/AGENT_REPORT.md.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-25_adv_confirm.sh
# (gate 9500 MiB, FF 18500 MiB announcing itself in $ROOT/waiting/ff.<pid>,
# MintFlow charged at its peak, settle delay, other queues' fresh locks
# respected). Markers are keyed by step AND dataset AND run; a relaunch resumes.
# Launch detached:
#   setsid nohup scripts/queue_2026-09-25_metrics.sh \
#     >> scripts/logs/metrics_2026-09-25/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
DATASETS="$GS $LU $OV $FF"

ROOT=scripts/logs/metrics_2026-09-25
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
WAIT=$ROOT/waiting
mkdir -p $QL $DONE $LOCK $FAILED $WAIT
GPUS="0 1"
NEED_OV=9500
NEED_FF=18500
TRIES=10
TRIES_OTHER=3
SETTLE=45
FRESH=600
CONTROL=wfix_warmup30_aw0.1_s
OVA="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6 --gat-sources type_only
     --w-warmup-epochs 30 --epochs 200 --patience 20 --figures-every 200"

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

# -- GPU picker ---------------------------------------------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000

gpu_free() {
  local g=$1 free pids pid mem held=0 found=0
  free=$(nvidia-smi --query-gpu=memory.total,memory.used \
         --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  pids=" $(pgrep -f "$GUARD_PATTERN" | tr '\n' ' ') "
  while IFS=', ' read -r pid mem; do
    [ -n "$pid" ] || continue
    case "$pids" in *" $pid "*) held=$((held + mem)); found=1 ;; esac
  done < <(nvidia-smi --query-compute-apps=pid,used_memory \
           --format=csv,noheader,nounits -i $g)
  if [ $found -eq 1 ] && [ $held -lt $GUARD_PEAK ]; then
    free=$((free - (GUARD_PEAK - held)))
  fi
  echo $free
}

gpu_idle() {  # no compute process at all on GPU $1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 | tr -d '[:space:]')" ]
}

held_elsewhere() {
  local g=$1 free=$2 role=$3 f pid dir now age
  now=$(date +%s)
  if [ "$role" != ff ] && [ "$free" -ge "$NEED_FF" ]; then
    for f in scripts/logs/*/waiting/ff.*; do
      [ -e "$f" ] || continue
      pid=${f##*.}
      kill -0 "$pid" 2>/dev/null && return 0
    done
  fi
  for dir in scripts/logs/*/locks/gpu$g; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB> <role: ov|ff|timing>
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      if [ "$role" = timing ]; then gpu_idle $g || continue; fi
      held_elsewhere $g $free $role && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue
      sleep $SETTLE
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $g $free $role \
         && { [ "$role" != timing ] || gpu_idle $g; }; then
        rm -f "$me"; echo $g; return 0
      fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    if [ $((waited % 10)) -eq 0 ]; then
      log "[$role] waiting for a GPU with ${need} MiB free (0: $(gpu_free 0), 1: $(gpu_free 1))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps --------------------------------------------------------------------
step() {  # step <dataset> <role|""> <marker> <cmd...>: a GPU step, retried on OOM
  local ds=$1 role=$2 name=$3; shift 3
  [ -n "$role" ] || role=$(role_of $ds)
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0 toucher=""
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $(need_of $ds) $role)
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
    if [ "$role" = timing ]; then     # keep the lock fresh: nobody joins
      ( while [ -d "$LOCK/gpu$gpu" ]; do touch "$LOCK/gpu$gpu" 2>/dev/null; sleep 60; done ) &
      toucher=$!
    fi
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    [ -n "$toucher" ] && { kill $toucher 2>/dev/null; toucher=""; }
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
    cp $QL/$name.log $QL/$name.try$try.log
    if grep -qE "OutOfMemoryError|CUDA out of memory" $QL/$name.log; then
      log "FAIL $name: CUDA OOM (try $try/$TRIES); see $QL/$name.try$try.log"
    else
      other=$((other + 1))
      log "FAIL $name (exit $rc; attempt $other/$TRIES_OTHER); see $QL/$name.try$try.log"
      [ $other -ge $TRIES_OTHER ] && break
    fi
    sleep 300
  done
  log "GIVING UP on $name"; touch "$FAILED/$name"; return 1
}

# -- item 3: the projection test ------------------------------------------------
fit_proj() {  # fit_proj <seed>
  local run=phiproj32_s$1
  if [ -f "$(runs $OV)/$run/metrics.json" ]; then
    touch "$DONE/$(marker fit $OV $run)"; log "skip fit $run (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  step $OV "" "$(marker fit $OV $run)" uv run python -m discell.model.train \
    --dataset $OV --run-name $run $OVA --seed $1 --phi-proj 32
}

battery_proj() {  # battery_proj <run>
  local run=$1
  [ -f "$(runs $OV)/$run/metrics.json" ] || { log "no fit for $run; no reads"; return 0; }
  step $OV "" "$(marker validate $OV $run)" uv run python -m discell.model.validate \
    --dataset $OV --run $run --analyses morans,niche,probe
  step $OV "" "$(marker degeneracy $OV $run)" uv run python -m discell.model.degeneracy \
    --dataset $OV --run $run
  step $OV "" "$(marker probe_regrade $OV $run)" uv run python -m \
    discell.experiments.probe_regrade --dataset $OV --run $run --force
  step $OV "" "$(marker recon_modes $OV $run)" uv run python -m \
    discell.experiments.recon_modes --dataset $OV --run $run
}

lane_phiproj() {
  local s
  for s in 0 1 2; do fit_proj $s; battery_proj phiproj32_s$s; done
  for s in 0 1 2; do
    step $OV "" "$(marker recon_modes $OV $CONTROL$s)" uv run python -m \
      discell.experiments.recon_modes --dataset $OV --run $CONTROL$s
  done
  log "phiproj lane finished"
}

# -- item 2: reconstruction modes -------------------------------------------------
lane_recon() {
  local ds s
  for ds in $DATASETS; do
    for s in 0 1 2; do
      step $ds "" "$(marker recon_modes $ds final_s$s)" uv run python -m \
        discell.experiments.recon_modes --dataset $ds --run final_s$s
    done
  done
  for s in 0 1 2; do
    step $GS "" "$(marker recon_modes_dual $GS final_s$s)" uv run python -m \
      discell.experiments.recon_modes --dataset $GS --run final_s$s \
      --eval-dataset $GD
  done
  log "recon lane finished"
}

# -- item 1: tile-bootstrap intervals ----------------------------------------------
lane_boot() {
  local ds s
  for ds in $GS $LU $OV $FF; do
    for s in 0 1 2; do
      step $ds "" "$(marker bootstrap $ds final_s$s)" uv run python -m \
        discell.experiments.bootstrap --dataset $ds --run final_s$s --n 1000
    done
  done
  log "boot lane finished"
}

# -- item 4: timing, alone on an idle GPU ---------------------------------------------
lane_timing() {
  local ds arm
  for ds in $GS $LU $OV $FF; do
    for arm in phi phi_zeroed phi_dropped; do
      step $ds timing "$(marker timing $ds $arm)" uv run python scripts/timing_mode.py \
        --dataset $ds --arm $arm --epochs 20
    done
  done
  step $OV timing "$(marker timing $OV phiproj32)" uv run python scripts/timing_mode.py \
    --dataset $OV --arm phiproj32 --epochs 20
  log "timing lane finished"
}

# -- preflight: the arm's flags reproduce the control's config.json but for phi_proj
preflight() {
  uv run python - "$OVA" "$(runs $OV)" "$CONTROL" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
flags, runs, control = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ALLOWED = {"run_name", "seed", "git", "phi_proj"}
default = asdict(TrainConfig(dataset=""))
bad = 0
for seed in (0, 1, 2):
    args = vars(build_parser().parse_args(
        ["--dataset", "xenium_prime_ovarian_cancer_ffpe", "--seed", str(seed),
         "--phi-proj", "32"] + shlex.split(flags)))
    args.pop("quiet"); args.pop("time_only")
    new = asdict(TrainConfig(**args))
    ref = json.loads((runs / f"{control}{seed}" / "config.json").read_text())
    d = [f"{k}: {new.get(k)!r} vs {ref.get(k, '<absent>')!r}"
         for k in sorted(set(new) | set(ref)) if k not in ALLOWED
         and ((k in ref and new.get(k) != ref[k])
              or (k not in ref and new[k] != default[k]))]
    ok = not d and new["phi_proj"] == 32 and ref["seed"] == seed
    print(f"phiproj32 s{seed}: " + (f"identical to {control}{seed} but for phi_proj=32"
                                    if ok else "DIFFERS: " + "; ".join(d)))
    bad += not ok
sys.exit(1 if bad else 0)
EOF
}

report() {
  {
    echo
    echo "## Queue finished $(date '+%F %T')"
    echo
    echo "* envelope tables with CIs: data/datasets/<ds>/experiments/envelope_table_ci.md,"
    echo "  combined $ROOT/envelope_tables_final_ci.md (read-out exit $1)"
    echo "* projection test: data/datasets/$OV/experiments/phi_projection.md (exit $2)"
    echo "* timing: data/datasets/<ds>/experiments/timing.md, $ROOT/timing_all.md (exit $3)"
    echo "* failed steps: $(ls $FAILED | tr '\n' ' ')"
    echo "* dead fits: $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
    echo
    echo '```'
    cat data/datasets/$OV/experiments/phi_projection.md 2>/dev/null | head -30
    echo '```'
  } >> $ROOT/AGENT_REPORT.md
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (gate $NEED_OV / FF $NEED_FF MiB, GPUs: $GPUS; code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi

PIDS=""
(lane_phiproj) & PIDS="$PIDS $!"
(sleep 20; lane_recon) & PIDS="$PIDS $!"
(sleep 40; lane_boot) & PIDS="$PIDS $!"
wait $PIDS
for s in 0 1 2; do
  f=$(runs $OV)/phiproj32_s$s/metrics.json
  if [ -f "$f" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('dead_w_channel') else 1)"; then
    grep -qx "phiproj32_s$s" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo phiproj32_s$s >> $ROOT/DEAD_RUNS.txt
  fi
done
log "parallel lanes finished; timing"
lane_timing

log "read-out"
uv run python scripts/envelope_tables.py --runs final_s0 final_s1 final_s2 --ci \
  --combined $ROOT/envelope_tables_final.md > $QL/envelope_ci.log 2>&1; rc1=$?
uv run python scripts/phi_projection_table.py > $QL/phi_projection.log 2>&1; rc2=$?
uv run python scripts/timing_mode.py --tables > $QL/timing_tables.log 2>&1; rc3=$?
report $rc1 $rc2 $rc3
log "queue finished"
