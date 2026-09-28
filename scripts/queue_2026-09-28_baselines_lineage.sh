#!/bin/bash
# SIMVI and MintFlow refitted at lineage labels (author's direction 2026-09-28:
# all baselines on lineage labels). Stage (f) of
# scripts/queue_2026-09-25_final_lineage.sh re-exported the five bundles with
# --label-key lineage (DisCell-baselines/data/lineage/, obs['cell_type'] = the
# lineage column) and refitted resolVI on them; SIMVI and MintFlow were only
# re-scored from their old-label fits. This queue refits them on the lineage
# exports, with the lane-2 runners unchanged:
#
#   SIMVI     GSE solo; GSE dual (a fit on the section -- SIMVI cannot
#             transfer); FF 100k-cell window. run_simvi.py defaults: our pruned
#             Delaunay, n_intrinsic 20 / n_spatial 6, batch 2,048, SIMVI's own
#             epoch rule.
#   MintFlow  GSE solo, 50 epochs, width_window 600, one fit predicted on both
#             GSE sections (the transfer to the dual).
#
# Latents: DisCell-baselines/results/<tool>/<dataset>_lineage/. Each fit is
# followed by its battery column against finalL_s0's cached context
# (baseline_battery_lineage.{json,md}) and its per-block probe grade against
# uncontrolledL_s* (experiments/probe_regrade_lineage/). At the end of each
# lane, swap_oldlabels.py moves the re-scored old-label columns whose lineage
# replacement is in to baseline_battery_lineage_oldlabels.{json,md} (and their
# probe records to probe_regrade_lineage_oldlabels/), the lineage_final probe
# tables are re-rendered, and SUMMARY.md sets each refit beside its old fit.
#
# SIMVI registers labels_key but its module reads only X and batch, so its
# lineage refit re-runs the old fit on identical inputs (same cells, counts,
# graph; checked): expect equality up to GPU nondeterminism. MintFlow consumes
# the label (cell-type one-hot, neighbourhood composition).
#
# Picker / lock / marker machinery from scripts/queue_2026-09-25_final_lineage.sh
# (other queues' fresh locks respected, FF waiters first, MintFlow charged at
# 9000 MiB while it runs); added: any FF process pinned to a card by its
# CUDA_VISIBLE_DEVICES is charged at 18500 MiB from launch, because the running
# ceiling/sensitivity queue pins cards without a picker. Gates: SIMVI 10000 MiB, 12000 for the FF window
# (measured peak 10.58 GB), MintFlow 8000. The battery columns and probe grades
# are CPU-only (discell.model.metrics never touches a device) and run as CPU
# steps, serialised by flock because they read-modify-write shared tables.
# Two lanes in parallel; our own lock keeps them on different cards.
#
#   mkdir -p scripts/logs/baselines_lineage_2026-09-28 && setsid nohup \
#     bash scripts/queue_2026-09-28_baselines_lineage.sh \
#     >> scripts/logs/baselines_lineage_2026-09-28/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
FF=xenium_prime_human_ovary_ff

ROOT=scripts/logs/baselines_lineage_2026-09-28
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks
FAILED=$ROOT/failed; BACKUP=$ROOT/backup; TLOCK=$ROOT/tables.lock
mkdir -p $QL $DONE $LOCK $FAILED $BACKUP
GPUS="0 1"
NEED_SIMVI=10000
NEED_SIMVI_FF=12000
NEED_MINTFLOW=8000
NEED_FF=18500         # an FF job's charge on its pinned card; another queue's FF waiter
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=2         # attempts on any other failure
SETTLE=45
FRESH=600
PY="uv run python"
CFG=finalL_s0         # the run whose split, labels and cached context score every column
TAG=_lineage
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag $TAG"

BASE=/home/rmolen/github/DisCell-baselines
BDATA=$BASE/data/lineage
BRES=$BASE/results

log() { echo "[$(date '+%F %T')] $*"; }

# -- GPU picker (from scripts/queue_2026-09-25_final_lineage.sh) ----------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
FF_PATTERN=$FF

gpu_free() {    # free MiB on GPU $1, net of the guarded neighbour's headroom
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
  # An FF job pinned to this card (CUDA_VISIBLE_DEVICES in its environment) is
  # charged at the FF gate from launch on: the ceiling/sensitivity queue pins
  # cards without a picker, and an FF read allocates ~17 GB only minutes after
  # it starts.
  local ffp=" " ffheld=0
  for pid in $(pgrep -f "$FF_PATTERN"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" \
      && ffp="$ffp$pid "
  done
  if [ "$ffp" != " " ]; then
    while IFS=', ' read -r pid mem; do
      [ -n "$pid" ] || continue
      case "$ffp" in *" $pid "*) ffheld=$((ffheld + mem)) ;; esac
    done < <(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader,nounits -i $g)
    [ $ffheld -lt $NEED_FF ] && free=$((free - (NEED_FF - ffheld)))
  fi
  echo $free
}

live_waiter() {  # live_waiter <glob>: a waiting file whose pid is alive
  local f
  for f in $1; do
    [ -e "$f" ] || continue
    kill -0 "${f##*.}" 2>/dev/null && return 0
  done
  return 1
}

held_elsewhere() {  # held_elsewhere <gpu> <free>: someone else has first call
  local g=$1 free=$2 dir now age
  now=$(date +%s)
  if [ "$free" -ge "$NEED_FF" ] && live_waiter "scripts/logs/*/waiting/ff.*"; then
    return 0                                       # an FF waiter first
  fi
  for dir in scripts/logs/*/locks/gpu$g; do        # may not have allocated yet
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB>
  local need=$1 g free waited=0
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      held_elsewhere $g $free && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # our other lane has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $g $free; then
        echo $g; return 0
      fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    if [ $((waited % 10)) -eq 0 ]; then
      log "waiting for a GPU with ${need} MiB free (0: $(gpu_free 0), 1: $(gpu_free 1))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps ------------------------------------------------------------------------
_step() {  # _step <need> <marker> <cmd...>: a GPU step, retried on OOM
  local need=$1 name=$2; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0 toucher=""
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $need)
    t0=$SECONDS
    if [ "${LOAD_MIN:-0}" -gt 0 ]; then   # a long host-side load: keep the lock fresh meanwhile
      ( for _ in $(seq $LOAD_MIN); do sleep 60; [ -d "$LOCK/gpu$gpu" ] || break
          touch "$LOCK/gpu$gpu"; done ) &
      toucher=$!
    fi
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    [ -n "$toucher" ] && { kill $toucher 2>/dev/null; toucher=""; }
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
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

cpu_step() {  # cpu_step <marker> <cmd...>: no GPU, serialised on the tables lock
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= flock $TLOCK "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}

backup() {  # backup <dataset>: the lineage tables and probe records, once (never clobber)
  local e=data/datasets/$1/experiments f
  mkdir -p $BACKUP/$1
  for f in baseline_battery_lineage.json baseline_battery_lineage.md \
           probe_regrade_lineage_final.json probe_regrade_lineage_final.md; do
    [ -f $e/$f ] && cp -n $e/$f $BACKUP/$1/
  done
  [ -d $e/probe_regrade_lineage ] && [ ! -d $BACKUP/$1/probe_regrade_lineage ] \
    && cp -r $e/probe_regrade_lineage $BACKUP/$1/
  return 0
}

# -- fits -------------------------------------------------------------------------
run_simvi() (  # run_simvi <dataset> [args]: as the lane-2 repair ran it (subshell: the cd stays inside)
  ds=$1; shift
  cd $BASE/simvi || exit 1
  exec ./.venv/bin/python run_simvi.py --h5ad $BDATA/$ds.h5ad \
    --out $BRES/simvi/${ds}_lineage "$@"
)

run_mintflow() (  # one fit on GSE solo, predicted on both sections (subshell)
  cd $BASE/mintflow || exit 1
  exec ./.venv/bin/python run_mintflow.py \
    --h5ad $BDATA/$GS.h5ad --out $BRES/mintflow/${GS}_lineage \
    --transfer-h5ad $BDATA/$GD.h5ad --transfer-out $BRES/mintflow/${GD}_lineage \
    --epochs 50 --width-window 600 --max-hours 24
)

score() {  # score <dataset> <config dataset> <method> <latents dir>: column + probe grade
  local ds=$1 from=$2 method=$3 dir=$4 extra=() tag
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  [ -f "$dir/latents.h5ad" ] || { log "$ds '$method': no latents at $dir -- not scored"; return 0; }
  tag=$(printf '%s' "$method" | tr -c 'A-Za-z0-9=.' '_')
  cpu_step "battery_${tag}_$ds" $PY -m discell.experiments.baseline_battery \
    --dataset $ds --method "$method" --latents "$dir/latents.h5ad" \
    --config-run $CFG --tag $TAG "${extra[@]}"
  # shellcheck disable=SC2086
  cpu_step "probe_${tag}_$ds" $PY -m discell.experiments.probe_regrade \
    --dataset $ds --run $CFG --baseline-latents "$dir/latents.h5ad" --method "$method" \
    --force $REFS "${extra[@]}"
}

swap() {  # old-label columns out once their replacement is in; probe tables re-rendered
  local name=$1
  log "swap after $name"
  CUDA_VISIBLE_DEVICES= flock $TLOCK $PY $ROOT/swap_oldlabels.py $GS $GD $FF \
    >> $QL/swap.log 2>&1 || log "FAIL swap after $name; see $QL/swap.log"
  # the render stage (g) of the final lineage queue used, scoped to these three sections
  # shellcheck disable=SC2086
  for ds in $GS $FF; do
    CUDA_VISIBLE_DEVICES= flock $TLOCK $PY -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' \
      $REFS >> $QL/probe_tables.log 2>&1 || log "FAIL probe table $ds; see $QL/probe_tables.log"
  done
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES= flock $TLOCK $PY -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --table lineage_final --baselines \
    --runs 'finalL_s*' 'uncontrolledL_s*' $REFS >> $QL/probe_tables.log 2>&1 \
    || log "FAIL probe table $GD; see $QL/probe_tables.log"
}

simvi_lane() {
  _step $NEED_SIMVI "simvi_$GS" run_simvi $GS
  score $GS $GS "SIMVI (lineage)" $BRES/simvi/${GS}_lineage
  _step $NEED_SIMVI "simvi_$GD" run_simvi $GD
  score $GD $GS "SIMVI (lineage, fit on this section)" $BRES/simvi/${GD}_lineage
  # the 15.5 GB export is read, filtered and windowed on the host for ~5-10 min
  # before SIMVI allocates: the lock is kept fresh for 10 of them
  LOAD_MIN=10 _step $NEED_SIMVI_FF "simvi_$FF" run_simvi $FF --max-cells 100000
  score $FF $FF "SIMVI (lineage, 100k-cell window)" $BRES/simvi/${FF}_lineage
  swap simvi_lane
  log "SIMVI lane finished"
}

mintflow_lane() {
  _step $NEED_MINTFLOW "mintflow_$GS" run_mintflow
  score $GS $GS "MintFlow (lineage)" $BRES/mintflow/${GS}_lineage
  score $GD $GS "MintFlow (lineage, transfer)" $BRES/mintflow/${GD}_lineage
  swap mintflow_lane
  log "MintFlow lane finished"
}

# -- main ---------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
log "queue start (gates: SIMVI $NEED_SIMVI MiB, FF window $NEED_SIMVI_FF, MintFlow $NEED_MINTFLOW; GPUs: $GPUS)"
for ds in $GS $GD $FF; do backup $ds; done
for f in $GS $GD $FF; do
  [ -f $BDATA/$f.h5ad ] || { log "no lineage export $BDATA/$f.h5ad -- stopping"; exit 1; }
done
[ -f data/datasets/$GS/experiments/baseline_context_${CFG}.npz ] \
  && [ -f data/datasets/$GD/experiments/baseline_context_${CFG}_from_$GS.npz ] \
  && [ -f data/datasets/$FF/experiments/baseline_context_${CFG}.npz ] \
  || { log "a cached $CFG context is missing -- stopping"; exit 1; }

mintflow_lane &
sleep 5            # MintFlow takes the first card; SIMVI the other
simvi_lane &
wait
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed)"
