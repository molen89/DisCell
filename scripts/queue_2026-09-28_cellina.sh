#!/bin/bash
# Cellina as a baseline (devlog "Cellina added as a baseline (motivation,
# 2026-09-28; author)", steps 4-5). PREPARED, NOT LAUNCHED.
#
# Cellina (Moeed et al., arXiv:2606.08493, github.com/PMBio/cellina v1.1.1 @
# c77b614) in its own venv, DisCell-baselines/cellina/, runner run_cellina.py:
# the published configuration (n_latent 64 per block, NB, batch 2,048, <= 100
# epochs with early stopping), cell-type classifier on obs['cell_type'] (the
# lineage label), NO domain adversary (our exports carry no spatial-domain
# label and none is invented), phi(v) aggregated over OUR pruned Delaunay
# graph. See DisCell-baselines/cellina/README.md.
#
#   GSE solo  one fit, transferred to the dual section (the same weights, phi
#             recomputed from the dual's own graph)
#   FF        the whole slide (no window: measured peak host RSS below; the
#             step waits for $NEED_HOST_FF GB of MemAvailable first)
#   ovarian, lung
#
# Latents: DisCell-baselines/results/cellina/<dataset>_lineage/. Each fit is
# followed by its battery column against finalL_s0's cached context
# (baseline_battery_lineage.{json,md}) and its per-block probe grade against
# uncontrolledL_s* (experiments/probe_regrade_lineage/), then the
# probe_regrade_lineage_final tables are re-rendered. Those tables are shared
# with scripts/queue_2026-09-28_baselines_lineage.sh, so the CPU steps take
# ITS tables lock, not one of our own.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-28_baselines_lineage.sh
# (other queues' fresh locks respected, FF waiters first, MintFlow charged at
# 9000 MiB, FF processes pinned by CUDA_VISIBLE_DEVICES charged at 18500).
# Cellina's GPU footprint is small and does not grow with the slide (a
# minibatch MLP VAE; measured in the smoke, README): gate $NEED_CELLINA MiB.
# Note the side effect: while the FF step runs, every picker that charges FF
# processes (this one and the lineage/ceiling queues) books 18.5 GB on its card.
#
# Estimate (measured s/epoch x the 100-epoch cap; early stopping can only
# shorten it): GSE solo + transfer ~10 min, FF ~2.1 h (4.5 min load, ~70
# s/epoch, 6.5 min write), ovarian ~30 min, lung ~20 min: ~3.1 h on one card
# at ~1.2 GB, sequential. CPU scoring ~1 min per GSE step at 2 threads, a few
# minutes per FF step.
#
#   mkdir -p scripts/logs/cellina_2026-09-28 && setsid nohup \
#     bash scripts/queue_2026-09-28_cellina.sh \
#     >> scripts/logs/cellina_2026-09-28/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
FF=xenium_prime_human_ovary_ff
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe

ROOT=scripts/logs/cellina_2026-09-28
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks
FAILED=$ROOT/failed; BACKUP=$ROOT/backup
TLOCK=scripts/logs/baselines_lineage_2026-09-28/tables.lock   # shared tables
mkdir -p $QL $DONE $LOCK $FAILED $BACKUP
GPUS="0 1"
NEED_CELLINA=3000     # MiB; measured peak 0.76 GB allocated, 1,228 MiB on nvidia-smi (GSE)
NEED_HOST_FF=60       # GB MemAvailable before the FF step loads; measured peak RSS 47.4 GB
NEED_FF=18500         # an FF job's charge on its pinned card; another queue's FF waiter
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=2         # attempts on any other failure
CPU_THREADS=2         # BLAS/torch threads for the CPU scoring steps: at load ~93 one probe
                      # MLP fit took 52.6 s at 8 threads, 5.1 s at 2, 2.3 s at 1 (measured)
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

# -- GPU picker (from scripts/queue_2026-09-28_baselines_lineage.sh) ------------
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
  # an FF job pinned to this card is charged at the FF gate from launch on
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
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue
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

wait_host() {  # wait_host <GB>: until MemAvailable reaches it (FF is host-bound)
  local need=$1 avail waited=0
  while true; do
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && return 0
    [ $((waited % 10)) -eq 0 ] && log "waiting for ${need} GB host memory (${avail} GB available)"
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps ------------------------------------------------------------------------
_step() {  # _step <need> <marker> <cmd...>: a GPU step, retried on OOM
  local need=$1 name=$2; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    [ "${HOST_GB:-0}" -gt 0 ] && wait_host $HOST_GB
    gpu=$(acquire_gpu $need)
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
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
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS \
    OPENBLAS_NUM_THREADS=$CPU_THREADS flock $TLOCK "$@" > $QL/$name.log 2>&1; rc=$?
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
run_cellina() (  # run_cellina <dataset> [args] (subshell: the cd stays inside)
  ds=$1; shift
  cd $BASE/cellina || exit 1
  exec ./.venv/bin/python run_cellina.py --h5ad $BDATA/$ds.h5ad \
    --out $BRES/cellina/${ds}_lineage "$@"
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

render() {  # the lineage_final probe tables, as the lineage queue renders them
  local ds
  # shellcheck disable=SC2086
  for ds in $GS $FF $OV $LU; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS \
      OPENBLAS_NUM_THREADS=$CPU_THREADS flock $TLOCK $PY -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' \
      $REFS >> $QL/probe_tables.log 2>&1 || log "FAIL probe table $ds; see $QL/probe_tables.log"
  done
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS \
    OPENBLAS_NUM_THREADS=$CPU_THREADS flock $TLOCK $PY -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --table lineage_final --baselines \
    --runs 'finalL_s*' 'uncontrolledL_s*' $REFS >> $QL/probe_tables.log 2>&1 \
    || log "FAIL probe table $GD; see $QL/probe_tables.log"
}

# -- main ---------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
log "queue start (gate: Cellina $NEED_CELLINA MiB, FF host $NEED_HOST_FF GB; GPUs: $GPUS)"
for ds in $GS $GD $FF $OV $LU; do backup $ds; done
for f in $GS $GD $FF $OV $LU; do
  [ -f $BDATA/$f.h5ad ] || { log "no lineage export $BDATA/$f.h5ad -- stopping"; exit 1; }
done
for f in $GS $FF $OV $LU; do
  [ -f data/datasets/$f/experiments/baseline_context_${CFG}.npz ] \
    || { log "cached $CFG context missing for $f -- stopping"; exit 1; }
done
[ -f data/datasets/$GD/experiments/baseline_context_${CFG}_from_$GS.npz ] \
  || { log "cached $CFG context missing for $GD -- stopping"; exit 1; }

# primaries first (GSE, FF), then ovarian and lung
_step $NEED_CELLINA "cellina_$GS" run_cellina $GS \
  --transfer-h5ad $BDATA/$GD.h5ad --transfer-out $BRES/cellina/${GD}_lineage
score $GS $GS "Cellina (lineage)" $BRES/cellina/${GS}_lineage
score $GD $GS "Cellina (lineage, transfer)" $BRES/cellina/${GD}_lineage

HOST_GB=$NEED_HOST_FF _step $NEED_CELLINA "cellina_$FF" run_cellina $FF
score $FF $FF "Cellina (lineage)" $BRES/cellina/${FF}_lineage

_step $NEED_CELLINA "cellina_$OV" run_cellina $OV
score $OV $OV "Cellina (lineage)" $BRES/cellina/${OV}_lineage

_step $NEED_CELLINA "cellina_$LU" run_cellina $LU
score $LU $LU "Cellina (lineage)" $BRES/cellina/${LU}_lineage

render
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed)"
