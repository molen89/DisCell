#!/bin/bash
# Cellina: the three additions the author approved (devlog "Cellina: three
# additions (motivation, 2026-09-29; author)"). Template: queue_2026-09-28_cellina.sh
# (picker, locks, markers, the shared tables lock, scoring through
# baseline_battery / probe_regrade). Report: $ROOT/AGENT_REPORT.md.
#
#   (1) nicheadv   Cellina with its domain adversary ON, the domain = our niche
#                  label (K = 10 k-means on neighbour composition, the
#                  validate.niche_labels recipe fitted on TRAINING cells, isolated
#                  cells = class 10; discell/experiments/cellina_niche_domain.py,
#                  sidecars in DisCell-baselines/data/niche_domain/). Published
#                  settings otherwise (run_cellina.py --domain-labels). All four
#                  datasets, GSE with its dual transfer. Latents:
#                  results/cellina/<ds>_lineage_nicheadv/; columns
#                  "cellina_nicheadv" / "cellina_nicheadv_transfer".
#   (2) owngraph   Cellina on its own graph (--graph own: Gaussian kernel,
#                  bandwidth 100 um, <= 200 neighbours, cutoff 0.1), adversary
#                  off, GSE solo + dual transfer: <ds>_lineage_owngraph,
#                  "cellina_owngraph" / "cellina_owngraph_transfer".
#   (3) cf         the counterfactual head-to-head: stage A refits Cellina on the
#                  training tiles and writes posterior-mean encodings
#                  (DisCell-baselines/cellina/cellina_cf_encode.py); stage B
#                  scores the neighbour-rewiring counterfactual with DisCell's
#                  transport reads and CIs, mirroring finalL_s0 draw for draw
#                  (discell/experiments/cellina_counterfactual.py). All four
#                  datasets: <ds>_lineage_cf/{transport/,bootstrap_ci.json,
#                  transport_side_by_side.md}.
# Every column is scored against finalL_s0 (lineage labels, the Unassigned
# mask, the q90 cycling set) into baseline_battery_lineage.* and
# probe_regrade_lineage/ (vs uncontrolledL_s0/s1), beside the main Cellina
# column, which is never touched. Tables re-rendered at the end.
#
# WAITS (poll 10 min, pgrep -f, as queue_2026-09-29_baselines_complete.sh)
# until queue_2026-09-28_unassigned.sh and queue_2026-09-28_sens_final.sh have
# ended. GPU 0 ONLY (GPU 1 belongs to the whole-section baseline queue): every
# GPU step goes through the picker restricted to GPU 0 (memory gate, other
# queues' fresh locks and FF waiters respected). FF steps wait for
# $NEED_HOST_FF GB MemAvailable (Cellina FF peaked at 47.4 GB host).
# Threads: scoring (battery, probe, tables) at OMP_NUM_THREADS=2; steps that
# compute k-means niche labels (niche domains, cf stage B) at 8 (issue T-omp).
#
# Order: primaries first -- GSE (1, 2, 3), FF (1, 3), ovarian (1, 3), lung (1, 3).
# Estimate (from the main Cellina queue's measured walls and DisCell's own
# transport + bootstrap walls; the adversary adds a discriminator step):
#   niche domains ~10 min CPU; nicheadv fits ~1.5 h (main: GSE 4.4 min, FF 33,
#   ovarian 12.5, lung 10.7, +~50 % adversary); owngraph ~10 min; cf stage A
#   ~1 h (fit on ~85 % of the cells); cf stage B ~1 h (smoke: GSE 97 s; DisCell
#   transport + bootstrap: lung 7.6 min, ovarian 9.3, FF 24); scoring ~40 min
#   (battery GSE 13 min at 2 threads). Total ~5 h once it starts, mostly GPU 0
#   at <= 2 GB; FF host-bound.
#
# Idempotent: dataset-qualified markers in $ROOT/done; a step that gives up
# leaves $ROOT/failed/<step> and the queue carries on.
#
#   mkdir -p scripts/logs/cellina_extra_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-09-29_cellina_extra.sh \
#     >> scripts/logs/cellina_extra_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
FF=xenium_prime_human_ovary_ff
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe

ROOT=scripts/logs/cellina_extra_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks
FAILED=$ROOT/failed; BACKUP=$ROOT/backup
TLOCK=scripts/logs/baselines_lineage_2026-09-28/tables.lock   # shared tables
mkdir -p $QL $DONE $LOCK $FAILED $BACKUP
GPUS="0"              # GPU 0 only
WAIT_FOR="unassigned sens_final"
POLL=600
NEED_CELLINA=3000     # MiB; Cellina measured 0.76 GB allocated, 1,228 MiB on nvidia-smi
NEED_CF=4000          # MiB; stage B (decoder + 2000-cell kernels + tile bootstrap)
NEED_HOST_FF=60       # GB MemAvailable before an FF step loads
NEED_FF=18500         # an FF job's charge on its pinned card (as the other pickers)
TRIES=10; TRIES_OTHER=2
CPU_THREADS=2
SETTLE=45; FRESH=600
PY="uv run python"
CFG=finalL_s0; TAG=_lineage
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag $TAG"

BASE=/home/rmolen/github/DisCell-baselines
BDATA=$BASE/data/lineage
ND=$BASE/data/niche_domain
BRES=$BASE/results

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# -- the waiter (as queue_2026-09-29_baselines_complete.sh) -----------------------
queues_running() {
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-28_${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}

# -- GPU picker (queue_2026-09-28_cellina.sh, restricted to $GPUS) ------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
FF_PATTERN=$FF

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

live_waiter() {
  local f
  for f in $1; do
    [ -e "$f" ] || continue
    kill -0 "${f##*.}" 2>/dev/null && return 0
  done
  return 1
}

held_elsewhere() {
  local g=$1 free=$2 dir now age
  now=$(date +%s)
  if [ "$free" -ge "$NEED_FF" ] && live_waiter "scripts/logs/*/waiting/ff.*"; then
    return 0
  fi
  for dir in scripts/logs/*/locks/gpu$g; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {
  local need=$1 g free waited=0
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      held_elsewhere $g $free && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue
      sleep $SETTLE
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $g $free; then
        echo $g; return 0
      fi
      rmdir "$LOCK/gpu$g" 2>/dev/null
    done
    [ $((waited % 10)) -eq 0 ] && log "waiting for GPU 0 with ${need} MiB free ($(gpu_free 0))" >&2
    waited=$((waited + 1))
    sleep 60
  done
}

wait_host() {
  local need=$1 avail waited=0
  while true; do
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && return 0
    [ $((waited % 10)) -eq 0 ] && log "waiting for ${need} GB host memory (${avail} GB available)"
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps --------------------------------------------------------------------------
_step() {  # _step <need MiB> <marker> <cmd...>: a GPU-0 step, retried on OOM
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
  log "GIVING UP on $name"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

cpu_run() {  # cpu_run <threads> <lock|-> <marker> <cmd...>: no GPU
  local th=$1 lk=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc pre=()
  [ "$lk" != "-" ] && pre=(flock "$lk")
  [ "${HOST_GB:-0}" -gt 0 ] && wait_host $HOST_GB
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th \
    NUMBA_NUM_THREADS=$th "${pre[@]}" "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

backup() {  # the lineage tables and probe records, once (never clobber)
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

# -- the work -------------------------------------------------------------------------
niche_domain() {  # niche_domain <dataset> [--config-from D]: k-means, 8 threads
  local ds=$1; shift
  cpu_run 8 - "niche_domain_$ds" $PY -m discell.experiments.cellina_niche_domain \
    --dataset $ds --config-run $CFG "$@"
}

cellina() (  # cellina <dataset> <out suffix> [args]: run_cellina.py (subshell: cd)
  ds=$1; suffix=$2; shift 2
  cd $BASE/cellina || exit 1
  exec ./.venv/bin/python run_cellina.py --h5ad $BDATA/$ds.h5ad \
    --out $BRES/cellina/${ds}_lineage_$suffix "$@"
)

cf_encode() (  # cf_encode <dataset>: stage A
  cd $BASE/cellina || exit 1
  exec ./.venv/bin/python cellina_cf_encode.py --h5ad $BDATA/$1.h5ad \
    --out $BRES/cellina/${1}_lineage_cf
)

score() {  # score <dataset> <config dataset> <method> <latents dir>
  local ds=$1 from=$2 method=$3 dir=$4 extra=()
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  [ -f "$dir/latents.h5ad" ] || { log "$ds '$method': no latents at $dir -- not scored"; return 0; }
  cpu_run $CPU_THREADS $TLOCK "battery_${method}_$ds" $PY -m discell.experiments.baseline_battery \
    --dataset $ds --method "$method" --latents "$dir/latents.h5ad" \
    --config-run $CFG --tag $TAG "${extra[@]}"
  # shellcheck disable=SC2086
  cpu_run $CPU_THREADS $TLOCK "probe_${method}_$ds" $PY -m discell.experiments.probe_regrade \
    --dataset $ds --run $CFG --baseline-latents "$dir/latents.h5ad" --method "$method" \
    --force $REFS "${extra[@]}"
}

nicheadv() {  # nicheadv <dataset>
  local ds=$1
  [ -f $DONE/niche_domain_$ds ] || { log "no niche label for $ds -- nicheadv skipped"; \
    echo "no niche label" > $FAILED/fit_nicheadv_$ds; return 0; }
  _step $NEED_CELLINA "fit_nicheadv_$ds" cellina $ds nicheadv --domain-labels $ND/$ds.npz \
    && score $ds $ds cellina_nicheadv $BRES/cellina/${ds}_lineage_nicheadv
}

cf() {  # cf <dataset>: stage A (Cellina venv) then stage B (8 threads: niche labels)
  local ds=$1
  _step $NEED_CELLINA "cf_encode_$ds" cf_encode $ds || return 0
  _step $NEED_CF "cf_score_$ds" $PY -m discell.experiments.cellina_counterfactual \
    --dataset $ds --cellina-dir $BRES/cellina/${ds}_lineage_cf --run $CFG
}

render() {  # the lineage_final probe tables (as the Cellina queue renders them)
  local ds
  for ds in $GS $FF $OV $LU; do
    # shellcheck disable=SC2086
    cpu_run $CPU_THREADS $TLOCK "render_$ds" $PY -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
  done
  # shellcheck disable=SC2086
  cpu_run $CPU_THREADS $TLOCK "render_$GD" $PY -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --table lineage_final --baselines \
    --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
}

# -- main -------------------------------------------------------------------------------
log "queue start (pid $$): waiting for: $WAIT_FOR"
n=0
while [ -n "$(queues_running)" ]; do
  [ $((n % 6)) -eq 0 ] && log "still running:$(queues_running) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the waited-for queues have ended; starting (GPU $GPUS only)"
rmdir $LOCK/gpu* 2>/dev/null      # one instance at a time: any lock is stale
for f in $GS $GD $FF $OV $LU; do
  [ -f $BDATA/$f.h5ad ] || { log "no lineage export $BDATA/$f.h5ad -- stopping"; exit 1; }
  backup $f
done
for f in $GS $FF $OV $LU; do
  [ -f data/datasets/$f/experiments/baseline_context_${CFG}.npz ] \
    || { log "cached $CFG context missing for $f -- stopping"; exit 1; }
done
[ -f data/datasets/$GD/experiments/baseline_context_${CFG}_from_$GS.npz ] \
  || { log "cached $CFG context missing for $GD -- stopping"; exit 1; }

# niche labels (CPU, 8 threads): the fitted sections, then the dual from GSE's centres
niche_domain $GS
[ -f $DONE/niche_domain_$GS ] && niche_domain $GD --config-from $GS
HOST_GB=50 niche_domain $FF   # measured peak RSS 43.4 GB (assemble)
niche_domain $OV
niche_domain $LU

# GSE: (1) niche adversary + dual transfer, (2) own graph + dual, (3) counterfactual
if [ -f $DONE/niche_domain_$GS ] && [ -f $DONE/niche_domain_$GD ]; then
  _step $NEED_CELLINA "fit_nicheadv_$GS" cellina $GS nicheadv \
    --domain-labels $ND/$GS.npz --transfer-h5ad $BDATA/$GD.h5ad \
    --transfer-out $BRES/cellina/${GD}_lineage_nicheadv --transfer-domain-labels $ND/$GD.npz \
    && { score $GS $GS cellina_nicheadv $BRES/cellina/${GS}_lineage_nicheadv
         score $GD $GS cellina_nicheadv_transfer $BRES/cellina/${GD}_lineage_nicheadv; }
else
  log "GSE niche labels missing -- nicheadv GSE skipped"; echo "no niche label" > $FAILED/fit_nicheadv_$GS
fi
_step $NEED_CELLINA "fit_owngraph_$GS" cellina $GS owngraph --graph own \
  --transfer-h5ad $BDATA/$GD.h5ad --transfer-out $BRES/cellina/${GD}_lineage_owngraph \
  && { score $GS $GS cellina_owngraph $BRES/cellina/${GS}_lineage_owngraph
       score $GD $GS cellina_owngraph_transfer $BRES/cellina/${GD}_lineage_owngraph; }
cf $GS

# FF (host-bound), then ovarian, then lung
HOST_GB=$NEED_HOST_FF nicheadv $FF
HOST_GB=$NEED_HOST_FF cf $FF
nicheadv $OV
cf $OV
nicheadv $LU
cf $LU

render
cat $BRES/cellina/*_lineage_cf/transport_side_by_side.md > $ROOT/transport_side_by_side_all.md 2>/dev/null
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
