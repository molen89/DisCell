#!/bin/bash
# Transport scored on held-out tiles only -- the sensitivity row (devlog
# "Transport scored on held-out tiles only (sensitivity row; motivation,
# 2026-09-29; author chose option b)"). The published read cross-fits the
# readout over five spatial folds, but ~85 % of fold-0 cells lie in DisCell's
# training tiles. This queue adds the row scored ONLY on cells of the model's
# held-out tiles (val_tiles), every model quantity (niche x type group means,
# predicted shifts, the HVG set) estimated from its training tiles
# (train_tiles). Everything else as published: k-means niches (K = 10, run
# seed, 8 threads), panels rule, trusted tier, cell-split ceiling (primary)
# and the tile-split one, half-tile CIs, the Unassigned mask, Read A
# own-target and twins. Code: `--scored-cells heldout-tiles` in
# discell/model/transport.py, discell/experiments/bootstrap.py and
# discell/experiments/cellina_counterfactual.py (default = the published read,
# bit-identical). Outputs never overwrite a published file:
#   data/datasets/<ds>/runs/finalL_s*/heldout_tiles/{transport/,bootstrap_ci.json}
#   DisCell-baselines/results/cellina/<ds>_lineage_cf/heldout_tiles/{transport/,
#     bootstrap_ci.json, transport_side_by_side.md}
#   $ROOT/transport_heldout_comparison.{md,json}  (scripts/transport_heldout_table.py)
#
# WAITS (poll 10 min, pgrep -f, as queue_2026-09-29_baselines_complete.sh)
# until queue_2026-09-28_unassigned.sh and queue_2026-09-28_sens_final.sh
# have ended. Then, primaries first (GSE, FF, ovarian, lung), per finalL_s0,
# s1, s2:
#   (1) transport --read both --hvg 1000 --scored-cells heldout-tiles
#   (2) bootstrap --reads transport_mean,transport_dist --n 1000
#       --scored-cells heldout-tiles           (both at OMP_NUM_THREADS=8:
#                                               k-means niche labels, T-omp)
# then the comparison table; then Cellina's counterfactual row per dataset
# (cellina_counterfactual --scored-cells heldout-tiles, mirroring finalL_s0),
# if and only if the Cellina-extra queue (queue_2026-09-29_cellina_extra.sh)
# has produced that dataset's encodings (its done/cf_encode_<ds> marker and
# encoded.npz): it waits for them -- and for that queue's own stage B of the
# dataset (done or failed), so the published Cellina row exists -- and
# records a FAILED entry instead when that queue gave up on the encoding or
# ended without it. The table is re-rendered at the end.
#
# GPU: transport and bootstrap run forward passes over every cell
# (collect_channels / collect_latents / collect_own_p) plus the MMD kernels
# and the half-tile draws; published walls on a 4090: transport --read both
# GSE ~85 s, lung ~290-390 s, ovarian ~370-470 s, FF ~1100-1200 s; bootstrap
# (all reads) GSE ~30 s, ovarian ~90 s, FF ~335 s. On CPU they are much slower
# (see AGENT_REPORT.md), so the steps use a GPU: GPU 0 ONLY, never GPU 1 (the
# whole-section baseline queue's), and only while the Cellina-extra queue is
# not using it (no $CX/locks/gpu0; after each step, while that queue lives,
# the card is left free for YIELD s so its picker can take it). Memory gate
# and lock conventions of the other queues: >= 9500 MiB free (FF 18500) twice
# SETTLE s apart, other queues' fresh locks respected, MintFlow charged at its
# peak. FF steps first wait for $NEED_HOST_FF GB MemAvailable.
#
# Estimate once it starts: DisCell ~2.3 GPU-h (GSE 0.1, lung 0.4, ovarian 0.5,
# FF 1.3) + Cellina stage B ~1 h (the Cellina-extra queue's estimate), plus
# settle/yield; ~4-8 h wall depending on how GPU 0 is shared.
#
# Idempotent: dataset-and-run-qualified markers in $ROOT/done; a step that
# gives up leaves $ROOT/failed/<step> (last lines of its log) and the queue
# carries on. Logs: $ROOT/logs/<step>.log; timings: $ROOT/STEPS.tsv.
#
#   mkdir -p scripts/logs/transport_heldout_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-09-29_transport_heldout.sh \
#     >> scripts/logs/transport_heldout_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
FF=xenium_prime_human_ovary_ff
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
RUNS="finalL_s0 finalL_s1 finalL_s2"

ROOT=scripts/logs/transport_heldout_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
CX=scripts/logs/cellina_extra_2026-09-29      # the Cellina-extra queue's root
GPU=0                 # GPU 0 only; GPU 1 is never used
WAIT_FOR="unassigned sens_final"
POLL=600
NEED_OV=9500          # MiB, every non-FF DisCell step (as the unassigned queue)
NEED_FF=18500         # MiB, an FF DisCell step
NEED_CF=4000          # MiB, Cellina stage B (the Cellina-extra queue's figure)
NEED_HOST_FF=60       # GB MemAvailable before an FF step loads (FF assemble ~43 GB)
TRIES=10; TRIES_OTHER=2
SETTLE=45; FRESH=600; YIELD=150
PY="uv run python"
BRES=/home/rmolen/github/DisCell-baselines/results

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
cellina_alive() { pgrep -f "queue_2026-09-29_cellina_extra\.sh" > /dev/null; }
cellina_on_gpu() { [ -d "$CX/locks/gpu$GPU" ]; }

# -- GPU 0 picker (the other queues' conventions) -------------------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000

gpu_free() {    # free MiB on GPU $GPU, net of the guarded neighbour's headroom
  local free pids pid mem held=0 found=0
  free=$(nvidia-smi --query-gpu=memory.total,memory.used \
         --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $1-$2}')
  pids=" $(pgrep -f "$GUARD_PATTERN" | tr '\n' ' ') "
  while IFS=', ' read -r pid mem; do
    [ -n "$pid" ] || continue
    case "$pids" in *" $pid "*) held=$((held + mem)); found=1 ;; esac
  done < <(nvidia-smi --query-compute-apps=pid,used_memory \
           --format=csv,noheader,nounits -i $GPU)
  if [ $found -eq 1 ] && [ $held -lt $GUARD_PEAK ]; then
    free=$((free - (GUARD_PEAK - held)))
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

held_elsewhere() {  # held_elsewhere <free> <role>: someone else has first call
  local free=$1 role=$2 dir now age
  cellina_on_gpu && return 0                        # the Cellina-extra queue, any age
  now=$(date +%s)
  if [ "$role" != ff ] && [ "$free" -ge "$NEED_FF" ] \
     && live_waiter "scripts/logs/*/waiting/ff.*"; then
    return 0                                       # an FF waiter first (any queue's)
  fi
  for dir in scripts/logs/*/locks/gpu$GPU; do      # may not have allocated yet
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB> <role: ov|ff>: prints 0 once it is ours
  local need=$1 role=$2 free waited=0
  while true; do
    free=$(gpu_free)
    if [ "$free" -ge "$need" ] && ! held_elsewhere $free $role \
       && mkdir "$LOCK/gpu$GPU" 2>/dev/null; then
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $free $role; then
        echo $GPU; return 0
      fi
      rmdir "$LOCK/gpu$GPU" 2>/dev/null
    fi
    if [ $((waited % 10)) -eq 0 ]; then
      log "[$role] waiting for GPU $GPU with ${need} MiB free ($(gpu_free) free;" \
          "Cellina-extra on it: $(cellina_on_gpu && echo yes || echo no))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

wait_host() {  # wait_host <GB>: MemAvailable
  local need=$1 avail waited=0
  while true; do
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && return 0
    [ $((waited % 10)) -eq 0 ] && log "waiting for ${need} GB host memory (${avail} GB available)"
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps ----------------------------------------------------------------------------
_step() {  # _step <need MiB> <role> <marker> <cmd...>: a GPU-0 step, retried on OOM
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    [ "$role" = ff ] && wait_host $NEED_HOST_FF
    gpu=$(acquire_gpu $need $role)
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
    # leave the card to the Cellina-extra queue's picker (it polls every 60 s)
    cellina_alive && sleep $YIELD
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

render() {  # the comparison table (CPU, reads files only; re-rendered each call)
  local t0=$SECONDS rc
  CUDA_VISIBLE_DEVICES= $PY scripts/transport_heldout_table.py \
    --out $ROOT/transport_heldout_comparison.md > $QL/render.log 2>&1; rc=$?
  log "rendered $ROOT/transport_heldout_comparison.md (exit $rc, $((SECONDS - t0))s)"
}

discell_row() {  # discell_row <dataset>: the variant on finalL_s0-s2
  local ds=$1 run need=$NEED_OV role=ov
  [ "$ds" = "$FF" ] && { need=$NEED_FF; role=ff; }
  for run in $RUNS; do
    if [ ! -f data/datasets/$ds/runs/$run/best.pt ]; then
      log "no fit for $ds/$run -- skipped"; echo "no best.pt" > "$FAILED/transport_ho_${ds}_$run"
      continue
    fi
    _step $need $role "transport_ho_${ds}_$run" $PY -m discell.model.transport \
      --dataset $ds --run $run --read both --hvg 1000 --scored-cells heldout-tiles \
      && _step $need $role "bootstrap_ho_${ds}_$run" $PY -m discell.experiments.bootstrap \
        --dataset $ds --run $run --reads transport_mean,transport_dist --n 1000 \
        --scored-cells heldout-tiles
  done
}

cellina_row() {  # cellina_row <dataset>: waits for the Cellina-extra queue's encodings
  local ds=$1 name="cf_ho_$1" dir=$BRES/cellina/${1}_lineage_cf n=0 role=ov
  [ "$ds" = "$FF" ] && role=ff
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  while true; do
    if [ -f $CX/failed/cf_encode_$ds ]; then
      log "$name: the Cellina-extra queue gave up on the encoding -- not run"
      { echo "no encodings: $CX/failed/cf_encode_$ds"; cat $CX/failed/cf_encode_$ds; } > "$FAILED/$name"
      return 1
    fi
    if [ -f $CX/done/cf_encode_$ds ] && [ -f $dir/encoded.npz ] \
       && { [ -f $CX/done/cf_score_$ds ] || [ -f $CX/failed/cf_score_$ds ] || ! cellina_alive; }; then
      break
    fi
    if ! cellina_alive && ! [ -f $CX/done/cf_encode_$ds ]; then
      log "$name: the Cellina-extra queue is not running and has no encoding -- not run"
      echo "no encodings: the Cellina-extra queue ended (or never started) without cf_encode_$ds" > "$FAILED/$name"
      return 1
    fi
    [ $((n % 6)) -eq 0 ] && log "$name: waiting for the Cellina-extra queue's encodings (cf_encode_$ds) and its stage B"
    n=$((n + 1)); sleep $POLL
  done
  _step $NEED_CF $role $name $PY -m discell.experiments.cellina_counterfactual \
    --dataset $ds --cellina-dir $dir --run finalL_s0 --scored-cells heldout-tiles
}

# -- main -------------------------------------------------------------------------------
log "queue start (pid $$): waiting for: $WAIT_FOR"
n=0
while [ -n "$(queues_running)" ]; do
  [ $((n % 6)) -eq 0 ] && log "still running:$(queues_running) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the waited-for queues have ended; starting (GPU $GPU only)"
rmdir $LOCK/gpu* 2>/dev/null      # one instance at a time: any lock is stale
grep -q "heldout-tiles" discell/model/transport.py \
  || { log "transport.py has no --scored-cells heldout-tiles -- stopping"; exit 1; }

for ds in $GS $FF $OV $LU; do discell_row $ds; done
render
for ds in $GS $FF $OV $LU; do cellina_row $ds; done
render
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
