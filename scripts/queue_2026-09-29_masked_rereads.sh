#!/bin/bash
# Four missing / pre-mask reads (devlog "Four missing or pre-mask reads
# (motivation, 2026-09-29; author approved)"; docs/results_manifest.md rows
# 15, 16, 17, 20). RE-READS ONLY, no refits, under the Unassigned mask
# (discell/model/eval_mask.py's default -- DISCELL_EVAL_INCLUDE_UNASSIGNED
# stays unset). Every command below reproduces the settings the existing
# files were built with (scripts/queue_2026-09-25_final_lineage.sh); only the
# mask differs, because it is now the code's default.
#
# (1) 6b.1 signalling-gene share, lung finalL_s{0,1,2} (never run before --
#     no backup needed; `ext signalling-share` in the 2026-09-25 queue).
# (2) 6b.3 MIG/MIC quadrant, lung finalL_s{0,1,2} (never run before; `ext
#     mi-quadrant`). The 2026-09-28 manifest's "all 4 datasets" for 6b.3 was
#     wrong -- lung was missing.
# (3) ovarian axis 2x2 (experiments/axis_2x2.{json,md}, axis_2x2.py):
#     pre-mask (2026-09-28 08:16). Its per-seed / alignment caches
#     (axis_2x2_seed{0,1,2}.json, axis_2x2_alignment.json) are backed up and
#     removed first -- axis_2x2.py reuses a cache verbatim if it is present,
#     so leaving them would silently keep the pre-mask numbers.
# (4) ovarian Phi-projection test (S53), projL384_s{0,1,2} / projL32_s{0,1,2}
#     (both arms already fitted -- best.pt on disk for all six runs; NO
#     REFIT). This is a re-read of existing checkpoints: validate
#     (--analyses morans,niche,probe) -> degeneracy -> probe_regrade --force
#     -> recon_modes on each of the six runs (the exact `proj_lane` pipeline
#     of the 2026-09-25 queue), then scripts/phi_projection_table.py --tag
#     _lineage assembles experiments/phi_projection_lineage.{json,md}.
#
# Every existing output this script overwrites is copied to
# <name>_premask.<ext> beside it first (once; cp -n, idempotent).
#
# GPU: GPU 0 ONLY, never GPU 1 (that belongs to the whole-section baseline
# queue). Memory gate and lock conventions of the other GPU-0 queues
# (queue_2026-09-29_cellina_extra.sh, queue_2026-09-29_transport_heldout.sh):
# >= 9500 MiB free twice SETTLE s apart, other queues' fresh
# scripts/logs/*/locks/gpu0 respected, MintFlow's neighbour charge guarded,
# one job of ours on the card at a time; each step yields YIELD s to the
# Cellina-extra queue's picker afterward, while it is still alive.
# OMP_NUM_THREADS=8 for steps that compute k-means niche labels (issue
# T-omp): signalling-share, mi-quadrant, validate, degeneracy. Everything
# else (axis_2x2 -- tumour-band only, no k-means; probe_regrade;
# recon_modes; the table render) runs at 2. WANDB_MODE=disabled.
#
# Waits (poll every 10 min, pgrep -f) until neither
# queue_2026-09-28_unassigned.sh nor queue_2026-09-28_sens_final.sh is
# running (the freeze), then starts.
#
# Estimate: (1)+(2) 6 external-criteria reads on lung (~0.5 GPU-h each in
# the 2026-09-25 queue's per-step means) ~3 GPU-h. (3) axis_2x2, 3 models x
# up to 4 read-columns x 3 seeds, ~0.5-1 GPU-h (the 2026-09-28 pre-mask run
# took well under an hour). (4) 6 runs x (validate ~0.5 + degeneracy ~0.1 +
# probe_regrade ~0.1 + recon_modes ~0.1) GPU-h ~5 GPU-h, + the CPU table
# render (seconds). Total ~9-10 GPU-h; wall time depends on how much GPU 0
# is shared with cellina_extra_2026-09-29 / transport_heldout_2026-09-29.
#
#   mkdir -p scripts/logs/masked_rereads_2026-09-29 && \
#   setsid nohup bash scripts/queue_2026-09-29_masked_rereads.sh \
#     >> scripts/logs/masked_rereads_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on (default)

OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe

ROOT=scripts/logs/masked_rereads_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
CX=scripts/logs/cellina_extra_2026-09-29      # the Cellina-extra queue's root
GPU=0                  # GPU 0 only; GPU 1 is never used
WAIT_FOR="unassigned sens_final"
POLL=600
NEED_OV=9500            # MiB, every DisCell read step (as the unassigned queue)
NEED_FF=18500           # MiB, an FF DisCell step (for the shared FF-waiter courtesy check)
TRIES=10; TRIES_OTHER=2
SETTLE=45; FRESH=600; YIELD=150
PY8="env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python"
PY2="env OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none --baseline-tag _lineage"

log() { echo "[$(date '+%F %T')] $*"; }

# -- one instance -----------------------------------------------------------------
exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# -- the waiter (queue_2026-09-29_baselines_complete.sh / transport_heldout) ------
queues_running() {
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-28_${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}
cellina_alive() { pgrep -f "queue_2026-09-29_cellina_extra\.sh" > /dev/null; }
cellina_on_gpu() { [ -d "$CX/locks/gpu$GPU" ]; }

# -- the mask must still be in the code (a rewrite would otherwise run an
#    unmasked read) --------------------------------------------------------------
MASKED_CODE="discell/model/validate.py discell/model/degeneracy.py
  discell/experiments/probe_regrade.py discell/experiments/external_criteria.py
  discell/experiments/recon_modes.py"
mask_code_ok() {
  local f
  [ -f discell/model/eval_mask.py ] || return 1
  for f in $MASKED_CODE; do grep -q "eval_mask\|EM\." $f || return 1; done
}
wait_mask_code() {
  local n=0
  until mask_code_ok; do
    [ $((n % 6)) -eq 0 ] && log "the mask is missing from the code (a rewrite?) -- waiting, no read runs"
    n=$((n + 1)); sleep 300
  done
}

# -- GPU 0 picker (the other GPU-0 queues' conventions) --------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000

gpu_free() {
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

live_waiter() {
  local f
  for f in $1; do
    [ -e "$f" ] || continue
    kill -0 "${f##*.}" 2>/dev/null && return 0
  done
  return 1
}

held_elsewhere() {  # held_elsewhere <free>: someone else has first call on GPU 0
  local free=$1 dir now age
  cellina_on_gpu && return 0
  now=$(date +%s)
  if [ "$free" -ge "$NEED_FF" ] && live_waiter "scripts/logs/*/waiting/ff.*"; then
    return 0                                       # an FF waiter first (any queue's)
  fi
  for dir in scripts/logs/*/locks/gpu$GPU; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB>: prints 0 once it is ours
  local need=$1 free waited=0
  while true; do
    free=$(gpu_free)
    if [ "$free" -ge "$need" ] && ! held_elsewhere $free \
       && mkdir "$LOCK/gpu$GPU" 2>/dev/null; then
      sleep $SETTLE
      free=$(gpu_free)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $free; then
        echo $GPU; return 0
      fi
      rmdir "$LOCK/gpu$GPU" 2>/dev/null
    fi
    if [ $((waited % 10)) -eq 0 ]; then
      log "waiting for GPU $GPU with ${need} MiB free ($(gpu_free) free;" \
          "Cellina-extra on it: $(cellina_on_gpu && echo yes || echo no))" >&2
    fi
    waited=$((waited + 1))
    sleep 60
  done
}

# -- steps --------------------------------------------------------------------------
gstep() {  # gstep <threads: 8|2> <marker> <cmd...>: a GPU-0 step, retried on OOM
  local threads=$1 name=$2; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local py=$PY8; [ "$threads" = 2 ] && py=$PY2
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    wait_mask_code
    gpu=$(acquire_gpu $NEED_OV)
    t0=$SECONDS
    log "GPU$gpu start $name (try $try, ${threads}t)"
    CUDA_VISIBLE_DEVICES=$gpu $py "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
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

cpu_step() {  # cpu_step <marker> <cmd...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  wait_mask_code
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= $PY2 "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}

# -- (0) backups: <name>_premask.<ext> beside every existing file we overwrite ---
premask_backup() {  # premask_backup <path...>: cp -n path path_premask.ext, once
  local p dir base stem ext
  for p in "$@"; do
    [ -f "$p" ] || continue
    dir=$(dirname "$p"); base=$(basename "$p")
    case "$base" in
      *.*) ext=${base##*.}; stem=${base%.*} ;;
      *)   ext=""; stem=$base ;;
    esac
    if [ -n "$ext" ]; then
      cp -n "$p" "$dir/${stem}_premask.${ext}"
    else
      cp -n "$p" "$dir/${stem}_premask"
    fi
  done
}

backup_and_clear_axis_2x2() {
  [ -f "$DONE/backup_axis_2x2" ] && return 0
  local d=data/datasets/$OV/experiments s
  premask_backup "$d/axis_2x2.json" "$d/axis_2x2.md" \
    "$d/axis_2x2_seed0.json" "$d/axis_2x2_seed1.json" "$d/axis_2x2_seed2.json" \
    "$d/axis_2x2_alignment.json"
  # axis_2x2.py reuses a per-seed / alignment cache verbatim if present, so a
  # re-read under the (now default) mask needs them gone, not just axis_2x2.{json,md}
  for s in 0 1 2; do rm -f "$d/axis_2x2_seed$s.json"; done
  rm -f "$d/axis_2x2_alignment.json"
  touch "$DONE/backup_axis_2x2"
}

backup_proj_run() {  # backup_proj_run <run>: files proj_lane's pipeline overwrites
  local run=$1 name="backup_proj_$run" d=data/datasets/$OV/runs/$run
  [ -f "$DONE/$name" ] && return 0
  premask_backup "$d/validation/probe_blocks.json" "$d/degeneracy.json" \
    "$d/recon_modes.json" "$d/recon_modes_cells.npz"
  touch "$DONE/$name"
}

backup_phi_table() {
  [ -f "$DONE/backup_phi_table" ] && return 0
  local d=data/datasets/$OV/experiments
  premask_backup "$d/phi_projection_lineage.json" "$d/phi_projection_lineage.md"
  touch "$DONE/backup_phi_table"
}

# -- (1) + (2): lung signalling-share / mi-quadrant, finalL_s{0,1,2} ---------------
lung_ext() {
  local s
  for s in 0 1 2; do
    gstep 8 "ext_signalling-share_${LU}_finalL_s$s" \
      -m discell.experiments.external_criteria signalling-share \
      --dataset $LU --run finalL_s$s
    gstep 8 "ext_mi-quadrant_${LU}_finalL_s$s" \
      -m discell.experiments.external_criteria mi-quadrant \
      --dataset $LU --run finalL_s$s
  done
}

# -- (3): ovarian axis 2x2 ---------------------------------------------------------
ovarian_axis_2x2() {
  backup_and_clear_axis_2x2
  gstep 2 "axis_2x2_${OV}" -m discell.experiments.axis_2x2 \
    --dataset $OV --seeds 0 1 2
}

# -- (4): ovarian Phi-projection test (existing runs only, no refit) --------------
phi_projection() {
  local s d run
  for s in 0 1 2; do
    for d in 384 32; do
      run=projL${d}_s$s
      if [ ! -f data/datasets/$OV/runs/$run/best.pt ]; then
        log "no fit for $OV/$run -- skipped (no refit)"
        echo "no best.pt" > "$FAILED/phi_$run"
        continue
      fi
      backup_proj_run $run
      gstep 8 "validate_${OV}_$run" -m discell.model.validate \
        --dataset $OV --run $run --analyses morans,niche,probe \
        && gstep 8 "degeneracy_${OV}_$run" -m discell.model.degeneracy \
          --dataset $OV --run $run \
        && gstep 2 "probe_regrade_${OV}_$run" -m discell.experiments.probe_regrade \
          --dataset $OV --run $run --force $REFS \
        && gstep 2 "recon_modes_${OV}_$run" -m discell.experiments.recon_modes \
          --dataset $OV --run $run
    done
  done
  backup_phi_table
  cpu_step "phi_projection_table_lineage" scripts/phi_projection_table.py \
    --full projL384_s --proj projL32_s --tag _lineage \
    --config "runs/finalL_s0/config.json at 200/20" \
    --reference "uncontrolledL_s{0,1}"
}

# -- main ---------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only, >= $NEED_OV MiB free per step; waiting for: $WAIT_FOR"
n=0
while [ -n "$(queues_running)" ]; do
  [ $((n % 6)) -eq 0 ] && log "still running:$(queues_running) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the waited-for queues have ended; starting (GPU $GPU only)"
rmdir $LOCK/gpu* 2>/dev/null      # one instance at a time: any lock is stale

lung_ext
ovarian_axis_2x2
phi_projection

log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
