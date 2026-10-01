#!/bin/bash
# Filling the breakdown figure's gaps (devlog "Filling the breakdown figure's gaps
# (motivation, 2026-10-01; author)"; family in
# scripts/logs/breakdown_2026-09-29/FAMILY.md; report in
# scripts/logs/breakdown_gaps_2026-10-01/AGENT_REPORT.md).
#
#   (a) the tumour-axis contrast on lung FFPE and ovarian FF: external_criteria
#       axis-test (with the 2026-09-30 gene-pairing fix) on finalL_s0-s2 and
#       sweepL_k{0,0.05,0.2,0.3,0.4}_s{0,1,2}, then breakdown_draws --groups axis.
#       Both sections pass the axis test's own requirements (checked 2026-10-01).
#   (b) the TMA serial section (GSE dual): the GSE core fits applied to it
#       (--eval-dataset, crossslide.load_applied) on the same 18 fits: the w guard
#       (degeneracy), transport --read both --hvg 1000, the signalling share,
#       then breakdown_draws --groups w_mi,signalling,transport_mean,transport_dist
#       (its cycle draws exist since 2026-09-29 and are not re-run).
#   (c) re-render: breakdown --all, paper_tables --only breakdown breakdown_traj,
#       fig_breakdown_data.py, latexmk in submission_paper/aistats.
#
# Order: lung axis -> GSE dual -> FF axis (the largest last). GPU 0 ONLY (GPU 1 runs
# the MintFlow refits: never touched); the memory gate and the other queues' fresh
# scripts/logs/*/locks/gpu0 respected, our own lock held while a step runs.
# OMP_NUM_THREADS=8 for steps computing k-means niche labels (the dual's w guard,
# transport, signalling share and draws; issue T-omp), 2 otherwise (axis test and
# its draws, the renders). FF steps wait for host MemAvailable >= FF_HOST_GB (the FF
# assembly alone peaks at 36 GB RSS). The Unassigned mask is on. WANDB_MODE=disabled;
# nothing leaves the machine.
#
# Idempotent: done/<step> markers; failed/<step> records (exit, last log lines). A
# CUDA OOM is retried (up to TRIES); any other failure is recorded once and the queue
# carries on. breakdown_draws exits 3 when a point estimate does not reproduce the
# read on disk: recorded as failed, its npz kept. Delete failed/<step> to retry it on
# a relaunch. The renders run every time the queue reaches them.
#
# Estimate (smoke 2026-10-01, GSE dual finalL_s0: w guard 17 s, signalling 22 s,
# transport 71 s, draws ~5 min at full n): dual 18 x ~7 min ~ 2 h; lung axis 18 x
# ~8 min ~ 2.5 h; FF axis 18 x ~15-20 min ~ 5-6 h; renders ~5 min. ~10 h on GPU 0.
#
#   mkdir -p scripts/logs/breakdown_gaps_2026-10-01 && setsid nohup \
#     bash scripts/queue_2026-10-01_breakdown_gaps.sh \
#     >> scripts/logs/breakdown_gaps_2026-10-01/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
KAPPAS_SWEEP="0 0.05 0.2 0.3 0.4"

ROOT=scripts/logs/breakdown_gaps_2026-10-01
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
GPU=0                  # GPU 0 only; GPU 1 is never used
NEED=9500              # MiB, a DisCell read step
NEED_FF=18500          # MiB, an FF step
FF_HOST_GB=50          # host MemAvailable before an FF step
TRIES=6
SETTLE=45; FRESH=600
PY8="env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python"
PY2="env OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# -- GPU 0 picker (the other GPU-0 queues' conventions, no yielding) -------------
GUARD_PATTERN='run_mintflow|baseline_battery.*MintFlow'
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
held_elsewhere() {  # another queue's fresh GPU-0 lock
  local dir now age
  now=$(date +%s)
  for dir in scripts/logs/*/locks/gpu$GPU; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}
acquire_gpu() {  # acquire_gpu <need MiB>
  local need=$1 free waited=0
  while true; do
    free=$(gpu_free)
    if [ "$free" -ge "$need" ] && ! held_elsewhere \
       && mkdir "$LOCK/gpu$GPU" 2>/dev/null; then
      sleep $SETTLE
      free=$(gpu_free)
      if [ "$free" -ge "$need" ] && ! held_elsewhere; then return 0; fi
      rmdir "$LOCK/gpu$GPU" 2>/dev/null
    fi
    [ $((waited % 10)) -eq 0 ] && log "waiting for GPU $GPU with ${need} MiB free ($(gpu_free) free)"
    waited=$((waited + 1)); sleep 60
  done
}
wait_host() {  # wait_host <GB>
  local n=0
  while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt "$1" ]; do
    [ $((n % 10)) -eq 0 ] && log "waiting for $1 GB host MemAvailable"
    n=$((n + 1)); sleep 60
  done
}

# gstep <threads 8|2> <dataset> <marker> <cmd...>: a GPU-0 step
gstep() {
  local threads=$1 ds=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  if [ -f "$FAILED/$name" ]; then log "skip $name (failed before; see $FAILED/$name)"; return 1; fi
  local py=$PY8; [ "$threads" = 2 ] && py=$PY2
  local need=$NEED; [ "$ds" = "$FF" ] && need=$NEED_FF
  local try=0 rc=1 t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    [ "$ds" = "$FF" ] && wait_host $FF_HOST_GB
    acquire_gpu $need
    t0=$SECONDS
    log "GPU$GPU start $name (try $try, ${threads}t)"
    CUDA_VISIBLE_DEVICES=$GPU $py "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$GPU" 2>/dev/null
    log "GPU$GPU done  $name (exit $rc, $((SECONDS - t0))s)"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
    cp $QL/$name.log $QL/$name.try$try.log
    if grep -qE "OutOfMemoryError|CUDA out of memory" $QL/$name.log; then
      log "FAIL $name: CUDA OOM (try $try/$TRIES)"; sleep 300; continue
    fi
    break
  done
  log "FAILED $name (exit $rc); see $QL/$name.log"
  { echo "exit $rc"; [ $rc -eq 3 ] && echo "point estimate not reproduced (npz kept)"; \
    tail -5 $QL/$name.log; } > "$FAILED/$name"
  return 1
}
cstep() {  # cstep <marker> <cmd...>: a CPU step (re-run every time it is reached)
  local name=$1; shift
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAILED $name; see $QL/$name.log"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

fits() {  # the 18 fits of a section (finalL = kappa 0.1)
  local k s
  for s in 0 1 2; do echo finalL_s$s; done
  for k in $KAPPAS_SWEEP; do for s in 0 1 2; do echo sweepL_k${k}_s$s; done; done
}
fitted() { [ -f data/datasets/$1/runs/$2/best.pt ]; }

# -- (a) the axis test and its draws ---------------------------------------------------
axis_section() {  # axis_section <dataset>
  local ds=$1 r
  for r in $(fits); do
    fitted $ds $r || { log "no fit $ds/$r"; echo "no best.pt" > "$FAILED/nofit_${ds}_$r"; continue; }
    gstep 2 $ds "axis_${ds}_$r" -m discell.experiments.external_criteria axis-test \
      --dataset $ds --run $r \
    && gstep 2 $ds "draws_axis_${ds}_$r" -m discell.experiments.breakdown_draws \
      --dataset $ds --run $r --groups axis
  done
}

# -- (b) the serial section through the core's fits -------------------------------------
dual_section() {
  local r ok
  for r in $(fits); do
    fitted $GS $r || { log "no fit $GS/$r"; echo "no best.pt" > "$FAILED/nofit_${GS}_$r"; continue; }
    ok=1
    gstep 8 $GS "dual_wguard_$r" -m discell.model.degeneracy --dataset $GS --run $r \
      --eval-dataset $GD || ok=0
    gstep 8 $GS "dual_transport_$r" -m discell.model.transport --dataset $GS --run $r \
      --read both --hvg 1000 --eval-dataset $GD || ok=0
    gstep 8 $GS "dual_signalling_$r" -m discell.experiments.external_criteria \
      signalling-share --dataset $GS --run $r --eval-dataset $GD || ok=0
    if [ $ok -eq 1 ]; then
      gstep 8 $GS "dual_draws_$r" -m discell.experiments.breakdown_draws --dataset $GS \
        --run $r --groups w_mi,signalling,transport_mean,transport_dist --eval-dataset $GD
    else
      log "skip dual_draws_$r: a read it reproduces failed"
    fi
  done
}

# -- (c) the renders ---------------------------------------------------------------------
render() {
  cstep render_breakdown $PY2 -m discell.experiments.breakdown --all
  cstep render_tables $PY2 scripts/paper_tables.py --only breakdown breakdown_traj
  cstep render_figure $PY2 submission_paper/aistats/figures/src/fig_breakdown_data.py
  cstep render_latex bash -c "cd submission_paper/aistats && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"
}

# -- main ---------------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only"
rmdir $LOCK/gpu$GPU 2>/dev/null      # one instance at a time: a lock of ours is stale
axis_section $LU
dual_section
axis_section $FF
render
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
