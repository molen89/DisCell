#!/bin/bash
# Todo 8.19, the breakdown-point layer, LEAN version (devlog "8.19 revised to
# a lean version (author, 2026-09-29)"; family and m_s in
# scripts/logs/breakdown_2026-09-29/FAMILY.md; code
# discell/experiments/breakdown_draws.py and breakdown.py).
#
# Members (m = 8 per section, 9 on ovarian; the GSE dual 1): cycle asymmetry
# (q90), I(niche;w) - floor, marker pairs, signalling share (response),
# axis test (ovarian) -- the NON-TRANSPORT members -- and Read A - type-mean,
# Read B twin margin, transport counterfactual - program_only / - leak_only
# -- the TRANSPORT members.
#
# Order (GSE first, FF last within every stage):
#   (1) missing reads on sweepL_k{0,0.05,0.2,0.3,0.4}_s{0,1,2}: the
#       signalling share (external_criteria signalling-share) on all four,
#       the axis test (external_criteria axis-test) on ovarian; plus lung
#       finalL_s{0,1,2} signalling share if queue_2026-09-29_masked_rereads.sh
#       has not produced it (waited for while that queue is alive and has
#       neither done nor failed it; only then run here -- never both at once).
#   (2) per-draw statistics, non-transport groups (breakdown_draws
#       cycle,w_mi,marker,signalling[,axis]) on the 18 fits of each section
#       (15 sweepL + finalL_s0-s2 for kappa = 0.1; into runs/<run>/breakdown/,
#       nothing published is overwritten), and the dual's cycle asymmetry
#       (each GSE fit applied to the dual section, as crossslide does).
#   (3) breakdown tables (partial: the non-transport members).
#   (4) TRANSPORT GATE: waits until queue_2026-09-29_transport_heldout.sh has
#       ended and its comparison table
#       scripts/logs/transport_heldout_2026-09-29/transport_heldout_comparison.json
#       exists. If ANY held-out-tiles value in it lies outside the published
#       read's CI (any row, both tiers), or no row is comparable, the
#       transport members are NOT computed: a HOLD note goes to
#       $ROOT/HOLD_transport.md and the queue log, for the author.
#   (5) otherwise: transport --read twins --hvg 1000 on sweepL (Read A/B's
#       distribution + twin read: finalL's `--read both` minus the mean read,
#       whose sweepL files already exist and are not rewritten), then
#       breakdown_draws transport_mean,transport_dist on the 18 fits.
#   (6) final breakdown tables (breakdown --all).
#
# Scheduling: waits ONLY for the freeze (queue_2026-09-28_unassigned.sh and
# queue_2026-09-28_sens_final.sh ended; poll 10 min, pgrep -f, as
# queue_2026-09-29_baselines_complete.sh). Then runs CONCURRENTLY with the
# other GPU-0 queues (cellina_extra, transport_heldout, masked_rereads): GPU
# 0 ONLY, never GPU 1 (the whole-section baselines and their timings); a
# memory gate (>= NEED MiB free, twice SETTLE s apart; MintFlow's neighbour
# charge guarded as the other queues do); other queues' fresh (< FRESH s)
# scripts/logs/*/locks/gpu0 respected and our own lock held while a step
# runs; no yielding. OMP_NUM_THREADS=8 for steps computing k-means niche
# labels (signalling-share; breakdown_draws w_mi, signalling, transport_*;
# transport; issue T-omp), 2 otherwise (axis-test; the dual's cycle; the
# tables). FF steps also wait for host MemAvailable >= FF_HOST_GB.
# WANDB_MODE=disabled; nothing leaves the machine.
#
# Idempotent: done/<step> markers; failed/<step> records (exit, last log
# lines). A CUDA OOM is retried (up to TRIES); any other failure is recorded
# once and the queue carries on. breakdown_draws exits 3 when a point
# estimate does not reproduce the number on disk: recorded as failed, its
# npz kept. Delete failed/<step> to retry it on a relaunch.
#
# Estimate (smoke runs 2026-09-29 and the 2026-09-28 step timings): (1) 60
# signalling reads 0.5-2.5 min + 15 axis reads ~0.7 min ~ 1.5 h; (2) 72 fits x
# (cycle + w_mi + marker + signalling at 4000 draws) ~2-10 min, ovarian axis
# +7 min, 18 dual fits ~0.5 min ~ 6-8 h; (5) 60 distribution/twin reads (GSE
# ~5 min .. FF ~45 min) + 72 transport draw steps ~ 10-14 h. Total ~18-24 h of
# GPU-0 time after the freeze, shared with the other GPU-0 queues.
#
#   mkdir -p scripts/logs/breakdown_2026-09-29 && setsid nohup \
#     bash scripts/queue_2026-10-01_breakdown.sh \
#     >> scripts/logs/breakdown_2026-09-29/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ORDER="$GS $OV $LU $FF"                        # GSE first, FF last
KAPPAS_SWEEP="0 0.05 0.2 0.3 0.4"

ROOT=scripts/logs/breakdown_2026-09-29
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
MR=scripts/logs/masked_rereads_2026-09-29     # the masked-rereads queue's root
TH=scripts/logs/transport_heldout_2026-09-29  # the transport-heldout queue's root
GPU=0                  # GPU 0 only; GPU 1 is never used
WAIT_FOR="unassigned sens_final"
POLL=600
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

queues_running() {
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-28_${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}

# -- GPU 0 picker (the other GPU-0 queues' conventions, no yielding) -------------
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
  CUDA_VISIBLE_DEVICES= $PY2 "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
  log "FAILED $name; see $QL/$name.log"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

fits() {  # the 18 fits of a section (finalL = kappa 0.1)
  local k s
  for s in 0 1 2; do echo finalL_s$s; done
  for k in $KAPPAS_SWEEP; do for s in 0 1 2; do echo sweepL_k${k}_s$s; done; done
}
sweep_fits() { local k s; for k in $KAPPAS_SWEEP; do for s in 0 1 2; do echo sweepL_k${k}_s$s; done; done; }
fitted() { [ -f data/datasets/$1/runs/$2/best.pt ]; }

# -- (1) missing reads ------------------------------------------------------------
lung_final_signalling() {  # never at the same time as the masked-rereads queue
  local s f name n
  for s in 0 1 2; do
    f=data/datasets/$LU/experiments/external_signalling_share_finalL_s$s.json
    name="ext_signalling-share_${LU}_finalL_s$s"
    n=0
    while [ ! -f "$f" ] && [ ! -f "$MR/done/$name" ] && [ ! -f "$MR/failed/$name" ] \
          && pgrep -f "queue_2026-09-29_masked_rereads\.sh" > /dev/null; do
      [ $((n % 6)) -eq 0 ] && log "lung finalL_s$s signalling share: waiting for the masked-rereads queue"
      n=$((n + 1)); sleep $POLL
    done
    if [ -f "$f" ]; then log "lung finalL_s$s signalling share on disk -- not re-run"; continue; fi
    gstep 8 $LU "$name" -m discell.experiments.external_criteria signalling-share \
      --dataset $LU --run finalL_s$s
  done
}
missing_reads() {
  local ds r
  for ds in $ORDER; do
    [ "$ds" = "$LU" ] && lung_final_signalling
    for r in $(sweep_fits); do
      fitted $ds $r || { log "no fit $ds/$r"; echo "no best.pt" > "$FAILED/nofit_${ds}_$r"; continue; }
      gstep 8 $ds "ext_signalling-share_${ds}_$r" -m discell.experiments.external_criteria \
        signalling-share --dataset $ds --run $r
      [ "$ds" = "$OV" ] && gstep 2 $ds "ext_axis-test_${ds}_$r" \
        -m discell.experiments.external_criteria axis-test --dataset $ds --run $r
    done
  done
}

# -- (2) per-draw statistics, non-transport ------------------------------------------
draws_nontransport() {
  local ds r groups
  for ds in $ORDER; do
    groups=cycle,w_mi,marker,signalling
    [ "$ds" = "$OV" ] && groups=$groups,axis
    for r in $(fits $ds); do
      fitted $ds $r || continue
      gstep 8 $ds "draws_nontransport_${ds}_$r" -m discell.experiments.breakdown_draws \
        --dataset $ds --run $r --groups $groups
      [ "$ds" = "$GS" ] && gstep 2 $ds "draws_dual_cycle_$r" -m discell.experiments.breakdown_draws \
        --dataset $GS --run $r --groups cycle --eval-dataset $GD
    done
  done
}

# -- (4) the transport gate ------------------------------------------------------------
hold_note() {  # hold_note <reason>
  printf '# HOLD: transport members not computed\n\n%s\n\n%s\n\nThe pre-set rule (devlog "Transport scored on held-out tiles only") may switch the headline transport read to held-out tiles, which would change how the transport members (Read A - type-mean, Read B twin margin, counterfactual - program_only / - leak_only) are scored. They wait for the author. To run them on the published read anyway: relaunch the queue with GATE_OVERRIDE=1 in its environment.\n' \
    "$(date '+%F %T')" "$1" > $ROOT/HOLD_transport.md
}
transport_gate() {  # 0 = go, 1 = hold
  local n=0 table=$TH/transport_heldout_comparison.json verdict
  while pgrep -f "queue_2026-09-29_transport_heldout\.sh" > /dev/null; do
    [ $((n % 6)) -eq 0 ] && log "transport gate: waiting for the transport-heldout queue to end"
    n=$((n + 1)); sleep $POLL
  done
  if [ ! -f $table ]; then
    verdict="HOLD the transport-heldout queue ended without $table"
  else
    verdict=$(CUDA_VISIBLE_DEVICES= $PY2 - "$table" <<'PYEOF'
import json, math, sys
rows = json.load(open(sys.argv[1]))
ok = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
out, comparable = [], 0
for r in rows:
    p, h = r.get("published") or {}, r.get("heldout") or {}
    for key in ("transport_of_ceiling_trusted", "transport_of_ceiling"):
        ci = (p.get(key) or {}).get("ci95")
        v = (h.get(key) or {}).get("value")
        if not ci or not ok(v) or not all(ok(c) for c in ci):
            continue
        comparable += 1
        if v < ci[0] or v > ci[1]:
            out.append(f"{r['dataset']} {r['method']} {r['run']} {key}: "
                       f"held-out {v:.3f} outside [{ci[0]:.3f}, {ci[1]:.3f}]")
if comparable == 0:
    print("HOLD no comparable row in the table")
elif out:
    print("HOLD " + "; ".join(out))
else:
    print(f"GO all {comparable} held-out values inside the published CIs")
PYEOF
)
  fi
  log "transport gate: $verdict"
  case "$verdict" in
    GO*) return 0 ;;
  esac
  hold_note "$verdict"
  if [ "${GATE_OVERRIDE:-0}" = 1 ]; then
    log "GATE_OVERRIDE=1: running the transport members despite the hold"; return 0
  fi
  return 1
}

# -- (5) transport members ----------------------------------------------------------------
transport_members() {
  local ds r
  for ds in $ORDER; do
    for r in $(sweep_fits); do
      fitted $ds $r || continue
      gstep 8 $ds "transport_twins_${ds}_$r" -m discell.model.transport --dataset $ds \
        --run $r --read twins --hvg 1000
    done
  done
  for ds in $ORDER; do
    for r in $(fits $ds); do
      fitted $ds $r || continue
      gstep 8 $ds "draws_transport_${ds}_$r" -m discell.experiments.breakdown_draws \
        --dataset $ds --run $r --groups transport_mean,transport_dist
    done
  done
}

# -- main ---------------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only; waiting for: $WAIT_FOR"
n=0
while [ -n "$(queues_running)" ]; do
  [ $((n % 6)) -eq 0 ] && log "still running:$(queues_running) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the freeze: the waited-for queues have ended; starting (GPU $GPU only, concurrent with the other GPU-0 queues)"
rmdir $LOCK/gpu$GPU 2>/dev/null      # one instance at a time: a lock of ours is stale

missing_reads
draws_nontransport
cstep tables_partial -m discell.experiments.breakdown --all
if transport_gate; then
  transport_members
  cstep tables_final -m discell.experiments.breakdown --all
else
  log "HOLD: transport members not computed; see $ROOT/HOLD_transport.md"
fi
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
