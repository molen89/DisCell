#!/bin/bash
# 8.9b: warm-up on BOTH KL terms (devlog 2026-09-24 pre-registration).
#
# One new arm, klwarm30 = --kl-warmup-epochs 30, tag wfixb:
#   FF      alpha_w 0.1, seeds 0 1 2 3  (they decide)
#   ovarian alpha_w 0.1, seeds 0 1 2
# The 8.9 control and w-only warm-up runs (wfix_control_aw0.1_s*,
# wfix_warmup30_aw0.1_s*) are the comparison rows and are NOT refitted.
#
# Picker / lock / marker machinery copied from
# scripts/queue_2026-09-23_wcollapse.sh. Three changes, all small:
#   * FF priority: while an FF step is waiting for a GPU, the ovarian lane
#     does not take a card that could host that FF step (>= NEED_FF free);
#     it only fills cards FF cannot use (today: GPU 0 beside MintFlow).
#   * the re-read after taking the lock comes after a SETTLE delay, so a
#     neighbour's job that has launched but not yet allocated is seen.
#   * MintFlow's card is charged at MintFlow's peak while it runs (gpu_free).
#
# Every step is idempotent: a DONE marker keyed by step AND dataset AND run
# (docs/issues.md W-q1: FF and ovarian share run names) -- and, for fits,
# metrics.json -- skips it, so relaunching resumes. Launch detached:
#   setsid nohup scripts/queue_2026-09-24_wcollapse_b.sh \
#     >> scripts/logs/wcollapse_b_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
# allocator-only setting: less fragmentation on a GPU shared with other agents
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FF=xenium_prime_human_ovary_ff
OV=xenium_prime_ovarian_cancer_ffpe
ROOT=scripts/logs/wcollapse_b_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting
mkdir -p $QL $DONE $LOCK $WAIT
GPUS="0 1"
# measured on this box (8.9): an FF 200/20 fit is ~17.2 GB resident, an
# ovarian fit over 6.1 GB; validate/degeneracy/transport reuse the fit gate
NEED_FF=18500
NEED_OV=9500
TRIES=10
SETTLE=45

TAG=wfixb
ARM=klwarm30
ARMFLAGS="--kl-warmup-epochs 30"
FFA="--label-key graphclust --alpha-z 0.00035 --alpha-w 0.1 --kappa 0.1 --d-w 6
     --gat-sources type_only --epochs 200 --patience 20 --figures-every 200"
OVA="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6
     --gat-sources type_only --epochs 200 --patience 20 --figures-every 200"

log() { echo "[$(date '+%F %T')] $*"; }

# The MintFlow baseline fit on GPU 0 must not be disturbed, and its usage
# swings (3-8 GB, measured 2026-09-24): while it runs, its GPU is charged at
# its PEAK, not at whatever it holds when the picker looks. That keeps FF fits
# off its card (15.5 GB is the most that can ever read free there) while an
# ovarian fit still fits beside it.
GUARD_PATTERN=run_mintflow.py
GUARD_PEAK=9000

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
  echo $free
}

# Block until some GPU has >= $1 MiB free AND is not already running one of our
# jobs, then take it. $2 is the role: "ff" steps announce themselves in $WAIT
# while they wait; "ov" steps skip any card an FF step could use while one is
# waiting. Echoes the index on stdout -- every other word goes to stderr.
acquire_gpu() {
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      if [ "$role" = ov ] && [ "$free" -ge "$NEED_FF" ] \
         && ls "$WAIT"/ff.* > /dev/null 2>&1; then
        continue                                      # FF goes first
      fi
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # someone of ours has it
      # re-read after taking the lock and a settle delay: a neighbour may have
      # just launched a job that has not allocated yet
      sleep $SETTLE
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ]; then
        [ "$role" = ff ] && rm -f "$me"
        echo $g; return 0
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

step() {  # step <need_MiB> <role> <marker> <cmd...>
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try rc gpu
  for try in $(seq $TRIES); do
    gpu=$(acquire_gpu $need $role)
    log "GPU$gpu start $name (try $try/$TRIES)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1
    rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
    log "GPU$gpu FAIL $name (exit $rc); see $QL/$name.log"
    sleep 300
  done
  log "GIVING UP on $name after $TRIES tries"
  return 1
}

# DONE markers are dataset-qualified (docs/issues.md W-q1)
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>

role_of() { [ "$1" = "$OV" ] && echo ov || echo ff; }
need_of() { [ "$1" = "$OV" ] && echo $NEED_OV || echo $NEED_FF; }

fit() {   # fit <dataset> <run> <flags...>
  local ds=$1 run=$2; shift 2
  local mk=$(marker fit $ds $run)
  if [ -f "data/datasets/$ds/runs/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  step $(need_of $ds) $(role_of $ds) "$mk" uv run python -m discell.model.train \
    --dataset $ds --run-name $run "$@"
}

post() {  # post <dataset> <run> [transport]
  local ds=$1 run=$2 tr=${3:-no}
  [ -f "$DONE/$(marker fit $ds $run)" ] || {
    log "no fit for $ds/$run; skipping post"; return 0; }
  local need=$(need_of $ds) role=$(role_of $ds)
  step $need $role "$(marker validate $ds $run)" \
    uv run python -m discell.model.validate \
    --dataset $ds --run $run --analyses morans,niche
  step $need $role "$(marker degeneracy $ds $run)" \
    uv run python -m discell.model.degeneracy --dataset $ds --run $run
  [ "$tr" = "transport" ] && step $need $role "$(marker transport $ds $run)" \
    uv run python -m discell.model.transport --dataset $ds --run $run \
    --read both --hvg 1000
  return 0
}

ff_seeds() {   # ff_seeds <seed>...
  local s run
  for s in "$@"; do
    run="${TAG}_${ARM}_aw0.1_s${s}"
    fit $FF "$run" $FFA --seed $s $ARMFLAGS
    post $FF "$run"
  done
}

# Two FF sub-lanes on DISJOINT seeds: while MintFlow holds GPU 0 only GPU 1
# can host an FF fit and they simply take turns on it; if GPU 0 frees, both
# run at once. Seeds 0 and 1 lead so a two-seed read exists first.
laneFF_A() { ff_seeds 0 2; log "FF sub-lane A finished"; }
laneFF_B() { ff_seeds 1 3; log "FF sub-lane B finished"; }

laneOV() {
  local s run
  for s in 0 1 2; do
    run="${TAG}_${ARM}_aw0.1_s${s}"
    fit $OV "$run" $OVA --seed $s $ARMFLAGS
    post $OV "$run" transport
  done
  log "ovarian lane finished"
}

# only one queue instance runs at a time, so any lock or wait flag is stale
rmdir $LOCK/gpu* 2>/dev/null
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, ovarian $NEED_OV MiB, GPUs: $GPUS)"

laneFF_A & PA=$!
# stagger B so A takes the first FF slot (seed 0 before seed 1)
(sleep 90; laneFF_B) & PB=$!
(sleep 10; laneOV) & POV=$!   # after A has announced itself
wait $PA $PB $POV
log "all lanes finished"

log "read-out"
uv run python scripts/wcollapse_b_table.py > $QL/wcollapse_b_table.log 2>&1
log "read-out done (exit $?); see $QL/wcollapse_b_table.log"
log "queue finished"
