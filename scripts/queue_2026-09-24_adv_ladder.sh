#!/bin/bash
# The adversary-capacity ladder (8.17; devlog 2026-09-24 21:20 "Two
# training-side candidates, run in parallel before one re-pin", part A), tag adv.
#
# Ovarian core, final configuration (warm-up 30, alpha_z 0.0035, alpha_w 0.1,
# kappa 0.1, d_w 6, type_only, 200/20), seeds 0 1:
#   control           wfix_warmup30_aw0.1_s{0,1}, reused (6 head steps, width
#                     64): the preflight re-derives its config from these flags
#                     and aborts otherwise; its reads (validate, probe_blocks,
#                     degeneracy, transport) exist and are reused
#   steps12           adv_steps12_s<seed>          --adv-head-steps 12
#   width128          adv_width128_s<seed>         --adv-head-width 128
#   ens3              adv_ens3_s<seed>             --adv-ensemble 3
#   comp3             adv_comp3_s<seed>            --adv-comp-weight 3
#   steps12_width128  adv_steps12_width128_s<seed> --adv-head-steps 12 --adv-head-width 128
# Per run: fit -> validate morans,niche,probe -> degeneracy -> probe_regrade
# --run --force (the control's record schema, graded against
# uncontrolled500_s{0,1}, with the in-trainer legacy check; it overwrites the
# validate probe record) -> transport --read both --hvg 1000 (k-means).
# Then scripts/adv_head_timing.py (head-step wall time, all arms in one
# process) and scripts/adv_table.py --at best -> experiments/adv_ladder.{json,md}
# and $ROOT/DECISION_ADV.json. A dead fit is recorded and read, not refitted.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_r12_arms.sh
# (gate 9500 MiB, one job of ours per GPU, MintFlow charged at its peak, a
# re-read after taking the lock, a settle delay). The courtesies to other
# queues scan every scripts/logs/*/ folder rather than a fixed list, so a
# queue launched after this one is covered: while any queue's FF step waits
# (scripts/logs/*/waiting/ff.<pid>, pid alive) a card with >= 18.5 GB free is
# left to it, and a card another queue locked less than FRESH s ago is
# skipped. Markers are keyed by step AND dataset AND run; a relaunch resumes.
# Launch detached:
#   setsid nohup scripts/queue_2026-09-24_adv_ladder.sh \
#     >> scripts/logs/adv_ladder_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OV=xenium_prime_ovarian_cancer_ffpe
ROOT=scripts/logs/adv_ladder_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
GPUS="0 1"
NEED_OV=9500          # an ovarian fit is ~6-7 GB resident; post-hoc reads reuse it
NEED_FF=18500         # an FF fit of another queue: such a card is theirs first
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=600
SEEDS="0 1"
CONTROL=wfix_warmup30_aw0.1_s   # + seed
OVA="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6 --gat-sources type_only
     --w-warmup-epochs 30 --epochs 200 --patience 20 --figures-every 200"
ARMS="steps12 width128 ens3 comp3 steps12_width128"

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$OV/runs; }
marker() { echo "${1}_${OV}_${2}"; }   # marker <step> <run>
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }
flags_of() {  # flags_of <arm>
  case $1 in
    steps12) echo "--adv-head-steps 12" ;;
    width128) echo "--adv-head-width 128" ;;
    ens3) echo "--adv-ensemble 3" ;;
    comp3) echo "--adv-comp-weight 3" ;;
    steps12_width128) echo "--adv-head-steps 12 --adv-head-width 128" ;;
    *) echo "UNKNOWN_ARM_$1" ;;
  esac
}

# -- GPU picker ---------------------------------------------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
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

foreign_hold() {  # foreign_hold <gpu> <free>: another queue has first call on it
  local g=$1 free=$2 f pid dir now age
  now=$(date +%s)
  if [ "$free" -ge "$NEED_FF" ]; then           # a waiting FF step goes first
    for f in scripts/logs/*/waiting/ff.*; do
      [ -e "$f" ] || continue
      case "$f" in $ROOT/*) continue ;; esac
      pid=${f##*.}
      kill -0 "$pid" 2>/dev/null && return 0
    done
  fi
  for dir in scripts/logs/*/locks/gpu$g; do     # may not have allocated yet
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

# Block until a GPU has >= $1 MiB free, no job of ours and no foreign first
# call, take it and echo its index. Everything else goes to stderr.
acquire_gpu() {
  local need=$1 g free waited=0
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      foreign_hold $g $free && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! foreign_hold $g $free; then
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

# -- steps --------------------------------------------------------------------
step() {  # step <marker> <cmd...>: a GPU step, retried on OOM
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $NEED_OV)
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
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

fit() {   # fit <arm> <seed>
  local run=adv_${1}_s$2 mk; mk=$(marker fit adv_${1}_s$2)
  if [ -f "$(runs)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2046
  step "$mk" uv run python -m discell.model.train --dataset $OV --run-name $run \
    $OVA --seed $2 $(flags_of $1)
}

battery() {  # battery <run>: every read of one fit
  local run=$1
  if [ ! -f "$(runs)/$run/metrics.json" ]; then
    log "no fit for $run; no reads"; return 0
  fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$(runs)/$run/metrics.json')).get('dead_w_channel') else 1)"; then
    log "$run: dead context channel flagged by the trainer -- recorded, reads continue"
    grep -qx "$run" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo $run >> $ROOT/DEAD_RUNS.txt
  fi
  step "$(marker validate $run)" uv run python -m discell.model.validate \
    --dataset $OV --run $run --analyses morans,niche,probe
  step "$(marker degeneracy $run)" uv run python -m discell.model.degeneracy \
    --dataset $OV --run $run
  step "$(marker probe_regrade $run)" uv run python -m \
    discell.experiments.probe_regrade --dataset $OV --run $run --force
  step "$(marker transport $run)" uv run python -m discell.model.transport \
    --dataset $OV --run $run --read both --hvg 1000
  return 0
}

lane_arm() {  # lane_arm <arm>
  local s
  for s in $SEEDS; do fit $1 $s; battery adv_${1}_s$s; done
  log "$1 lane finished"
}

check_control() {  # the control's reads exist (reused, never refitted)
  local s run f ok=0
  for s in $SEEDS; do
    run=$CONTROL$s
    for f in metrics.json history.jsonl validation/validation.json \
             validation/probe_blocks.json degeneracy.json \
             transport/transport_distribution.json; do
      if [ ! -f "$(runs)/$run/$f" ]; then log "control $run lacks $f"; ok=1; fi
    done
  done
  return $ok
}

# -- preflight: the arms' flags reproduce the control's config.json -----------
preflight() {
  uv run python - "$OVA" "$(runs)" "$CONTROL" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
flags, runs, control = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ARMS = {"steps12": ("--adv-head-steps 12", {"adv_steps": 12}),
        "width128": ("--adv-head-width 128", {"adv_hidden": 128}),
        "ens3": ("--adv-ensemble 3", {"adv_ensemble": 3}),
        "comp3": ("--adv-comp-weight 3", {"adv_comp_weight": 3.0}),
        "steps12_width128": ("--adv-head-steps 12 --adv-head-width 128",
                             {"adv_steps": 12, "adv_hidden": 128})}
ALLOWED = {"run_name", "seed", "git"}
default = asdict(TrainConfig(dataset=""))
def diffs(new, ref, allowed):
    # a key the reference predates must sit at its default
    return [f"{k}: {new.get(k)!r} vs {ref.get(k, '<absent>')!r}"
            for k in sorted(set(new) | set(ref)) if k not in allowed
            and ((k in ref and new.get(k) != ref[k])
                 or (k not in ref and new[k] != default[k]))]
bad = 0
for arm, (arm_flags, knobs) in ARMS.items():
    for seed in (0, 1):
        args = vars(build_parser().parse_args(
            ["--dataset", "xenium_prime_ovarian_cancer_ffpe", "--seed", str(seed)]
            + shlex.split(flags) + shlex.split(arm_flags)))
        args.pop("quiet")
        new = asdict(TrainConfig(**args))
        ref = json.loads((runs / f"{control}{seed}" / "config.json").read_text())
        d = diffs(new, ref, ALLOWED | set(knobs))
        ok = not d and all(new[k] == v for k, v in knobs.items()) \
            and ref["seed"] == seed
        print(f"{arm} s{seed}: " + (f"identical to {control}{seed} but for {knobs}"
                                    if ok else "DIFFERS: " + "; ".join(d)))
        bad += not ok
# the sensitivity control (decides nothing): the fresh reference pair
for seed in (0, 1):
    path = runs / f"aw_ref0.1_s{seed}" / "config.json"
    ref = json.loads((runs / f"{control}{seed}" / "config.json").read_text())
    if path.exists():
        d = diffs({**default, **json.loads(path.read_text())}, ref, ALLOWED)
        print(f"sensitivity aw_ref0.1_s{seed}: "
              + ("same configuration as the control" if not d else "DIFFERS: " + "; ".join(d)))
sys.exit(1 if bad else 0)
EOF
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
trap 'rmdir $LOCK/gpu* 2>/dev/null' EXIT
log "queue start (gate $NEED_OV MiB, GPUs: $GPUS; code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi
if ! check_control; then log "CONTROL READS MISSING -- stopping"; exit 1; fi

PIDS=""; delay=0
for arm in $ARMS; do
  (sleep $delay; lane_arm $arm) & PIDS="$PIDS $!"
  delay=$((delay + 30))
done
wait $PIDS
log "all lanes finished"

step "$(marker head_timing adv_all)" uv run python scripts/adv_head_timing.py

log "read-out"
uv run python scripts/adv_table.py --at best > $QL/adv_table.log 2>&1
rc=$?
log "read-out done (exit $rc); see $QL/adv_table.log"
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Read-out (exit $rc): data/datasets/$OV/experiments/adv_ladder.{json,md};"
  echo "decision: $ROOT/DECISION_ADV.json"
  python3 -c "import json; d=json.load(open('$ROOT/DECISION_ADV.json')); print('Decision:', d['reading'], '(status', d['status'] + ')'); print('Sensitivity (fresh control, decides nothing):', d['sensitivity_fresh_control']['reading'])" 2>&1
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (trainer flag): $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
