#!/bin/bash
# The composition-weight confirmation (8.17 follow-up; devlog 2026-09-25
# morning, 8.17: "comp3 and comp5 on a third ovarian seed and on GSE and FF
# (2 seeds each) vs their controls"), tag advc.
#
# Final configuration (per-dataset flags of scripts/queue_2026-09-24_final.sh,
# but 200/20, --figures-every 200, --w-warmup-epochs 30); the only change is
# --adv-comp-weight c. Fits, in this order (GSE first, ovarian, FF last):
#   GSE core  advc_comp3_s{0,1}  advc_comp5_s{0,1}
#             control aw_ref0.1_s{0,1} (reused)
#   ovarian   adv_comp3_s2  adv_comp5_s{0,1,2}   (adv_comp3_s{0,1}: the ladder)
#             control wfix_warmup30_aw0.1_s{0,1,2} + aw_ref0.1_s{0,1} (reused)
#   FF        advc_comp3_s{0,1}
#             control wfix_warmup30_aw0.1_s{0,1} (= runs/aw_ref0.1_s{0,1}, reused)
# The preflight re-derives each arm's TrainConfig from these flags and aborts
# unless it equals every control's config.json (and the ladder's adv_comp3)
# but for run name, seed, git and adv_comp_weight; the controls' reads must
# exist (never refitted).
# Per run: fit -> validate morans,niche,probe -> degeneracy -> probe_regrade
# --force (graded against that dataset's uncontrolled500_s*) -> transport
# --read both --hvg 1000. A dead fit is recorded and read, never refitted.
# Two workers pull whole runs from the one ordered list; the partial table is
# refreshed after every run; at the end scripts/adv_confirm_table.py --at best
# writes <ovarian>/experiments/adv_confirm.{json,md} and $ROOT/DECISION_ADVC.json.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_adv_ladder.sh:
# FF steps gate at 18500 MiB, the others at 9500; one job of ours per GPU;
# MintFlow charged at its peak; a re-read after taking the lock and a settle
# delay; other queues (every scripts/logs/*/ folder): a card one of them locked
# less than FRESH s ago is skipped; a non-FF step leaves a card with >= 18.5 GB
# free to any live FF waiter (theirs or ours: $ROOT/waiting/ff.<pid>). Markers
# are keyed by step AND dataset AND run; a relaunch resumes.
# Launch detached:
#   setsid nohup scripts/queue_2026-09-25_adv_confirm.sh \
#     >> scripts/logs/adv_confirm_2026-09-25/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff

ROOT=scripts/logs/adv_confirm_2026-09-25
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
CLAIM=$ROOT/claims; WAIT=$ROOT/waiting
mkdir -p $QL $DONE $LOCK $FAILED $CLAIM $WAIT
GPUS="0 1"
NEED_OV=9500          # every non-FF dataset
NEED_FF=18500
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=600
WORKERS=2
COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 200 --patience 20 --figures-every 200 --w-warmup-epochs 30"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$FF]="--alpha-z 0.00035 --label-key graphclust")
declare -A PREFIX=([$GS]=advc [$OV]=adv [$FF]=advc)
# <dataset>:<comp weight>:<seed>, in run order
JOBS="$GS:3:0 $GS:3:1 $GS:5:0 $GS:5:1
      $OV:3:2 $OV:5:0 $OV:5:1 $OV:5:2
      $FF:3:0 $FF:3:1"
# <dataset>:<run> whose config and reads are reused
CONTROLS="$GS:aw_ref0.1_s0 $GS:aw_ref0.1_s1
          $OV:wfix_warmup30_aw0.1_s0 $OV:wfix_warmup30_aw0.1_s1
          $OV:wfix_warmup30_aw0.1_s2 $OV:aw_ref0.1_s0 $OV:aw_ref0.1_s1
          $OV:adv_comp3_s0 $OV:adv_comp3_s1
          $FF:wfix_warmup30_aw0.1_s0 $FF:wfix_warmup30_aw0.1_s1"

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
run_of() { echo ${PREFIX[$1]}_comp${2}_s$3; }   # run_of <dataset> <c> <seed>
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

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

held_elsewhere() {  # held_elsewhere <gpu> <free> <role>: someone has first call
  local g=$1 free=$2 role=$3 f pid dir now age
  now=$(date +%s)
  if [ "$role" = ov ] && [ "$free" -ge "$NEED_FF" ]; then  # an FF waiter first
    for f in scripts/logs/*/waiting/ff.*; do
      [ -e "$f" ] || continue
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

# Block until a GPU has >= $1 MiB free, no job of ours and nobody's first
# call, take it and echo its index. Role ff announces itself in $WAIT while
# waiting. Everything else goes to stderr.
acquire_gpu() {
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      held_elsewhere $g $free $role && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $g $free $role; then
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
step() {  # step <dataset> <marker> <cmd...>: a GPU step, retried on OOM
  local ds=$1 name=$2; shift 2
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $(need_of $ds) $(role_of $ds))
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

fit() {   # fit <dataset> <c> <seed>
  local ds=$1 run; run=$(run_of $1 $2 $3)
  local mk; mk=$(marker fit $ds $run)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  step $ds "$mk" uv run python -m discell.model.train --dataset $ds \
    --run-name $run ${FLAGS[$ds]} $COMMON --seed $3 --adv-comp-weight $2
}

battery() {  # battery <dataset> <run>: every read of one fit
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then
    log "no fit for $ds/$run; no reads"; return 0
  fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$(runs $ds)/$run/metrics.json')).get('dead_w_channel') else 1)"; then
    log "$ds/$run: dead context channel flagged by the trainer -- recorded, reads continue"
    grep -qx "$ds/$run" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo "$ds/$run" >> $ROOT/DEAD_RUNS.txt
  fi
  step $ds "$(marker validate $ds $run)" uv run python -m discell.model.validate \
    --dataset $ds --run $run --analyses morans,niche,probe
  step $ds "$(marker degeneracy $ds $run)" uv run python -m discell.model.degeneracy \
    --dataset $ds --run $run
  step $ds "$(marker probe_regrade $ds $run)" uv run python -m \
    discell.experiments.probe_regrade --dataset $ds --run $run --force
  step $ds "$(marker transport $ds $run)" uv run python -m discell.model.transport \
    --dataset $ds --run $run --read both --hvg 1000
  return 0
}

table() {  # the partial (or final) read-out; never fails the queue
  uv run python scripts/adv_confirm_table.py --at best > $QL/adv_confirm_table.log 2>&1
}

worker() {  # worker <k>: claim runs from the ordered list until none is left
  local job ds c s run
  for job in $JOBS; do
    IFS=: read -r ds c s <<< "$job"
    run=$(run_of $ds $c $s)
    mkdir "$CLAIM/${ds}_$run" 2>/dev/null || continue
    log "worker $1: $ds/$run"
    fit $ds $c $s
    battery $ds $run
    table || log "partial table failed (exit $?); see $QL/adv_confirm_table.log"
  done
  log "worker $1 finished"
}

check_controls() {  # the controls' reads exist (reused, never refitted)
  local x ds run f ok=0
  for x in $CONTROLS; do
    ds=${x%%:*}; run=${x#*:}
    for f in metrics.json history.jsonl validation/probe_blocks.json \
             degeneracy.json transport/transport_distribution.json; do
      if [ ! -f "$(runs $ds)/$run/$f" ]; then log "control $ds/$run lacks $f"; ok=1; fi
    done
  done
  return $ok
}

# -- preflight: the arms' flags reproduce every control's config.json ---------
preflight() {
  local specs=() ds
  for ds in $GS $OV $FF; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  uv run python - "$CONTROLS" "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
controls = [x.split(":", 1) for x in sys.argv[1].split()]
flags = dict(s.split("::", 1) for s in sys.argv[2:])
ALLOWED = {"run_name", "seed", "git", "adv_comp_weight"}
default = asdict(TrainConfig(dataset=""))
bad = 0
for ds, run in controls:
    ref = json.loads(Path(f"data/datasets/{ds}/runs/{run}/config.json").read_text())
    ok, d = True, []
    for c in (3.0, 5.0):
        args = vars(build_parser().parse_args(
            ["--dataset", ds, "--seed", str(ref["seed"])] + shlex.split(flags[ds])
            + ["--adv-comp-weight", str(c)]))
        args.pop("quiet")
        new = asdict(TrainConfig(**args))
        # a key the reference predates must sit at its default
        d += [f"{k}: {new.get(k)!r} vs {ref.get(k, '<absent>')!r}"
              for k in sorted(set(new) | set(ref)) if k not in ALLOWED
              and ((k in ref and new.get(k) != ref[k])
                   or (k not in ref and new[k] != default[k]))]
        ok = ok and not d and new["adv_comp_weight"] == c \
            and new["epochs"] == 200 and new["patience"] == 20 \
            and new["w_warmup_epochs"] == 30
    bad += not ok
    print(f"{ds} {run} (seed {ref['seed']}, adv_comp_weight "
          f"{ref.get('adv_comp_weight', '<absent>')}): "
          + ("the arms' config but for run name, seed and comp weight"
             if ok else "DIFFERS: " + "; ".join(d)))
sys.exit(1 if bad else 0)
EOF
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -rf $CLAIM/* $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, others $NEED_OV MiB, GPUs: $GPUS; code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi
if ! check_controls; then log "CONTROL READS MISSING -- stopping"; exit 1; fi

PIDS=""
for k in $(seq 1 $WORKERS); do
  (sleep $(((k - 1) * 20)); worker $k) & PIDS="$PIDS $!"
done
wait $PIDS
log "all workers finished"

log "read-out"
table; rc=$?
log "read-out done (exit $rc); see $QL/adv_confirm_table.log"
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Read-out (exit $rc): data/datasets/$OV/experiments/adv_confirm.{json,md};"
  echo "decision: $ROOT/DECISION_ADVC.json"
  python3 -c "import json; d=json.load(open('$ROOT/DECISION_ADVC.json')); print('Decision:', d['reading'], '(status', d['status'] + ')'); print('comp5 (dose, same rule, decides nothing):', d['comp5_dose']['passes_same_rule']); print('Ovarian control subsets (sensitivity):', d['sensitivity_ovarian_controls'])" 2>&1
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (trainer flag): $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
