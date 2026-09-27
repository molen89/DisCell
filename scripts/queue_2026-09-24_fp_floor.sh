#!/bin/bash
# The fixed false-positive floor, with vs without (todo 8.15b; devlog
# 2026-09-24 21:20 "Two training-side candidates, run in parallel before one
# re-pin", part B), tag fp.
#
# p_i = (1 - kappa - eta_i) rho_i + kappa rho_bar_i + eta_i u, u = 1/G,
# eta_i = min(lambda / l_i, 0.2), one lambda per section (the step-1 area
# rule: within-type Spearman of control counts vs segmented area 0.075
# ovarian / 0.034 GSE core, below 0.3 -> per section; step1.json here).
# Final configuration, kappa 0.1, 200/20, seeds 0 1 2, --fp-floor:
#   ovarian  fp_s<seed>          --alpha-z 0.0035
#            control reused: wfix_warmup30_aw0.1_s{0,1,2} (every read exists)
#   GSE core fp_s<seed>          --alpha-z 0.0018 --variant pdl018d --tile-cells 2048
#            control reused: aw_ref0.1_s{0,1} (+ the 6b.1 signalling share,
#            which they lack); fitted here: fp_control_s2 (the same
#            configuration without the floor, seed 2)
#   ovarian  fp_area_s<seed>     --fp-floor --fp-area (coordinator, 21:50:
#            the area arm, lambda_i proportional to segmented area with the
#            section total fixed -- the Spearman rule has no power on 97-99 %
#            zero control counts, recorded as a rule defect); after the
#            per-section fp runs, same post-processing
#   all      --alpha-w 0.1 --kappa 0.1 --d-w 6 --gat-sources type_only
#            --w-warmup-epochs 30 --epochs 200 --patience 20 --figures-every 200
# Per fit: validate morans,niche,probe -> degeneracy -> probe_regrade --run
# --force (graded against uncontrolled500_s{0,1}) -> transport --read both
# --hvg 1000 -> external_criteria signalling-share. Then the probe_regrade
# table (group fp) per dataset and scripts/fp_table.py --at best ->
# <ovarian>/experiments/fp_floor.{json,md}. A dead fit is recorded and read,
# not refitted (failed legs are findings).
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_r12_arms.sh
# (gate 9500 MiB for ovarian and GSE, one job of ours per GPU, MintFlow
# charged at its peak, a re-read after taking the lock, a settle delay), with
# the courtesies of scripts/queue_2026-09-24_adv_ladder.sh scanning every
# scripts/logs/*/ folder: while any queue's FF step waits (waiting/ff.<pid>,
# pid alive) a card with >= 18.5 GB free is left to it, and a card another
# queue locked less than FRESH s ago is skipped. Markers are keyed by step
# AND dataset AND run; a relaunch resumes. Launch detached:
#   setsid nohup scripts/queue_2026-09-24_fp_floor.sh \
#     >> scripts/logs/fp_floor_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OV=xenium_prime_ovarian_cancer_ffpe
GS=gse315411_pdltma06_11_prime_solo
ROOT=scripts/logs/fp_floor_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $FAILED
GPUS="0 1"
NEED_OV=9500          # ovarian and GSE-core fits and reads
NEED_FF=18500         # an FF step of another queue: such a card is theirs first
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=300
SEEDS="0 1 2"
COMMON="--alpha-w 0.1 --kappa 0.1 --d-w 6 --gat-sources type_only
        --w-warmup-epochs 30 --epochs 200 --patience 20 --figures-every 200"
declare -A FLAGS=(
  [$OV]="--alpha-z 0.0035"
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048")
declare -A CONTROLS=(
  [$OV]="wfix_warmup30_aw0.1_s0 wfix_warmup30_aw0.1_s1 wfix_warmup30_aw0.1_s2"
  [$GS]="aw_ref0.1_s0 aw_ref0.1_s1 fp_control_s2")

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              discell/model/fp_floor.py discell/model/transport.py \
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

cpu_step() {  # cpu_step <marker> <cmd...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}

has() {  # has <dataset> <run> <file>: an existing output counts as the step
  [ -f "$(runs $1)/$2/$3" ] || [ -f "data/datasets/$1/$3" ]
}

fit() {   # fit <dataset> <run> <seed> [extra flags]
  local ds=$1 run=$2 seed=$3 mk; shift 3
  mk=$(marker fit $ds $run)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  step "$mk" uv run python -m discell.model.train --dataset $ds --run-name $run \
    ${FLAGS[$ds]} $COMMON --seed $seed "$@"
}

battery() {  # battery <dataset> <run>: every read of one fit, the missing ones
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then
    log "no fit for $ds/$run; no reads"; return 0
  fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$(runs $ds)/$run/metrics.json')).get('dead_w_channel') else 1)"; then
    log "$ds/$run: dead context channel flagged by the trainer -- recorded, reads continue"
    grep -qx "$ds/$run" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo "$ds/$run" >> $ROOT/DEAD_RUNS.txt
  fi
  has $ds $run validation/probe_blocks.json && has $ds $run validation/validation.json \
    && touch "$DONE/$(marker validate $ds $run)" "$DONE/$(marker probe_regrade $ds $run)"
  has $ds $run degeneracy.json && touch "$DONE/$(marker degeneracy $ds $run)"
  has $ds $run transport/transport_distribution.json \
    && touch "$DONE/$(marker transport $ds $run)"
  has $ds $run experiments/external_signalling_share_$run.json \
    && touch "$DONE/$(marker signalling $ds $run)"
  step "$(marker validate $ds $run)" uv run python -m discell.model.validate \
    --dataset $ds --run $run --analyses morans,niche,probe
  step "$(marker degeneracy $ds $run)" uv run python -m discell.model.degeneracy \
    --dataset $ds --run $run
  step "$(marker probe_regrade $ds $run)" uv run python -m \
    discell.experiments.probe_regrade --dataset $ds --run $run --force
  step "$(marker transport $ds $run)" uv run python -m discell.model.transport \
    --dataset $ds --run $run --read both --hvg 1000
  step "$(marker signalling $ds $run)" uv run python -m \
    discell.experiments.external_criteria signalling-share --dataset $ds --run $run
  return 0
}

lane() {  # lane <dataset>: the controls' missing reads, then the fp fits
  local ds=$1 s run
  for run in ${CONTROLS[$ds]}; do
    case $run in fp_control_s*) fit $ds $run ${run#fp_control_s} ;; esac
    battery $ds $run
  done
  for s in $SEEDS; do fit $ds fp_s$s $s --fp-floor; battery $ds fp_s$s; done
  if [ $ds = $OV ]; then        # the area arm, after the per-section runs
    for s in $SEEDS; do
      fit $ds fp_area_s$s $s --fp-floor --fp-area; battery $ds fp_area_s$s
    done
  fi
  cpu_step "$(marker probe_table2 $ds fp)" uv run python -m \
    discell.experiments.probe_regrade --dataset $ds --table fp \
    --runs 'fp_s*' 'fp_area_s*' 'fp_control_s*' ${CONTROLS[$ds]}
  log "$ds lane finished"
}

# -- preflight: --fp-floor is the only difference from each control ------------
preflight() {
  uv run python - "$COMMON" "${FLAGS[$OV]}" "${FLAGS[$GS]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
common, flags_ov, flags_gs = sys.argv[1:4]
OV, GS = "xenium_prime_ovarian_cancer_ffpe", "gse315411_pdltma06_11_prime_solo"
REF = {OV: ["wfix_warmup30_aw0.1_s0", "wfix_warmup30_aw0.1_s1",
            "wfix_warmup30_aw0.1_s2"],
       GS: ["aw_ref0.1_s0", "aw_ref0.1_s1", "aw_ref0.1_s0"]}  # s2: vs s0 but seed
ALLOWED = {"run_name", "seed", "git", "fp_floor", "fp_area", "fp_lambda",
           "fp_cap_share"}
default = asdict(TrainConfig(dataset=""))
bad = 0
for ds, flags in ((OV, flags_ov), (GS, flags_gs)):
    for seed in (0, 1, 2):
        for fp, area in ((True, False), (False, False), (True, True)):
            if not fp and not (ds == GS and seed == 2):
                continue                         # only fp_control_s2 is fitted
            if area and ds != OV:
                continue                         # the area arm: ovarian only
            argv = ["--dataset", ds, "--seed", str(seed)] + shlex.split(flags) \
                + shlex.split(common) + (["--fp-floor"] if fp else []) \
                + (["--fp-area"] if area else [])
            args = vars(build_parser().parse_args(argv))
            args.pop("quiet")
            new = asdict(TrainConfig(**args))
            ref = json.loads((Path("data/datasets") / ds / "runs"
                              / REF[ds][seed] / "config.json").read_text())
            diffs = [f"{k}: {new.get(k)!r} vs control {ref.get(k, '<absent>')!r}"
                     for k in sorted(set(new) | set(ref)) if k not in ALLOWED
                     and ((k in ref and new.get(k) != ref[k])
                          or (k not in ref and new[k] != default[k]))]
            ok = not diffs and new["fp_floor"] == fp and new["fp_area"] == area
            tag = "fp_area" if area else "fp" if fp else "fp_control"
            print(f"{ds} {tag}_s{seed}: " + (
                "identical to the control but for the floor" if ok
                else "DIFFERS: " + "; ".join(diffs)))
            bad += not ok
sys.exit(1 if bad else 0)
EOF
}

# -- main -----------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
trap 'rmdir $LOCK/gpu* 2>/dev/null' EXIT
log "queue start (gate $NEED_OV MiB, GPUs: $GPUS; code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi

lane $OV & P1=$!
(sleep 60; lane $GS) & P2=$!
wait $P1 $P2
log "all lanes finished"

log "read-out"
uv run python scripts/fp_table.py --at best > $QL/fp_table.log 2>&1
rc=$?
log "read-out done (exit $rc); see $QL/fp_table.log"
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Read-out (exit $rc): data/datasets/$OV/experiments/fp_floor.{json,md}"
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (trainer flag): $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
  echo
  sed -n '/Moved by more than one/p' data/datasets/$OV/experiments/fp_floor.md 2>/dev/null
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
