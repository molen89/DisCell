#!/bin/bash
# The uncontrolled reference of the per-block probe at the final budget
# (coordinator, 2026-09-24 ~17:30: the 200/20 references of
# scripts/queue_2026-09-24_uncontrolled.sh stopped far earlier than the final
# fits on FF -- epoch 54 vs 179). alpha_a = 0 fits at each dataset's final_s0
# flags, 500/40, run names uncontrolled500_s<seed>: seed 0 on all four
# datasets, seed 1 on GSE and ovarian. Every fraction is re-expressed against
# the seed mean of these; the 200/20 fraction stays in the records. Each fit is
# graded on the CPU (GSE also on the dual section; FF only with >= 45 GB RAM
# free) and every record of that dataset is re-verdicted.
#
# GPU gates as the other queues (FF 18500 MiB, others 9500; one job of ours
# per GPU; another queue's waiting FF step has first call on any card that
# could host it). A fresh lock of another queue reserves its largest job on
# that card (18500 for the final re-pin and 8.9b queues, which run FF steps;
# 9500 for the R12 arms and alpha_w ladder stage 1, which do not, and only
# for 180 s, by which a non-FF job has allocated) -- amended 16:50 after the
# lane starved behind back-to-back short foreign steps with both cards 70 %
# free. A preflight re-derives every TrainConfig
# field from the flags and aborts unless only alpha_a / epochs / patience /
# seed / run name differ from final_s0/config.json. DONE markers are
# keyed by step AND dataset; a relaunch resumes.
#
#   setsid nohup bash scripts/queue_2026-09-24_uncontrolled500.sh \
#     >> scripts/logs/uncontrolled500_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ROOT=scripts/logs/uncontrolled500_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting
FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $WAIT $FAILED
GPUS="0 1"
NEED_FF=18500
NEED_OV=9500
TRIES=10; TRIES_OTHER=3; SETTLE=45; FRESH=600

# final_s0's flags (scripts/queue_2026-09-24_final.sh) with alpha_a 0
COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30"
MIN_RAM_FF=45         # GB MemAvailable before an FF grade on the CPU
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002 --label-key graphclust"
  [$FF]="--alpha-z 0.00035 --label-key graphclust")

log() { echo "[$(date '+%F %T')] $*"; }
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
code_hash() { sha256sum discell/model/train.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

# -- GPU picker (scripts/queue_2026-09-24_awladder.sh) --------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
# other queues: <log root>:<process pattern>:<reserve MiB>. A fresh lock of
# theirs (launched, maybe not yet allocated) reserves their largest job on
# that card: 18500 for queues that run FF steps (the card is theirs), 9500 for
# the non-FF ones (R12 arms: ovarian; alpha_w ladder stage 1: GSE + ovarian,
# SCOPE.json -- 18500 again should its scope gain FF).
AW_RESERVE=9500
grep -q "$FF" scripts/logs/awladder_2026-09-24/SCOPE.json 2>/dev/null && AW_RESERVE=18500
AW_FRESH=180
[ $AW_RESERVE -gt 9500 ] && AW_FRESH=$FRESH
# 4th field: how long a lock counts as fresh (not yet allocated): a non-FF
# job has allocated within ~2 min; past that its memory is in "free" already
FOREIGN="scripts/logs/final_2026-09-24:queue_2026-09-24_final.sh:18500:$FRESH
scripts/logs/r12_arms_2026-09-24:queue_2026-09-24_r12_arms.sh:9500:180
scripts/logs/awladder_2026-09-24:queue_2026-09-24_awladder.sh:$AW_RESERVE:$AW_FRESH
scripts/logs/wcollapse_b_2026-09-24:queue_2026-09-24_wcollapse_b.sh:18500:$FRESH"

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
  echo $free
}

foreign_hold() {  # foreign_hold <gpu> <free> <need>: GPU $1 is not ours to take
  local g=$1 free=$2 need=$3 root pat reserve fresh now age
  now=$(date +%s)
  while IFS=: read -r root pat reserve fresh; do
    [ -n "$root" ] || continue
    pgrep -f "$pat" > /dev/null || continue
    if [ "$free" -ge "$NEED_FF" ] && ls "$root"/waiting/ff.* > /dev/null 2>&1; then
      return 0                                        # their FF step goes first
    fi
    if [ -d "$root/locks/gpu$g" ]; then
      age=$((now - $(stat -c %Y "$root/locks/gpu$g")))
      [ $age -lt $fresh ] && free=$((free - reserve))
    fi
  done <<< "$FOREIGN"
  [ "$free" -lt "$need" ]
}

acquire_gpu() {
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      foreign_hold $g $free $need && continue
      if [ "$role" = ov ] && [ "$free" -ge "$NEED_FF" ] \
         && ls "$WAIT"/ff.* > /dev/null 2>&1; then
        continue                                      # our FF fit goes first
      fi
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue
      sleep $SETTLE
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! foreign_hold $g $free $need; then
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

gpu_step() {  # gpu_step <need> <role> <marker> <cmd...>
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $need $role)
    printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$(code_hash)" \
      >> $ROOT/code_hashes.tsv
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
    cp $QL/$name.log $QL/$name.try$try.log
    if grep -qE "OutOfMemoryError|CUDA out of memory" $QL/$name.log; then
      log "FAIL $name: CUDA OOM (try $try/$TRIES)"
    else
      other=$((other + 1))
      log "FAIL $name (exit $rc; attempt $other/$TRIES_OTHER)"
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
  case $name in *"$FF"*)                       # an FF assembly is ~30 GB
    until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" \
            -ge $MIN_RAM_FF ]; do sleep 60; done ;;
  esac
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}

MOD="uv run python -m discell.experiments.probe_regrade"

lane() {  # lane <dataset> <seed>: fit, grade, re-verdict the dataset's records
  local ds=$1 seed=$2 run=uncontrolled500_s$2
  gpu_step $(need_of $ds) $(role_of $ds) "fit_${ds}_${run}" \
    uv run python -m discell.model.train --dataset $ds --run-name $run \
    ${FLAGS[$ds]} $COMMON --seed $seed || return 1
  cpu_step "grade_${ds}_${run}" $MOD --dataset $ds --run $run
  cpu_step "reverdict_${ds}_${run}" $MOD --dataset $ds --reverdict
  if [ "$ds" = "$GS" ]; then
    cpu_step "grade_${GD}_${run}" $MOD --dataset $GD --config-from $GS --run $run
    cpu_step "reverdict_${GD}_${run}" $MOD --dataset $GD --config-from $GS --reverdict
  fi
}

preflight() {
  local specs=() ds
  for ds in $GS $OV $LU $FF; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  uv run python - "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
ALLOWED = {"run_name", "git", "alpha_a"}
default = asdict(TrainConfig(dataset=""))
bad = 0
for spec in sys.argv[1:]:
    ds, flags = spec.split("::", 1)
    args = vars(build_parser().parse_args(["--dataset", ds, "--seed", "0"]
                                          + shlex.split(flags)))
    args.pop("quiet")
    new = asdict(TrainConfig(**args))
    ref = json.load(open(f"data/datasets/{ds}/runs/final_s0/config.json"))
    diffs = [f"{k}: {new.get(k)!r} vs final_s0 {ref.get(k, '<absent>')!r}"
             for k in sorted(set(new) | set(ref)) if k not in ALLOWED
             and ((k in ref and new.get(k) != ref[k])
                  or (k not in ref and new[k] != default[k]))]
    print(f"{ds}: alpha_a {new['alpha_a']}, {new['epochs']}/{new['patience']}, "
          f"seed {new['seed']}: " + ("identical to final_s0 otherwise" if not diffs
                                     else "DIFFERS: " + "; ".join(diffs)))
    bad += bool(diffs) or new["alpha_a"] != 0
sys.exit(1 if bad else 0)
EOF
}

rmdir $LOCK/gpu* 2>/dev/null
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (gates FF $NEED_FF / others $NEED_OV MiB; trainer $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi
lane $FF 0 & PF=$!
(sleep 20; lane $GS 0; lane $GS 1; lane $OV 0; lane $OV 1; lane $LU 0) & PS=$!
wait $PF $PS
log "queue finished"
