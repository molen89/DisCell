#!/bin/bash
# The three query/prior ablations (devlog 2026-09-23 pre-registration):
#   control | --query type_free | --query image | --prior-type-free
# Ovarian core, seeds 0 1 2, 200/20, at whatever objective the w-collapse test
# selected -- the queue WAITS for that verdict before it fits anything.
# One fit at a time, on whichever GPU has room at launch time (both cards are
# shared with other agents' queues). Every step is idempotent: a DONE marker
# (and, for fits, metrics.json) skips it, so relaunching resumes. Launch:
#   setsid nohup scripts/queue_2026-09-23_queryprior.sh \
#     > scripts/logs/queryprior_2026-09-23/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OV=xenium_prime_ovarian_cancer_ffpe
ROOT=scripts/logs/queryprior_2026-09-23
QL=$ROOT/logs; DONE=$ROOT/done; mkdir -p $QL $DONE
WFIX=scripts/logs/wcollapse_2026-09-23
DECISION=$WFIX/DECISION.json

# the pre-registered operating point (ovarian core, devlog 2026-09-23)
BASE="--alpha-z 0.0035 --alpha-w 0.1 --kappa 0.1 --d-w 6
      --gat-sources type_only --epochs 200 --patience 20 --figures-every 200"
SEEDS="0 1 2"
ARMS="control query_type_free query_image prior_type_free"

log() { echo "[$(date '+%F %T')] $*"; }

arm_flags() {
  case "$1" in
    control)         echo "" ;;
    query_type_free) echo "--query type_free" ;;
    query_image)     echo "--query image" ;;
    prior_type_free) echo "--prior-type-free" ;;
    *) log "unknown arm $1"; exit 2 ;;
  esac
}

# -- the wait: the collapse-fix agent's verdict ----------------------------
# DECISION.json is the hand-off (scripts/wcollapse_table.py writes it and
# withholds it while that grid is incomplete). If it never appears but the
# agent's report names a selected arm, derive the flags from the report --
# the mapping is the pre-registration's own: warm-up -> --w-warmup-epochs 30,
# free bits -> --w-free-bits 0.05. Otherwise poll every 10 minutes forever.
decision_from_report() {
  local report=$WFIX/AGENT_REPORT.md flags="" line
  [ -f "$report" ] || return 1
  line=$(grep -i -m1 '^\**Selected' "$report")
  [ -n "$line" ] || return 1
  case "$line" in *none*|*"grid incomplete"*) return 1 ;; esac
  case "$line" in *warmup30*|*warm-up*) flags="--w-warmup-epochs 30" ;; esac
  case "$line" in *fb0.05*|*"free bits"*|*free-bits*)
      flags="$flags --w-free-bits 0.05" ;; esac
  [ -n "$flags" ] || case "$line" in *control*) flags=" " ;; esac
  [ -n "$flags" ] || return 1
  log "no DECISION.json; read the selected arm off AGENT_REPORT.md: $line"
  printf '{"selected_arm": "from AGENT_REPORT.md", "flags": "%s", "source": "AGENT_REPORT.md"}\n' \
         "$(echo $flags)" > $DECISION
  return 0
}

wait_for_decision() {
  while [ ! -f "$DECISION" ]; do
    decision_from_report && break
    log "waiting for $DECISION (or a selected arm in $WFIX/AGENT_REPORT.md)"
    sleep 600
  done
  FIXFLAGS=$(uv run python -c "import json,sys; print(json.load(open('$DECISION')).get('flags',''))")
  log "collapse-fix verdict: $(cat $DECISION | tr -d '\n')"
  log "collapse-fix flags for every fit here: '${FIXFLAGS}'"
}

# -- GPU choice: whichever card has room, checked at launch time ------------
NEED=12000
pick_gpu() {
  local free0 free1
  while true; do
    free0=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 \
            | awk -F', ' '{print $1-$2}')
    free1=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 1 \
            | awk -F', ' '{print $1-$2}')
    if [ "$free0" -ge "$NEED" ] && [ "$free0" -ge "$free1" ]; then echo 0; return; fi
    if [ "$free1" -ge "$NEED" ]; then echo 1; return; fi
    log "GPU0 ${free0} MiB / GPU1 ${free1} MiB free, need ${NEED}; waiting" >&2
    sleep 120
  done
}

TRIES=10
step() {  # step <marker> <cmd...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try rc gpu
  for try in $(seq $TRIES); do
    gpu=$(pick_gpu)
    log "GPU$gpu start $name (try $try/$TRIES)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1
    rc=$?
    log "GPU$gpu done  $name (exit $rc)"
    if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
    log "GPU$gpu FAIL $name (exit $rc); see $QL/$name.log"
    sleep 300
  done
  log "GIVING UP on $name after $TRIES tries"
  return 1
}

fit() {   # fit <run> <arm flags...>
  local run=$1; shift
  if [ -f "data/datasets/$OV/runs/$run/metrics.json" ]; then
    touch "$DONE/fit_$run"; log "skip fit_$run (metrics.json present)"; return 0
  fi
  step "fit_$run" uv run python -m discell.model.train \
    --dataset $OV --run-name $run "$@"
}

post() {  # post <run>
  local run=$1
  [ -f "$DONE/fit_$run" ] || { log "no fit for $run; skipping post"; return 0; }
  step "validate_$run" uv run python -m discell.model.validate \
    --dataset $OV --run $run --analyses morans,niche
  step "degeneracy_$run" uv run python -m discell.model.degeneracy \
    --dataset $OV --run $run
  step "transport_$run" uv run python -m discell.model.transport \
    --dataset $OV --run $run --read both --hvg 1000
  step "transport_band_$run" uv run python -m discell.model.transport \
    --dataset $OV --run $run --read both --hvg 1000 --niche-source tumour-band
  step "mi_quadrant_$run" uv run python -m discell.experiments.external_criteria \
    mi-quadrant --dataset $OV --run $run
  # arm (ii)'s read; run on every arm so the entropies have a reference
  step "attention_$run" uv run python -m discell.model.attention_read \
    --dataset $OV --run $run
}

# --compare-runs reads the sibling's atlas/programs.npy, which only the
# sibling's own atlas run writes: so two passes once an arm's three seeds are
# all fitted -- build every atlas, then rebuild each against the other two.
atlas_arm() {   # atlas_arm <arm>
  local arm=$1 s other
  for s in $SEEDS; do
    [ -f "$DONE/fit_qp_${arm}_s${s}" ] || { log "atlas $arm: seed $s missing"; return 0; }
  done
  for s in $SEEDS; do
    step "atlasbuild_qp_${arm}_s${s}" uv run python -m discell.model.atlas \
      --dataset $OV --run "qp_${arm}_s${s}"
  done
  for s in $SEEDS; do
    step "atlas_qp_${arm}_s${s}" uv run python -m discell.model.atlas \
      --dataset $OV --run "qp_${arm}_s${s}" --compare-runs \
      $(for other in $SEEDS; do [ "$other" = "$s" ] || echo "qp_${arm}_s${other}"; done)
  done
}

wait_for_decision

for arm in $ARMS; do
  for s in $SEEDS; do
    run="qp_${arm}_s${s}"
    fit "$run" $BASE --seed $s $(arm_flags $arm) ${FIXFLAGS}
    post "$run"
  done
  atlas_arm "$arm"
done

log "read-out"
uv run python scripts/queryprior_table.py > $QL/queryprior_table.log 2>&1
log "read-out done (exit $?); see $QL/queryprior_table.log"
log "queue finished"
