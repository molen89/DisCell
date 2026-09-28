#!/bin/bash
# Sensitivity rows at the final configuration, tag sensF (devlog 2026-09-28,
# "Author's directions on the handover audit", item 3: nothing is quoted from
# pre-final settings; the kappa-form arms, the false-positive floor, the
# alpha_w ladder and the adversary arms are rerun at the final configuration
# as sensitivity rows beside finalL). Motivation of each family: devlog
# "R12 test 2" (kappa form), "Two training-side candidates" part B and the
# 2026-09-25 decisions (fp floor), "alpha_w in units of 1/l-bar" (ladder),
# part A of "Two training-side candidates" and 8.17 (adversary).
#
# Controls = finalL_s{0,1,2} of each dataset: reused, never refit. The arms
# (seed s shares finalL_s<s>'s data split):
#   (a) kappa form, ovarian, seeds 0 1 2:
#       sensF_kdepth_s*    --kappa-mode depth
#       sensF_kgene_s*     --kappa-mode gene
#       sensF_kdensity_s*  --kappa-mode density
#   (b) false-positive floor, seeds 0 1 2:
#       sensF_fp_s*        --fp-floor               ovarian and GSE core
#       sensF_fparea_s*    --fp-floor --fp-area     ovarian
#   (c) alpha_w = m / l-bar (l-bar = 1/(2 alpha_z), the R19 ladder's
#       convention, scripts/logs/awladder_2026-09-24/LADDER.json), seeds 0 1:
#       sensF_awm<m>_s*    m in 1 2 5 10 25 on GSE core and ovarian, 5 10 on FF
#   (d) adversary, ovarian, seeds 0 1:
#       sensF_advsteps12_s*   --adv-head-steps 12
#       sensF_advwidth128_s*  --adv-head-width 128
#       sensF_advens3_s*      --adv-ensemble 3
#       sensF_advcomp1_s*     --adv-comp-weight 1 (the old default, now the ablation)
#       sensF_advcomp5_s*     --adv-comp-weight 5
# Every flag of the final configuration is passed explicitly (the TrainConfig
# defaults are being changed by another lane): COMMON below plus the per-
# dataset alpha_z (and GSE's variant/tiles); the arm's flag comes last. Each
# fit runs only after a config check, on the GPU it holds, re-derives its
# TrainConfig with the code of that moment and finds it equal to
# finalL_s<seed>/config.json but for the arm's own fields (and the fields
# the trainer sets at setup); after the fit its written config.json is checked
# the same way (CONFIG_CHECK.tsv). The launch preflight checks every arm.
#
# Per run: validate morans,niche,probe -> degeneracy (dead fits recorded in
# DEAD_RUNS.tsv, read, never refitted) -> probe_regrade --force against
# uncontrolledL_s{0,1} -> transport --read both --hvg 1000 (k-means; on
# ovarian also --niche-source tumour-band) -> (a) atlas, 6b.1 signalling share,
# GO localisation (CPU, own stem); (b) signalling share; (c) w_deviation ->
# recon_modes -> the tile bootstrap (bootstrap_ci.json; the CI columns).
# Controls get the family reads the re-pin did not make: GO per run on
# ovarian, the signalling share on GSE, w_deviation on GSE, ovarian and FF.
# The read-out, scripts/sensF_tables.py, is refreshed after every run and at
# the end -> <ovarian>/experiments/sensF_{kappa_form,fp_floor,alpha_w,
# adversary}.{md,json}.
#
# Picker / lock / dataset-qualified marker machinery from
# scripts/queue_2026-09-25_final_lineage.sh: gate 9500 MiB (FF 18500, its
# waiter announced in $ROOT/waiting/ff.<pid> and served first), MintFlow
# charged at its peak, settle delay, other queues' fresh locks respected, one
# job of ours per GPU. Markers are keyed by step AND dataset AND run; a
# relaunch resumes. SENSF_DRYRUN=1: preflight and job list only.
#
# GPU-hour estimate (per-step means of the 500/40 lineage re-pin and the
# 2026-09-24 arm queues): ovarian run ~25 min compute (+ ~7 min settle) x 35,
# GSE ~4.5 (+6) x 13, FF ~95 (+6) x 4, control reads ~0.5 h: ~23 GPU-h of
# compute, ~28 card-hours with the settle; ~14 h wall on two free cards.
#
#   mkdir -p scripts/logs/sens_final_2026-09-28 && \
#   setsid nohup scripts/queue_2026-09-28_sens_final.sh \
#     >> scripts/logs/sens_final_2026-09-28/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff

ROOT=scripts/logs/sens_final_2026-09-28
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting; FAILED=$ROOT/failed
mkdir -p $QL $DONE $LOCK $WAIT $FAILED
GPUS="0 1"
NEED_FF=18500
NEED_OV=9500          # every non-FF dataset
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
TRIES_FIT=6           # ... for a fit (its config check may meet a trainer mid-edit)
SETTLE=45
FRESH=600
PY="uv run python"

# the final configuration, every flag explicit (runs/finalL_s0/config.json)
COMMON="--adv-comp-weight 3 --w-warmup-epochs 30 --label-key lineage --alpha-w 0.1
        --kappa 0.1 --d-w 6 --alpha-a 0.3 --gat-sources type_only
        --epochs 500 --patience 40 --figures-every 100"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$FF]="--alpha-z 0.00035")
# alpha_w = m / l-bar = 2 m alpha_z (checked against finalL's alpha_z)
declare -A AW=(
  [$GS:1]=0.0036 [$GS:2]=0.0072 [$GS:5]=0.018 [$GS:10]=0.036 [$GS:25]=0.09
  [$OV:1]=0.007 [$OV:2]=0.014 [$OV:5]=0.035 [$OV:10]=0.07 [$OV:25]=0.175
  [$FF:5]=0.0035 [$FF:10]=0.007)
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              discell/model/fp_floor.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

arm_run() { echo sensF_${1}_s$2; }    # arm_run <arm> <seed>
arm_flags() {  # arm_flags <dataset> <arm>
  case $2 in
    kdepth) echo "--kappa-mode depth" ;;
    kgene) echo "--kappa-mode gene" ;;
    kdensity) echo "--kappa-mode density" ;;
    fp) echo "--fp-floor" ;;
    fparea) echo "--fp-floor --fp-area" ;;
    awm*) echo "--alpha-w ${AW[$1:${2#awm}]}" ;;
    advsteps12) echo "--adv-head-steps 12" ;;
    advwidth128) echo "--adv-head-width 128" ;;
    advens3) echo "--adv-ensemble 3" ;;
    advcomp1) echo "--adv-comp-weight 1" ;;
    advcomp5) echo "--adv-comp-weight 5" ;;
    *) echo "UNKNOWN_ARM_$2" ;;
  esac
}
fit_flags() { echo "${FLAGS[$1]} $COMMON $(arm_flags $1 $2)"; }   # the arm's flag last
family_of() {
  case $1 in k*) echo kappa ;; fp*) echo fp ;; awm*) echo aw ;; adv*) echo adv ;; esac
}

# -- GPU picker (scripts/queue_2026-09-25_final_lineage.sh) ----------------------
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

live_waiter() {  # live_waiter <glob>: a waiting file whose pid is alive
  local f
  for f in $1; do
    [ -e "$f" ] || continue
    kill -0 "${f##*.}" 2>/dev/null && return 0
  done
  return 1
}

held_elsewhere() {  # held_elsewhere <gpu> <free> <role>: someone has first call
  local g=$1 free=$2 role=$3 dir now age
  now=$(date +%s)
  if [ "$role" != ff ] && [ "$free" -ge "$NEED_FF" ] \
     && live_waiter "scripts/logs/*/waiting/ff.*"; then
    return 0                                       # an FF waiter first (any queue's)
  fi
  for dir in scripts/logs/*/locks/gpu$g; do        # may not have allocated yet
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB> <role: ov|ff>
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

# -- steps ------------------------------------------------------------------------
_step() {  # _step <need> <role> <marker> <cmd...>: a GPU step, retried on OOM
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 max_other=$TRIES_OTHER rc gpu t0
  case $name in fit_*) max_other=$TRIES_FIT ;; esac
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $need $role)
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
      log "FAIL $name (exit $rc; attempt $other/$max_other); see $QL/$name.try$try.log"
      [ $other -ge $max_other ] && break
    fi
    sleep 300
  done
  log "GIVING UP on $name"; touch "$FAILED/$name"; return 1
}
gstep() { local ds=$1; shift; _step $(need_of $ds) $(role_of $ds) "$@"; }

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

# -- the config check ----------------------------------------------------------------
# cfg_check <pre|post> <spec>...; spec = dataset|run|seed|arm|flags. pre derives
# the TrainConfig from the flags with the current code, post reads the run's
# config.json; both compare with finalL_s<seed>/config.json. Exit 0 = equal
# but for the arm's own fields (and those the trainer sets at setup), 1 = a
# difference, 2 = could not check (the trainer does not import / parse).
cfg_check() {
  $PY - "$@" <<'EOF'
import dataclasses, json, math, shlex, sys
from pathlib import Path
mode, specs = sys.argv[1], sys.argv[2:]
try:
    from discell.model.train import TrainConfig, build_parser
    default = dataclasses.asdict(TrainConfig(dataset=""))
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
except Exception as exc:                      # a trainer mid-edit
    print(f"CANNOT CHECK: the trainer does not import ({exc!r})")
    sys.exit(2)
EXPECT = {"kdepth": {"kappa_mode": "depth"}, "kgene": {"kappa_mode": "gene"},
          "kdensity": {"kappa_mode": "density"},
          "fp": {"fp_floor": True, "fp_area": False},
          "fparea": {"fp_floor": True, "fp_area": True},
          "advsteps12": {"adv_steps": 12}, "advwidth128": {"adv_hidden": 128},
          "advens3": {"adv_ensemble": 3}, "advcomp1": {"adv_comp_weight": 1.0},
          "advcomp5": {"adv_comp_weight": 5.0}}
SETUP = {"kdepth": {"kappa_ratio_mean"}, "kdensity": {"kappa_ratio_mean"},
         "kgene": {"kappa_gene_source"}, "fp": {"fp_lambda", "fp_cap_share"},
         "fparea": {"fp_lambda", "fp_cap_share"}}
FINAL = {"label_key": "lineage", "adv_comp_weight": 3.0, "w_warmup_epochs": 30,
         "epochs": 500, "patience": 40, "kappa": 0.1, "alpha_w": 0.1}

def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return a == b

worst = 0
for spec in specs:
    ds, run, seed, arm, flags = spec.split("|", 4)
    seed = int(seed)
    runs = Path("data/datasets") / ds / "runs"
    ref = json.loads((runs / f"finalL_s{seed}" / "config.json").read_text())
    expect = dict(EXPECT.get(arm, {}))
    if arm.startswith("awm"):
        expect["alpha_w"] = round(2 * int(arm[3:]) * ref["alpha_z"], 10)
    if not expect:
        print(f"{mode} {ds} {run}: UNKNOWN ARM {arm}"); worst = max(worst, 1); continue
    try:
        if mode == "pre":
            args = vars(build_parser().parse_args(
                ["--dataset", ds, "--seed", str(seed), "--run-name", run]
                + shlex.split(flags)))
            new = dataclasses.asdict(TrainConfig(
                **{k: v for k, v in args.items() if k in fields}))
        else:
            new = json.loads((runs / run / "config.json").read_text())
    except (Exception, SystemExit) as exc:
        print(f"{mode} {ds} {run}: CANNOT CHECK ({exc!r})"); worst = max(worst, 2); continue
    allowed = {"run_name", "seed", "git"} | set(expect) | SETUP.get(arm, set())
    diffs = [f"{k}: {new.get(k, '<absent>')!r} vs finalL_s{seed} {ref.get(k, '<absent>')!r}"
             for k in sorted(set(new) | set(ref)) if k not in allowed
             and not same(new.get(k, default.get(k)), ref.get(k, default.get(k)))]
    want = {**FINAL, "seed": seed, **expect}
    wrong = [f"{k}={new.get(k)!r} (want {v!r})" for k, v in want.items()
             if not same(new.get(k), v)]
    if mode == "post":
        wrong += [f"{k} not set at setup" for k in SETUP.get(arm, set())
                  if new.get(k) is None]
    ok = not diffs and not wrong
    print(f"{mode} {ds} {run}: " + (
        f"finalL_s{seed}'s configuration but for "
        + ", ".join(f"{k}={new.get(k)!r}" for k in sorted(expect))
        if ok else "DIFFERS: " + "; ".join(diffs + wrong)))
    worst = max(worst, 0 if ok else 1)
sys.exit(worst)
EOF
}

checked_train() {  # checked_train <dataset> <run> <seed> <arm>: check on the held GPU, then fit
  local ds=$1 run=$2 seed=$3 arm=$4
  cfg_check pre "$ds|$run|$seed|$arm|$(fit_flags $ds $arm)" || return 1
  # shellcheck disable=SC2046
  $PY -m discell.model.train --dataset $ds --run-name $run $(fit_flags $ds $arm) --seed $seed
}

fit() {   # fit <dataset> <arm> <seed>
  local ds=$1 arm=$2 seed=$3 run mk line; run=$(arm_run $2 $3); mk=$(marker fit $1 $run)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"
  else
    gstep $ds "$mk" checked_train $ds $run $seed $arm || return 1
  fi
  grep -qP "\t$ds\t$run\t" $ROOT/CONFIG_CHECK.tsv 2>/dev/null && return 0
  line=$(cfg_check post "$ds|$run|$seed|$arm|" 2>&1 | tail -1)
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $ds $run "$line" >> $ROOT/CONFIG_CHECK.tsv
  case "$line" in *DIFFERS*|*CANNOT*) log "CONFIG $ds/$run: $line" ;; esac
  return 0
}

dead_reason() {  # prints why <dataset>/<run> is a dead fit; nothing when live
  python3 - "$(runs $1)/$2" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); why = []
if json.loads((d / "metrics.json").read_text()).get("dead_w_channel"):
    why.append("metrics.json dead_w_channel")
g = d / "degeneracy.json"
w = (json.loads(g.read_text()).get("w_channel") or {}) if g.exists() else {}
if w.get("dead_context_channel"):
    why.append(f"guard: I(niche;w) excess {w.get('w_niche_mi_excess'):+.4f}")
print("; ".join(why))
EOF
}

record_dead() {  # record_dead <dataset> <run>: DEAD_RUNS.tsv, once
  local why
  why=$(dead_reason $1 $2) || return 0
  [ -n "$why" ] || return 0
  grep -qP "\t$1\t$2\t" $ROOT/DEAD_RUNS.tsv 2>/dev/null || \
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $1 $2 "$why" >> $ROOT/DEAD_RUNS.tsv
  log "$1/$2 DEAD ($why) -- recorded, read, not refitted"
}

ext() {  # ext <read> <dataset> <run>
  gstep $2 "$(marker ext_$1 $2 $3)" $PY -m discell.experiments.external_criteria $1 \
    --dataset $2 --run $3
}

table() {  # the read-out; cheap, CPU, never fatal
  CUDA_VISIBLE_DEVICES= $PY scripts/sensF_tables.py --quiet > $QL/table_${1:-partial}.log 2>&1 \
    || log "read-out (${1:-partial}) failed; see $QL/table_${1:-partial}.log"
}

# -- one arm run ----------------------------------------------------------------------
sens_reads() {  # sens_reads <dataset> <run> <family>
  local ds=$1 run=$2 fam=$3
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche,probe
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  record_dead $ds $run
  # after validate: its in-module probe grades against the old references
  # shellcheck disable=SC2086
  gstep $ds "$(marker probe_regrade $ds $run)" $PY -m discell.experiments.probe_regrade \
    --dataset $ds --run $run --force $REFS
  gstep $ds "$(marker transport $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read both --hvg 1000
  if [ "$ds" = "$OV" ]; then
    gstep $ds "$(marker transport_band $ds $run)" $PY -m discell.model.transport \
      --dataset $ds --run $run --read both --hvg 1000 --niche-source tumour-band
  fi
  case $fam in
    kappa)
      gstep $ds "$(marker atlas $ds $run)" $PY -m discell.model.atlas --dataset $ds --run $run
      ext signalling-share $ds $run
      cpu_step "$(marker go_localisation $ds $run)" $PY -m discell.experiments.go_localisation \
        --dataset $ds --run $run --out-stem go_localisation_$run ;;
    fp) ext signalling-share $ds $run ;;
    aw) gstep $ds "$(marker w_deviation $ds $run)" $PY -m discell.experiments.w_deviation \
          --dataset $ds --run $run ;;
  esac
  gstep $ds "$(marker recon_modes $ds $run)" $PY -m discell.experiments.recon_modes \
    --dataset $ds --run $run
  gstep $ds "$(marker bootstrap $ds $run)" $PY -m discell.experiments.bootstrap \
    --dataset $ds --run $run --n 1000
  return 0
}

job() {  # job <dataset> <arm> <seed>: the fit and every read of one run
  local ds=$1 arm=$2 seed=$3 run; run=$(arm_run $2 $3)
  fit $ds $arm $seed || { log "$ds/$run: no fit -- no reads"; return 0; }
  sens_reads $ds $run $(family_of $arm)
  table partial
}

jobs_of() {  # jobs_of <lane>: "dataset arm seed" lines in lane order (seed-major)
  local s a m
  case $1 in
    kappa) for s in 0 1 2; do for a in kdepth kgene kdensity; do echo "$OV $a $s"; done; done ;;
    fp) for s in 0 1 2; do echo "$OV fp $s"; echo "$OV fparea $s"; echo "$GS fp $s"; done ;;
    aw) for s in 0 1; do for m in 1 2 5 10 25; do echo "$GS awm$m $s"; echo "$OV awm$m $s"; done; done ;;
    awff) for s in 0 1; do for m in 5 10; do echo "$FF awm$m $s"; done; done ;;
    adv) for s in 0 1; do
           for a in advsteps12 advwidth128 advens3 advcomp1 advcomp5; do echo "$OV $a $s"; done
         done ;;
  esac
}
LANES="awff kappa fp aw adv"

lane() {  # lane <name>: its jobs in order; the list on fd 3, out of the jobs' reach
  local ds arm seed
  while read -r ds arm seed <&3; do job $ds $arm $seed < /dev/null; done 3< <(jobs_of $1)
  log "lane $1 finished"
}

lane_controls() {  # the family reads on finalL that the re-pin did not make (never a refit)
  local s ds
  for s in 0 1 2; do
    cpu_step "$(marker go_localisation $OV finalL_s$s)" $PY -m discell.experiments.go_localisation \
      --dataset $OV --run finalL_s$s --out-stem go_localisation_finalL_s$s
    ext signalling-share $GS finalL_s$s
    for ds in $GS $OV $FF; do
      gstep $ds "$(marker w_deviation $ds finalL_s$s)" $PY -m discell.experiments.w_deviation \
        --dataset $ds --run finalL_s$s
    done
  done
  table partial
  log "lane controls finished"
}

preflight() {  # every arm's derived config vs its control's; the controls exist
  local specs=() lane ds arm seed bad=0
  for ds in $GS $OV $FF; do
    for seed in 0 1 2; do
      [ -f "$(runs $ds)/finalL_s$seed/best.pt" ] || { log "missing control $ds/finalL_s$seed"; bad=1; }
    done
  done
  for lane in $LANES; do
    while read -r ds arm seed; do
      specs+=("$ds|$(arm_run $arm $seed)|$seed|$arm|$(fit_flags $ds $arm)")
    done < <(jobs_of $lane)
  done
  log "preflight: ${#specs[@]} arm runs"
  cfg_check pre "${specs[@]}" || bad=1
  return $bad
}

# -- main -------------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, others $NEED_OV MiB, GPUs: $GPUS; model code $(code_hash))"
if ! preflight; then log "PREFLIGHT FAILED -- stopping"; exit 1; fi
if [ "${SENSF_DRYRUN:-0}" = 1 ]; then
  for lane in $LANES; do log "lane $lane: $(jobs_of $lane | wc -l) runs"; done
  log "dry run: preflight and job list only"; exit 0
fi

PIDS=""
lane awff & PIDS="$PIDS $!"           # the long FF pole takes the first free card
(sleep 5; lane_controls) & PIDS="$PIDS $!"
d=15
for l in kappa fp aw adv; do (sleep $d; lane $l) & PIDS="$PIDS $!"; d=$((d + 10)); done
# shellcheck disable=SC2086
wait $PIDS
log "all lanes finished"

table final
log "read-out: data/datasets/$OV/experiments/sensF_{kappa_form,fp_floor,alpha_w,adversary}.md (see $QL/table_final.log)"
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Read-out: data/datasets/$OV/experiments/sensF_{kappa_form,fp_floor,alpha_w,adversary}.{md,json}"
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (recorded, read, not refitted): $(cut -f2,3 $ROOT/DEAD_RUNS.tsv 2>/dev/null | tr '\t\n' '/ ')"
  echo "Config checks not ok: $(grep -cvP '\tpost [^\t]*: finalL_s' $ROOT/CONFIG_CHECK.tsv 2>/dev/null) of $(wc -l < $ROOT/CONFIG_CHECK.tsv 2>/dev/null) (CONFIG_CHECK.tsv)"
  echo "Model code at fit starts: $(cut -f3 $ROOT/code_hashes.tsv 2>/dev/null | sort -u | wc -l) distinct hash set(s) (code_hashes.tsv)"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
