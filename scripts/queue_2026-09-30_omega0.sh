#!/bin/bash
# The omega = 0 ablation (devlog "Author's decisions on the open flags
# (2026-09-30)" item 1, and "omega = 0 ablation (motivation, 2026-09-30)"):
# section 2.3 calls the intrinsic path load-bearing -- it breaks the z/w
# allocation tie toward z -- and no fit tests it.
#
# Fits: omega0L_s{0,1,2} on the four trained sections, launched exactly as
# the finalL fits were (scripts/queue_2026-09-25_final_lineage.sh: every
# model flag explicit, per-dataset alpha_z, GSE's variant pdl018d / tiles
# 2048, --label-key lineage, 500/40, --w-warmup-epochs 30,
# --adv-comp-weight 3) plus --omega 0 (the intrinsic path dropped; the
# (1+omega) z-KL factor then reads 1). Same seeds, hence the same tile
# splits, as finalL_s{0,1,2}. A preflight re-derives every arm's TrainConfig
# and stops unless it equals finalL_s<seed>/config.json but for omega (and
# run_name / git). A dead w channel is recorded in DEAD_RUNS.tsv and NOT
# refitted: the seed is the split, and a dead channel at omega = 0 is itself
# a finding.
#
# Reads after each fit (finalL's read set, under the Unassigned mask):
#   degeneracy (guard + the in-trainer battery at best.pt, which carries the
#   q90 cycle rows: cycle_r2_{z,w}_q90) -> validate morans,niche (its
#   in-module probe is left out, as in the Unassigned re-read of finalL:
#   probe_regrade --force writes the same record) -> probe_regrade --force vs
#   uncontrolledL_s{0,1} (GSE also on the dual section) -> GSE: crossslide
#   on the dual (the held-out section's q90 cycle rows) -> the NMI of mu_w
#   with the lineage type (scripts/omega0_nmi_w.py: the battery's own k-means
#   NMI, applied to mu_w; its mu_z value must reproduce degeneracy.json).
#   The same NMI read runs once on finalL_s{0,1,2} (the comparison).
#   cycle_reread.py is not needed: a fit made with today's code writes the
#   q90 rows itself.
# Control: GSE finalL_s0 refitted with today's code as omega0ctl_finalL_s0
# (finalL's flags, omega 1), compared with finalL_s0's metrics.json -- the
# model code changed since the finalL fits (train.py hash), so this says
# whether code drift alone moves a fit.
# At the end: scripts/omega0_readout.py writes $ROOT/READOUT.md.
#
# Order: GSE, ovarian, lung, FF; within a section, the three fits' reads
# follow each fit. Scheduling: waits (pgrep -f, poll 10 min, as
# scripts/queue_2026-09-29_baselines_complete.sh) until
# scripts/queue_2026-10-01_breakdown.sh has ended. Then GPU 0 ONLY (never
# GPU 1: the whole-section baselines whose wall times go into the paper),
# with a memory gate (>= NEED MiB free, twice SETTLE s apart; MintFlow's
# neighbour charge guarded as the other queues do; other queues' fresh
# scripts/logs/*/locks/gpu0 respected, our own held while a step runs). FF
# steps also wait for host MemAvailable >= FF_HOST_GB.
# Threads (issue T-omp): OMP_NUM_THREADS=8 for the fits (their NMI k-means
# picks the checkpoint; finalL was fitted at 8), degeneracy (k-means niches),
# validate (niche), crossslide and the NMI read; 2 for probe_regrade and the
# read-out. WANDB_MODE=disabled; nothing leaves the machine.
#
# Idempotent: done/<step> markers (a fit is also skipped when its
# metrics.json exists); failed/<step> records (exit code, last log lines). A
# CUDA OOM is retried (up to TRIES); any other failure is recorded once and
# the queue carries on. Delete failed/<step> to retry it on a relaunch.
#
# Estimate (finalL fit minutes; the Unassigned queue's read timings): fits
# GSE 4 + ovarian 27 + lung 14 + FF 116 min; reads ~5 / 13 / 11 / 45 min;
# the finalL NMI reads ~20 min; settle 45 s per step (~55 steps) ~40 min.
# ~4.5-5 h of GPU 0 after the breakdown queue ends (omega = 0 may stop at
# other epochs: 4-7 h).
#
#   mkdir -p scripts/logs/omega0_2026-09-30 && setsid nohup \
#     bash scripts/queue_2026-09-30_omega0.sh \
#     >> scripts/logs/omega0_2026-09-30/queue.log 2>&1 < /dev/null & disown
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
ORDER="$GS $OV $LU $FF"                        # the author's order
SEEDS="0 1 2"
ARM=omega0L

ROOT=scripts/logs/omega0_2026-09-30
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed; NMI=$ROOT/nmi_w
mkdir -p $QL $DONE $LOCK $FAILED $NMI
GPU=0                  # GPU 0 only; GPU 1 is never used
WAIT_FOR="queue_2026-10-01_breakdown"
POLL=600
NEED=9500              # MiB, a non-FF step (the finalL queue's gate)
NEED_FF=18500          # MiB, an FF step
FF_HOST_GB=60          # host MemAvailable before an FF step
TRIES=6
SETTLE=45; FRESH=600
PY8="env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python"
PY2="env OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"

# finalL's launch, flag for flag (scripts/queue_2026-09-25_final_lineage.sh)
COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30
        --adv-comp-weight 3 --label-key lineage"
OMEGA0="--omega 0"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002"
  [$FF]="--alpha-z 0.00035")
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"
CTL=omega0ctl_finalL_s0                        # the code-drift control (GSE)

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

queues_running() {  # anchored on the bash that runs the queue: a shell whose
  # command line merely mentions the queue's name (e.g. another waiter's
  # pgrep loop) must not hold this queue back
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "^(/usr)?(/bin/)?bash .*${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}

# -- GPU 0 picker (scripts/queue_2026-10-01_breakdown.sh's) -----------------------
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
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
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
  { echo "exit $rc"; tail -5 $QL/$name.log; } > "$FAILED/$name"
  return 1
}
cstep() {  # cstep <marker> <cmd...>: a CPU step (re-run every time it is reached)
  local name=$1; shift
  local t0=$SECONDS rc
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= $PY2 "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAILED $name; see $QL/$name.log"; tail -5 $QL/$name.log > "$FAILED/$name"; return 1
}

# -- preflight -----------------------------------------------------------------------
preflight_config() {  # every omega0L arm's TrainConfig vs finalL_s<seed>/config.json
  local specs=() ds
  for ds in $ORDER; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  CUDA_VISIBLE_DEVICES= $PY2 - "$OMEGA0" "$SEEDS" "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
omega0, seeds, bad = sys.argv[1], [int(s) for s in sys.argv[2].split()], 0
BASE = {"run_name", "git"}
default = asdict(TrainConfig(dataset=""))
def derive(ds, flags, seed):
    args = vars(build_parser().parse_args(["--dataset", ds, "--seed", str(seed)] + shlex.split(flags)))
    for k in ("quiet", "time_only"):
        args.pop(k, None)
    return asdict(TrainConfig(**args))
for spec in sys.argv[3:]:
    ds, flags = spec.split("::", 1)
    for seed in seeds:
        ref = json.load(open(f"data/datasets/{ds}/runs/finalL_s{seed}/config.json"))
        for name, arm_flags, extra, want in (
                (f"finalL_s{seed} (control)", flags, set(), {"omega": 1.0}),
                (f"omega0L_s{seed}", f"{flags} {omega0}", {"omega"}, {"omega": 0.0})):
            new = derive(ds, arm_flags, seed)
            diffs = [f"{k}: {new.get(k)!r} vs finalL_s{seed} {ref.get(k, '<absent>')!r}"
                     for k in sorted(set(new) | set(ref)) if k not in BASE | extra
                     and ((k in ref and new.get(k) != ref[k])
                          or (k not in ref and new[k] != default[k]))]
            wrong = [f"{k}={new[k]!r} (want {v!r})" for k, v in
                     {**want, "seed": seed, "epochs": 500, "patience": 40,
                      "label_key": "lineage", "adv_comp_weight": 3.0,
                      "w_warmup_epochs": 30}.items() if new[k] != v]
            ok = not diffs and not wrong
            bad += not ok
            print(f"{ds} {name}: " + ("finalL's config but for the arm's own fields"
                                      if ok else "DIFFERS: " + "; ".join(diffs + wrong)))
sys.exit(1 if bad else 0)
EOF
}

# -- fits and reads ---------------------------------------------------------------------
fit() {   # fit <dataset> <run> <seed> [extra flags...]
  local ds=$1 run=$2 seed=$3 mk; shift 3; mk=fit_${ds}_$run
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  gstep 8 $ds "$mk" -m discell.model.train --dataset $ds --run-name $run \
    ${FLAGS[$ds]} $COMMON --seed $seed "$@"
}

record_dead() {  # record_dead <dataset> <run>: DEAD_RUNS.tsv, once; never refitted
  local why
  why=$(python3 - "$(runs $1)/$2" <<'EOF'
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
) || return 0
  [ -n "$why" ] || return 0
  grep -qP "\t$1\t$2\t" $ROOT/DEAD_RUNS.tsv 2>/dev/null || \
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $1 $2 "$why" "recorded (ablation; not refitted)" >> $ROOT/DEAD_RUNS.tsv
  log "$1/$2 DEAD ($why) -- recorded, not refitted"
}

nmi_w() {  # nmi_w <dataset> <run>
  gstep 8 $1 "nmi_w_${1}_$2" scripts/omega0_nmi_w.py --dataset $1 --run $2 \
    --out $NMI/${1}__$2.json
}

reads() {  # reads <dataset> <run>: finalL's read set
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  gstep 8 $ds "degeneracy_${ds}_$run" -m discell.model.degeneracy --dataset $ds --run $run \
    && record_dead $ds $run
  gstep 8 $ds "validate_${ds}_$run" -m discell.model.validate --dataset $ds --run $run \
    --analyses morans,niche
  # shellcheck disable=SC2086
  gstep 2 $ds "probe_regrade_${ds}_$run" -m discell.experiments.probe_regrade \
    --dataset $ds --run $run --force $REFS
  if [ "$ds" = "$GS" ]; then
    # shellcheck disable=SC2086
    gstep 2 $ds "probe_regrade_${GD}_$run" -m discell.experiments.probe_regrade \
      --dataset $GD --config-from $GS --run $run --force $REFS
    gstep 8 $ds "crossslide_${ds}_$run" -m discell.model.crossslide --dataset $GS \
      --run $run --eval-dataset $GD
  fi
  nmi_w $ds $run
  return 0
}

control() {  # GSE finalL_s0 refitted with today's code; compared with the original
  fit $GS $CTL 0 || return 1
  CUDA_VISIBLE_DEVICES= $PY2 - "$(runs $GS)" "$CTL" "$ROOT/control.json" <<'EOF' > $QL/control_compare.log 2>&1
import json, sys
from pathlib import Path
d, ctl, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
best = lambda r: json.loads((d / r / "metrics.json").read_text())["best"]
ref, new = best("finalL_s0"), best(ctl)
seeds = [best(f"finalL_s{s}") for s in (0, 1, 2)]
row = {k: {"finalL_s0": ref[k], ctl: new[k],
           "finalL_range": [min(s[k] for s in seeds), max(s[k] for s in seeds)]}
       for k in ("epoch", "recon_val", "nmi")}
same = all(ref[k] == new[k] for k in ("epoch", "recon_val", "nmi"))
inside = all(r["finalL_range"][0] <= r[ctl] <= r["finalL_range"][1] for r in row.values())
text = ("reproduces finalL_s0 exactly (best epoch, recon_val, NMI)" if same else
        "differs from finalL_s0: " + "; ".join(
            f"{k} {r['finalL_s0']:.6g} -> {r[ctl]:.6g}" for k, r in row.items())
        + (" -- inside finalL's seed range on all three" if inside
           else " -- OUTSIDE finalL's seed range on at least one"))
out.write_text(json.dumps({"run": ctl, "same": same, "inside_seed_range": inside,
                           "reads": row, "text": text}, indent=1))
print(text)
EOF
  log "control: $(cat $QL/control_compare.log)"
}

section() {  # section <dataset>: three fits, each followed by its reads
  local ds=$1 s
  for s in $SEEDS; do
    # shellcheck disable=SC2086
    fit $ds ${ARM}_s$s $s $OMEGA0 && reads $ds ${ARM}_s$s
  done
  for s in $SEEDS; do nmi_w $ds finalL_s$s; done   # the comparison's NMI mu_w
  log "$ds finished"
}

# -- main ----------------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only; waiting for: $WAIT_FOR; model code $(code_hash)"
if ! preflight_config > $QL/preflight.log 2>&1; then
  cat $QL/preflight.log; log "PREFLIGHT FAILED: an arm's config is not finalL's -- stopping"; exit 1
fi
cat $QL/preflight.log
n=0
while [ -n "$(queues_running)" ]; do
  [ $((n % 6)) -eq 0 ] && log "still running:$(queues_running) -- waiting"
  n=$((n + 1)); sleep $POLL
done
log "the waited-for queue has ended; starting (GPU $GPU only)"
rmdir $LOCK/gpu$GPU 2>/dev/null      # one instance at a time: a lock of ours is stale

control
for ds in $ORDER; do section $ds; done
cstep readout scripts/omega0_readout.py $ROOT
log "queue finished ($(ls $DONE | wc -l) steps done, $(ls $FAILED | wc -l) failed: $(ls $FAILED | tr '\n' ' '))"
