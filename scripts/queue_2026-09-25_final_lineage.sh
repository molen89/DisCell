#!/bin/bash
# The single re-pin at lineage labels (devlog 2026-09-25 "Lineage-level labels
# (R9, motivation)", "Lineage mapping tables drafted (R9, results)",
# "Adversary composition weight, confirmation (8.17)", "Metrics package for
# the final tables"). Generalises scripts/queue_2026-09-24_final.sh.
#
# Environment (all read once at start):
#   ADV_COMP_WEIGHT  adversary composition weight          (default 3)
#   LABEL_KEY        obs column that becomes t              (default lineage;
#                    the run names finalL/uncontrolledL/sweepL assume it)
#   SOX2OT           unassigned | tumour    REQUIRED, the author's choice
#   CYST             mesothelial | tumour   REQUIRED, the author's choice
#   NAME_GAPS_OK     1 = launch although a name-matched read (tumour bands on
#                    ovarian) cannot find its class in the new vocabulary;
#                    those steps are then skipped and logged (default 0: stop)
#   SWEEP_READS      light | full (default light). light: every sweep point
#                    gets validate morans,niche,probe, degeneracy, the probe
#                    grade, transport --read mean and (GSE) crossslide, plus the
#                    per-dataset sweep report and kappa survival; the full reads
#                    (transport both/hvg 1000, atlas + compare, recon_modes,
#                    report, bootstrap, external criteria, GO) run on finalL_s*
#                    only (uncontrolledL_s*: see (c)). full: the sweep points
#                    also get the per-run full battery and the per-kappa atlas
#                    compare.
#
# Final configuration = runs/final_s0/config.json of each dataset (alpha_z at
# 1/2 x 1/mean-count, 500/40, --w-warmup-epochs 30, kappa 0.1, d_w 6, alpha_w
# 0.1, alpha_a 0.3, type_only) plus --adv-comp-weight $ADV_COMP_WEIGHT
# --label-key $LABEL_KEY. A preflight re-derives every arm's TrainConfig and
# stops unless it differs from final_s0 only where the arm says it does.
#
# Stages (every step logged to $ROOT/logs/<marker>.log; markers keyed by step
# AND dataset AND run under $ROOT/done; a relaunch resumes):
#  (a) apply_lineage on ovarian, GSE solo (+ the dual section, one
#      vocabulary), lung, FF -- CPU, waits until no process holds the bundle.
#  (b) finalL_s{0,1,2} per dataset, 500/40, dead-channel rule: a slot whose
#      fit is dead (metrics.json dead_w_channel or degeneracy.json w_channel.
#      dead_context_channel) is refitted with the next unused seed 3..8 and
#      recorded in DEAD_RUNS.tsv.
#  (c) uncontrolledL_s{0,1} per dataset (alpha_a 0, 500/40): degeneracy and
#      the per-block probe grade (GSE also on the dual section) -- they are
#      the reference every lineage probe fraction is taken against -- then,
#      after the finalL slots, validate morans,niche (without its in-module
#      probe, which would re-grade the reference against the old
#      uncontrolled500 fits), recon_modes (+ dual) and transport --read mean
#      (the "no adversary" reference row). No bootstrap, external criteria,
#      GO, atlas, compare or report on them.
#  (d) sweepL_k{0,0.05,0.1,0.2,0.3,0.4}_s{0,1,2} per dataset at the
#      references' budget and configuration (R28), one train.py call per fit
#      (sweep.py cannot pass the warm-up or the comp weight); dead fits are
#      recorded, never refitted. Seeds outer: a cut queue leaves whole grids.
#      sweepL_k0.1_s<s> is a symlink to finalL_s<s> (same config, seed and
#      split -- the preflight checks the config), never refitted and never
#      re-read (its reads are finalL's); SWEEP_LINKS.tsv. The sweep report and
#      kappa survival read through the link.
#  (e) full per-run reads (finalL; sweepL only with
#      SWEEP_READS=full): validate morans,niche,probe -> degeneracy ->
#      probe_regrade --force against uncontrolledL_s* (after both are graded)
#      -> atlas -> transport --read both --hvg 1000 (+ tumour-band, ovarian)
#      -> GSE crossslide on the dual -> recon_modes (+ dual) ; per seed group:
#      atlas --compare-runs, report ; per sweep: sweep.py --report-only and
#      validate --sweep-tag kappa_survival ; finalL: tile bootstrap. Light
#      sweep reads: see SWEEP_READS.
#  (f) baselines at $LABEL_KEY: re-export the five bundles
#      (DisCell-baselines/data/lineage/), refit resolVI on all five
#      (results/resolvi_lineage/; the dual via transfer or its fallback),
#      battery columns (tag _lineage: baseline_battery_lineage.{json,md};
#      DisCell finalL slot 0 last so the cached context is slot 0's) for
#      DisCell, resolVI, SIMVI, MintFlow, and their per-block probe grades
#      (experiments/probe_regrade_lineage/).
#  (g) envelope tables --at best --ci over the finalL triples, probe tables,
#      external criteria 6b.1 (ovarian, FF) / 6b.3 (ovarian, GSE, FF) / 6b.4
#      (ovarian) and GO localisation (go_localisation_lineage) on every finalL
#      seed; the S53 projection test re-read at lineage labels (ovarian,
#      projL384_s* vs projL32_s*, 200/20) and the timing mode at finalL's
#      config (experiments/timing_lineage/), alone on an idle GPU, last.
#  (h) runs/best -> the slot-0 run (finalL_s0) per dataset; previous target
#      in REPIN.tsv.
#
# Order: (a) -> phase 1 = (b)+(c)+(e) on the references -> phase 2 = (d)+(e)
# on the sweep and phase 3 = (f)+(g)+(h) in parallel (a sweep step leaves a
# freed card to a waiting phase-3 step) -> scoped re-verdict, probe tables,
# timing, READOUT.md.
#
# GPU-hour estimate, SWEEP_READS=light (per-step means of the 2026-09-24
# final re-pin, the uncontrolled500 and confirmation queues and lane 2;
# bootstrap, recon_modes, sweep report, kappa survival and timing not yet
# measured -- rough): (a) 0; (b) 3.0 (+0.8 per dead FF refit); (c) 1.3 fits
# + 0.6 guard/grade + 1.2 validate/recon_modes/mean read; (d) 15.0 (15 fits
# per dataset, k0.1 linked); (e) 5.1 finalL + 10.2 sweepL light + ~3.2 sweep
# report/kappa survival + ~4.3 bootstrap (finalL); (f) 4.5; (g) 0.9 external
# + 1.1 projection + ~1.0 timing; (h) 0. Total ~51 GPU-h, FF ~34; ~26 h wall
# at best on two cards. SWEEP_READS=full adds ~15 (~67 total).
# Breakdown: scripts/logs/final_prep_2026-09-25/gpu_hours_estimate_final.txt.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-25_metrics.sh
# (gate 9500 MiB, FF 18500 announcing itself in $ROOT/waiting/ff.<pid>,
# resolVI 4000, MintFlow charged at its peak, settle delay, other queues'
# fresh locks respected; timing only on a card with no compute process).
#
#   mkdir -p scripts/logs/final_lineage_2026-09-25 && \
#   SOX2OT=unassigned CYST=mesothelial setsid nohup \
#     scripts/queue_2026-09-25_final_lineage.sh \
#     >> scripts/logs/final_lineage_2026-09-25/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ADV_COMP_WEIGHT=${ADV_COMP_WEIGHT:-3}
LABEL_KEY=${LABEL_KEY:-lineage}
SOX2OT=${SOX2OT:-}
CYST=${CYST:-}
NAME_GAPS_OK=${NAME_GAPS_OK:-0}
SWEEP_READS=${SWEEP_READS:-light}

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ALL="$GS $OV $LU $FF"

ROOT=scripts/logs/final_lineage_2026-09-25
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting
SLOTS=$ROOT/slots; SEEDS=$ROOT/seeds; ACC=$ROOT/accepted; FAILED=$ROOT/failed
BACKUP=$ROOT/backup_pre_lineage
mkdir -p $QL $DONE $LOCK $WAIT $SLOTS $SEEDS $ACC $FAILED $BACKUP
GPUS="0 1"
NEED_FF=18500
NEED_OV=9500          # every non-FF dataset
NEED_LIGHT=4000       # resolVI (peak 3.1 GB on FF)
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=600
MAX_SEED=8            # replacement seeds 3..8 per dataset (finalL only)
REF_WAIT=43200        # longest a probe grade waits for its uncontrolled refs
KAPPAS="0 0.05 0.1 0.2 0.3 0.4"
TAG=_lineage          # battery / baseline probe records / timing outputs
PY="uv run python"

BASE=/home/rmolen/github/DisCell-baselines
BDATA=$BASE/data/lineage
BRES=$BASE/results
RVI=$BRES/resolvi_lineage

COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30
        --adv-comp-weight $ADV_COMP_WEIGHT --label-key $LABEL_KEY"
UNCONTROLLED="--alpha-a 0"                        # appended: last flag wins
PROJ="--epochs 200 --patience 20 --figures-every 200"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002"
  [$FF]="--alpha-z 0.00035")
declare -A VARIANT=([$GS]=pdl018d [$GD]=pdl018d [$OV]=full [$LU]=full [$FF]=full)
declare -A TILES=([$GS]=2048 [$GD]=2048 [$OV]=4096 [$LU]=4096 [$FF]=4096)
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag $TAG"
TUMOUR_BAND=1         # set to 0 by the preflight when NAME_GAPS_OK=1 lets a gap through

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
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

gpu_idle() {  # no compute process at all on GPU $1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 | tr -d '[:space:]')" ]
}

live_waiter() {  # live_waiter <glob> [<dir to skip>]: a waiting file whose pid is alive
  local f
  for f in $1; do
    [ -e "$f" ] || continue
    [ -n "${2:-}" ] && case "$f" in $2/*) continue ;; esac
    kill -0 "${f##*.}" 2>/dev/null && return 0
  done
  return 1
}

# First call on a freed card: a phase-3 step (the tables) > an FF step > the
# rest. A phase-3 step still yields to another queue's FF waiter; a sweep
# step (phase 2) yields to a waiting phase-3 step, whatever its size.
held_elsewhere() {  # held_elsewhere <gpu> <free> <role>: someone has first call
  local g=$1 free=$2 role=$3 dir now age skip=""
  now=$(date +%s)
  [ "${P3:-0}" = 1 ] && skip=$WAIT
  if [ "$role" != ff ] && [ "$free" -ge "$NEED_FF" ] \
     && live_waiter "scripts/logs/*/waiting/ff.*" "$skip"; then
    return 0                                       # an FF waiter first
  fi
  if [ "${P2:-0}" = 1 ] && live_waiter "$WAIT/p3.*"; then
    return 0                                       # the sweep yields to phase 3
  fi
  for dir in scripts/logs/*/locks/gpu$g; do        # may not have allocated yet
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    age=$((now - $(stat -c %Y "$dir")))
    [ $age -lt $FRESH ] && return 0
  done
  return 1
}

acquire_gpu() {  # acquire_gpu <need MiB> <role: ov|ff|light|timing>
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID" p3="$WAIT/p3.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  [ "${P3:-0}" = 1 ] && touch "$p3"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      if [ "$role" = timing ]; then gpu_idle $g || continue; fi
      held_elsewhere $g $free $role && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! held_elsewhere $g $free $role \
         && { [ "$role" != timing ] || gpu_idle $g; }; then
        rm -f "$me" "$p3"; echo $g; return 0
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
_step() {  # _step <need> <role> <marker> <cmd...>: a GPU step, retried on OOM
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0 toucher=""
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    gpu=$(acquire_gpu $need $role)
    case $name in fit_*) printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" \
                           "$(code_hash)" >> $ROOT/code_hashes.tsv ;; esac
    if [ "$role" = timing ]; then     # keep the lock fresh: nobody joins
      ( while [ -d "$LOCK/gpu$gpu" ]; do touch "$LOCK/gpu$gpu" 2>/dev/null; sleep 60; done ) &
      toucher=$!
    fi
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    [ -n "$toucher" ] && { kill $toucher 2>/dev/null; toucher=""; }
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

backup() {  # backup <file>...: keep the pre-lineage version once (never clobber)
  local f dst
  for f in "$@"; do
    [ -f "$f" ] || continue
    dst=$BACKUP/${f#data/datasets/}; mkdir -p "$(dirname $dst)"
    cp -n "$f" "$dst"
  done
}

# -- (a) the relabel ------------------------------------------------------------
bundles_in_use() {  # every bundle file of the four datasets and the dual
  local ds
  for ds in $ALL $GD; do ls data/datasets/$ds/bundle/*.h5ad; done
}

apply_one() {  # apply_one <dataset>: wait until nobody holds a bundle, apply
  local ds=$1 n=0
  if [ -f "$DONE/$(marker apply_lineage $ds all)" ]; then log "skip apply_lineage $ds (done)"; return 0; fi
  # shellcheck disable=SC2046
  while fuser $(bundles_in_use) > /dev/null 2>&1; do
    [ $((n % 10)) -eq 0 ] && log "(a) a process holds a bundle open -- waiting before $ds"
    n=$((n + 1)); sleep 60
  done
  cpu_step "$(marker apply_lineage $ds all)" $PY -m discell.experiments.apply_lineage \
    --dataset $ds --sox2ot $SOX2OT --cyst $CYST
}

stage_a() {
  local ds bad=0
  for ds in $OV $GS $LU $FF; do apply_one $ds || bad=1; done   # GS also writes GD
  return $bad
}

# -- preflight -------------------------------------------------------------------
preflight_config() {  # every arm's TrainConfig vs runs/final_s0/config.json
  local specs=() ds
  for ds in $ALL; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  $PY - "$LABEL_KEY" "$ADV_COMP_WEIGHT" "$UNCONTROLLED" "$PROJ" "$KAPPAS" "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
label_key, comp, unc, proj, kappas = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5].split()
BASE = {"run_name", "seed", "git", "label_key", "adv_comp_weight"}
default = asdict(TrainConfig(dataset=""))
bad = 0
def derive(ds, flags):
    args = vars(build_parser().parse_args(["--dataset", ds, "--seed", "0"] + shlex.split(flags)))
    for k in ("quiet", "time_only"):
        args.pop(k, None)
    return asdict(TrainConfig(**args))
for spec in sys.argv[6:]:
    ds, flags = spec.split("::", 1)
    ref = json.load(open(f"data/datasets/{ds}/runs/final_s0/config.json"))
    arms = [("finalL", flags, set(), {"alpha_a": 0.3, "kappa": 0.1, "epochs": 500, "patience": 40, "phi_proj": 0}),
            ("uncontrolledL", f"{flags} {unc}", {"alpha_a"}, {"alpha_a": 0.0, "epochs": 500})]
    arms += [(f"sweepL_k{k}", f"{flags} --kappa {k}", {"kappa"}, {"kappa": float(k), "epochs": 500, "patience": 40})
             for k in kappas]
    if ds == "xenium_prime_ovarian_cancer_ffpe":
        arms += [(f"projL{d}", f"{flags} {proj}" + (" --phi-proj 32" if d == 32 else ""),
                  {"epochs", "patience", "figures_every", "phi_proj"},
                  {"epochs": 200, "patience": 20, "phi_proj": 32 if d == 32 else 0})
                 for d in (384, 32)]
    for name, arm_flags, extra, want in arms:
        new = derive(ds, arm_flags)
        diffs = [f"{k}: {new.get(k)!r} vs final_s0 {ref.get(k, '<absent>')!r}"
                 for k in sorted(set(new) | set(ref)) if k not in BASE | extra
                 and ((k in ref and new.get(k) != ref[k])
                      or (k not in ref and new[k] != default[k]))]
        want = {**want, "label_key": label_key, "adv_comp_weight": comp, "w_warmup_epochs": 30}
        wrong = [f"{k}={new[k]!r} (want {v!r})" for k, v in want.items() if new[k] != v]
        ok = not diffs and not wrong
        bad += not ok
        print(f"{ds} {name}: " + ("final_s0's config but for the arm's own fields"
                                  if ok else "DIFFERS: " + "; ".join(diffs + wrong)))
sys.exit(1 if bad else 0)
EOF
}

preflight_labels() {  # the label column where every step will read it; name matchers
  $PY - "$LABEL_KEY" "$GS" "$GD" "$OV" "$LU" "$FF" <<'EOF'
import sys, types
import h5py, numpy as np
from anndata.io import read_elem
from discell.model.transport import tumour_band_labels
from discell.model.labels import is_endothelial, is_smooth_muscle, is_tumour
key, gs, gd, ov, lu, ff = sys.argv[1:]
use = [(gs, "pdl018d"), (gd, "pdl018d"), (ov, "full"), (lu, "full"), (ff, "full"), (gs, "full"), (gd, "full")]
seen, bad = {}, 0
for ds, variant in use:
    with h5py.File(f"data/datasets/{ds}/bundle/{variant}.h5ad", "r") as f:
        if key not in f["obs"]:
            print(f"MISSING obs[{key!r}] in {ds}/{variant}"); bad += 1; continue
        values = np.asarray(read_elem(f["obs"][key])).astype(str)
        default = f["uns"]["default_label"][()]
        old = np.asarray(read_elem(f["obs"][default.decode() if isinstance(default, bytes) else default])).astype(str)
    seen[(ds, variant)] = (sorted(set(values)), sorted(set(old) - {"nan"}))
    print(f"{ds}/{variant}: {len(set(values))} classes in obs[{key!r}]")
for variant in ("pdl018d", "full"):
    if seen.get((gs, variant), [None])[0] != seen.get((gd, variant), [None])[0]:
        print(f"GSE solo / dual {variant}: class sets differ -- crossslide would fail"); bad += 1
# readers that find classes by name (discell.model.labels, as transport and
# validate call them): which classes each one sees, old -> new
MATCH = {"tumour band / interface (transport, external 6b.3-4, atlas/validate landmarks)": is_tumour,
         "vasculature landmark (is_endothelial)": is_endothelial,
         "pericyte landmark ('Pericyte')": lambda n: "Pericyte" in n,
         "smooth-muscle landmark (is_smooth_muscle)": is_smooth_muscle}
for ds in (ov, gs, lu, ff):
    new, old = seen[(ds, "full")]
    for what, hit in MATCH.items():
        a, b = [n for n in old if hit(n)], [n for n in new if hit(n)]
        if a or b:
            print(f"  {ds}: {what}: {len(a)} old classes -> {len(b)} new {b}"
                  + ("   <-- LOST" if a and not b else ""))
new = seen[(ov, "full")][0]
fake = types.SimpleNamespace(type_names=np.asarray(new), t=np.zeros(4, int),
                             graph=types.SimpleNamespace(n_cells=4, degrees=np.ones(4)),
                             positions=np.random.default_rng(0).random((4, 2)))
try:
    tumour_band_labels(fake)
    print("TUMOUR_BAND ok: transport.tumour_band_labels finds a tumour class on ovarian")
except ValueError as exc:
    print(f"TUMOUR_BAND GAP: {exc}")
    sys.exit(3 if not bad else 1)
sys.exit(1 if bad else 0)
EOF
}

preflight_env() {
  case "$SOX2OT" in unassigned|tumour) ;; *)
    log "SOX2OT must be set to unassigned or tumour (the author's choice) -- stopping"; return 1 ;; esac
  case "$CYST" in mesothelial|tumour) ;; *)
    log "CYST must be set to mesothelial or tumour (the author's choice) -- stopping"; return 1 ;; esac
  case "$SWEEP_READS" in light|full) ;; *)
    log "SWEEP_READS must be light or full -- stopping"; return 1 ;; esac
  $PY -m discell.model.train --help 2>/dev/null | grep -q -- "--adv-comp-weight" \
    || { log "the trainer has no --adv-comp-weight -- stopping"; return 1; }
  log "ADV_COMP_WEIGHT=$ADV_COMP_WEIGHT LABEL_KEY=$LABEL_KEY SOX2OT=$SOX2OT CYST=$CYST NAME_GAPS_OK=$NAME_GAPS_OK SWEEP_READS=$SWEEP_READS"
}

# -- fits --------------------------------------------------------------------------
fit() {   # fit <dataset> <run> <seed> [extra flags...]
  local ds=$1 run=$2 seed=$3 mk; shift 3; mk=$(marker fit $ds $run)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$mk"; log "skip $mk (metrics.json present)"; return 0
  fi
  # shellcheck disable=SC2086
  gstep $ds "$mk" $PY -m discell.model.train --dataset $ds --run-name $run \
    ${FLAGS[$ds]} $COMMON --seed $seed "$@"
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

record_dead() {  # record_dead <dataset> <run> <action>: DEAD_RUNS.tsv, once
  local why
  why=$(dead_reason $1 $2) || return 0
  [ -n "$why" ] || return 0
  grep -qP "\t$1\t$2\t" $ROOT/DEAD_RUNS.tsv 2>/dev/null || \
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $1 $2 "$why" "$3" >> $ROOT/DEAD_RUNS.tsv
  log "$1/$2 DEAD ($why) -- $3"
}

next_seed() {  # next_seed <dataset>: claim the smallest unused seed >= 3
  local s
  for s in $(seq 3 $MAX_SEED); do
    mkdir $SEEDS/$1/s$s 2>/dev/null && { echo $s; return 0; }
  done
  return 1
}

# -- (e) the per-run reads -----------------------------------------------------------
wait_refs() {  # wait_refs <dataset>: both uncontrolledL grades done (or given up)
  local ds=$1 t0=$SECONDS n=0 s ok mk
  while true; do
    ok=1
    for s in 0 1; do
      for mk in "$(marker probe_regrade $ds uncontrolledL_s$s)" \
                $([ "$ds" = "$GS" ] && marker probe_regrade $GD uncontrolledL_s$s); do
        [ -f "$DONE/$mk" ] || [ -f "$FAILED/$mk" ] || [ -f "$FAILED/$(marker fit $ds uncontrolledL_s$s)" ] || ok=0
      done
    done
    [ $ok -eq 1 ] && return 0
    if [ $((SECONDS - t0)) -ge $REF_WAIT ]; then
      log "$ds: uncontrolledL grades still missing after ${REF_WAIT}s -- grading without them"; return 0
    fi
    [ $((n % 10)) -eq 0 ] && log "$ds: waiting for the uncontrolledL grades before a probe grade"
    n=$((n + 1)); sleep 60
  done
}

grade() {  # grade <dataset> <run>: the per-block probe on its own section (+ the dual)
  local ds=$1 run=$2
  # shellcheck disable=SC2086
  gstep $ds "$(marker probe_regrade $ds $run)" $PY -m discell.experiments.probe_regrade \
    --dataset $ds --run $run --force $REFS
  if [ "$ds" = "$GS" ]; then
    # shellcheck disable=SC2086
    gstep $GD "$(marker probe_regrade $GD $run)" $PY -m discell.experiments.probe_regrade \
      --dataset $GD --config-from $GS --run $run --force $REFS
  fi
}

reads() {  # reads <dataset> <run> [validate analyses]: every per-run read before the cross-seed atlas
  local ds=$1 run=$2 analyses=${3:-morans,niche,probe}
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses $analyses
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  wait_refs $ds
  grade $ds $run          # after validate: its in-module probe grades against the old refs
  gstep $ds "$(marker atlas $ds $run)" $PY -m discell.model.atlas --dataset $ds --run $run
  gstep $ds "$(marker transport $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read both --hvg 1000
  if [ "$ds" = "$OV" ]; then
    if [ $TUMOUR_BAND -eq 1 ]; then
      gstep $ds "$(marker transport_band $ds $run)" $PY -m discell.model.transport \
        --dataset $ds --run $run --read both --hvg 1000 --niche-source tumour-band
    else
      log "SKIP transport_band $ds/$run: no tumour class by name (NAME_GAPS_OK=1)"
    fi
  fi
  if [ "$ds" = "$GS" ]; then
    gstep $ds "$(marker crossslide $ds $run)" $PY -m discell.model.crossslide --dataset $GS \
      --run $run --eval-dataset $GD
    gstep $ds "$(marker recon_modes_dual $ds $run)" $PY -m discell.experiments.recon_modes \
      --dataset $GS --run $run --eval-dataset $GD
  fi
  gstep $ds "$(marker recon_modes $ds $run)" $PY -m discell.experiments.recon_modes \
    --dataset $ds --run $run
  return 0
}

compare_group() {  # compare_group <dataset> <run>...: cross-seed atlas, then reports
  local ds=$1 r others live=""; shift
  for r in "$@"; do [ -f "$(runs $ds)/$r/metrics.json" ] && live="$live $r"; done
  for r in $live; do
    others=$(echo $live | tr ' ' '\n' | grep -vx $r | tr '\n' ' ')
    if [ -n "$others" ]; then
      # shellcheck disable=SC2086
      gstep $ds "$(marker atlas_cmp $ds $r)" $PY -m discell.model.atlas --dataset $ds \
        --run $r --compare-runs $others
    else
      log "$ds/$r: no other seed -- no cross-seed atlas"
    fi
    gstep $ds "$(marker report $ds $r)" $PY -m discell.model.report --dataset $ds --run $r
  done
}

# -- (c) + (b): the references ---------------------------------------------------------
unc_reads() {  # unc_reads <dataset> <run>: the "no adversary" reference row's reads
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  # no in-module probe: it would re-grade the reference against the old fits
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche
  gstep $ds "$(marker recon_modes $ds $run)" $PY -m discell.experiments.recon_modes \
    --dataset $ds --run $run
  if [ "$ds" = "$GS" ]; then
    gstep $ds "$(marker recon_modes_dual $ds $run)" $PY -m discell.experiments.recon_modes \
      --dataset $GS --run $run --eval-dataset $GD
  fi
  gstep $ds "$(marker transport_mean $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read mean
}

uncontrolled() {  # uncontrolled <dataset> <seed>: fit, guard, grade (the probe reference)
  local ds=$1 run=uncontrolledL_s$2
  # shellcheck disable=SC2086
  fit $ds $run $2 $UNCONTROLLED || { log "$ds/$run: fit failed -- no reference"; return 1; }
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  record_dead $ds $run "recorded (reference; not refitted)"
  grade $ds $run
}

slot() {  # slot <dataset> <k>: seed k; while the fit is dead, the next seed
  local ds=$1 k=$2 chain=$SLOTS/${1}_slot$2 seed run why
  [ -s $chain ] || echo $k > $chain
  seed=$(tail -1 $chain)
  while true; do
    run=finalL_s$seed
    if ! fit $ds $run $seed; then log "SLOT $ds/$k: fit $run failed -- slot left empty"; return 1; fi
    gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy \
      --dataset $ds --run $run || log "SLOT $ds/$k: guard failed on $run -- deciding on metrics.json alone"
    if ! why=$(dead_reason $ds $run); then
      log "SLOT $ds/$k: could not read the flags of $run -- slot left empty"; return 1
    fi
    if [ -z "$why" ]; then
      echo $run > $ACC/${ds}_slot$k; log "SLOT $ds/$k: $run live"; break
    fi
    record_dead $ds $run "refitted with the next seed"
    if ! seed=$(next_seed $ds); then
      log "SLOT $ds/$k: $run DEAD ($why) and no unused seed <= $MAX_SEED left"; return 1
    fi
    echo $seed >> $chain
    log "SLOT $ds/$k: $run DEAD ($why) -- refitting as finalL_s$seed"
  done
  reads $ds $run
}

accepted() {  # the live run of each slot, slot order (0 1 2)
  local k
  for k in 0 1 2; do [ -s $ACC/${1}_slot$k ] && cat $ACC/${1}_slot$k; done
}
accepted_s0_last() {  # slots 1 2 0: whatever runs last leaves the battery cache
  local k
  for k in 1 2 0; do [ -s $ACC/${1}_slot$k ] && cat $ACC/${1}_slot$k; done
}
slot0() { cat $ACC/${1}_slot0 2>/dev/null; }

ref_lane() {  # ref_lane <dataset> "<uncontrolled seeds>" "<slots>"
  local ds=$1 s
  for s in $2; do uncontrolled $ds $s; done
  for s in $3; do slot $ds $s; done
  for s in $2; do unc_reads $ds uncontrolledL_s$s; done   # after the slots
}
laneGS() { ref_lane $GS "0 1" "0 1 2"; compare_group $GS $(accepted $GS); log "GSE references finished"; }
laneOV() { ref_lane $OV "0 1" "0 1 2"; compare_group $OV $(accepted $OV); log "ovarian references finished"; }
laneLU() { ref_lane $LU "0 1" "0 1 2"; compare_group $LU $(accepted $LU); log "lung references finished"; }
laneFF_A() { ref_lane $FF "0" "0 2"; log "FF reference sub-lane A finished"; }
laneFF_B() { ref_lane $FF "1" "1"; log "FF reference sub-lane B finished"; }

# -- (d): the kappa sweep ------------------------------------------------------------------
sweep_run() { echo sweepL_k${1}_s$2; }

light_reads() {  # light_reads <dataset> <run>: the sweep point's reads (SWEEP_READS=light)
  local ds=$1 run=$2
  if [ ! -f "$(runs $ds)/$run/metrics.json" ]; then log "no fit for $ds/$run -- no reads"; return 0; fi
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche,probe
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  wait_refs $ds
  grade $ds $run          # after validate: its in-module probe grades against the old refs
  gstep $ds "$(marker transport_mean $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read mean
  if [ "$ds" = "$GS" ]; then   # the held-out section at every kappa (cheap)
    gstep $ds "$(marker crossslide $ds $run)" $PY -m discell.model.crossslide --dataset $GS \
      --run $run --eval-dataset $GD
  fi
  return 0
}

link_reference() {  # link_reference <dataset> <seed>: sweepL_k0.1_s<seed> -> finalL_s<seed>
  local ds=$1 s=$2 d link target=finalL_s$2; d=$(runs $1); link=$d/sweepL_k0.1_s$2
  if [ -L $link ]; then log "$ds/sweepL_k0.1_s$s -> $(readlink $link) (linked)"; return 0; fi
  if [ -e $link ]; then log "$ds/sweepL_k0.1_s$s exists as a fitted run -- kept, not linked"; return 1; fi
  if [ ! -f $d/$target/metrics.json ]; then
    log "$ds: $target has no metrics.json -- sweepL_k0.1_s$s is fitted instead"; return 1
  fi
  ln -s $target $link
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $ds sweepL_k0.1_s$s $target >> $ROOT/SWEEP_LINKS.tsv
  log "$ds: sweepL_k0.1_s$s -> $target (same config, seed and split; not refitted)"
}

sweep_lane() {  # sweep_lane <dataset>: 15 fits + 3 links, reads by SWEEP_READS, sweep reads
  local ds=$1 s k run group linked
  export P2=1
  for s in 0 1 2; do
    for k in $KAPPAS; do
      run=$(sweep_run $k $s)
      if [ "$k" = 0.1 ] && link_reference $ds $s; then
        record_dead $ds $run "recorded (sweep point = finalL_s$s)"
        continue                    # finalL's reads are this point's reads
      fi
      if fit $ds $run $s --kappa $k; then
        gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
        record_dead $ds $run "recorded (sweep; not refitted)"
        if [ "$SWEEP_READS" = full ]; then reads $ds $run; else light_reads $ds $run; fi
      fi
    done
  done
  if [ "$SWEEP_READS" = full ]; then
    for k in $KAPPAS; do
      group=""; linked=0
      for s in 0 1 2; do
        group="$group $(sweep_run $k $s)"
        [ -L "$(runs $ds)/$(sweep_run $k $s)" ] && linked=1
      done
      [ $linked -eq 1 ] && continue   # a linked point: finalL's compare stands for it
      # shellcheck disable=SC2086
      compare_group $ds $group
    done
  fi
  # shellcheck disable=SC2086
  gstep $ds "$(marker sweep_report $ds sweepL)" $PY -m discell.model.sweep --dataset $ds \
    --report-only --tag sweepL --param kappa --values $KAPPAS --seeds 0 1 2 \
    --label-key $LABEL_KEY ${FLAGS[$ds]} --epochs 500 --patience 40
  # writes the untagged atlas_kappa_survival{,_internal}.json (the old
  # published numbers are carried under legacy) -- keep the sweep3 copies
  backup data/datasets/$ds/experiments/atlas_kappa_survival.json \
         data/datasets/$ds/experiments/atlas_kappa_survival_internal.json
  # shellcheck disable=SC2086
  gstep $ds "$(marker kappa_survival $ds sweepL)" $PY -m discell.model.validate --dataset $ds \
    --sweep-tag sweepL --kappas $KAPPAS --seeds 0 1 2 \
    --analyses kappa_survival,morans,niche \
    --survival-reference "$(slot0 $ds || echo finalL_s0)"
  log "$ds sweep finished"
}

proj_lane() {  # the S53 projection test at lineage labels (ovarian, 200/20)
  local s run arm d
  export P2=1
  for s in 0 1 2; do
    for d in 384 32; do
      run=projL${d}_s$s; arm=""; [ $d = 32 ] && arm="--phi-proj 32"
      # shellcheck disable=SC2086
      fit $OV $run $s $PROJ $arm || continue
      gstep $OV "$(marker validate $OV $run)" $PY -m discell.model.validate --dataset $OV \
        --run $run --analyses morans,niche,probe
      gstep $OV "$(marker degeneracy $OV $run)" $PY -m discell.model.degeneracy --dataset $OV --run $run
      record_dead $OV $run "recorded (projection test)"
      wait_refs $OV; grade $OV $run
      gstep $OV "$(marker recon_modes $OV $run)" $PY -m discell.experiments.recon_modes \
        --dataset $OV --run $run
    done
  done
  log "projection lane finished"
}

# -- (f): baselines at the new labels ---------------------------------------------------
seed_of() { echo "${1##*_s}"; }

export_one() {  # export_one <dataset> <config run>: the bundle as the baselines see it
  local ds=$1 cfg=$2 out=$BDATA/$1.h5ad tmp=$BDATA/tmp_$1
  cpu_step "$(marker export $ds lineage)" bash -c "
    set -e; mkdir -p '$tmp'
    $PY -m discell.experiments.export_for_baselines --dataset $ds \
      --variant ${VARIANT[$ds]} --tile-cells ${TILES[$ds]} --label-key $LABEL_KEY \
      --seed $(seed_of $cfg) --out-dir '$tmp'
    mv '$tmp/${VARIANT[$ds]}_baselines.h5ad' '$out'; rmdir '$tmp'"
}

resolvi() {  # resolvi <dataset> [transfer dataset]
  local ds=$1 args=(--h5ad "$BDATA/$1.h5ad" --out "$RVI/$1")
  [ $# -ge 2 ] && args+=(--transfer-h5ad "$BDATA/$2.h5ad" --transfer-out "$RVI/$2")
  [ -f "$BDATA/$ds.h5ad" ] || { log "no lineage export for $ds -- resolVI not refitted"; return 1; }
  _step $NEED_LIGHT light "$(marker resolvi $ds lineage)" $PY "$BASE/resolvi/run_resolvi.py" "${args[@]}"
}

battery_columns() {  # battery_columns <dataset> [<config dataset>]: DisCell finalL, slot 0 last
  local ds=$1 from=${2:-$1} r extra=() cfg; cfg=$(slot0 $from)
  [ -n "$cfg" ] || { log "$ds: no live slot 0 on $from -- no battery columns"; return 1; }
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  for r in $(accepted_s0_last $from); do
    gstep $ds "$(marker battery_discell $ds $r)" $PY -m discell.experiments.baseline_battery \
      --dataset $ds --discell-run $r --config-run $cfg --tag $TAG "${extra[@]}"
  done
}

baseline() {  # baseline <dataset> <config dataset> <method> <latents dir>: column + probe grade
  local ds=$1 from=$2 method=$3 dir=$4 extra=() cfg tag
  cfg=$(slot0 $from)
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  if [ -z "$cfg" ] || [ ! -f "$DONE/$(marker battery_discell $ds $cfg)" ]; then
    log "$ds '$method': the slot-0 DisCell column did not run -- no cached context, not scored"; return 0
  fi
  [ -f "$dir/latents.h5ad" ] || { log "$ds '$method': no latents at $dir -- not scored"; return 0; }
  tag=$(printf '%s' "$method" | tr -c 'A-Za-z0-9=.' '_')
  gstep $ds "$(marker battery_$tag $ds $cfg)" $PY -m discell.experiments.baseline_battery \
    --dataset $ds --method "$method" --latents "$dir/latents.h5ad" --config-run $cfg \
    --tag $TAG "${extra[@]}"
  # shellcheck disable=SC2086
  gstep $ds "$(marker probe_$tag $ds $cfg)" $PY -m discell.experiments.probe_regrade \
    --dataset $ds --run $cfg --baseline-latents "$dir/latents.h5ad" --method "$method" \
    --force $REFS "${extra[@]}"
}

stage_f() {
  local ds
  mkdir -p $BDATA
  for ds in $GS $OV $LU $FF; do [ -n "$(slot0 $ds)" ] && export_one $ds "$(slot0 $ds)"; done
  [ -n "$(slot0 $GS)" ] && export_one $GD "$(slot0 $GS)"
  resolvi $GS $GD; resolvi $OV; resolvi $LU
  battery_columns $GS
  baseline $GS $GS resolVI $RVI/$GS
  baseline $GS $GS SIMVI $BRES/simvi/$GS
  baseline $GS $GS MintFlow $BRES/mintflow/$GS
  baseline $GS $GS "MintFlow (w=600, 5/50 epochs)" $BRES/mintflow/${GS}_w600
  baseline $GS $GS "MintFlow (50 epochs, w=600)" $BRES/mintflow/${GS}_full
  battery_columns $GD $GS
  baseline $GD $GS resolVI $RVI/$GD
  baseline $GD $GS "SIMVI (fit on this section)" $BRES/simvi/${GD}_fit_on_target
  baseline $GD $GS "MintFlow (transfer, 5/50 epochs)" $BRES/mintflow/${GD}_transfer
  baseline $GD $GS "MintFlow (transfer, 50 epochs)" $BRES/mintflow/${GD}_transfer_full
  battery_columns $OV; baseline $OV $OV resolVI $RVI/$OV
  battery_columns $LU; baseline $LU $LU resolVI $RVI/$LU
  resolvi $FF
  battery_columns $FF
  baseline $FF $FF resolVI $RVI/$FF
  baseline $FF $FF "SIMVI (100k-cell window)" $BRES/simvi/$FF
  baseline $FF $FF MintFlow $BRES/mintflow/$FF      # skipped: the FF MintFlow fit left no latents
  log "(f) baselines finished"
}

# -- (g) + (h) ------------------------------------------------------------------------------
ext() {  # ext <read> <dataset> <run>
  gstep $2 "$(marker ext_$1 $2 $3)" $PY -m discell.experiments.external_criteria $1 \
    --dataset $2 --run $3
}

envelope() {
  local ds std=1 args
  for ds in $ALL; do
    backup data/datasets/$ds/experiments/envelope_table_ci_at_best.md
    [ "$(accepted $ds | tr '\n' ' ')" = "finalL_s0 finalL_s1 finalL_s2 " ] || std=0
  done
  if [ $std -eq 1 ]; then
    cpu_step envelope_finalL_all $PY scripts/envelope_tables.py --datasets $ALL \
      --runs finalL_s0 finalL_s1 finalL_s2 --at best --ci \
      --combined $ROOT/envelope_tables_finalL.md
  else   # a refitted slot: each dataset with its own live triple
    for ds in $ALL; do
      args=$(accepted $ds)
      [ -n "$args" ] || { log "$ds: no live run -- no envelope table"; continue; }
      # shellcheck disable=SC2086
      cpu_step "$(marker envelope $ds finalL)" $PY scripts/envelope_tables.py --datasets $ds \
        --runs $args --at best --ci --combined $ROOT/envelope_tables_finalL_$ds.md
    done
  fi
}

repin() {  # repin <dataset>: runs/best -> the slot-0 run
  local ds=$1 d new old; d=$(runs $1); new=$(slot0 $ds)
  if [ -z "$new" ]; then log "$ds: slot 0 has no live run -- best NOT re-pinned"; return 0; fi
  if [ -L $d/best ] && [ "$(readlink $d/best)" = "$new" ]; then
    log "$ds: best already -> $new"; return 0
  fi
  if [ -L $d/best ]; then
    old=$(readlink $d/best)
  elif [ -d $d/best ]; then
    old="directory, moved to best_pre_lineage"; mv $d/best $d/best_pre_lineage
  else
    old=none
  fi
  ln -sfn $new $d/best.lineage_tmp && mv -T $d/best.lineage_tmp $d/best
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" $ds "$old" $new >> $ROOT/REPIN.tsv
  log "$ds: best was -> $old; re-pinned to $new"
}

phase3() {  # (e) bootstrap on finalL, (f), (g) but timing, (h); one step at a time
  local ds r args
  export P3=1
  for ds in $GS $OV $LU $FF; do
    for r in $(accepted $ds); do
      gstep $ds "$(marker bootstrap $ds $r)" $PY -m discell.experiments.bootstrap \
        --dataset $ds --run $r --n 1000
    done
  done
  stage_f
  for r in $(accepted $OV); do
    ext signalling-share $OV $r; ext mi-quadrant $OV $r
    if [ $TUMOUR_BAND -eq 1 ]; then ext axis-test $OV $r
    else log "SKIP axis-test $OV/$r: no tumour class by name (NAME_GAPS_OK=1)"; fi
  done
  for r in $(accepted $FF); do ext signalling-share $FF $r; ext mi-quadrant $FF $r; done
  for r in $(accepted $GS); do ext mi-quadrant $GS $r; done
  for ds in $ALL; do
    args=(); for r in $(accepted $ds); do args+=(--run $r); done
    [ ${#args[@]} -gt 0 ] || { log "$ds: no live run -- no GO read"; continue; }
    cpu_step "$(marker go_localisation $ds finalL)" $PY -m discell.experiments.go_localisation \
      --dataset $ds "${args[@]}" --out-stem go_localisation$TAG
  done
  envelope
  for ds in $ALL; do repin $ds; done
  log "phase 3 finished"
}

# -- after everything: verdicts against the lineage resolVI, tables, timing -----------------
tables() {
  local ds pats=('finalL_s*' 'uncontrolledL_s*' 'sweepL_*' 'projL*')
  for ds in $ALL; do   # scoped: the old-label records are never touched
    # shellcheck disable=SC2086
    cpu_step "$(marker reverdict $ds lineage)" $PY -m discell.experiments.probe_regrade \
      --dataset $ds --reverdict --runs "${pats[@]}" $REFS
    # shellcheck disable=SC2086
    cpu_step "$(marker probe_table $ds lineage_final)" $PY -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
    # shellcheck disable=SC2086
    cpu_step "$(marker probe_table $ds lineage_sweep)" $PY -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_sweep --runs 'sweepL_*' 'finalL_s*' $REFS
  done
  # shellcheck disable=SC2086
  cpu_step "$(marker reverdict $GD lineage)" $PY -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --reverdict --runs "${pats[@]}" $REFS
  # shellcheck disable=SC2086
  cpu_step "$(marker probe_table $GD lineage_final)" $PY -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --table lineage_final --baselines \
    --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
}

timing_lane() {
  local ds arm cfg
  for ds in $GS $LU $OV $FF; do
    cfg=$(slot0 $ds); [ -n "$cfg" ] || { log "$ds: no slot 0 -- no timing"; continue; }
    for arm in phi phi_zeroed phi_dropped; do
      _step $(need_of $ds) timing "$(marker timing $ds $arm)" $PY scripts/timing_mode.py \
        --dataset $ds --arm $arm --epochs 20 --ref-run $cfg --tag $TAG
    done
  done
  cfg=$(slot0 $OV)
  [ -n "$cfg" ] && _step $NEED_OV timing "$(marker timing $OV phiproj32)" $PY scripts/timing_mode.py \
    --dataset $OV --arm phiproj32 --epochs 20 --ref-run $cfg --tag $TAG
  cpu_step "$(marker timing_tables all lineage)" $PY scripts/timing_mode.py --tables \
    --tag $TAG --ref-run "$(slot0 $OV || echo finalL_s0)" --combined $ROOT/timing_all$TAG.md
  cpu_step "$(marker phi_projection $OV lineage)" $PY scripts/phi_projection_table.py \
    --full projL384_s --proj projL32_s --tag $TAG \
    --config "runs/finalL_s0/config.json at 200/20" --reference "uncontrolledL_s{0,1}"
  log "timing finished"
}

readout() {
  $PY - $ROOT $ALL > $QL/readout.log 2>&1 <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); datasets = sys.argv[2:]
def load(p):
    try: return json.loads(p.read_text())
    except Exception: return None
def f(x, n=4):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
acc = {}
for p in (root / "accepted").glob("*_slot*"):
    ds, k = p.name.rsplit("_slot", 1); acc.setdefault(ds, {})[int(k)] = p.read_text().strip()
L = ["# Lineage re-pin read-out (scripts/queue_2026-09-25_final_lineage.sh)", "",
     "All reads at the accepted checkpoint (`best`).", "",
     "| dataset | run | slot | label_key | comp w | best epoch | recon_val | NMI | dead_w | "
     "I(niche;w) excess | invariance_pass (lineage refs) | minutes |", "|" + "---|" * 12]
for ds in datasets:
    slots = {v: k for k, v in acc.get(ds, {}).items()}
    rd = pathlib.Path(f"data/datasets/{ds}/runs")
    for pat in ("finalL_s*", "uncontrolledL_s*", "sweepL_*", "projL*"):
        for d in sorted(p for p in rd.glob(pat) if p.is_dir() and not p.is_symlink()):
            m = load(d / "metrics.json") or {}; g = load(d / "degeneracy.json") or {}
            c = load(d / "config.json") or {}; pb = load(d / "validation" / "probe_blocks.json") or {}
            b = m.get("best") or {}; w = g.get("w_channel") or {}
            L.append(f"| {ds} | {d.name} | {slots.get(d.name, '')} | {c.get('label_key')} | "
                     f"{c.get('adv_comp_weight')} | {f(b.get('epoch'))} | {f(b.get('recon_val'))} | "
                     f"{f(b.get('nmi'))} | {f(m.get('dead_w_channel'))} | {f(w.get('w_niche_mi_excess'))} | "
                     f"{f(pb.get('invariance_pass'))} | {f(m.get('minutes'), 1)} |")
L += ["", "## Live finalL triples (slot 0 is runs/best)", ""]
L += [f"* {ds}: " + ", ".join(acc.get(ds, {}).get(k, "EMPTY") for k in (0, 1, 2)) for ds in datasets]
for name, title in (("DEAD_RUNS.tsv", "Dead fits (time, dataset, run, why, action)"),
                    ("SWEEP_LINKS.tsv", "Sweep points linked to the references (time, dataset, link, target)"),
                    ("REPIN.tsv", "runs/best re-pins (time, dataset, previous, new)")):
    p = root / name
    L += ["", f"## {title}", "", "```", p.read_text().strip() if p.exists() else "(none)", "```"]
failed = sorted(p.name for p in (root / "failed").iterdir())
L += ["", "## Failed steps (logs in logs/<step>.log)", ""] + ([f"* {n}" for n in failed] or ["(none)"])
hashes = root / "code_hashes.tsv"
if hashes.exists():
    rows = [r.split("\t") for r in hashes.read_text().strip().splitlines()]
    distinct = sorted({r[2].strip() for r in rows})
    L += ["", "## Model code (sha256[:12] of train, networks, equations, elbo) at each fit start", ""]
    L += [f"* {h}: {sum(r[2].strip() == h for r in rows)} fit launches" for h in distinct]
    if len(distinct) > 1:
        L.append("* **the model code changed between fits** -- see code_hashes.tsv")
L += ["", "## Outputs", ""]
outs = [root / "envelope_tables_finalL_at_best.md", root / "timing_all_lineage.md"]
for ds in datasets:
    e = pathlib.Path(f"data/datasets/{ds}/experiments")
    outs += [e / n for n in ("envelope_table_ci_at_best.md", "baseline_battery_lineage.md",
                             "probe_regrade_lineage_final.md", "probe_regrade_lineage_sweep.md",
                             "kappa_sweep_sweepL.json", "validation_sweepL.json",
                             "go_localisation_lineage.md", "timing_lineage.md")]
    outs += [pathlib.Path(f"data/datasets/{ds}/labels/lineage_map_applied.csv")]
outs += [pathlib.Path("data/datasets/gse315411_pdltma06_10_prime_dual/experiments/baseline_battery_lineage.md"),
         pathlib.Path("data/datasets/gse315411_pdltma06_10_prime_dual/labels/lineage_map_applied.csv"),
         pathlib.Path("data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/phi_projection_lineage.md")]
L += [f"* {'ok     ' if p.exists() else 'MISSING'} {p}" for p in outs]
out = root / "READOUT.md"; out.write_text("\n".join(L) + "\n"); print("\n".join(L))
EOF
  log "read-out written: $ROOT/READOUT.md (exit $?)"
}

# -- main ---------------------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -f $WAIT/ff.* $WAIT/p3.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* $WAIT/p3.* 2>/dev/null' EXIT
log "queue start (FF gate $NEED_FF MiB, others $NEED_OV MiB, GPUs: $GPUS; model code $(code_hash))"
preflight_env || exit 1

log "=== (a) relabel ==="
if ! stage_a; then log "(a) FAILED: the relabel did not apply everywhere -- stopping"; exit 1; fi

log "=== preflight ==="
if ! preflight_config; then log "PREFLIGHT FAILED: an arm's config is not final_s0's -- stopping"; exit 1; fi
preflight_labels; rc=$?
if [ $rc -eq 3 ]; then
  if [ "$NAME_GAPS_OK" = 1 ]; then
    TUMOUR_BAND=0
    log "tumour-band reads (transport band, 6b.4) will be SKIPPED on ovarian (NAME_GAPS_OK=1)"
  else
    log "PREFLIGHT: no ovarian class matches the tumour-band reader ('Tumor Cells' / 'Malignant')"
    log "  -- fix the matcher (transport.py / validate.py) or relaunch with NAME_GAPS_OK=1; stopping"
    exit 1
  fi
elif [ $rc -ne 0 ]; then
  log "PREFLIGHT FAILED: the label column is missing or the GSE sections disagree -- stopping"; exit 1
fi
for ds in $ALL; do mkdir -p $SEEDS/$ds/s0 $SEEDS/$ds/s1 $SEEDS/$ds/s2; done

log "=== phase 1: references (b, c, e) ==="
laneFF_A & PFA=$!
(sleep 90; laneFF_B) & PFB=$!    # A takes the first FF slot (seed 0 first)
(sleep 10; laneGS) & PGS=$!
(sleep 20; laneOV) & POV=$!
(sleep 30; laneLU) & PLU=$!
wait $PFA $PFB
compare_group $FF $(accepted $FF); log "FF references finished"
wait $PGS $POV $PLU
log "phase 1 finished"

log "=== phase 2 (d, e) and phase 3 (f, g, h) ==="
phase3 & P3PID=$!
(sleep 30; sweep_lane $FF) & PSF=$!
(sleep 40; sweep_lane $GS) & PSG=$!
(sleep 50; sweep_lane $OV; proj_lane) & PSO=$!
(sleep 60; sweep_lane $LU) & PSL=$!
wait $P3PID $PSF $PSG $PSO $PSL
log "phases 2 and 3 finished"

log "=== verdicts, tables, timing ==="
tables
timing_lane
readout
log "queue finished"
