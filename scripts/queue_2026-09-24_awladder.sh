#!/bin/bash
# The alpha_w multiplier ladder (review R19; devlog pre-registration "alpha_w
# in units of 1/l-bar: the multiplier ladder (R19, motivation, 2026-09-24
# 13:50)"), tag awladder.
#
# alpha_w = m / l-bar, m in {1, 2, 5, 10, 25}, plus the reference rung alpha_w
# 0.1; per-rung values fixed once in $ROOT/LADDER.json (scripts/awladder_table.py
# --init-ladder; l-bar = 1/(2 alpha_z), see that script's docstring). Final
# configuration otherwise, 200/20, seeds 0 1 2, runs aw_<rung>_s<seed>:
#   GSE     --alpha-z 0.0018 --variant pdl018d --tile-cells 2048
#   lung    --alpha-z 0.002 --label-key graphclust
#   ovarian --alpha-z 0.0035
#   FF      --alpha-z 0.00035 --label-key graphclust
#   all     --kappa 0.1 --d-w 6 --gat-sources type_only --w-warmup-epochs 30
#           --epochs 200 --patience 20 --figures-every 200
# Reused, not refitted (LADDER.json "links", symlinks in runs/, preflighted):
#   ovarian and FF aw_ref0.1_s<k> -> wfix_warmup30_aw0.1_s<k> (the w-collapse
#   grid's warm-up arm: this exact configuration); lung aw_m25_s<k> ->
#   aw_ref0.1_s<k> (25 / l-bar = 0.1 on lung: the same configuration).
# SCOPE (author, 2026-09-24, before any fit counted; replaces the
# pre-registered 4 datasets x 3 seeds): stage 1 (default, AWLADDER_STAGE=1) =
# GSE and ovarian, seeds 0 1, all six rungs (ovarian reference reused, GSE
# reference fitted) -> 22 fits. Stage 2 (AWLADDER_STAGE=2, ONLY on the
# coordinator's word, after stage 1 has finished) = FF, seeds 0 1, the
# reference (reused) plus AWLADDER_FF_RUNGS, the stage-1 candidates, e.g.
#   AWLADDER_STAGE=2 AWLADDER_FF_RUNGS="m2 m5" setsid nohup ...
# Lung is dropped. Each launch merges its scope into $ROOT/SCOPE.json, which
# the read-out follows.
# Per run: fit -> degeneracy (I(niche;w) guard) -> validate morans,niche ->
# w_deviation (the pre-registered deviation read). A dead fit is recorded in
# DEAD_RUNS.txt and read, never refitted (failed legs are findings).
# Order: GSE, lung, ovarian, then FF; within a dataset seed-major with the
# rungs interleaved (ref, m1, m2, m5, m10, m25 per seed), so a partial table
# has every rung early. Two workers pull jobs from that one list; the partial
# table is refreshed after every job; the read-out at the end writes
# <ovarian>/experiments/awladder.{json,md} and, once complete, DECISION_AW.json.
#
# Picker / lock / marker machinery from scripts/queue_2026-09-24_r12_arms.sh:
# gate 9500 MiB (FF steps 18500), one job of ours per GPU, MintFlow charged at
# its peak, a re-read after taking the lock and a settle delay; courtesies to
# the other queues (their locks and waiters live in their own folders): a
# card with >= 18.5 GB free is left to their waiting FF step, and a card one
# of them locked less than FRESH s ago is skipped. Markers are keyed by step
# AND dataset AND run; a relaunch resumes. Launch detached:
#   setsid nohup scripts/queue_2026-09-24_awladder.sh \
#     >> scripts/logs/awladder_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GS=gse315411_pdltma06_11_prime_solo
LU=xenium_prime_human_lung_cancer_ffpe
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff

ROOT=scripts/logs/awladder_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; FAILED=$ROOT/failed
CLAIM=$ROOT/claims; WAIT=$ROOT/waiting
LADDER=$ROOT/LADDER.json
mkdir -p $QL $DONE $LOCK $FAILED $CLAIM $WAIT
GPUS="0 1"
NEED_OV=9500          # every non-FF dataset
NEED_FF=18500
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=90            # coordinator 2026-09-24 ~15:05: 300 starved the ladder
WORKERS=2
RUNGS="ref0.1 m1 m2 m5 m10 m25"
STAGE=${AWLADDER_STAGE:-1}
SEEDS="0 1"
case $STAGE in
  1) SCOPE_DS="$GS $OV"; FF_RUNGS="" ;;
  2) SCOPE_DS="$FF"; FF_RUNGS="${AWLADDER_FF_RUNGS:-}"
     [ -n "$FF_RUNGS" ] || { echo "stage 2 needs AWLADDER_FF_RUNGS"; exit 1; } ;;
  *) echo "unknown AWLADDER_STAGE=$STAGE"; exit 1 ;;
esac
# FF rungs as given ("5 10" or "m5 m10"), as rung names, each checked
FF_RUNGS=$(for x in $FF_RUNGS; do echo -n "m${x#m} "; done)
for x in $FF_RUNGS; do
  case " $RUNGS " in *" $x "*) ;; *) echo "unknown FF rung $x"; exit 1 ;; esac
done
rungs_of() { [ "$1" = "$FF" ] && echo "ref0.1 $FF_RUNGS" || echo "$RUNGS"; }
# the transport reads (coordinator, 2026-09-24 evening): the reference and the
# stage-2 rungs, on every dataset; the ovarian reused reference is read under
# its alias ref0.1wfix (its existing transport files count)
TRANSPORT_RUNGS="ref0.1 ref0.1wfix m5 m10"
COMMON="--kappa 0.1 --d-w 6 --gat-sources type_only --w-warmup-epochs 30
        --epochs 200 --patience 20 --figures-every 200"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$LU]="--alpha-z 0.002 --label-key graphclust"
  [$OV]="--alpha-z 0.0035"
  [$FF]="--alpha-z 0.00035 --label-key graphclust")

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
code_hash() { sha256sum discell/model/train.py discell/model/networks.py \
              discell/model/equations.py discell/model/elbo.py \
              | awk '{printf "%s ", substr($1, 1, 12)}'; }
alpha_w() {  # alpha_w <dataset> <rung>: the ladder's value, fixed once
  python3 -c "import json,sys; print(json.load(open('$LADDER'))['datasets'][sys.argv[1]]['alpha_w'][sys.argv[2]])" $1 ${2/ref0.1wfix/ref0.1}
}
link_of() {  # link_of <dataset> <run>: the run it is read through, if any
  python3 -c "import json,sys; print(json.load(open('$LADDER'))['datasets'][sys.argv[1]]['links'].get(sys.argv[2], ''))" $1 $2
}

# -- GPU picker ---------------------------------------------------------------
GUARD_PATTERN='run_mintflow.py|baseline_battery.*MintFlow'
GUARD_PEAK=9000
# other queues: <log root>:<process pattern>
FOREIGN="scripts/logs/final_2026-09-24:queue_2026-09-24_final.sh
scripts/logs/r12_arms_2026-09-24:queue_2026-09-24_r12_arms.sh
scripts/logs/wcollapse_b_2026-09-24:queue_2026-09-24_wcollapse_b.sh
scripts/logs/uncontrolled500_2026-09-24:queue_2026-09-24_uncontrolled500.sh"

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
  local g=$1 free=$2 root pat now age
  now=$(date +%s)
  while IFS=: read -r root pat; do
    [ -n "$root" ] || continue
    pgrep -f "$pat" > /dev/null || continue           # that queue is gone
    if [ "$free" -ge "$NEED_FF" ] && ls "$root"/waiting/ff.* > /dev/null 2>&1; then
      return 0                                        # its FF step goes first
    fi
    if [ -d "$root/locks/gpu$g" ]; then
      age=$((now - $(stat -c %Y "$root/locks/gpu$g")))
      [ $age -lt $FRESH ] && return 0                 # may not have allocated yet
    fi
  done <<< "$FOREIGN"
  return 1
}

# Block until a GPU has >= $1 MiB free, no job of ours and no foreign first
# call, take it and echo its index. Role "ff" announces itself in $WAIT while
# waiting (for the record; other queues do not read it). Everything else goes
# to stderr.
acquire_gpu() {
  local need=$1 role=$2 g free waited=0 me="$WAIT/ff.$BASHPID"
  [ "$role" = ff ] && touch "$me"
  while true; do
    for g in $GPUS; do
      free=$(gpu_free $g)
      [ "$free" -ge "$need" ] || continue
      foreign_hold $g $free && continue
      mkdir "$LOCK/gpu$g" 2>/dev/null || continue     # one of ours has it
      sleep $SETTLE      # a neighbour's job may have launched, not allocated
      free=$(gpu_free $g)
      if [ "$free" -ge "$need" ] && ! foreign_hold $g $free; then
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

# -- steps --------------------------------------------------------------------
step() {  # step <need> <role> <marker> <cmd...>: a GPU step, retried on OOM
  local need=$1 role=$2 name=$3; shift 3
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local try=0 other=0 rc gpu t0
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
      log "FAIL $name (exit $rc; attempt $other/$TRIES_OTHER); see $QL/$name.try$try.log"
      [ $other -ge $TRIES_OTHER ] && break
    fi
    sleep 300
  done
  log "GIVING UP on $name"; touch "$FAILED/$name"; return 1
}

reuse() {  # reuse <step> <dataset> <run> <file>: an existing output counts
  local name; name=$(marker $1 $2 $3)
  [ -f "$DONE/$name" ] && return 0
  if [ -f "$(runs $2)/$3/$4" ]; then touch "$DONE/$name"; log "reuse $name ($4 present)"; fi
}

table() {  # the partial (or final) read-out; cheap, CPU, never fatal
  CUDA_VISIBLE_DEVICES= uv run python scripts/awladder_table.py --at best \
    > $QL/table_${1}.log 2>&1 || log "read-out ($1) failed; see $QL/table_${1}.log"
}

job() {  # job <dataset> <rung> <seed>: fit and every read of one run
  local ds=$1 rung=$2 seed=$3 run need role aw
  run=aw_${rung}_s$seed; need=$(need_of $ds); role=$(role_of $ds)
  aw=$(alpha_w $ds $rung)
  if [ -f "$(runs $ds)/$run/metrics.json" ]; then
    touch "$DONE/$(marker fit $ds $run)"
  else
    step $need $role "$(marker fit $ds $run)" uv run python -m discell.model.train \
      --dataset $ds --run-name $run ${FLAGS[$ds]} $COMMON --alpha-w $aw --seed $seed \
      || { log "$ds/$run: no fit -- no reads"; return 0; }
  fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$(runs $ds)/$run/metrics.json')).get('dead_w_channel') else 1)"; then
    log "$ds/$run: dead context channel flagged by the trainer -- recorded, reads continue"
    grep -qx "$ds/$run" $ROOT/DEAD_RUNS.txt 2>/dev/null || echo "$ds/$run" >> $ROOT/DEAD_RUNS.txt
  fi
  reuse degeneracy $ds $run degeneracy.json
  step $need $role "$(marker degeneracy $ds $run)" uv run python -m \
    discell.model.degeneracy --dataset $ds --run $run
  reuse validate $ds $run validation/validation.json
  step $need $role "$(marker validate $ds $run)" uv run python -m \
    discell.model.validate --dataset $ds --run $run --analyses morans,niche
  reuse w_deviation $ds $run w_deviation.json
  step $need $role "$(marker w_deviation $ds $run)" uv run python -m \
    discell.experiments.w_deviation --dataset $ds --run $run
  case " $TRANSPORT_RUNGS " in
    *" $rung "*)
      reuse_all transport $ds $run transport/transport.json \
        transport/transport_distribution.json transport/transport_twins.json
      step $need $role "$(marker transport $ds $run)" uv run python -m \
        discell.model.transport --dataset $ds --run $run --read both --hvg 1000
      if [ "$ds" = "$OV" ]; then   # only the ovarian slide is tumour-annotated
        reuse_all transport_band $ds $run transport/transport_tumour-band.json \
          transport/transport_tumour-band_distribution.json \
          transport/transport_tumour-band_twins.json
        step $need $role "$(marker transport_band $ds $run)" uv run python -m \
          discell.model.transport --dataset $ds --run $run --read both \
          --hvg 1000 --niche-source tumour-band
      fi ;;
  esac
  table partial
  return 0
}

reuse_all() {  # reuse_all <step> <dataset> <run> <file>...: all present -> done
  local name f; name=$(marker $1 $2 $3)
  [ -f "$DONE/$name" ] && return 0
  for f in "${@:4}"; do [ -f "$(runs $2)/$3/$f" ] || return 0; done
  touch "$DONE/$name"; log "reuse $name (outputs present)"
}

joblist() {  # every (dataset, rung, seed) to run, in queue order; links skipped
  if [ "$STAGE" = 2 ]; then   # FF jobs, each followed by two transport-only jobs
    paste -d '\n' <(joblist_of "$FF") \
      <(transport_jobs | awk 'NR % 2 == 1') <(transport_jobs | awk 'NR % 2 == 0') \
      | grep -v '^$'
    return
  fi
  joblist_of "$SCOPE_DS"
}

transport_jobs() {  # GSE and ovarian: the transport rungs whose fits are done
  local ds s r
  for ds in $GS $OV; do
    for s in $SEEDS; do
      for r in $TRANSPORT_RUNGS; do
        [ -f "$(runs $ds)/aw_${r}_s$s/metrics.json" ] && echo "$ds $r $s"
      done
    done
  done
}

joblist_of() {
  local ds s r
  for ds in $1; do
    for s in $SEEDS; do
      for r in $(rungs_of $ds); do
        case "$(link_of $ds aw_${r}_s$s)" in
          ''|wfix_*) echo "$ds $r $s" ;;   # a reused reference still gets its reads
          *) ;;                            # read through another rung's run
        esac
      done
    done
  done
}

worker() {  # worker <k> [noff]: claim the next unclaimed job, run it, repeat
  local ds r s
  # the list on fd 3: nothing a job runs can swallow the remaining lines
  while read -r ds r s <&3; do
    # "noff": never an FF job, so the 9.5 GB reads are not stuck behind FF
    # steps waiting for an 18.5 GB card
    [ "${2:-}" = noff ] && [ "$ds" = "$FF" ] && continue
    mkdir "$CLAIM/${ds}_aw_${r}_s$s" 2>/dev/null || continue
    log "worker $1: $ds aw_${r}_s$s"
    job $ds $r $s < /dev/null
  done 3< $ROOT/JOBS.txt
  log "worker $1 finished"
}

# -- preflight -------------------------------------------------------------------
# (1) every rung's flags reproduce runs/final_s0/config.json (the frozen final
#     configuration) but for the budget (200/20), figures, seed, name and
#     alpha_w, and alpha_w is the ladder's; (2) the reused references are this
#     exact configuration.
preflight() {
  local specs=() ds
  for ds in $SCOPE_DS; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
  uv run python - "$LADDER" "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from pathlib import Path
from discell.model.train import TrainConfig, build_parser
ladder = json.load(open(sys.argv[1]))
default = asdict(TrainConfig(dataset=""))
bad = 0
def diff(new, ref, allowed):
    return [f"{k}: {new.get(k)!r} vs {ref.get(k, '<absent>')!r}"
            for k in sorted(set(new) | set(ref)) if k not in allowed
            and ((k in ref and new.get(k) != ref[k])
                 or (k not in ref and new[k] != default[k]))]
for spec in sys.argv[2:]:
    ds, flags = spec.split("::", 1)
    runs = Path(f"data/datasets/{ds}/runs")
    entry = ladder["datasets"][ds]
    final = json.load(open(runs / "final_s0" / "config.json"))
    for rung, aw in entry["alpha_w"].items():
        for seed in (0, 1, 2):
            args = vars(build_parser().parse_args(
                ["--dataset", ds, "--seed", str(seed), "--run-name",
                 f"aw_{rung}_s{seed}", "--alpha-w", str(aw)] + shlex.split(flags)))
            args.pop("quiet")
            new = asdict(TrainConfig(**args))
            d = diff(new, final, {"run_name", "seed", "git", "epochs", "patience",
                                  "figures_every", "alpha_w"})
            ok = (not d and new["alpha_w"] == aw and new["epochs"] == 200
                  and new["patience"] == 20 and new["w_warmup_epochs"] == 30)
            link = entry["links"].get(f"aw_{rung}_s{seed}")
            if link and link.startswith("aw_"):   # another rung of this ladder
                target = link[len("aw_"):].rsplit("_s", 1)[0]
                ok = ok and entry["alpha_w"][target] == aw
                if entry["alpha_w"][target] != aw:
                    d.append(f"link {link} has alpha_w {entry['alpha_w'][target]}")
            elif link:                            # an existing fit, reused
                ref = json.load(open(runs / link / "config.json"))
                d2 = diff(new, ref, {"run_name", "git"})
                ok = ok and not d2
                d += [f"vs link {link}: " + x for x in d2]
            if not ok or seed == 0:
                print(f"{ds[:24]} aw_{rung}_s{seed}: alpha_w {aw}"
                      + (f" (read through {link})" if link else "")
                      + (": identical to final_s0 but for budget/alpha_w"
                         if ok else ": DIFFERS " + "; ".join(d)))
            bad += not ok
sys.exit(1 if bad else 0)
EOF
}

write_scope() {  # merge this stage's scope into SCOPE.json (the read-out's)
  local ds args=()
  for ds in $SCOPE_DS; do args+=("$ds" "$(rungs_of $ds)"); done
  python3 - "$ROOT/SCOPE.json" "$SEEDS" "${args[@]}" <<'EOF'
import json, os, sys
path, seeds, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
scope = json.load(open(path)) if os.path.exists(path) else {"datasets": {}}
scope["seeds"] = [int(x) for x in seeds.split()]
for ds, rungs in zip(pairs[::2], pairs[1::2]):
    scope["datasets"][ds] = rungs.split()
json.dump(scope, open(path, "w"), indent=1)
print("scope:", json.dumps(scope))
EOF
}

links() {  # the reused runs in scope, as relative symlinks inside runs/
  python3 - "$LADDER" "$ROOT/SCOPE.json" <<'EOF'
import json, os, sys
scope = json.load(open(sys.argv[2]))
in_scope = {(ds, f"aw_{r}_s{s}") for ds, rungs in scope["datasets"].items()
            for r in rungs for s in scope["seeds"]}
for ds, e in json.load(open(sys.argv[1]))["datasets"].items():
    for name, target in e["links"].items():
        if (ds, name) not in in_scope and not target.startswith("wfix_"):
            continue                 # the reused fits' aliases are always kept
        path = f"data/datasets/{ds}/runs/{name}"
        if os.path.islink(path):
            assert os.readlink(path) == target, (path, os.readlink(path), target)
            continue
        assert not os.path.exists(path), f"{path} exists and is not a link"
        os.symlink(target, path)
        print(f"linked {path} -> {target}")
EOF
}

# -- main -------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -rf $CLAIM/* $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start, stage $STAGE (gates: FF $NEED_FF MiB, others $NEED_OV MiB; GPUs: $GPUS; code $(code_hash))"
[ -f "$LADDER" ] || { log "no $LADDER -- run scripts/awladder_table.py --init-ladder"; exit 1; }
if ! preflight > $QL/preflight.log 2>&1; then
  cat $QL/preflight.log; log "PREFLIGHT FAILED -- stopping"; exit 1
fi
cat $QL/preflight.log
write_scope || { log "SCOPE FAILED -- stopping"; exit 1; }
links || { log "LINKS FAILED -- stopping"; exit 1; }
joblist > $ROOT/JOBS.txt
log "stage $STAGE: $(wc -l < $ROOT/JOBS.txt) jobs ($SCOPE_DS; seeds $SEEDS; seed-major, rungs interleaved)"
if [ "${AWLADDER_DRYRUN:-0}" = 1 ]; then log "dry run: preflight, links and job list only"; exit 0; fi

PIDS=""
for k in $(seq 1 $WORKERS); do
  worker $k & PIDS="$PIDS $!"
  sleep 20
done
if [ "$STAGE" = 2 ]; then worker $((WORKERS + 1)) noff & PIDS="$PIDS $!"; fi
wait $PIDS
log "all workers finished"

table final
log "read-out done; see $QL/table_final.log"
{
  echo
  echo "## Queue (stage $STAGE) finished $(date '+%F %T')"
  echo
  echo "Read-out: data/datasets/$OV/experiments/awladder.{json,md}"
  echo "Decision: $( [ -f $ROOT/DECISION_AW.json ] && cat $ROOT/DECISION_AW.json || echo 'withheld (grid incomplete)')"
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Dead fits (trainer flag): $(cat $ROOT/DEAD_RUNS.txt 2>/dev/null | tr '\n' ' ')"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
