#!/bin/bash
# MintFlow refits with a corrected export (devlog 2026-10-01, "MintFlow refits
# with a corrected export"; issue B-mf1). The original lineage fits exported
# the encoder's split Xint + Xmic, which sums back to the cell's own counts, as
# the "decoded rate", and saved no weights. The corrected runner
# DisCell-baselines/mintflow/run_mintflow_decoded.py is run_mintflow.py with
# two changes only: decoded_heldout.npz holds MintFlow's decoded rate at the
# posterior mean (decode_mintflow.decoded_rate) and the fitted model is dumped
# to <out>/model.pt. Same MintFlow (0.3.0, same venv), same settings (50
# epochs, width_window 600, MintFlow's own squidpy Delaunay graph, its default
# batching), same lineage exports (DisCell-baselines/data/lineage/), same
# >= 5-count filter. Preflight: scripts/logs/mintflow_refit_2026-10-01/
# AGENT_REPORT.md and preflight/.
#
# Order, strictly sequential (never two MintFlow fits at once, so never two on
# one card):
#   (1) TMA core, 50 epochs, --max-hours 24, predicted on the serial section
#       as well (the transfer)                                     est. ~9.5 h
#   (2) lung FFPE whole section, --max-hours 80                    est. ~15 h
#   (3) ovarian FFPE whole section, --max-hours 80                 est. ~16 h
# GPU: GPU 1 when it has >= NEED_MIB free (twice, 60 s apart), no MintFlow fit
# on it and no other queue's fresh lock; else GPU 0 on the same terms; else
# wait. Host: a fit starts only with >= HOST_NEED_GB available (free -g), as
# the 2026-09-29 queue gated MintFlow FF (here ~2x the largest measured peak,
# 23.6 GB on ovarian). Our own lock scripts/logs/mintflow_refit_2026-10-01/
# locks/gpu<g> is kept fresh while a fit runs so other queues' pickers keep
# off the card.
#
# Results: DisCell-baselines/results/mintflow_refit_lineage/<dataset>/
# (latents.h5ad, decoded_heldout.npz, config.json, model.pt,
# input_filtered.h5ad). After each successful fit (CPU, 2 threads, on the
# baseline tables' lock), as the 2026-09-28/29 queues scored MintFlow:
#   battery column  baseline_battery --config-run finalL_s0 --tag _lineage
#                   (same column name as before, so it replaces the old one)
#   probe regrade   probe_regrade --force vs uncontrolledL_s{0,1}, tag _lineage
#   probe table     the section's lineage_final probe table re-rendered
#   context grade   context_grade grade --only MintFlow
#   timing row      $ROOT/timing.tsv (setup, train, wall, s/epoch, peaks)
#   feasibility row DisCell-baselines/results/feasibility.tsv; the reason
#                   starts "refit (corrected export)" (scripts/paper_tables.py
#                   reads it; a refit that did not finish leaves the old row)
# The serial section is scored with --config-from the core, as before. Before
# anything is scored, every table the scoring rewrites is copied once into
# $ROOT/backup/<dataset>/ (never clobbered): the old values the READOUT
# compares against. Once a section's refit is scored, its original folder
# DisCell-baselines/results/mintflow/<dataset>_lineage is MOVED (not deleted)
# to results/_archive_mintflow_export_bug/mintflow/.
#
# At the end: $ROOT/READOUT.md (old vs new, every MintFlow metric, material
# moves flagged; $ROOT/readout.py) and a preview of the paper tables rendered
# into $ROOT/tables_preview/ (the generator's default output under
# submission_paper/ is NOT written by this queue).
#
# Idempotent: a fit whose outcome is recorded ($ROOT/done/fit_<dataset>) and
# every scoring step with a marker in $ROOT/done/ is skipped on a relaunch;
# delete a marker to redo that step.
#
#   mkdir -p scripts/logs/mintflow_refit_2026-10-01 && setsid nohup \
#     bash scripts/queue_2026-10-01_mintflow_refit.sh \
#     >> scripts/logs/mintflow_refit_2026-10-01/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
LU=xenium_prime_human_lung_cancer_ffpe
OV=xenium_prime_ovarian_cancer_ffpe

ROOT=scripts/logs/mintflow_refit_2026-10-01
QL=$ROOT/logs; DONE=$ROOT/done; ATT=$ROOT/attempts; BACKUP=$ROOT/backup; LOCK=$ROOT/locks
mkdir -p $QL $DONE $ATT $BACKUP $LOCK
GPUS="1 0"                # preference order
NEED_MIB=20480            # >= 20 GB free on the card before a fit (as 2026-09-29)
HOST_NEED_GB=48           # host memory available before a fit
FRESH=600                 # another queue's lock younger than this holds its card
OTHER_MIB=2048            # another process's memory that makes an OOM "theirs"
TLOCK=scripts/logs/baselines_lineage_2026-09-28/tables.lock
PY2="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"
TAG="refit (corrected export)"   # = scripts/paper_tables.py MF_REFIT_TAG

BASE=/home/rmolen/github/DisCell-baselines
BDATA=$BASE/data/lineage
BRES=$BASE/results
NEW=$BRES/mintflow_refit_lineage
ARCH=$BRES/_archive_mintflow_export_bug/mintflow
FEAS=$BRES/feasibility.tsv
TIMING=$ROOT/timing.tsv

log() { echo "[$(date '+%F %T')] $*"; }

# -- one instance ---------------------------------------------------------------
exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }
rmdir $LOCK/gpu* 2>/dev/null     # one instance: any lock of ours is stale

# -- helpers ----------------------------------------------------------------------
cpu_step() {  # cpu_step <marker> <cmd...>: a CPU step with a done marker
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc
  log "CPU  start $name"
  "$@" > "$QL/$name.log" 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; return 1
}

gpu_free() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
    -i $1 | awk -F', ' '{print $1-$2}'
}

mintflow_on() {  # mintflow_on <gpu>: a MintFlow fit runs (or is pinned) on the card
  local g=$1 pid on
  on=" $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $g | tr '\n' ' ') "
  for pid in $(pgrep -f "run_mintflow"); do
    case "$on" in *" $pid "*) return 0 ;; esac
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" && return 0
  done
  return 1
}

held_elsewhere() {  # another queue's fresh lock on the card
  local g=$1 dir now
  now=$(date +%s)
  for dir in scripts/logs/*/locks/gpu$g; do
    [ -d "$dir" ] || continue
    case "$dir" in $ROOT/*) continue ;; esac
    [ $((now - $(stat -c %Y "$dir"))) -lt $FRESH ] && return 0
  done
  return 1
}

card_ok() {
  local g=$1
  [ "$(gpu_free $g)" -ge $NEED_MIB ] && ! mintflow_on $g && ! held_elsewhere $g
}

host_avail_gb() { free -g | awk '/^Mem:/{print $7}'; }

acquire_gpu() {  # prints the card: GPU 1 first, GPU 0 if 1 is busy; waits otherwise
  local n=0 g
  while true; do
    if [ "$(host_avail_gb)" -ge $HOST_NEED_GB ]; then
      for g in $GPUS; do
        card_ok $g || continue
        mkdir "$LOCK/gpu$g" 2>/dev/null || continue
        sleep 60
        if card_ok $g; then echo $g; return 0; fi
        rmdir "$LOCK/gpu$g" 2>/dev/null
      done
    fi
    [ $((n % 10)) -eq 0 ] && log "waiting: host $(host_avail_gb) GB available (need $HOST_NEED_GB); free MiB GPU1 $(gpu_free 1), GPU0 $(gpu_free 0) (need $NEED_MIB, no MintFlow on the card)" >&2
    n=$((n + 1)); sleep 60
  done
}

tree_pids() { local c; echo $1; for c in $(pgrep -P $1); do tree_pids $c; done; }

sampler() {  # sampler <root pid> <file> <gpu>: every 15 s "t own_gpu_mib used_gpu_mib rss_mib avail_mib"
  local root=$1 f=$2 g=$3 pids own tot rss avail pid mem
  while kill -0 $root 2>/dev/null; do
    pids=" $(tree_pids $root | tr '\n' ' ') "
    own=0
    while IFS=', ' read -r pid mem; do
      [ -n "$pid" ] || continue
      case "$pids" in *" $pid "*) own=$((own + mem)) ;; esac
    done < <(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader,nounits -i $g 2>/dev/null)
    tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    rss=$(ps -o rss= -p "$(echo $pids | tr ' ' ',')" 2>/dev/null \
          | awk '{s += $1} END {print int(s / 1024)}')
    avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    echo "$(date +%s) $own $tot $rss $avail" >> "$f"
    touch "$LOCK/gpu$g" 2>/dev/null       # keep our lock fresh for other pickers
    sleep 15
  done
}

feas_row() {  # feas_row <tool> <dataset> <cells> <outcome> <wall_h> <gpu_gb> <host_gb> <reason>
  [ -f $FEAS ] || printf 'tool\tdataset\tcells\toutcome\twall_h\tpeak_gpu_gb\tpeak_host_gb\treason\n' > $FEAS
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> $FEAS
  log "feasibility: $*"
}

timing_row() {  # timing_row <dataset> <out dir> <wall s> <gpu>
  [ -f $TIMING ] || printf 'dataset\tfit\tgpu\tepochs_completed\tsetup_s\ttrain_s\ts_per_epoch\twall_s\tpeak_gpu_gb\tpeak_host_gb\tn_cells\n' > $TIMING
  python3 - "$1" "$2" "$3" "$4" >> $TIMING <<'PYEOF'
import json, sys
ds, out, wall, gpu = sys.argv[1:]
c = json.load(open(out + "/config.json"))
e = c["epochs_completed"]
print("\t".join(str(x) for x in (ds, out, gpu, e, c["setup_s"], c["train_s"],
      round(c["train_s"] / max(e, 1), 1), wall, c["peak_gpu_gb"], c["peak_host_gb"], c["n_cells"])))
PYEOF
}

backup() {  # backup <dataset>: every table the scoring rewrites, once (never clobber)
  local e=data/datasets/$1/experiments f
  mkdir -p $BACKUP/$1
  for f in baseline_battery_lineage.json baseline_battery_lineage.md \
           probe_regrade_lineage_final.json probe_regrade_lineage_final.md \
           context_grade.json context_grade.md; do
    [ -f $e/$f ] && cp -n $e/$f $BACKUP/$1/
  done
  [ -d $e/probe_regrade_lineage ] && [ ! -d $BACKUP/$1/probe_regrade_lineage ] \
    && cp -r $e/probe_regrade_lineage $BACKUP/$1/
  return 0
}

# -- one attempt ------------------------------------------------------------------
# attempt <name> <dataset> <out dir> <cmd...>: a fit on the acquired card with the
# sampler; classifies the outcome, writes $ATT/<name>.json, the feasibility and
# timing rows and the outcome marker. Returns 0 when latents were written.
attempt() {
  local name=$1 ds=$2 out=$3; shift 3
  if [ -f "$DONE/fit_$name" ]; then
    log "skip fit_$name (outcome recorded: $(cat $DONE/fit_$name))"
    grep -qxE "ok|truncated" "$DONE/fit_$name" && [ -f "$out/latents.h5ad" ]; return
  fi
  local try=0 rc t0 t_start wall pid spid gpu
  while true; do
    try=$((try + 1))
    gpu=$(acquire_gpu)
    local slog="$QL/fit_$name.try$try.log" samples="$ATT/$name.try$try.samples"
    : > "$samples"
    log "GPU$gpu start fit_$name (try $try; host $(host_avail_gb) GB available): $*"
    t0=$SECONDS; t_start=$(date +%s)
    ( export CUDA_VISIBLE_DEVICES=$gpu; "$@" ) > "$slog" 2>&1 &   # "$@" may be a function
    pid=$!
    sampler $pid "$samples" $gpu &
    spid=$!
    wait $pid; rc=$?
    wall=$((SECONDS - t0))
    kill $spid 2>/dev/null; wait $spid 2>/dev/null
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  fit_$name (exit $rc, ${wall}s)"
    journalctl -k --since "@$t_start" --no-pager 2>/dev/null \
      | grep -iE "out of memory|oom-kill|killed process" | tail -5 > "$ATT/$name.try$try.kernel" || true
    # classify (the 2026-09-29 queue's rules); prints
    # outcome, cells, gpu_gb, host_gb, other_mib, reason (separated by \x1f)
    local res
    res=$(CUDA_VISIBLE_DEVICES= python3 - "$ds" "$out" "$slog" "$samples" "$rc" "$wall" \
          "$ATT/$name.json" "$try" "$t_start" "$ATT/$name.try$try.kernel" "$gpu" <<'PYEOF'
import json, os, re, sys
ds, out, slog, samples, rc, wall, dst, tryn, t_start, kern, gpu = sys.argv[1:]
kernel = open(kern).read().strip() if os.path.exists(kern) else ""
rc, wall, t_start = int(rc), int(wall), int(t_start)
text = open(slog, errors="replace").read()
rows = [list(map(int, l.split())) for l in open(samples) if len(l.split()) == 5]
MIB = 1.048576e6 / 1e9
own_gpu = max((r[1] for r in rows), default=0) * MIB
own_rss = max((r[3] for r in rows), default=0) * MIB
avail_min = min((r[4] for r in rows), default=-1) * MIB
t_end = rows[-1][0] if rows else 0
other_mib = max((r[2] - r[1] for r in rows if r[0] >= t_end - 60), default=0)
m = re.findall(r"-> \((\d+), \d+\)", text)
cells = m[0] if m else ""
cfgp, lat = os.path.join(out, "config.json"), os.path.join(out, "latents.h5ad")
rec = dict(tool="MintFlow", dataset=ds, attempt=int(tryn), exit=rc, wall_s=wall,
           started=t_start, gpu=int(gpu), log=slog, samples=samples,
           sampled_peak_gpu_gb_nvidia_smi=round(own_gpu, 2),
           sampled_peak_host_rss_gb=round(own_rss, 2),
           min_host_available_gb=round(avail_min, 2),
           other_process_gpu_mib_last_minute=other_mib)
gpu_gb = host = ""
if rc == 0 and os.path.exists(lat) and os.path.exists(cfgp) \
        and os.path.getmtime(lat) >= t_start:
    cfg = json.load(open(cfgp))
    cells = cfg.get("n_cells", cells)
    gpu_gb, host = cfg.get("peak_gpu_gb", ""), cfg.get("peak_host_gb", "")
    if not cfg.get("decoded_rate") or not os.path.exists(os.path.join(out, "model.pt")):
        outcome = "failed"
        reason = "latents written without the corrected decode or the model dump; not scored"
    elif cfg.get("truncated"):
        outcome = "truncated"
        reason = ("wall-clock cap %s h reached: %s/%s epochs completed; latents written; "
                  "setup_s=%s train_s=%s" % (cfg.get("max_hours"), cfg.get("epochs_completed"),
                  cfg.get("num_training_epochs"), cfg.get("setup_s"), cfg.get("train_s")))
    else:
        outcome = "ok"
        reason = "whole section; train_s=%s; epochs %s/%s" % (
            cfg.get("train_s"), cfg.get("epochs_completed"), cfg.get("num_training_epochs"))
    rec["config"] = cfg
else:
    gpu_gb, host = round(own_gpu, 2), round(own_rss, 2)
    oom = re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|"
                    r"CUBLAS_STATUS_ALLOC_FAILED", text)
    tried = re.search(r"Tried to allocate ([\d.]+ [KMG]iB)", text)
    alloc = re.search(r"([\d.]+ [KMG]iB) already allocated|this process has ([\d.]+ [KMG]iB) memory in use", text)
    if oom:
        outcome = "failed"
        reason = "CUDA OOM on a 24 GB RTX 4090 (GPU %s)" % gpu
        if tried:
            reason += ": tried to allocate %s" % tried.group(1)
        if alloc:
            reason += " with %s already in use by the fit" % (alloc.group(1) or alloc.group(2))
        reason += "; sampled peak %.1f GB (nvidia-smi)" % own_gpu
        rec["error_class"] = "CUDA OOM"
    elif rc in (137, -9):
        outcome = "failed"
        host_oom = bool(re.search(r"out of memory|oom-kill", kernel, re.I))
        reason = ("%s (exit %d): peak host RSS %.1f GB (sampled), %.1f GB host memory "
                  "available at the lowest sample, 125 GB host"
                  % ("host OOM, killed by the kernel OOM killer" if host_oom
                     else "killed by SIGKILL, no kernel OOM record", rc, own_rss, avail_min))
        rec["error_class"] = "host OOM (SIGKILL)" if host_oom else "SIGKILL"
        rec["kernel_log"] = kernel
    else:
        outcome = "failed"
        errs = re.findall(r"^(\w[\w.]*(?:Error|Exception)\b.*)$", text, re.M)
        reason = "exit %d: %s" % (rc, errs[-1][:200] if errs else "no exception in the log")
        rec["error_class"] = errs[-1].split(":")[0] if errs else "exit %d" % rc
rec.update(outcome=outcome, cells=cells, reason=reason)
json.dump(rec, open(dst, "w"), indent=2, default=str)
print("\x1f".join(str(x).replace("\t", " ") for x in      # \x1f: empty fields survive read
                (outcome, cells, gpu_gb, host, other_mib, reason.replace("\n", " "))))
PYEOF
)
    local outcome cells ggb hgb other reason wall_h
    IFS=$'\x1f' read -r outcome cells ggb hgb other reason <<< "$res"
    [ -n "$outcome" ] || { outcome=failed; reason="classification failed (exit $rc); see $slog"; }
    wall_h=$(awk -v s=$wall 'BEGIN{printf "%.2f", s/3600}')
    if [ "$outcome" = failed ] && [ $try -eq 1 ] && grep -qE "CUDA out of memory|OutOfMemoryError" "$slog" \
       && [ "${other:-0}" -ge $OTHER_MIB ]; then
      log "fit_$name: CUDA OOM with another process holding ${other} MiB on GPU $gpu -- retrying once"
      feas_row MintFlow "$ds" "$cells" failed "$wall_h" "$ggb" "$hgb" \
        "$TAG: $reason; another process held ${other} MiB on the card -- retried"
      sleep 300; continue
    fi
    feas_row MintFlow "$ds" "$cells" "$outcome" "$wall_h" "$ggb" "$hgb" "$TAG: $reason"
    case "$outcome" in ok|truncated) timing_row "$ds" "$out" "$wall" "$gpu" ;; esac
    echo "$outcome" > "$DONE/fit_$name"
    [ "$outcome" = ok ] || [ "$outcome" = truncated ]
    return
  done
}

# -- scoring ----------------------------------------------------------------------
mintflow_name() {  # mintflow_name <dir> <base name>: the column name, from config.json
  python3 - "$1/config.json" "$2" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
print("MintFlow (lineage, truncated %s/%s epochs)" % (c["epochs_completed"], c["num_training_epochs"])
      if c.get("truncated") and sys.argv[2] == "MintFlow (lineage)" else sys.argv[2])
PYEOF
}

score() {  # score <dataset> <config dataset> <method> <latents dir>
  local ds=$1 from=$2 method=$3 dir=$4 extra=() ok=0
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  if [ "$method" != "MintFlow (lineage)" ] && [ "$method" != "MintFlow (lineage, transfer)" ] \
     && [ -f data/datasets/$ds/experiments/probe_regrade_lineage/MintFlow_lineage.json ]; then
    # a truncated refit gets its own column name: the old probe record (backed
    # up) leaves, so the context grade sees one MintFlow record
    mv data/datasets/$ds/experiments/probe_regrade_lineage/MintFlow_lineage.json \
      $BACKUP/$ds/MintFlow_lineage.json.moved_by_refit
  fi
  cpu_step "battery_$ds" flock $TLOCK $PY2 -m discell.experiments.baseline_battery \
    --dataset $ds --method "$method" --latents "$dir/latents.h5ad" \
    --config-run finalL_s0 --tag _lineage "${extra[@]}" || ok=1
  # shellcheck disable=SC2086
  cpu_step "probe_$ds" flock $TLOCK $PY2 -m discell.experiments.probe_regrade \
    --dataset $ds --run finalL_s0 --baseline-latents "$dir/latents.h5ad" \
    --method "$method" --force $REFS "${extra[@]}" || ok=1
  # shellcheck disable=SC2086
  cpu_step "probe_table_$ds" flock $TLOCK $PY2 -m discell.experiments.probe_regrade \
    --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' \
    $REFS "${extra[@]}" || ok=1
  cpu_step "context_$ds" flock $TLOCK $PY2 -m discell.experiments.context_grade grade \
    --dataset $ds "${extra[@]}" --only MintFlow || ok=1
  return $ok
}

archive_old() {  # archive_old <dataset>: the export-bug fit moves to the archive
  local src=$BRES/mintflow/${1}_lineage
  if [ -d "$src" ] && [ ! -e "$ARCH/${1}_lineage" ]; then
    mkdir -p $ARCH && mv "$src" "$ARCH/" && log "archived $src -> $ARCH/${1}_lineage"
  fi
}

run_fit() {  # run_fit <dataset> <max hours> [transfer args]: the corrected runner
  local ds=$1 hours=$2; shift 2
  # the environment of the original fits: this queue's exports (OMP 8,
  # WANDB_MODE=disabled, PCI bus order), cwd mintflow/, the same flags
  bash -c "cd $BASE/mintflow && exec ./.venv/bin/python run_mintflow_decoded.py \
    --h5ad $BDATA/$ds.h5ad --out $NEW/$ds --epochs 50 --width-window 600 \
    --max-hours $hours $*"
}

refit_core() {
  if attempt "$GS" "$GS" "$NEW/$GS" run_fit $GS 24 \
       --transfer-h5ad $BDATA/$GD.h5ad --transfer-out $NEW/$GD; then
    score $GS $GS "MintFlow (lineage)" $NEW/$GS && archive_old $GS
    score $GD $GS "MintFlow (lineage, transfer)" $NEW/$GD && archive_old $GD
  fi
}

refit_whole() {  # refit_whole <dataset>
  local ds=$1 name
  if attempt "$ds" "$ds" "$NEW/$ds" run_fit $ds 80; then
    name=$(mintflow_name $NEW/$ds "MintFlow (lineage)")
    log "fit_$ds -> column '$name'"
    score $ds $ds "$name" $NEW/$ds && archive_old $ds
  fi
}

# -- main -------------------------------------------------------------------------
log "queue start (pid $$): GPUs $GPUS in that order, >= $NEED_MIB MiB free, host >= $HOST_NEED_GB GB"
for ds in $GS $GD $LU $OV; do
  [ -f $BDATA/$ds.h5ad ] || { log "no lineage export $BDATA/$ds.h5ad -- stopping"; exit 1; }
  backup $ds
done
[ -f data/datasets/$GS/experiments/baseline_context_finalL_s0.npz ] \
  && [ -f data/datasets/$GD/experiments/baseline_context_finalL_s0_from_$GS.npz ] \
  && [ -f data/datasets/$LU/experiments/baseline_context_finalL_s0.npz ] \
  && [ -f data/datasets/$OV/experiments/baseline_context_finalL_s0.npz ] \
  || { log "a cached finalL_s0 context is missing -- stopping"; exit 1; }
mkdir -p $NEW

refit_core
refit_whole $LU
refit_whole $OV

cpu_step readout env CUDA_VISIBLE_DEVICES= python3 $ROOT/readout.py
cpu_step tables_preview env CUDA_VISIBLE_DEVICES= uv run python scripts/paper_tables.py \
  --out $ROOT/tables_preview
log "queue finished ($(ls $DONE | wc -l) markers; $(grep -c . $DONE/fit_* | wc -l) fits recorded)"
