#!/bin/bash
# Completing the baseline table: SIMVI and MintFlow on WHOLE sections only
# (devlog 2026-09-28, "Completing the baseline table", its Amendment "no
# windows" and the binding Amendment 2 (author, 15:10): whole sections only,
# in order of likelihood, with time caps; no windows anywhere).
#
# Waits (poll every 10 min, pgrep -f) until none of these is running:
#   queue_2026-09-28_unassigned.sh, queue_2026-09-28_sens_final.sh,
#   queue_2026-09-28_baselines_lineage.sh, queue_2026-09-28_cellina.sh
# then runs strictly in sequence on GPU 1 only (CUDA_VISIBLE_DEVICES=1 on
# every GPU step; >= 20 GB free on GPU 1 before each fit):
#   (1) MintFlow lung whole     50 epochs, width 600, --max-hours 80   est. ~46 h
#   (2) MintFlow ovarian whole  same                                   est. ~67 h
#   (3) SIMVI lung whole        attempt, `timeout 24h`                 est. OOM in minutes, else ~7-13 h
#   (4) SIMVI ovarian whole     same                                   est. OOM in minutes, else ~10-19 h
#   (5) SIMVI FF whole          same                                   est. OOM in minutes (dense 23 GB), else cap 24 h
#   (6) MintFlow FF whole       only if `free -g` shows >= 110 GB available
#                               at its start, else a "not attempted: host
#                               memory" record citing the 2026-09-22 failure  est. host OOM, else cap 80 h (~21 epochs)
# Estimates: MintFlow 816 s/epoch at 69k cells (GSE, 50 epochs 40,807 s),
# scaled by cells; SIMVI 6,438-11,585 s at 69k, scaled by cells (its epoch
# rule makes the total ~linear in n). --max-hours caps MintFlow's training
# loop only (checked after each whole epoch); setup and prediction add ~1 h.
#
# SIMVI and --max-cells: run_simvi.py's --max-cells DEFAULTS TO 200,000, and
# lung (278k), ovarian (407k) and FF (1.16M) all exceed it, so leaving the
# flag out would silently fit a 200k-cell window. The whole-section intent
# is honoured by passing a no-op cap (--max-cells $NO_CAP, larger than any
# slide) and checking `subsampled` is null in the fit's config.json; a fit
# that was windowed anyway is recorded as a failure and not scored.
#
# OOM: a CUDA OOM is retried once only when another process held >= 2 GiB on
# GPU 1 during the last minute before the failure (sampled every 15 s);
# otherwise the failure is recorded with the measured memory. Failures never
# stop the queue.
#
# After each successful fit (CPU, OMP_NUM_THREADS=2, CUDA_VISIBLE_DEVICES
# empty, serialised on the baseline tables' lock): the battery column
# (baseline_battery --config-run finalL_s0 --tag _lineage: lineage labels,
# the Unassigned evaluation mask, the label-free top-decile cycling set) and
# the probe grade (probe_regrade --force vs uncontrolledL_s{0,1}, tag
# _lineage). Columns: "MintFlow (lineage)" / "SIMVI (lineage)"; a MintFlow
# fit stopped by the cap is "MintFlow (lineage, truncated E/50 epochs)" and
# its config.json carries truncated / epochs_completed (the runner writes
# them).
#
# Every attempt (success, cap, failure, not attempted) appends one row to
# DisCell-baselines/results/feasibility.tsv:
#   tool dataset cells outcome wall_h peak_gpu_gb peak_host_gb reason
# outcome: ok | truncated | capped | failed | not_attempted. On success the
# peaks are the runner's (torch max allocated; ru_maxrss); on a failure they
# are sampled (nvidia-smi per-process memory on GPU 1, RSS of the process
# tree), and the reason carries the at-failure numbers from the OOM message.
# Per-attempt details: $ROOT/attempts/<step>.json (+ .samples).
#
# Tables: at start and end, archive_windows.py moves any window column out of
# baseline_battery_lineage*.{json,md} into baseline_battery_windows_archived
# (and its probe record into probe_regrade_windows_archived/); at the end the
# battery markdown and the lineage_final probe tables are re-rendered.
#
# Idempotent: a step whose outcome is recorded ($ROOT/done/fit_<tool>_<ds>,
# holding the outcome) is skipped on a relaunch; delete it to re-attempt.
# Scoring steps have their own dataset-qualified markers.
#
#   mkdir -p scripts/logs/baselines_complete_2026-09-28 && setsid nohup \
#     bash scripts/queue_2026-09-29_baselines_complete.sh \
#     >> scripts/logs/baselines_complete_2026-09-28/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

LU=xenium_prime_human_lung_cancer_ffpe
OV=xenium_prime_ovarian_cancer_ffpe
FF=xenium_prime_human_ovary_ff
GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual

ROOT=scripts/logs/baselines_complete_2026-09-28
QL=$ROOT/logs; DONE=$ROOT/done; ATT=$ROOT/attempts
mkdir -p $QL $DONE $ATT
GPU=1
NEED_MIB=20480            # >= 20 GB free on GPU 1 before each fit
HOST_NEED_GB=110          # MintFlow FF is attempted only with this much available
POLL=600                  # waiter poll, s
SIMVI_CAP=24h
MINTFLOW_HOURS=80
NO_CAP=100000000          # --max-cells above any slide: no window (see header)
OTHER_MIB=2048            # another process's GPU 1 memory that makes an OOM "theirs"
WAIT_FOR="unassigned sens_final baselines_lineage cellina"
TLOCK=scripts/logs/baselines_lineage_2026-09-28/tables.lock
PY="uv run python"
PY2="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"

BASE=/home/rmolen/github/DisCell-baselines
BDATA=$BASE/data/lineage
BRES=$BASE/results
FEAS=$BRES/feasibility.tsv

log() { echo "[$(date '+%F %T')] $*"; }

# -- one instance ---------------------------------------------------------------
exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

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

archive_windows() {  # archive_windows [--render]: window columns out of the lineage tables
  cpu_step "archive_windows${1:+_render}_$2" flock $TLOCK $PY2 \
    $ROOT/archive_windows.py ${1:-} $GS $GD $OV $LU $FF
}

queues_running() {  # the names of the queues we wait for that are still running
  local q out=""
  for q in $WAIT_FOR; do
    pgrep -f "queue_2026-09-28_${q}\.sh" > /dev/null && out="$out $q"
  done
  echo $out
}

gpu_free() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
    -i $GPU | awk -F', ' '{print $1-$2}'
}

wait_gpu() {  # >= NEED_MIB free on GPU 1, twice 60 s apart
  local n=0 f
  while true; do
    f=$(gpu_free)
    if [ "$f" -ge $NEED_MIB ]; then
      sleep 60; f=$(gpu_free)
      [ "$f" -ge $NEED_MIB ] && { log "GPU $GPU: $f MiB free"; return 0; }
    fi
    [ $((n % 10)) -eq 0 ] && log "waiting for $NEED_MIB MiB free on GPU $GPU ($f free)"
    n=$((n + 1)); sleep 60
  done
}

host_avail_gb() { free -g | awk '/^Mem:/{print $7}'; }

tree_pids() {  # tree_pids <pid>: the pid and all its descendants
  local c; echo $1
  for c in $(pgrep -P $1); do tree_pids $c; done
}

sampler() {  # sampler <root pid> <file>: every 15 s "t own_gpu_mib used_gpu_mib rss_mib avail_mib"
  local root=$1 f=$2 pids own tot rss avail pid mem
  while kill -0 $root 2>/dev/null; do
    pids=" $(tree_pids $root | tr '\n' ' ') "
    own=0
    while IFS=', ' read -r pid mem; do
      [ -n "$pid" ] || continue
      case "$pids" in *" $pid "*) own=$((own + mem)) ;; esac
    done < <(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    rss=$(ps -o rss= -p "$(echo $pids | tr ' ' ',')" 2>/dev/null \
          | awk '{s += $1} END {print int(s / 1024)}')
    avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    echo "$(date +%s) $own $tot $rss $avail" >> "$f"
    sleep 15
  done
}

feas_row() {  # feas_row <tool> <dataset> <cells> <outcome> <wall_h> <gpu_gb> <host_gb> <reason>
  [ -f $FEAS ] || printf 'tool\tdataset\tcells\toutcome\twall_h\tpeak_gpu_gb\tpeak_host_gb\treason\n' > $FEAS
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> $FEAS
  log "feasibility: $*"
}

# -- one attempt ------------------------------------------------------------------
# attempt <name> <tool> <dataset> <out dir> <cap s or 0> <cmd...>
# Runs cmd (already carrying cd / env / timeout) on GPU 1 with the sampler,
# classifies the outcome, writes $ATT/<name>.json, appends the feasibility row
# and the outcome marker. Returns 0 when latents were written.
attempt() {
  local name=$1 tool=$2 ds=$3 out=$4 cap=$5; shift 5
  if [ -f "$DONE/fit_$name" ]; then
    log "skip fit_$name (outcome recorded: $(cat $DONE/fit_$name))"
    grep -qxE "ok|truncated" "$DONE/fit_$name" && [ -f "$out/latents.h5ad" ]; return
  fi
  local try=0 rc t0 t_start wall pid spid
  while true; do
    try=$((try + 1))
    wait_gpu
    local slog="$QL/fit_$name.try$try.log" samples="$ATT/$name.try$try.samples"
    : > "$samples"
    log "GPU$GPU start fit_$name (try $try): $*"
    t0=$SECONDS; t_start=$(date +%s)
    ( exec env CUDA_VISIBLE_DEVICES=$GPU "$@" ) > "$slog" 2>&1 &
    pid=$!
    sampler $pid "$samples" &
    spid=$!
    wait $pid; rc=$?
    wall=$((SECONDS - t0))
    kill $spid 2>/dev/null; wait $spid 2>/dev/null
    log "GPU$GPU done  fit_$name (exit $rc, ${wall}s)"
    # the kernel's OOM-killer lines since the start (a SIGKILL's cause)
    journalctl -k --since "@$t_start" --no-pager 2>/dev/null \
      | grep -iE "out of memory|oom-kill|killed process" | tail -5 > "$ATT/$name.try$try.kernel" || true
    # classify; python prints: outcome<TAB>cells<TAB>gpu_gb<TAB>host_gb<TAB>other_mib<TAB>reason
    local res
    res=$(CUDA_VISIBLE_DEVICES= python3 - "$tool" "$ds" "$out" "$slog" "$samples" \
          "$rc" "$wall" "$cap" "$ATT/$name.json" "$try" "$BDATA/$ds.h5ad" "$t_start" "$ATT/$name.try$try.kernel" <<'PYEOF'
import json, os, re, sys
tool, ds, out, slog, samples, rc, wall, cap, dst, tryn, h5, t_start, kern = sys.argv[1:]
kernel = open(kern).read().strip() if os.path.exists(kern) else ""
rc, wall, cap, t_start = int(rc), int(wall), int(cap), int(t_start)
text = open(slog, errors="replace").read()
rows = [list(map(int, l.split())) for l in open(samples) if len(l.split()) == 5]
MIB = 1.048576e6 / 1e9                                   # MiB -> GB
own_gpu = max((r[1] for r in rows), default=0) * MIB
own_rss = max((r[3] for r in rows), default=0) * MIB
avail_min = min((r[4] for r in rows), default=-1) * MIB
t_end = rows[-1][0] if rows else 0
other_mib = max((r[2] - r[1] for r in rows if r[0] >= t_end - 60), default=0)
m = re.findall(r"loaded \((\d+), \d+\)|-> \((\d+), \d+\)", text)
cells = next((a or b for a, b in m), "")
if not cells:
    try:
        import h5py
        with h5py.File(h5, "r") as f:
            cells = "%d (export, before the >=5-count filter)" % f["layers/counts"].attrs["shape"][0]
    except Exception:
        cells = ""
cfgp = os.path.join(out, "config.json")
lat = os.path.join(out, "latents.h5ad")
rec = dict(tool=tool, dataset=ds, attempt=int(tryn), exit=rc, wall_s=wall, started=t_start,
           cap_s=cap, log=slog, samples=samples,
           sampled_peak_gpu_gb_nvidia_smi=round(own_gpu, 2),
           sampled_peak_host_rss_gb=round(own_rss, 2),
           min_host_available_gb=round(avail_min, 2),
           other_process_gpu_mib_last_minute=other_mib)
gpu = host = ""
if rc == 0 and os.path.exists(lat) and os.path.exists(cfgp) \
        and os.path.getmtime(lat) >= t_start:
    cfg = json.load(open(cfgp))
    cells = cfg.get("n_cells", cells)
    gpu, host = cfg.get("peak_gpu_gb", ""), cfg.get("peak_host_gb", "")
    if cfg.get("subsampled"):
        outcome = "failed"
        reason = "runner fitted a %s-cell window, not the whole section; not scored" % cfg["subsampled"]
    elif cfg.get("truncated"):
        outcome = "truncated"
        reason = ("wall-clock cap %s h reached: %s/%s epochs completed; latents written; "
                  "setup_s=%s train_s=%s" % (cfg.get("max_hours"), cfg.get("epochs_completed"),
                  cfg.get("num_training_epochs"), cfg.get("setup_s"), cfg.get("train_s")))
    else:
        outcome = "ok"
        reason = "whole section; train_s=%s%s" % (cfg.get("train_s"),
                  "; epochs %s/%s" % (cfg.get("epochs_completed"), cfg.get("num_training_epochs"))
                  if tool == "MintFlow" else "")
    rec["config"] = cfg
else:
    gpu, host = round(own_gpu, 2), round(own_rss, 2)
    oom = re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|"
                    r"CUBLAS_STATUS_ALLOC_FAILED", text)
    tried = re.search(r"Tried to allocate ([\d.]+ [KMG]iB)", text)
    alloc = re.search(r"([\d.]+ [KMG]iB) already allocated|this process has ([\d.]+ [KMG]iB) memory in use", text)
    if cap and (rc == 124 or (rc == 137 and wall >= cap - 120)):
        outcome = "capped"
        reason = ("wall-clock cap %d h reached before the fit finished (no latents: the "
                  "runner writes them only at the end)" % (cap // 3600))
    elif oom:
        outcome = "failed"
        reason = "CUDA OOM on a 24 GB RTX 4090 (GPU 1)"
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
    elif re.search(r"\bMemoryError\b|Unable to allocate", text):
        outcome = "failed"
        reason = "host memory: %s; host RSS %.1f GB" % (
            re.findall(r"(MemoryError.*|Unable to allocate.*)", text)[-1][:160], own_rss)
        rec["error_class"] = "host MemoryError"
    else:
        outcome = "failed"
        errs = re.findall(r"^(\w[\w.]*(?:Error|Exception)\b.*)$", text, re.M)
        reason = "exit %d: %s" % (rc, errs[-1][:200] if errs else "no exception in the log")
        rec["error_class"] = errs[-1].split(":")[0] if errs else "exit %d" % rc
    rec["at_failure"] = dict(tried_to_allocate=tried.group(1) if tried else None,
                             in_use=(alloc.group(1) or alloc.group(2)) if alloc else None)
rec.update(outcome=outcome, cells=cells, reason=reason)
json.dump(rec, open(dst, "w"), indent=2, default=str)
print("\t".join(str(x).replace("\t", " ") for x in
                (outcome, cells, gpu, host, other_mib, reason.replace("\n", " "))))
PYEOF
)
    local outcome cells gpu host other reason
    IFS=$'\t' read -r outcome cells gpu host other reason <<< "$res"
    [ -n "$outcome" ] || { outcome=failed; reason="classification failed (exit $rc); see $slog"; }
    if [ "$outcome" = failed ] && [ $try -eq 1 ] && grep -qE "CUDA out of memory|OutOfMemoryError" "$slog" \
       && [ "${other:-0}" -ge $OTHER_MIB ]; then
      log "fit_$name: CUDA OOM with another process holding ${other} MiB on GPU $GPU -- retrying once"
      feas_row "$tool" "$ds" "$cells" "failed" "$(awk -v s=$wall 'BEGIN{printf "%.2f", s/3600}')" \
        "$gpu" "$host" "$reason; another process held ${other} MiB on GPU $GPU -- retried"
      continue
    fi
    feas_row "$tool" "$ds" "$cells" "$outcome" "$(awk -v s=$wall 'BEGIN{printf "%.2f", s/3600}')" \
      "$gpu" "$host" "$reason"
    echo "$outcome" > "$DONE/fit_$name"
    [ "$outcome" = ok ] || [ "$outcome" = truncated ]
    return
  done
}

score() {  # score <dataset> <method> <latents dir> <key>: battery column + probe grade
  local ds=$1 method=$2 dir=$3 key=$4
  cpu_step "battery_${key}_$ds" flock $TLOCK $PY2 -m discell.experiments.baseline_battery \
    --dataset $ds --method "$method" --latents "$dir/latents.h5ad" \
    --config-run finalL_s0 --tag _lineage
  # shellcheck disable=SC2086
  cpu_step "probe_${key}_$ds" flock $TLOCK $PY2 -m discell.experiments.probe_regrade \
    --dataset $ds --run finalL_s0 --baseline-latents "$dir/latents.h5ad" \
    --method "$method" --force $REFS
}

mintflow_name() {  # the column name, from the fit's config.json
  python3 - "$1/config.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
print("MintFlow (lineage, truncated %s/%s epochs)" % (c["epochs_completed"], c["num_training_epochs"])
      if c.get("truncated") else "MintFlow (lineage)")
PYEOF
}

mintflow() {  # mintflow <dataset>: whole section, 50 epochs, width 600, cap 80 h
  local ds=$1 out=$BRES/mintflow/${1}_lineage name
  mkdir -p $out
  if attempt "mintflow_$ds" MintFlow $ds $out 0 \
      bash -c "cd $BASE/mintflow && exec ./.venv/bin/python run_mintflow.py \
        --h5ad $BDATA/$ds.h5ad --out $out --epochs 50 --width-window 600 \
        --max-hours $MINTFLOW_HOURS"; then
    name=$(mintflow_name $out)
    log "fit_mintflow_$ds -> column '$name'"
    score $ds "$name" $out "mintflow"
  fi
}

simvi() {  # simvi <dataset>: whole-section attempt, wall-clock cap 24 h
  local ds=$1 out=$BRES/simvi/${1}_lineage
  mkdir -p $out
  if attempt "simvi_$ds" SIMVI $ds $out 86400 \
      bash -c "cd $BASE/simvi && exec timeout --kill-after=10m $SIMVI_CAP \
        ./.venv/bin/python run_simvi.py --h5ad $BDATA/$ds.h5ad --out $out \
        --max-cells $NO_CAP"; then
    score $ds "SIMVI (lineage)" $out "simvi"
  fi
}

mintflow_ff() {  # MintFlow FF whole: only with >= HOST_NEED_GB available at its start
  local name=mintflow_$FF avail
  if [ -f "$DONE/fit_$name" ]; then log "skip fit_$name (outcome recorded: $(cat $DONE/fit_$name))"
  else
    wait_gpu
    avail=$(host_avail_gb)
    if [ "$avail" -lt $HOST_NEED_GB ]; then
      log "fit_$name: $avail GB host memory available < $HOST_NEED_GB -- not attempted"
      feas_row MintFlow $FF 1157343 not_attempted "" "" "" \
        "host memory: ${avail} GB available at the start of this step (< ${HOST_NEED_GB} GB required to retry). Earlier measured failure (2026-09-22, lane-2 REPAIR_REPORT section C): host OOM-killed (exit 137) after 574 s inside MintFlow's setup_data, before training; 1,157,343 cells x 5,001 genes, 956 M non-zeros; the process floor estimated ~92 GB (devlog: >= 45 GB before training) on a 125 GB host"
      echo not_attempted > "$DONE/fit_$name"
      return 0
    fi
    log "fit_$name: $avail GB host memory available -- attempting"
  fi
  mintflow $FF
}

# -- main -------------------------------------------------------------------------
log "queue start (pid $$): GPU $GPU only, >= $NEED_MIB MiB free per fit; waiting for: $WAIT_FOR"
archive_windows "" start
for f in $LU $OV $FF; do
  [ -f $BDATA/$f.h5ad ] || log "WARNING: no lineage export $BDATA/$f.h5ad"
  [ -f data/datasets/$f/experiments/baseline_context_finalL_s0.npz ] \
    || log "WARNING: no cached finalL_s0 context for $f (its scoring will fail)"
done

n=0
while true; do
  running=$(queues_running)
  [ -z "$running" ] && break
  [ $((n % 6)) -eq 0 ] && log "waiting: still running:$running"
  n=$((n + 1)); sleep $POLL
done
log "no queue we wait for is running -- starting (host: $(host_avail_gb) GB available, GPU $GPU: $(gpu_free) MiB free)"

mintflow $LU
mintflow $OV
simvi $LU
simvi $OV
simvi $FF
mintflow_ff

# the closing re-render: window columns out (again), battery markdown, probe tables
archive_windows --render end
for ds in $LU $OV $FF; do
  # shellcheck disable=SC2086
  cpu_step "probe_table_lineage_final_$ds" flock $TLOCK $PY2 -m discell.experiments.probe_regrade \
    --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
done
log "feasibility table:"; cat $FEAS
log "queue finished ($(ls $DONE | wc -l) markers)"
