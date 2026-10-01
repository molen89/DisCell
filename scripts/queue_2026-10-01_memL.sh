#!/bin/bash
# Peak memory of DISCELL, VRAM and host RAM (devlog 2026-10-01, "Peak memory,
# VRAM and RAM, for every method (motivation, 2026-10-01; author)").
#
# One measurement fit per section at the final configuration: seed 0, the
# flags of finalL_s0 (FLAGS + COMMON of scripts/queue_2026-09-25_final_lineage.sh,
# hence the same tile split, which assemble() draws from the seed), run name
# memL_s0. A preflight re-derives each fit's TrainConfig and stops unless it
# equals runs/finalL_s0/config.json but for run_name / git / device. The fits
# are used for memory and wall time only.
#
# Each fit runs under `/usr/bin/time -v` (maximum resident set size, elapsed
# wall clock) inside a small wrapper that calls discell.model.train.main and
# then records torch.cuda.max_memory_allocated / max_memory_reserved and the
# process's own ru_maxrss: the same counters as the baselines' config.json
# (DisCell-baselines/common.py: max_memory_allocated over the whole process;
# ru_maxrss of the fitting process).
#
# GPU 0 only (MintFlow refits run on GPU 1; never touched): a fit starts when
# GPU 0 has no compute process. While this queue runs it keeps a fresh lock
# $ROOT/locks/gpu0, which the MintFlow refit queue's picker respects, so the
# refits stay off GPU 0. Order: TMA core, ovarian FFPE, lung FFPE, then FF,
# which starts only with >= FF_HOST_GB host memory available (MemAvailable).
#
# Outputs: $ROOT/logs/<dataset>.log (train log), $ROOT/<dataset>.time
# (/usr/bin/time -v), $ROOT/<dataset>.torch.json (wrapper), and
# $ROOT/memL_summary.json (all four sections). Then the timing table and the
# cost figure are regenerated:
#   uv run python scripts/paper_tables.py --only timing
#   uv run python submission_paper/aistats/figures/src/fig_cost.py
# Idempotent: a section with $ROOT/<dataset>.torch.json (exit 0) is skipped.
#
#   mkdir -p scripts/logs/memL_2026-10-01 && setsid nohup \
#     bash scripts/queue_2026-10-01_memL.sh \
#     >> scripts/logs/memL_2026-10-01/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=offline CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # as the final_lineage queue

GS=gse315411_pdltma06_11_prime_solo
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ORDER="$GS $OV $LU $FF"
RUN=memL_s0
GPU=0
FF_HOST_GB=70

ROOT=scripts/logs/memL_2026-10-01
QL=$ROOT/logs; LOCK=$ROOT/locks
mkdir -p $QL $LOCK
PY=.venv/bin/python

# the final configuration (copied from scripts/queue_2026-09-25_final_lineage.sh)
COMMON="--gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 --alpha-a 0.3
        --epochs 500 --patience 40 --figures-every 100 --w-warmup-epochs 30
        --adv-comp-weight 3 --label-key lineage"
declare -A FLAGS=(
  [$GS]="--alpha-z 0.0018 --variant pdl018d --tile-cells 2048"
  [$OV]="--alpha-z 0.0035"
  [$LU]="--alpha-z 0.002"
  [$FF]="--alpha-z 0.00035")

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>> $ROOT/instance.lock
flock -n 9 || { log "another instance holds $ROOT/instance.lock -- exiting"; exit 1; }

# keep our GPU 0 lock fresh while the queue lives
mkdir -p $LOCK/gpu0
( while kill -0 $$ 2>/dev/null; do touch $LOCK/gpu0; sleep 60; done ) &
KEEPER=$!
trap 'kill $KEEPER 2>/dev/null; rmdir $LOCK/gpu0 2>/dev/null' EXIT

gpu0_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU)" ]; }
host_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

# -- preflight: every fit's TrainConfig is finalL_s0's -----------------------
specs=()
for ds in $ORDER; do specs+=("$ds::${FLAGS[$ds]} $COMMON"); done
if ! $PY - "${specs[@]}" <<'EOF'
import json, shlex, sys
from dataclasses import asdict
from discell.model.train import TrainConfig, build_parser
bad = 0
for spec in sys.argv[1:]:
    ds, flags = spec.split("::", 1)
    args = vars(build_parser().parse_args(
        ["--dataset", ds, "--seed", "0", "--run-name", "memL_s0"] + shlex.split(flags)))
    for k in ("quiet", "time_only"):
        args.pop(k, None)
    new = asdict(TrainConfig(**args))
    ref = json.load(open(f"data/datasets/{ds}/runs/finalL_s0/config.json"))
    skip = {"run_name", "git", "device"}
    default = asdict(TrainConfig(dataset=""))
    diffs = [f"{k}: {new.get(k)!r} vs finalL_s0 {ref.get(k, '<absent>')!r}"
             for k in sorted(set(new) | set(ref)) if k not in skip
             and ((k in ref and new.get(k) != ref[k])
                  or (k not in ref and new[k] != default[k]))]
    bad += bool(diffs)
    print(f"{ds}: " + ("finalL_s0's config" if not diffs else "DIFFERS: " + "; ".join(diffs)))
sys.exit(1 if bad else 0)
EOF
then log "preflight FAILED: a memL config differs from finalL_s0; nothing run"; exit 1; fi
log "preflight ok"

fit() {  # fit <dataset>
  local ds=$1 tj=$ROOT/$1.torch.json rc t0
  if [ -f $tj ] && grep -q '"exit": 0' $tj; then log "skip $ds ($tj present)"; return 0; fi
  if [ -d data/datasets/$ds/runs/$RUN ]; then
    log "$ds: runs/$RUN exists from an unfinished attempt -- moved aside"
    mv data/datasets/$ds/runs/$RUN data/datasets/$ds/runs/${RUN}_aborted_$(date +%s)
  fi
  if [ $ds = $FF ]; then
    local n=0
    until [ "$(host_avail_gb)" -ge $FF_HOST_GB ]; do
      [ $((n % 10)) -eq 0 ] && log "FF waits: $(host_avail_gb) GiB available (need $FF_HOST_GB)"
      n=$((n + 1)); sleep 60
    done
  fi
  local n=0
  until gpu0_idle; do
    [ $((n % 10)) -eq 0 ] && log "$ds waits: a compute process is on GPU $GPU"
    n=$((n + 1)); sleep 60
  done
  log "GPU$GPU start $ds ($(host_avail_gb) GiB host available)"
  t0=$SECONDS
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU /usr/bin/time -v -o $ROOT/$ds.time \
    $PY -c '
import json, resource, sys, time
import torch
from discell.model import train
out, argv = sys.argv[1], sys.argv[2:]
t0 = time.time()
rc = train.main(argv)
json.dump({"exit": rc, "argv": argv, "train_main_s": time.time() - t0,
           "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(),
           "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(),
           "ru_maxrss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "gpu": torch.cuda.get_device_name(0)}, open(out, "w"), indent=2)
' $tj --dataset $ds --run-name $RUN ${FLAGS[$ds]} $COMMON --seed 0 \
    > $QL/$ds.log 2>&1; rc=$?
  log "GPU$GPU done  $ds (exit $rc, $((SECONDS - t0))s)"
  [ $rc -eq 0 ] || { log "FAIL $ds; see $QL/$ds.log"; return 1; }
}

for ds in $ORDER; do fit $ds; done

# -- summary --------------------------------------------------------------------
$PY - $ROOT $ORDER <<'EOF'
import json, re, sys
from pathlib import Path
root, out = Path(sys.argv[1]), {}
for ds in sys.argv[2:]:
    tj, tt = root / f"{ds}.torch.json", root / f"{ds}.time"
    if not tj.exists() or not tt.exists():
        out[ds] = {"status": "missing"}
        continue
    t, txt = json.loads(tj.read_text()), tt.read_text()
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt).group(1))
    h, m, s = (["0"] + re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", txt)
               .group(1).split(":"))[-3:]
    m_json = Path(f"data/datasets/{ds}/runs/memL_s0/metrics.json")
    mt = json.loads(m_json.read_text()) if m_json.exists() else {}
    out[ds] = {"status": "ok" if t["exit"] == 0 else "failed",
               "run": f"data/datasets/{ds}/runs/memL_s0",
               "max_rss_kib_time_v": rss,
               "max_rss_gib_time_v": rss / 2 ** 20,
               "ru_maxrss_self_kib": t["ru_maxrss_self_kib"],
               "torch_max_allocated_gib": t["torch_max_allocated_bytes"] / 2 ** 30,
               "torch_max_reserved_gib": t["torch_max_reserved_bytes"] / 2 ** 30,
               "wall_s_time_v": int(h) * 3600 + int(m) * 60 + float(s),
               "train_main_s": t["train_main_s"],
               "fit_minutes": mt.get("minutes"),
               "last_epoch": mt.get("last_epoch"),
               "gpu": t["gpu"]}
(root / "memL_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF
log "summary written: $ROOT/memL_summary.json"

uv run python scripts/paper_tables.py --only timing > $QL/paper_tables_timing.log 2>&1 \
  && log "tab:timing regenerated" || log "FAIL paper_tables --only timing; see $QL/paper_tables_timing.log"
if [ -f submission_paper/aistats/figures/src/fig_cost.py ]; then
  uv run python submission_paper/aistats/figures/src/fig_cost.py > $QL/fig_cost.log 2>&1 \
    && log "fig_cost regenerated" || log "FAIL fig_cost.py; see $QL/fig_cost.log"
fi
log "queue finished"
