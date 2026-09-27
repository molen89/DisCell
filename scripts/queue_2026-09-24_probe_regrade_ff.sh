#!/bin/bash
# FF collapse-grid re-grade (wfix_*, wfixb_*), split out of
# scripts/queue_2026-09-24_probe_regrade.sh after its FF step was stopped to
# relieve RAM (three FF-sized processes, 8 GB left; 2026-09-24 16:10). Waits
# for the FF uncontrolled reference (scripts/queue_2026-09-24_uncontrolled.sh)
# and >= 50 GB MemAvailable, then grades on the CPU (graded runs are skipped),
# re-verdicts FF and renders the FF collapse tables.
#
#   setsid nohup bash scripts/queue_2026-09-24_probe_regrade_ff.sh \
#     >> scripts/logs/probe_regrade_2026-09-24/queue_ff.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6 MKL_NUM_THREADS=6
FF=xenium_prime_human_ovary_ff
ROOT=scripts/logs/probe_regrade_2026-09-24
MOD="uv run python -m discell.experiments.probe_regrade"
REF=scripts/logs/uncontrolled_2026-09-24/done/reverdict_$FF
log() { echo "[$(date '+%F %T')] $*"; }
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

log "waiting for $REF and >= 50 GB free"
until [ -f "$REF" ] && [ "$(avail)" -ge 50 ]; do sleep 60; done
runs=$(for d in data/datasets/$FF/runs/wfix_* data/datasets/$FF/runs/wfixb_*; do
         [ -f "$d/best.pt" ] && [ -f "$d/metrics.json" ] && basename "$d"; done)
log "start FF collapse grids ($(echo $runs | wc -w) runs, $(avail) GB free)"
$MOD --dataset $FF --run $runs > $ROOT/logs/grade_wfix_${FF}_redo.log 2>&1 \
  && touch $ROOT/done/grade_wfix_$FF && log "done" || log "FAILED (see logs)"
$MOD --dataset $FF --reverdict >> $ROOT/logs/grade_wfix_${FF}_redo.log 2>&1
$MOD --dataset $FF --table wfix --runs 'wfix_*' > $ROOT/logs/table_wfix_$FF.log 2>&1
$MOD --dataset $FF --table wfixb --runs 'wfixb_*' 'wfix_control_aw0.1_s*' \
  'wfix_warmup30_aw0.1_s*' > $ROOT/logs/table_wfixb_$FF.log 2>&1
log "FF tables written"
