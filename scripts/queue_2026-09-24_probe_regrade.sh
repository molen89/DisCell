#!/bin/bash
# The invariance probe re-graded per block, with a nonlinear grader
# (devlog 2026-09-24 15:00, R20 + R22). Inference only, CPU only: every step
# runs with CUDA_VISIBLE_DEVICES="" so no GPU queue (final re-pin, R12 kappa
# arms, alpha_w ladder) is ever blocked. Two lanes, each one process at a time:
#   lane S (small): GSE solo + dual section, lung, ovarian;
#   lane F: FF (its assembly + resident tiles are ~30 GB of RAM).
# Order: (a) final_s* / best_s* (dual via the crossslide protocol) and
# (b) the baselines' stored latents -> tables; then (c) ovarian sweep3,
# alpha_z ladder, collapse grids, query/prior, R12 arms (+ FF collapse grids);
# then (d) polls aw_* (every dataset) and r12_* (ovarian) until the alpha_w
# ladder and R12 queues exit, grading each finished fit as it lands.
# Idempotent: a run whose probe_blocks.json exists is skipped by the module;
# DONE markers are keyed by step AND dataset.
#
#   setsid nohup bash scripts/queue_2026-09-24_probe_regrade.sh \
#     >> scripts/logs/probe_regrade_2026-09-24/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6 MKL_NUM_THREADS=6

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
BRES=/home/rmolen/github/DisCell-baselines/results
SPLIT=final_s0          # the split and targets the baselines are graded on
POLL=600

ROOT=scripts/logs/probe_regrade_2026-09-24
QL=$ROOT/logs; DONE=$ROOT/done
mkdir -p $QL $DONE
MOD="uv run python -m discell.experiments.probe_regrade"

log() { echo "[$(date '+%F %T')] $*"; }

# finished fits (best.pt AND metrics.json) matching patterns; a symlinked
# name (best_s0 -> best_az0.5_s0) yields its target, graded once
finished() {  # finished <dataset> <pattern>...
  local ds=$1; shift
  local d p
  for p in "$@"; do
    for d in data/datasets/$ds/runs/$p; do
      [ -d "$d" ] && [ -f "$d/best.pt" ] && [ -f "$d/metrics.json" ] \
        && basename "$(readlink -f "$d")"
    done
  done | sort -u
}

step() {  # step <marker> <cmd...>: run once, log to $QL/<marker>.log
  local m=$1; shift
  [ -f "$DONE/$m" ] && { log "skip $m (done)"; return 0; }
  log "start $m"
  if "$@" > "$QL/$m.log" 2>&1; then
    touch "$DONE/$m"; log "done  $m"
  else
    log "FAILED $m (exit $?) -- see $QL/$m.log"
  fi
}

grade() {  # grade <dataset> <step name> <pattern>... : DisCell runs, own section
  local ds=$1 name=$2; shift 2
  local runs; runs=$(finished $ds "$@")
  [ -n "$runs" ] || { log "$ds $name: no finished runs"; return 0; }
  step "grade_${name}_${ds}" $MOD --dataset $ds --run $runs
}

grade_dual() {  # grade_dual <pattern>... : GSE runs on the dual section
  local runs; runs=$(finished $GS "$@")
  step "grade_final_${GD}" $MOD --dataset $GD --config-from $GS --run $runs
}

baseline() {  # baseline <dataset> <split source> <method> <latents dir>
  local ds=$1 from=$2 method=$3 dir=$4 extra=() tag
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  [ -f "$dir/latents.h5ad" ] || { log "$ds '$method': no latents at $dir"; return 0; }
  tag=$(printf '%s' "$method" | tr -c 'A-Za-z0-9=.' '_')
  step "baseline_${tag}_${ds}" $MOD --dataset $ds --run $SPLIT "${extra[@]}" \
    --baseline-latents "$dir/latents.h5ad" --method "$method"
}

table() {  # table <dataset> <group> [--config-from X] [--baselines] --runs <patterns>
  local ds=$1 group=$2; shift 2
  $MOD --dataset $ds --table $group "$@" > "$QL/table_${group}_${ds}.log" 2>&1 \
    && log "table $group $ds" || log "table $group $ds FAILED"
}

tables_ab() {
  local ds
  for ds in $GS $OV $LU $FF; do
    table $ds final --runs 'final_s*' 'best_s*'
    table $ds baselines --baselines --runs 'final_s*' 'best_s*'
  done
  table $GD final --config-from $GS --runs 'final_s*' 'best_s*'
  table $GD baselines --config-from $GS --baselines --runs 'final_s*' 'best_s*'
}

tables_c() {
  table $OV sweep3_k --runs 'sweep3_k*'
  table $OV sweep3_aw --runs 'sweep3_aw*'
  table $OV ladder_az --runs 'ladder_az_*' 'sweep3_k0.1_s*'
  table $OV wfix --runs 'wfix_*'
  table $OV wfixb --runs 'wfixb_*' 'wfix_control_aw0.1_s*' 'wfix_warmup30_aw0.1_s*'
  table $OV qp --runs 'qp_*'
  table $OV r12 --runs 'r12_*' 'wfix_warmup30_aw0.1_s*'
  table $FF wfix --runs 'wfix_*'
  table $FF wfixb --runs 'wfixb_*' 'wfix_control_aw0.1_s*' 'wfix_warmup30_aw0.1_s*'
}

tables_d() {
  local ds
  for ds in $GS $OV $LU $FF; do
    [ -n "$(finished $ds 'aw_*')" ] && table $ds awladder --runs 'aw_*'
  done
}

lane_small() {
  # (a)
  grade $GS final 'final_s*' 'best_s*'
  grade_dual 'final_s*' 'best_s*'
  grade $LU final 'final_s*' 'best_s*'
  grade $OV final 'final_s*' 'best_s*'
  # (b)
  baseline $GS $GS resolVI $BRES/resolvi/$GS
  baseline $GS $GS SIMVI $BRES/simvi/$GS
  baseline $GS $GS MintFlow $BRES/mintflow/$GS
  baseline $GS $GS "MintFlow (w=600, 5/50 epochs)" $BRES/mintflow/${GS}_w600
  baseline $GS $GS "MintFlow (50 epochs, w=600)" $BRES/mintflow/${GS}_full
  baseline $GD $GS resolVI $BRES/resolvi/$GD
  baseline $GD $GS "SIMVI (fit on this section)" $BRES/simvi/${GD}_fit_on_target
  baseline $GD $GS "MintFlow (transfer, 5/50 epochs)" $BRES/mintflow/${GD}_transfer
  baseline $GD $GS "MintFlow (transfer, 50 epochs)" $BRES/mintflow/${GD}_transfer_full
  baseline $LU $LU resolVI $BRES/resolvi/$LU
  baseline $OV $OV resolVI $BRES/resolvi/$OV
  touch $DONE/lane_small_ab
}

lane_ff_ab() {
  grade $FF final 'final_s*' 'best_s*'
  baseline $FF $FF resolVI $BRES/resolvi/$FF
  baseline $FF $FF "SIMVI (100k-cell window)" $BRES/simvi/$FF
  touch $DONE/lane_ff_ab
}

lane_small_c() {
  grade $OV sweep3 'sweep3_k*' 'sweep3_aw*'
  grade $OV ladder_az 'ladder_az_*'
  grade $OV wfix 'wfix_*' 'wfixb_*'
  grade $OV qp 'qp_*'
  grade $OV r12 'r12_*'
}

lane_ff_c() {
  grade $FF wfix 'wfix_*' 'wfixb_*'
}

# (d): the module skips graded runs, so each poll only grades new fits
poll() {
  local n=0 ds live
  while :; do
    for ds in $GS $OV $LU $FF; do
      [ -n "$(finished $ds 'aw_*')" ] || continue
      rm -f $DONE/grade_awpoll_$ds
      grade $ds awpoll 'aw_*'
    done
    rm -f $DONE/grade_r12poll_$OV
    grade $OV r12poll 'r12_*'
    tables_c; tables_d
    live=$(pgrep -f 'queue_2026-09-24_awladder.sh|queue_2026-09-24_r12_arms.sh' | wc -l)
    if [ "$live" -eq 0 ]; then
      log "alpha_w ladder and R12 queues have exited -- last poll done"
      break
    fi
    [ $((n % 6)) -eq 0 ] && log "polling aw_* / r12_* (every ${POLL}s)"
    n=$((n + 1)); sleep $POLL
  done
}

log "queue start (CPU only; OMP $OMP_NUM_THREADS per process; split $SPLIT)"
log "=== (a) + (b) ==="
lane_small & PS=$!
(sleep 30; lane_ff_ab) & PF=$!
wait $PS $PF
tables_ab
log "(a) + (b) tables written"
touch $DONE/phase_ab
log "=== (c) ==="
lane_small_c & PS=$!
(sleep 30; lane_ff_c) & PF=$!
wait $PS $PF
tables_c
log "(c) tables written"
touch $DONE/phase_c
log "=== (d) ==="
poll
tables_ab
log "queue finished"
