#!/bin/bash
# The evaluation mask, re-read without refits (tag excl). Devlog 2026-09-28,
# "Unassigned is a training class and a neighbour, never a metric target
# (author's decision)": Unassigned cells stay in training, the graph, the
# composition and the leak influx; they are excluded as TARGETS of every
# evaluation read (discell/model/eval_mask.py; the switch
# DISCELL_EVAL_INCLUDE_UNASSIGNED=1 restores the pre-rule reads). Nothing is
# refitted; every read below re-reads best.pt under the mask and overwrites
# its file in place (the excluded version is primary from here). The pre-mask
# files are copied once to $ROOT/backup_pre_mask/ before anything runs.
#
# Order (every GPU step through the picker below; markers keyed by step AND
# dataset AND run under $ROOT/done; a relaunch resumes):
#  (0) backup of every pre-mask read, then the one-time "incl. Unassigned"
#      companion of the finals' envelope table, rendered through the switch
#      from the still-unmasked files (envelope_table_ci_at_best_incl_unassigned.md).
#  per dataset, one lane each (GSE, ovarian, lung, FF), in this order:
#  (b0) uncontrolledL_s{0,1}: degeneracy (guard + the masked in-trainer
#      battery), the probe grade (GSE also on the dual) -- the reference every
#      fraction below is taken against, so it goes first;
#  (validate runs without its in-module probe: probe_regrade --force writes the
#  same record -- probe_blocks_for_run, same seed and draws -- graded against
#  the right references, so the validate probe would be computed twice.)
#  (a) finalL_s{0,1,2}: validate morans,niche -> degeneracy ->
#      probe_regrade --force vs uncontrolledL_s* -> atlas -> transport --read
#      both --hvg 1000 (k-means; + tumour-band on ovarian) -> GSE: crossslide
#      on the dual, its probe grade and recon_modes -> recon_modes; then atlas
#      --compare-runs per seed, the tile bootstrap, external criteria (the
#      reads on disk for finalL), GO localisation, subtype recovery;
#  (b) uncontrolledL_s{0,1}: validate morans,niche, recon_modes (+ dual),
#      transport --read mean;
#  (c) sweepL_k{0,0.05,0.2,0.3,0.4}_s{0,1,2} (k0.1 = finalL, links): validate
#      morans,niche,probe, degeneracy, the probe grade (+ dual), transport
#      --read mean (+ GSE crossslide); the sweep report, kappa survival, the
#      sweep transport JSON rebuilt from the per-run files (FF's 11 missing
#      points included; supersedes the ceiling agent's ff_queue.sh), marker
#      pairs on finalL + sweepL with the kappa summary;
#  (d) after scripts/queue_2026-09-28_baselines_lineage.sh has finished:
#      the battery columns (DisCell finalL, slot 0 last) and the probe grades
#      of every baseline latent on disk (the lineage refit where present,
#      else the old-label fit) under the mask, serialised on that queue's
#      tables lock; re-verdicts (fractions of the re-graded resolVI) and the
#      probe tables;
#  (e) the envelope tables --at best --ci over finalL (the mask primary),
#      the sweep probe table; after the sensitivity queue has finished, its
#      arm reads made before the mask (listed in $ROOT/SENSF_PRE_MASK.tsv at
#      launch) re-read under it (SENSF_REREAD=0 skips) and sensF_tables.py.
# The top-decile cycle rows (cycling set, author's decision 2026-09-28) come
# with the same re-read: Trainer.evaluate carries them, so degeneracy.json's
# masked battery (every finalL / uncontrolledL / sweepL run), crossslide's
# held-out section (GSE) and every battery column (DisCell and baselines)
# hold them; the tables read them from there. cell_cycle.cycle_eligible is
# the shared mask. metrics.json is never rewritten (cycle_reread.py is not
# needed for the tables and is not run).
# Transport and bootstrap run at OMP_NUM_THREADS=8 (the k-means niche labels
# of the published reads; other thread counts move one cell).
#
# Picker / lock / dataset-qualified marker machinery from
# scripts/queue_2026-09-28_sens_final.sh (gate 9500 MiB, FF 18500 with its
# waiter announced in $ROOT/waiting/ff.<pid>, MintFlow charged at its peak,
# settle delay, other queues' fresh locks respected, one job of ours per GPU).
#
# GPU-hour estimate (per-step means of the 2026-09-25 lineage queue, which
# ran these reads on the same slides): (a) ~6.4 (transport 1.9 with the
# Unassigned panels still drawn, marker pairs 1.2, validate 0.7, bootstrap
# 0.5, atlas 0.4, external 0.5, subtype 0.5); (b) ~0.9; (c) ~8.8 (validate
# 3.4 of it FF-heavy, kappa survival 2.4, the rest ~1 each); (d) ~0.5; (e) ~0
# (+ the sensF re-reads, ~1). ~17 GPU-h of compute, ~21 card-hours with the
# settle; wall ~20-30 h on cards shared with the sensitivity and baseline
# queues (an FF step waits for 18.5 GB free).
#
#   mkdir -p scripts/logs/unassigned_2026-09-28 && \
#   setsid nohup scripts/queue_2026-09-28_unassigned.sh \
#     >> scripts/logs/unassigned_2026-09-28/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the mask is on (primary)

GS=gse315411_pdltma06_11_prime_solo
GD=gse315411_pdltma06_10_prime_dual
OV=xenium_prime_ovarian_cancer_ffpe
LU=xenium_prime_human_lung_cancer_ffpe
FF=xenium_prime_human_ovary_ff
ALL="$GS $OV $LU $FF"

ROOT=scripts/logs/unassigned_2026-09-28
QL=$ROOT/logs; DONE=$ROOT/done; LOCK=$ROOT/locks; WAIT=$ROOT/waiting
FAILED=$ROOT/failed; BACKUP=$ROOT/backup_pre_mask
mkdir -p $QL $DONE $LOCK $WAIT $FAILED $BACKUP
GPUS="0 1"
NEED_FF=18500
NEED_OV=9500          # every non-FF dataset
TRIES=10              # attempts when a step dies on a CUDA OOM
TRIES_OTHER=3         # attempts on any other failure
SETTLE=45
FRESH=600
PY="uv run python"
#: reads that compute no k-means niche labels (probe grades, battery columns,
#: subtype recovery, marker pairs, recon modes) run at 2 threads: their MLP
#: probes take ~53 s per fit at 8 threads on the loaded machine, ~5 s at 2.
#: Every step that computes niche labels (validate, degeneracy, transport,
#: bootstrap, external criteria, sweep report) and every other step stays at
#: 8, the published reads' count. Each output records it (eval_mask).
PY2="env OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python"
KAPPAS="0 0.05 0.1 0.2 0.3 0.4"
SWEEP_KAPPAS="0 0.05 0.2 0.3 0.4"             # 0.1 links to finalL
REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none
      --baseline-tag _lineage"
declare -A SWEEP_FLAGS=(
  [$GS]="--variant pdl018d --tile-cells 2048 --alpha-z 0.0018"
  [$OV]="--alpha-z 0.0035" [$LU]="--alpha-z 0.002" [$FF]="--alpha-z 0.00035")
BASE=/home/rmolen/github/DisCell-baselines
BRES=$BASE/results
RVI=$BRES/resolvi_lineage
BL_ROOT=scripts/logs/baselines_lineage_2026-09-28
TLOCK=$BL_ROOT/tables.lock          # the baseline tables' read-modify-write lock
SENSF_ROOT=scripts/logs/sens_final_2026-09-28
SENSF_REREAD=${SENSF_REREAD:-1}

log() { echo "[$(date '+%F %T')] $*"; }
runs() { echo data/datasets/$1/runs; }
marker() { echo "${1}_${2}_${3}"; }   # marker <step> <dataset> <run>
need_of() { [ "$1" = "$FF" ] && echo $NEED_FF || echo $NEED_OV; }
role_of() { [ "$1" = "$FF" ] && echo ff || echo ov; }
fitted() { [ -f "$(runs $1)/$2/metrics.json" ] && [ -f "$(runs $1)/$2/best.pt" ]; }

#: the modules whose reads carry the mask; each must still hold it when a
#: step starts (another agent's rewrite would otherwise run a read unmasked)
MASKED_CODE="discell/model/train.py discell/model/validate.py discell/model/degeneracy.py
  discell/model/transport.py discell/model/atlas.py discell/model/sweep.py
  discell/experiments/bootstrap.py discell/experiments/recon_modes.py
  discell/experiments/baseline_battery.py discell/experiments/probe_regrade.py
  discell/experiments/external_criteria.py discell/experiments/subtype_recovery.py
  discell/experiments/marker_pairs.py discell/experiments/w_deviation.py
  discell/experiments/at_best.py scripts/envelope_tables.py"
mask_code_ok() {
  local f
  [ -f discell/model/eval_mask.py ] || return 1
  for f in $MASKED_CODE; do grep -q "eval_mask" $f || return 1; done
}
wait_mask_code() {
  local n=0
  until mask_code_ok; do
    [ $((n % 6)) -eq 0 ] && log "the mask is missing from the code (a rewrite?) -- waiting, no read runs"
    n=$((n + 1)); sleep 300
  done
}

# -- GPU picker (scripts/queue_2026-09-28_sens_final.sh) ------------------------
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
  local try=0 other=0 rc gpu t0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    wait_mask_code
    gpu=$(acquire_gpu $need $role)
    t0=$SECONDS
    log "GPU$gpu start $name (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > $QL/$name.log 2>&1; rc=$?
    rmdir "$LOCK/gpu$gpu" 2>/dev/null
    log "GPU$gpu done  $name (exit $rc, $((SECONDS - t0))s)"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
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
  [ "$name" = envelope_incl_unassigned ] || wait_mask_code
  log "CPU  start $name"
  CUDA_VISIBLE_DEVICES= "$@" > $QL/$name.log 2>&1; rc=$?
  log "CPU  done  $name (exit $rc, $((SECONDS - t0))s)"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$((SECONDS - t0))" >> $ROOT/STEPS.tsv
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; rm -f "$FAILED/$name"; return 0; fi
  log "FAIL $name; see $QL/$name.log"; touch "$FAILED/$name"; return 1
}
locked_cpu_step() { local name=$1; shift; cpu_step "$name" flock $TLOCK "$@"; }

# -- (0) the pre-mask backup and the incl. companion ---------------------------------
backup_pre_mask() {  # every read file a step below may overwrite, once (cp -n)
  local ds d
  [ -f "$DONE/backup_pre_mask" ] && return 0
  for ds in $ALL $GD; do
    d=data/datasets/$ds
    find -L $d/experiments -maxdepth 2 -type f \( -name '*.json' -o -name '*.md' \) \
      2>/dev/null | while read -r f; do
        mkdir -p "$BACKUP/$(dirname ${f#data/datasets/})"; cp -n "$f" "$BACKUP/${f#data/datasets/}"; done
    for r in $d/runs/finalL_s* $d/runs/uncontrolledL_s* $d/runs/sweepL_*; do
      [ -d "$r" ] || continue
      [ -L "$r" ] && continue                      # k0.1 links: finalL's files
      find $r -maxdepth 2 -type f \( -name '*.json' -o -name 'programs.npy' \) \
        ! -name 'history.jsonl' | while read -r f; do
          mkdir -p "$BACKUP/$(dirname ${f#data/datasets/})"; cp -n "$f" "$BACKUP/${f#data/datasets/}"; done
    done
  done
  touch "$DONE/backup_pre_mask"
  log "pre-mask reads backed up to $BACKUP ($(du -sh $BACKUP | cut -f1))"
}

incl_companion() {  # the one-time "both ways" table, from the still-unmasked files
  cpu_step envelope_incl_unassigned $PY scripts/envelope_tables.py --datasets $ALL \
    --runs finalL_s0 finalL_s1 finalL_s2 --at best --ci --eval-include-unassigned \
    --combined $ROOT/envelope_tables_finalL.md
}

# -- the reads ----------------------------------------------------------------------
grade() {  # grade <dataset> <run>: the per-block probe on its own section (+ the dual)
  local ds=$1 run=$2
  # shellcheck disable=SC2086
  gstep $ds "$(marker probe_regrade $ds $run)" $PY2 -m discell.experiments.probe_regrade \
    --dataset $ds --run $run --force $REFS
  if [ "$ds" = "$GS" ]; then
    # shellcheck disable=SC2086
    gstep $GD "$(marker probe_regrade $GD $run)" $PY2 -m discell.experiments.probe_regrade \
      --dataset $GD --config-from $GS --run $run --force $REFS
  fi
}

reference() {  # (b0) reference <dataset> <run>: the guard, the battery, the probe grade
  local ds=$1 run=$2
  fitted $ds $run || { log "no fit for $ds/$run -- no reference reads"; return 0; }
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  grade $ds $run
}

final_reads() {  # (a) final_reads <dataset> <run>
  local ds=$1 run=$2
  fitted $ds $run || { log "no fit for $ds/$run -- no reads"; return 0; }
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  if [ "$ds" = "$GS" ]; then   # before the dual's grade: its legacy check reads it
    gstep $ds "$(marker crossslide $ds $run)" $PY -m discell.model.crossslide --dataset $GS \
      --run $run --eval-dataset $GD
  fi
  grade $ds $run          # after validate: its in-module probe grades against the old refs
  gstep $ds "$(marker atlas $ds $run)" $PY -m discell.model.atlas --dataset $ds --run $run
  gstep $ds "$(marker transport $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read both --hvg 1000
  if [ "$ds" = "$OV" ]; then
    gstep $ds "$(marker transport_band $ds $run)" $PY -m discell.model.transport \
      --dataset $ds --run $run --read both --hvg 1000 --niche-source tumour-band
  fi
  if [ "$ds" = "$GS" ]; then
    gstep $ds "$(marker recon_modes_dual $ds $run)" $PY2 -m discell.experiments.recon_modes \
      --dataset $GS --run $run --eval-dataset $GD
  fi
  gstep $ds "$(marker recon_modes $ds $run)" $PY2 -m discell.experiments.recon_modes \
    --dataset $ds --run $run
}

compare_group() {  # compare_group <dataset> <run>...: cross-seed atlas
  local ds=$1 r others live=""; shift
  for r in "$@"; do fitted $ds $r && live="$live $r"; done
  for r in $live; do
    others=$(echo $live | tr ' ' '\n' | grep -vx $r | tr '\n' ' ')
    [ -n "$others" ] || continue
    # shellcheck disable=SC2086
    gstep $ds "$(marker atlas_cmp $ds $r)" $PY -m discell.model.atlas --dataset $ds \
      --run $r --compare-runs $others
  done
}

ext() {  # ext <read> <dataset> <run>
  gstep $2 "$(marker ext_$1 $2 $3)" $PY -m discell.experiments.external_criteria $1 \
    --dataset $2 --run $3
}

final_extras() {  # (a, cont.) bootstrap, external criteria, GO, subtype recovery
  local ds=$1 r read args=() dev
  for r in finalL_s0 finalL_s1 finalL_s2; do
    fitted $ds $r || continue
    gstep $ds "$(marker bootstrap $ds $r)" $PY -m discell.experiments.bootstrap \
      --dataset $ds --run $r --n 1000
  done
  for r in finalL_s0 finalL_s1 finalL_s2; do
    for read in signalling-share mi-quadrant axis-test; do
      # the external reads on disk for finalL (the lineage queue's set, plus
      # any added since, e.g. the GSE signalling share of the sensF controls)
      [ -f data/datasets/$ds/experiments/external_${read//-/_}_$r.json ] && ext $read $ds $r
    done
  done
  for r in finalL_s0 finalL_s1 finalL_s2; do fitted $ds $r && args+=(--run $r); done
  [ ${#args[@]} -gt 0 ] && cpu_step "$(marker go_localisation $ds finalL)" \
    $PY -m discell.experiments.go_localisation --dataset $ds "${args[@]}" \
    --out-stem go_localisation_lineage
  for r in finalL_s0 finalL_s1 finalL_s2; do
    [ -f data/datasets/$ds/experiments/go_localisation_$r.json ] && cpu_step \
      "$(marker go_localisation_run $ds $r)" $PY -m discell.experiments.go_localisation \
      --dataset $ds --run $r --out-stem go_localisation_$r
  done
  # subtype recovery: the FF encode does not fit beside the other queues' jobs
  # on a card, so FF runs on the CPU (~40 GB RAM, as its own queue ran it)
  local subtype=(subtype_recovery --dataset $ds --run finalL_s0 finalL_s1 finalL_s2 --force)
  if [ "$ds" = "$FF" ]; then
    wait_ram 45
    cpu_step "$(marker subtype $ds finalL)" $PY2 -m discell.experiments."${subtype[0]}" \
      "${subtype[@]:1}" --device cpu
  else
    gstep $ds "$(marker subtype $ds finalL)" $PY2 -m discell.experiments."${subtype[0]}" \
      "${subtype[@]:1}" --device cuda
  fi
}

wait_ram() {  # wait_ram <GB>: until that much RAM is available (a CPU FF read)
  local n=0
  while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt "$1" ]; do
    [ $((n % 10)) -eq 0 ] && log "waiting for $1 GB of free RAM"
    n=$((n + 1)); sleep 60
  done
}

unc_reads() {  # (b) unc_reads <dataset> <run>
  local ds=$1 run=$2
  fitted $ds $run || return 0
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche
  gstep $ds "$(marker recon_modes $ds $run)" $PY2 -m discell.experiments.recon_modes \
    --dataset $ds --run $run
  if [ "$ds" = "$GS" ]; then
    gstep $ds "$(marker recon_modes_dual $ds $run)" $PY2 -m discell.experiments.recon_modes \
      --dataset $GS --run $run --eval-dataset $GD
  fi
  gstep $ds "$(marker transport_mean $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read mean
}

light_reads() {  # (c) light_reads <dataset> <run>: a sweep point
  local ds=$1 run=$2
  fitted $ds $run || { log "no fit for $ds/$run -- no reads"; return 0; }
  gstep $ds "$(marker validate $ds $run)" $PY -m discell.model.validate --dataset $ds \
    --run $run --analyses morans,niche
  gstep $ds "$(marker degeneracy $ds $run)" $PY -m discell.model.degeneracy --dataset $ds --run $run
  if [ "$ds" = "$GS" ]; then   # the held-out section at every kappa, before its grade
    gstep $ds "$(marker crossslide $ds $run)" $PY -m discell.model.crossslide --dataset $GS \
      --run $run --eval-dataset $GD
  fi
  grade $ds $run
  gstep $ds "$(marker transport_mean $ds $run)" $PY -m discell.model.transport --dataset $ds \
    --run $run --read mean
}

sweep_tail() {  # (c, cont.) the per-dataset sweep reads
  local ds=$1 runs_list=() k s
  # shellcheck disable=SC2086
  gstep $ds "$(marker sweep_report $ds sweepL)" $PY -m discell.model.sweep --dataset $ds \
    --report-only --tag sweepL --param kappa --values $KAPPAS --seeds 0 1 2 \
    --label-key lineage ${SWEEP_FLAGS[$ds]} --epochs 500 --patience 40
  # shellcheck disable=SC2086
  gstep $ds "$(marker kappa_survival $ds sweepL)" $PY -m discell.model.validate --dataset $ds \
    --sweep-tag sweepL --kappas $KAPPAS --seeds 0 1 2 \
    --analyses kappa_survival,morans,niche --survival-reference finalL_s0
  cpu_step "$(marker sweep_transport_json $ds sweepL)" $PY $ROOT/merge_sweep_transport.py $ds
  runs_list=(finalL_s0 finalL_s1 finalL_s2)
  for k in $SWEEP_KAPPAS; do for s in 0 1 2; do runs_list+=(sweepL_k${k}_s$s); done; done
  gstep $ds "$(marker marker_pairs $ds finalL_sweepL)" $PY2 -m discell.experiments.marker_pairs \
    --dataset $ds --run "${runs_list[@]}" --kappa --overwrite
}

lane() {  # lane <dataset>: everything of one dataset, in the order above
  local ds=$1 s k
  for s in 0 1; do reference $ds uncontrolledL_s$s; done
  for s in 0 1 2; do final_reads $ds finalL_s$s; done
  compare_group $ds finalL_s0 finalL_s1 finalL_s2
  final_extras $ds
  for s in 0 1; do unc_reads $ds uncontrolledL_s$s; done
  for s in 0 1 2; do for k in $SWEEP_KAPPAS; do light_reads $ds sweepL_k${k}_s$s; done; done
  sweep_tail $ds
  log "lane $ds finished"
}

# -- (d) the baselines ------------------------------------------------------------------
baselines_done() {  # the baselines queue has finished (or is not running)
  pgrep -f queue_2026-09-28_baselines_lineage.sh > /dev/null && return 1
  return 0
}

battery_columns() {  # battery_columns <dataset> [<config dataset>]: DisCell, slot 0 last
  local ds=$1 from=${2:-$1} r extra=()
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  for r in finalL_s1 finalL_s2 finalL_s0; do
    fitted $from $r || continue
    gstep $ds "$(marker battery_discell $ds $r)" flock $TLOCK $PY2 -m \
      discell.experiments.baseline_battery --dataset $ds --discell-run $r \
      --config-run finalL_s0 --tag _lineage "${extra[@]}"
  done
}

baseline() {  # baseline <dataset> <config dataset> <method> <latents dir>: column + grade
  local ds=$1 from=$2 method=$3 dir=$4 extra=() tag
  [ "$from" != "$ds" ] && extra=(--config-from $from)
  [ -f "$dir/latents.h5ad" ] || { log "$ds '$method': no latents at $dir -- not scored"; return 0; }
  tag=$(printf '%s' "$method" | tr -c 'A-Za-z0-9=.' '_')
  locked_cpu_step "$(marker battery_$tag $ds finalL_s0)" $PY2 -m \
    discell.experiments.baseline_battery --dataset $ds --method "$method" \
    --latents "$dir/latents.h5ad" --config-run finalL_s0 --tag _lineage "${extra[@]}"
  # shellcheck disable=SC2086
  locked_cpu_step "$(marker probe_$tag $ds finalL_s0)" $PY2 -m \
    discell.experiments.probe_regrade --dataset $ds --run finalL_s0 \
    --baseline-latents "$dir/latents.h5ad" --method "$method" --force $REFS "${extra[@]}"
}

pick() {  # pick <dataset> <config> <lineage method> <lineage dir> [<old method> <old dir>]...
  # the lineage refit where its latents are on disk, else every old-label fit
  local ds=$1 from=$2 m=$3 d=$4; shift 4
  if [ -f "$d/latents.h5ad" ]; then baseline $ds $from "$m" "$d"; return; fi
  while [ $# -ge 2 ]; do baseline $ds $from "$1" "$2"; shift 2; done
}

cellina() {  # Cellina (installed in parallel): scored only when its latents are on
  # disk and baseline_battery.LATENT_KEYS names its intrinsic key; else skipped
  local d name ds from rest method
  if ! $PY -c "from discell.experiments.baseline_battery import LATENT_KEYS; import sys; sys.exit(0 if 'cellina' in LATENT_KEYS else 1)" 2>/dev/null; then
    ls $BRES/cellina/*/latents.h5ad > /dev/null 2>&1 \
      && log "(d) Cellina latents on disk but LATENT_KEYS has no 'cellina' entry -- skipped (scored on its own)"
    return 0
  fi
  for d in $BRES/cellina/*/; do
    [ -f "$d/latents.h5ad" ] || continue
    name=$(basename $d); ds=""
    for c in $GD $GS $OV $LU $FF; do case $name in $c|${c}_*) ds=$c; break ;; esac; done
    [ -n "$ds" ] || { log "(d) Cellina $name: no dataset of ours -- skipped"; continue; }
    from=$ds; [ "$ds" = "$GD" ] && from=$GS
    rest=${name#$ds}; rest=${rest#_}
    method="Cellina${rest:+ ($rest)}"
    baseline $ds $from "$method" "${d%/}"
  done
}

stage_d() {
  local n=0 ds pats=('finalL_s*' 'uncontrolledL_s*' 'sweepL_*')
  until baselines_done; do
    [ $((n % 30)) -eq 0 ] && log "(d) waiting for the baselines queue to finish"
    n=$((n + 1)); sleep 60
  done
  log "(d) baselines under the mask"
  battery_columns $GS
  baseline $GS $GS resolVI $RVI/$GS
  pick $GS $GS "SIMVI (lineage)" $BRES/simvi/${GS}_lineage SIMVI $BRES/simvi/$GS
  pick $GS $GS "MintFlow (lineage)" $BRES/mintflow/${GS}_lineage \
    MintFlow $BRES/mintflow/$GS "MintFlow (w=600, 5/50 epochs)" $BRES/mintflow/${GS}_w600 \
    "MintFlow (50 epochs, w=600)" $BRES/mintflow/${GS}_full
  battery_columns $GD $GS
  baseline $GD $GS resolVI $RVI/$GD
  pick $GD $GS "SIMVI (lineage, fit on this section)" $BRES/simvi/${GD}_lineage \
    "SIMVI (fit on this section)" $BRES/simvi/${GD}_fit_on_target
  pick $GD $GS "MintFlow (lineage, transfer)" $BRES/mintflow/${GD}_lineage \
    "MintFlow (transfer, 5/50 epochs)" $BRES/mintflow/${GD}_transfer \
    "MintFlow (transfer, 50 epochs)" $BRES/mintflow/${GD}_transfer_full
  battery_columns $OV; baseline $OV $OV resolVI $RVI/$OV
  battery_columns $LU; baseline $LU $LU resolVI $RVI/$LU
  battery_columns $FF; baseline $FF $FF resolVI $RVI/$FF
  pick $FF $FF "SIMVI (lineage, 100k-cell window)" $BRES/simvi/${FF}_lineage \
    "SIMVI (100k-cell window)" $BRES/simvi/$FF
  cellina
  # re-verdicts: every DisCell record's fraction of the re-graded resolVI
  for ds in $ALL; do
    # shellcheck disable=SC2086
    locked_cpu_step "$(marker reverdict $ds excl)" $PY2 -m discell.experiments.probe_regrade \
      --dataset $ds --reverdict --runs "${pats[@]}" $REFS
    # shellcheck disable=SC2086
    locked_cpu_step "$(marker probe_table $ds lineage_final)" $PY2 -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
    # shellcheck disable=SC2086
    locked_cpu_step "$(marker probe_table $ds lineage_sweep)" $PY2 -m discell.experiments.probe_regrade \
      --dataset $ds --table lineage_sweep --runs 'sweepL_*' 'finalL_s*' $REFS
  done
  # shellcheck disable=SC2086
  locked_cpu_step "$(marker reverdict $GD excl)" $PY2 -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --reverdict --runs "${pats[@]}" $REFS
  # shellcheck disable=SC2086
  locked_cpu_step "$(marker probe_table $GD lineage_final)" $PY2 -m discell.experiments.probe_regrade \
    --dataset $GD --config-from $GS --table lineage_final --baselines \
    --runs 'finalL_s*' 'uncontrolledL_s*' $REFS
  log "(d) finished"
}

# -- (e) the tables ---------------------------------------------------------------------
sensf_done() {
  pgrep -f queue_2026-09-28_sens_final.sh > /dev/null && return 1
  return 0
}

sensf_reread() {  # the sensF reads whose files predate the mask, re-read under it
  local ds run step
  $PY $ROOT/sensf_scan.py "$SENSF_ROOT" > $ROOT/SENSF_PRE_MASK.tsv 2> $QL/sensf_scan.log
  log "sensF: $(wc -l < $ROOT/SENSF_PRE_MASK.tsv) reads predate the mask (SENSF_PRE_MASK.tsv)"
  while IFS=$'\t' read -r step ds run <&3; do
    case $step in
      validate) gstep $ds "$(marker sensF_validate $ds $run)" $PY -m discell.model.validate \
                  --dataset $ds --run $run --analyses morans,niche ;;
      degeneracy) gstep $ds "$(marker sensF_degeneracy $ds $run)" $PY -m discell.model.degeneracy \
                  --dataset $ds --run $run ;;
      probe_regrade) # shellcheck disable=SC2086
                  gstep $ds "$(marker sensF_probe_regrade $ds $run)" $PY2 -m \
                  discell.experiments.probe_regrade --dataset $ds --run $run --force $REFS ;;
      transport) gstep $ds "$(marker sensF_transport $ds $run)" $PY -m discell.model.transport \
                  --dataset $ds --run $run --read both --hvg 1000 ;;
      transport_band) gstep $ds "$(marker sensF_transport_band $ds $run)" $PY -m \
                  discell.model.transport --dataset $ds --run $run --read both --hvg 1000 \
                  --niche-source tumour-band ;;
      atlas) gstep $ds "$(marker sensF_atlas $ds $run)" $PY -m discell.model.atlas \
                  --dataset $ds --run $run ;;
      ext_signalling-share) gstep $ds "$(marker sensF_ext_ss $ds $run)" $PY -m \
                  discell.experiments.external_criteria signalling-share --dataset $ds --run $run ;;
      go_localisation) cpu_step "$(marker sensF_go $ds $run)" $PY -m \
                  discell.experiments.go_localisation --dataset $ds --run $run \
                  --out-stem go_localisation_$run ;;
      w_deviation) gstep $ds "$(marker sensF_w_deviation $ds $run)" $PY -m \
                  discell.experiments.w_deviation --dataset $ds --run $run ;;
      recon_modes) gstep $ds "$(marker sensF_recon_modes $ds $run)" $PY2 -m \
                  discell.experiments.recon_modes --dataset $ds --run $run ;;
      bootstrap) gstep $ds "$(marker sensF_bootstrap $ds $run)" $PY -m \
                  discell.experiments.bootstrap --dataset $ds --run $run --n 1000 ;;
    esac
  done 3< $ROOT/SENSF_PRE_MASK.tsv
  # every sensF record's guard fractions against the re-graded references
  for ds in $GS $OV $FF; do
    # shellcheck disable=SC2086
    cpu_step "$(marker sensF_reverdict $ds excl)" $PY2 -m discell.experiments.probe_regrade \
      --dataset $ds --reverdict --runs 'sensF_*' $REFS
  done
}

stage_e() {
  local n=0
  cpu_step envelope_finalL_excl $PY scripts/envelope_tables.py --datasets $ALL \
    --runs finalL_s0 finalL_s1 finalL_s2 --at best --ci \
    --combined $ROOT/envelope_tables_finalL.md
  until sensf_done; do
    [ $((n % 30)) -eq 0 ] && log "(e) waiting for the sensitivity queue to finish"
    n=$((n + 1)); sleep 60
  done
  if [ "$SENSF_REREAD" = 1 ]; then sensf_reread; else log "SENSF_REREAD=0: sensF pre-mask reads left"; fi
  cpu_step sensF_tables_excl $PY scripts/sensF_tables.py --quiet
  log "(e) finished"
}

# -- main -------------------------------------------------------------------------------
rmdir $LOCK/gpu* 2>/dev/null     # one instance at a time: any lock is stale
rm -f $WAIT/ff.* 2>/dev/null
trap 'rmdir $LOCK/gpu* 2>/dev/null; rm -f $WAIT/ff.* 2>/dev/null' EXIT
log "queue start (tag excl; FF gate $NEED_FF MiB, others $NEED_OV MiB, GPUs: $GPUS)"
if [ "${EXCL_DRYRUN:-0}" = 1 ]; then
  for ds in $ALL; do
    for r in uncontrolledL_s0 uncontrolledL_s1 finalL_s0 finalL_s1 finalL_s2; do
      fitted $ds $r || log "dry run: MISSING $ds/$r"
    done
    for s in 0 1 2; do for k in $SWEEP_KAPPAS; do
      fitted $ds sweepL_k${k}_s$s || log "dry run: MISSING $ds/sweepL_k${k}_s$s"; done; done
  done
  $PY $ROOT/sensf_scan.py "$SENSF_ROOT" > $ROOT/SENSF_PRE_MASK.dryrun.tsv
  log "dry run: $(wc -l < $ROOT/SENSF_PRE_MASK.dryrun.tsv) sensF reads predate the mask so far"
  exit 0
fi
backup_pre_mask
incl_companion || { log "the incl. Unassigned companion failed -- stopping before any masked read"; exit 1; }

PIDS=""
lane $FF & PIDS="$PIDS $!"            # the long FF pole first
(sleep 20; lane $OV) & PIDS="$PIDS $!"
(sleep 40; lane $GS) & PIDS="$PIDS $!"
(sleep 60; lane $LU) & PIDS="$PIDS $!"
# shellcheck disable=SC2086
wait $PIDS
log "all dataset lanes finished"
stage_d
stage_e
{
  echo
  echo "## Queue finished $(date '+%F %T')"
  echo
  echo "Failed steps: $(ls $FAILED | tr '\n' ' ')"
  echo "Steps run: $(wc -l < $ROOT/STEPS.tsv); GPU seconds: $(awk -F'\t' '$2 !~ /^(envelope|go_|sweep_transport|sensF_tables|battery_resolVI|battery_SIMVI|battery_MintFlow|probe_resolVI|probe_SIMVI|probe_MintFlow|reverdict|probe_table)/ {s+=$4} END {print s}' $ROOT/STEPS.tsv)"
  echo "Tables: data/datasets/<ds>/experiments/envelope_table_ci_at_best.md (mask) and"
  echo "envelope_table_ci_at_best_incl_unassigned.md (the one-time companion);"
  echo "$ROOT/envelope_tables_finalL_at_best_ci{,_incl_unassigned}.md"
} >> $ROOT/AGENT_REPORT.md
log "queue finished"
