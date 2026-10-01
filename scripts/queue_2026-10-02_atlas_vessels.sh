#!/bin/bash
# The vessel landmark on the lung sections (devlog "Vessel landmark misses
# lung 'EC' labels (bug; fix queued 2026-10-01)"). discell/model/labels.py
# is_endothelial now also matches the lung vocabulary's whole-word "EC" token
# (EC aerocyte capillary, EC general capillary, EC venous, Lymphatic EC), so
# landmark_inventory builds a "vasculature" class on lung FFPE and the TMA
# core. The ovarian vocabularies match exactly as before ("Endothelial cells"
# only; checked on the assembled ovarian FFPE data: identical inventory).
#
# Steps:
#  (0) back up data/datasets/<ds>/runs/finalL_s{0,1,2}/atlas -> atlas_prevessel
#      (once; never overwritten);
#  (1) atlas --compare-runs <other seeds> for finalL_s0-s2 on lung FFPE and
#      the GSE TMA core (the atlas.json-producing command, as in
#      scripts/queue_2026-09-28_unassigned.sh compare_group). The atlas needs a
#      forward pass (collect_latents), so: waits until the breakdown-gaps and
#      no-image queues have ended, then GPU 0 ONLY (never GPU 1);
#  (2) CPU: recomb_fig_programmes.py; recomb_fig_hallmarks.py only if any
#      programme's loadings or hallmark labels moved (they should not: the
#      landmarks enter the drivers only); then submission_paper/recomb/build.sh;
#  (3) $ROOT/READOUT.md: landmark classes, driver R2, leading programme and
#      hallmarks, before (atlas_prevessel) and after.
# Idempotent: done/<step> markers; a relaunch resumes. No .tex text is edited.
#
#   mkdir -p scripts/logs/atlas_vessels_2026-10-02 && setsid nohup \
#     bash scripts/queue_2026-10-02_atlas_vessels.sh \
#     >> scripts/logs/atlas_vessels_2026-10-02/queue.log 2>&1 < /dev/null & disown
set -u
cd /home/rmolen/github/DisCell
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISCELL_EVAL_INCLUDE_UNASSIGNED          # the Unassigned mask is on

GS=gse315411_pdltma06_11_prime_solo
LU=xenium_prime_human_lung_cancer_ffpe
DATASETS="$LU $GS"
SEEDS="finalL_s0 finalL_s1 finalL_s2"
ROOT=scripts/logs/atlas_vessels_2026-10-02
QL=$ROOT/logs; DONE=$ROOT/done
mkdir -p $QL $DONE
PY="uv run python"
NEED_MB=10000
log() { echo "[$(date '+%F %T')] $*"; }

step() {  # step <marker> <cmd...>; env CUDA_VISIBLE_DEVICES set by caller
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then log "skip $name (done)"; return 0; fi
  local t0=$SECONDS rc
  log "start $name"
  "$@" > $QL/$name.log 2>&1; rc=$?
  log "done  $name (exit $rc, $((SECONDS - t0))s)"
  [ $rc -eq 0 ] && touch "$DONE/$name"
  return $rc
}

log "queue start (pid $$)"
# -- (0) backups ------------------------------------------------------------------
for ds in $DATASETS; do for run in $SEEDS; do
  d=data/datasets/$ds/runs/$run
  if [ ! -d $d/atlas_prevessel ]; then
    cp -a $d/atlas $d/atlas_prevessel && log "backup $d/atlas_prevessel" \
      || { log "BACKUP FAILED for $d -- stopping"; exit 1; }
  fi
done; done

# -- (1) atlas on GPU 0, after the two queues ---------------------------------------
while pgrep -f '[q]ueue_2026-10-01_breakdown_gaps.sh' > /dev/null \
   || pgrep -f '[q]ueue_2026-10-01_noimg.sh' > /dev/null; do
  log "waiting for the breakdown-gaps / no-image queues"; sleep 600
done
while :; do
  free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "${free:-0}" -ge $NEED_MB ] && break
  log "GPU 0 has ${free} MiB free (< $NEED_MB); waiting"; sleep 300
done
for ds in $DATASETS; do
  for run in $SEEDS; do
    others=$(echo $SEEDS | tr ' ' '\n' | grep -vx $run | tr '\n' ' ')
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 step atlas_${ds}_${run} \
      $PY -m discell.model.atlas --dataset $ds --run $run --compare-runs $others \
      || log "FAIL atlas $ds $run; see $QL/atlas_${ds}_${run}.log"
  done
done

# -- (2) figures and the RECOMB build (CPU) -------------------------------------------
export CUDA_VISIBLE_DEVICES=
SRC=submission_paper/aistats/figures/src
step fig_programmes $PY $SRC/recomb_fig_programmes.py \
  || log "FAIL fig_programmes; see $QL/fig_programmes.log"
# hallmarks figure only if a programme's loadings or labels moved
if $PY - $DATASETS <<'EOF' > $QL/hallmark_check.log 2>&1
import json, sys
import numpy as np
moved = []
for ds in sys.argv[1:]:
    for run in ("finalL_s0", "finalL_s1", "finalL_s2"):
        d = f"data/datasets/{ds}/runs/{run}"
        old = json.load(open(f"{d}/atlas_prevessel/atlas.json"))
        new = json.load(open(f"{d}/atlas/atlas.json"))
        po, pn = np.load(f"{d}/atlas_prevessel/programs.npy"), np.load(f"{d}/atlas/programs.npy")
        same_load = po.shape == pn.shape and np.allclose(po, pn, atol=1e-6)
        same_hm = [p["hallmarks"] for p in old["programs"]] == [p["hallmarks"] for p in new["programs"]]
        print(ds, run, "loadings same" if same_load else "LOADINGS MOVED",
              "hallmarks same" if same_hm else "HALLMARKS MOVED")
        if not (same_load and same_hm):
            moved.append((ds, run))
sys.exit(1 if moved else 0)
EOF
then log "programmes and hallmarks unchanged -> recomb_fig_hallmarks.py not re-run"
else
  log "programmes or hallmarks moved (see $QL/hallmark_check.log) -> re-running recomb_fig_hallmarks.py"
  step fig_hallmarks env DISCELL_VOCAB=recomb $PY $SRC/recomb_fig_hallmarks.py \
    || log "FAIL fig_hallmarks; see $QL/fig_hallmarks.log"
fi
rm -f $DONE/build_recomb
step build_recomb bash submission_paper/recomb/build.sh \
  || log "FAIL build_recomb; see $QL/build_recomb.log"

# -- (3) read-out ---------------------------------------------------------------------
$PY - $ROOT $DATASETS <<'EOF' > $QL/readout.log 2>&1 || log "FAIL readout; see $QL/readout.log"
import json, re, sys
from pathlib import Path
import numpy as np
import anndata as ad
from discell.model.labels import is_endothelial

root, datasets = Path(sys.argv[1]), sys.argv[2:]
seeds = ["finalL_s0", "finalL_s1", "finalL_s2"]
L = ["# Vessel landmark on the lung sections: atlas re-run (2026-10-02)", "",
     "Devlog: \"Vessel landmark misses lung 'EC' labels (bug; fix queued 2026-10-01)\". "
     "Before = `atlas_prevessel/`, after = `atlas/`. Queue: "
     "`scripts/queue_2026-10-02_atlas_vessels.sh`.", ""]

L += ["## Endothelial matcher per section (lineage labels)", "",
      "| section | labels matched by is_endothelial |", "|---|---|"]
for f in sorted(Path("../DisCell-baselines/data/lineage").glob("*.h5ad")):
    a = ad.read_h5ad(f, backed="r")
    cats = sorted(map(str, a.obs["cell_type"].astype("category").cat.categories))
    L.append(f"| {f.stem} | {', '.join(c for c in cats if is_endothelial(c)) or '(none)'} |")
L.append("")

def fmt(x):
    return "--" if x is None else f"{x:.3f}"

for ds in datasets:
    L += [f"## {ds}", ""]
    for run in seeds:
        d = Path(f"data/datasets/{ds}/runs/{run}")
        old = json.loads((d / "atlas_prevessel/atlas.json").read_text())
        new = json.loads((d / "atlas/atlas.json").read_text())
        L += [f"### {run}", "",
              f"- landmark classes before: {old['landmark_classes']}",
              f"- landmark classes after:  {new['landmark_classes']}"]
        logf = root / "logs" / f"atlas_{ds}_{run}.log"
        if logf.exists():
            for line in logf.read_text().splitlines():
                if re.search(r"landmark class|pericyte co-location", line):
                    L.append(f"  - log: `{line.split(' - ')[-1].strip()}`")
        po = np.load(d / "atlas_prevessel/programs.npy")
        pn = np.load(d / "atlas/programs.npy")
        same = po.shape == pn.shape and np.allclose(po, pn, atol=1e-6)
        L += [f"- rank before/after: {old['rank']} / {new['rank']}; programme "
              f"loadings {'identical' if same else 'CHANGED (max |diff| %s)' % (np.abs(po - pn).max() if po.shape == pn.shape else 'shape')}",
              "", "| programme | block | marginal R2 before | after | unique R2 before | after |",
              "|---|---|---|---|---|---|"]
        for k, (p_o, p_n) in enumerate(zip(old["programs"], new["programs"])):
            do, dn = p_o["drivers"], p_n["drivers"]
            for b in dn["marginal"]:
                L.append(f"| {k} | {b} | {fmt(do['marginal'].get(b))} | {fmt(dn['marginal'][b])} "
                         f"| {fmt(do['partial'].get(b))} | {fmt(dn['partial'][b])} |")
            L.append(f"| {k} | joint | {fmt(do['joint'])} | {fmt(dn['joint'])} | | |")
        L.append("")
        lead_o, lead_n = old["programs"][0], new["programs"][0]
        sig = lambda p: [h["hallmark"] for h in p["hallmarks"] if h["significant"]][:3]
        top = lambda p: [g for g, _ in p["signature_high"][:5]]
        L += [f"- leading programme (0): share {lead_o['variance_share']:.3f} -> {lead_n['variance_share']:.3f}; "
              f"top genes {top(lead_o)} -> {top(lead_n)}",
              f"- leading programme hallmarks (top 3 significant): {sig(lead_o)} -> {sig(lead_n)}"
              + ("" if sig(lead_o) == sig(lead_n) else "  **CHANGED**"),
              f"- all programmes' hallmark records "
              + ("identical" if [p["hallmarks"] for p in old["programs"]] == [p["hallmarks"] for p in new["programs"]]
                 else "**CHANGED**"), ""]

L += ["## Figures and build", ""]
for name in ("hallmark_check", "fig_programmes", "build_recomb"):
    f = root / "logs" / f"{name}.log"
    if f.exists():
        L += [f"### {name}", "```", *f.read_text().splitlines()[-40:], "```", ""]
L.append("The S5 sentence on the vessel gap and the drivers sentence in 03_4 are NOT edited by this queue.")
(root / "READOUT.md").write_text("\n".join(L) + "\n")
print("wrote", root / "READOUT.md")
EOF
log "queue end"
