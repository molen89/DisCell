#!/usr/bin/env bash
# End-to-end shared-label run for GSE315411. Usage:  bash run_all.sh <gpu-index>
# Detach with:  setsid nohup bash run_all.sh 1 </dev/null >run.log 2>&1 & disown
# Stages that already have their output are skipped, so a rerun is cheap.
set -euo pipefail

GPU="${1:-1}"
ROOT="/home/rmolen/cellxgene_all_visium/highres_raw_with_images/GEO_Xenium/GSE315411"
WORK="$ROOT/annotate"
REFDIR="$WORK/references"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"

echo "GPU=$GPU  ROOT=$ROOT  $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

if [ ! -f "$WORK/query_raw.h5ad" ]; then
  "$PY" "$HERE/00_build_query.py" --root "$ROOT" --inspect
  "$PY" "$HERE/00_build_query.py" --root "$ROOT"
else echo "SKIP stage 0: query_raw.h5ad exists"; fi

if [ ! -f "$WORK/query_emb.h5ad" ]; then
  "$PY" "$HERE/01_embed_cluster.py" --root "$ROOT" --gpu "$GPU" --resolution 2.0 --max-epochs 30
else echo "SKIP stage 1: query_emb.h5ad exists"; fi

# wait for the references if the download is still running
while [ ! -f "$REFDIR/hlca_core.h5ad" ] || [ ! -f "$REFDIR/lungmap_cellref.h5ad" ]; do
  echo "waiting for references ... $(date -Is)"; sleep 120
done

if [ ! -f "$WORK/transfer_hlca.csv" ]; then
  "$PY" "$HERE/02_transfer.py" --root "$ROOT" --gpu "$GPU" \
      --ref "$REFDIR/hlca_core.h5ad" --ref-label-key ann_finest_level \
      --ref-batch-key dataset --ref-gene-col feature_name --ref-use-raw \
      --tag hlca --ref-subsample 300000 --query-epochs 30
else echo "SKIP stage 2 hlca"; fi

# CellRef's cell-type column is set after inspecting the h5ad: references/cellref_label.txt
CELLREF_LABEL="$(cat "$REFDIR/cellref_label.txt" 2>/dev/null || echo celltype_level3)"
echo "CellRef label column: $CELLREF_LABEL"
if [ ! -f "$WORK/transfer_cellref.csv" ]; then
  "$PY" "$HERE/02_transfer.py" --root "$ROOT" --gpu "$GPU" \
      --ref "$REFDIR/lungmap_cellref.h5ad" --ref-label-key "$CELLREF_LABEL" \
      --ref-batch-key donor_id --ref-gene-col feature_name --ref-use-raw \
      --tag cellref --query-epochs 30
else echo "SKIP stage 2 cellref"; fi

"$PY" "$HERE/03_name_and_report.py" --root "$ROOT" --tags hlca cellref --primary hlca
"$PY" "$HERE/04_write_cell_groups.py" --root "$ROOT" --tag hlca

echo "done $(date -Is). labels are in $WORK/labels_hlca.csv and beside each outs dir"
