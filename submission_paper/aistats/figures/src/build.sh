#!/usr/bin/env bash
# Regenerate every manuscript figure from the stored bundles.
# Run from anywhere inside the repository:  bash submission_paper/aistats/figures/src/build.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=submission_paper/aistats/figures/src
python "$SRC/fig_voronoi_face.py"      # fig:voronoi-face  (appendix)
python "$SRC/fig_contact_kernel.py"    # fig:contact-kernel (appendix)
python "$SRC/fig_graph_compare.py"     # fig:graph-compare (appendix)
python "$SRC/fig_breakdown.py"         # fig:breakdown (appendix, schematic)
python "$SRC/fig_batching.py"          # fig:batching (appendix)
python "$SRC/fig_kappa_bound.py"       # fig:kappa-bound (appendix, schematic)
python "$SRC/fig_mask_radius.py"       # fig:mask-radius (appendix)
python "$SRC/fig_kronos_umap.py"       # fig:kronos-umap (appendix; UMAP layouts cached in data/)
python "$SRC/fig_probe.py"             # fig:probe (appendix, schematic)
python "$SRC/fig_sections.py"          # fig:sections (appendix)
python "$SRC/fig_kl_maps.py"           # fig:kl-seeds (appendix); also fig_kl_maps, fig_kl_hist, not in the text since the 2026-09-29 trim
python "$SRC/fig_latents.py"           # fig:latents-umap (appendix; embeddings cached in data/)
python "$SRC/fig_model_tissue.py"      # fig:overview a, b (main text)
python "$SRC/fig_tradeoff.py"          # fig:tradeoff (leakage vs cycle R^2, per section; via scripts/paper_tables.py)
python "$SRC/fig_kappa_sweep.py"       # fig:kappa-sweep (main text; diagnostics over kappa; via scripts/paper_tables.py)
python "$SRC/fig_probe_data.py"        # fig:probe-data (main text; probe dumbbells, as tab:probe)
python "$SRC/fig_sensitivity.py"       # fig:sensitivity (main text; forest plot of tab:sensitivity's moves)
python "$SRC/fig_breakdown_data.py"    # fig:breakdown-data (main text; rerun when the breakdown queue writes the transport members)
python "$SRC/fig_planted_percell.py"   # fig:planted-percell (appendix)
python "$SRC/fig_transport_heldout.py" # fig:transport-heldout (appendix)
python "$SRC/fig_battery.py"           # fig:battery (as tab:battery)
python "$SRC/fig_synthetic_misspec.py" # fig:synthetic-misspec (appendix)
python "$SRC/fig_separation.py"        # fig:separation (main text; context vs intrinsic composition, as tab:context)
( cd "$SRC" && pdflatex -interaction=nonstopmode fig_model_graph.tex > /dev/null \
  && cp fig_model_graph.pdf ../ )      # fig:overview c (TikZ)
echo "all figures rebuilt in submission_paper/aistats/figures/"
