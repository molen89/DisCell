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
python "$SRC/recomb_fig1_overview.py"  # RECOMB fig:overview (a: real tissue + morphology inset; b-d schematic)
python "$SRC/recomb_tradeoff.py"     # RECOMB fig:tradeoff (same data; axis worded "residual niche signal")
python "$SRC/recomb_fig_programmes.py"  # RECOMB fig:programmes (leading response programme per section; GPU-free, reads atlases + kl_maps caches)
DISCELL_VOCAB=recomb python "$SRC/recomb_fig_hallmarks.py"  # RECOMB fig:hallmarks (supplement; hallmark AUC of every programme, from the finalL atlases)
python "$SRC/fig_kappa_sweep.py"       # fig:kappa-sweep (main text; diagnostics over kappa; via scripts/paper_tables.py)
python "$SRC/fig_probe_data.py"        # fig:probe-data (main text; probe dumbbells, as tab:probe)
python "$SRC/fig_sensitivity.py"       # fig:sensitivity (main text; forest plot of tab:sensitivity's moves)
python "$SRC/fig_breakdown_data.py"    # fig:breakdown-data (main text; rerun when the breakdown queue writes the transport members)
python "$SRC/fig_planted_percell.py"   # fig:planted-percell (appendix)
python "$SRC/fig_transport_heldout.py" # fig:transport-heldout (appendix)
python "$SRC/fig_battery.py"           # fig:battery (as tab:battery)
python "$SRC/fig_synthetic_misspec.py" # fig:synthetic-misspec (appendix)
python "$SRC/fig_separation.py"        # fig:separation (main text; context vs intrinsic composition, as tab:context)
DISCELL_VOCAB=recomb python "$SRC/fig_breakdown_data.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_breakdown_data
DISCELL_VOCAB=recomb python "$SRC/fig_breakdown.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_breakdown
DISCELL_VOCAB=recomb python "$SRC/fig_kappa_sweep.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_kappa_sweep
DISCELL_VOCAB=recomb python "$SRC/fig_sensitivity.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_sensitivity
DISCELL_VOCAB=recomb python "$SRC/fig_synthetic_misspec.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_synthetic_misspec
DISCELL_VOCAB=recomb python "$SRC/fig_batching.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_batching
DISCELL_VOCAB=recomb python "$SRC/fig_contact_kernel.py"  # RECOMB variant (spill-over / relocation wording) -> recomb_fig_contact_kernel
( cd "$SRC" && pdflatex -interaction=nonstopmode fig_model_graph.tex > /dev/null \
  && cp fig_model_graph.pdf ../ )      # fig:overview c (TikZ)
( cd "$SRC" && pdflatex -interaction=nonstopmode recomb_fig_model_graph.tex > /dev/null \
  && cp recomb_fig_model_graph.pdf ../ )      # RECOMB fig:model-detail (TikZ, spill-over wording)
echo "all figures rebuilt in submission_paper/aistats/figures/"
