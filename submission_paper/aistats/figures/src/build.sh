#!/usr/bin/env bash
# Regenerate every manuscript figure from the stored bundles.
# Run from anywhere inside the repository:  bash submission_paper/aistats/figures/src/build.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=submission_paper/aistats/figures/src
python "$SRC/fig_voronoi_face.py"      # fig:voronoi-face  (appendix)
python "$SRC/fig_contact_kernel.py"    # fig:contact-kernel (appendix)
python "$SRC/fig_graph_compare.py"     # fig:graph-compare (appendix)
python "$SRC/fig_model_tissue.py"      # fig:overview a, b (main text)
( cd "$SRC" && pdflatex -interaction=nonstopmode fig_model_graph.tex > /dev/null \
  && cp fig_model_graph.pdf ../ )      # fig:overview c (TikZ)
echo "all figures rebuilt in submission_paper/aistats/figures/"
