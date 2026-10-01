# Brief for the supplement trimmers (2026-10-01)

Goal: **publication at RECOMB.** The supplement must be lean, easy to navigate, and truthful. Read SUPPLEMENT_TRIM_MAP.md (approved by the author) and follow your row of its "Split among four parallel agents" table and its rules exactly. Also read STYLE.md and the main text (sections/*.tex) so you know what the pointers expect.

Hard rules:
- Edit only your files. Keep every label in the pointer inventory. No new numbers: use only numbers already in the files, in generated tables, or in result files you open and cite in a % comment.
- Every honest limitation and negative result survives, condensed at most. Never cut something to look better.
- Development history becomes at most one sentence: what was chosen and why.
- Each result is shown once, as a table OR a figure, unless both carry different information. Captions are as short as needed to read the float.
- Vocabulary per STYLE.md. No code references in rendered text.
- Build into a private directory to avoid clashes (`latexmk -pdf -interaction=nonstopmode -outdir=_build_<you> supplement.tex`). The assembler (agent D) runs ./build.sh at the end.
- Report in `_reports/trim_<you>.md`: pages before and after, every float cut or merged, labels kept, cross-file requests, and any item you could not resolve.

Additions decided after the map (author-approved):
1. **Agent D:** a compact **contrast-definitions table** in the sweep section (new S7). One row per claimed contrast: the name as in main Table 4 (tab:kappa-star), its formula, what positive means, allocation or biological, and the sections it is read on. Take each formula from S5 (readouts) and the breakdown code's definitions (/home/rmolen/github/DisCell/scripts/logs/breakdown_2026-09-29/FAMILY.md). Then: **cut tab:breakdown** (it duplicates main Table 4; retarget its references to tab:kappa-star); keep **fig:breakdown-data** with a shorter caption that points to the definitions table; **merge fig:kappa-sweep and tab:breakdown-traj** into one diagnostics-and-trajectories item (prefer the figure, plus a compact table only for numbers the figure cannot show). Also add a small **probe-across-κ** read: the main text says the held-out probe is checked at each κ. Source: data/datasets/<ds>/experiments/probe_regrade_lineage_sweep.{json,md}; as a short table or a sentence with the range, with a source comment.
2. **Agent B:** the **subtype metric gap.** The main text quotes balanced accuracies (0.95/0.94 for the ovarian fibroblasts) while app:subtype reports AUC. Find the balanced-accuracy source (data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/subtype_recovery.{json,md}) and report the values BA-consistently in the condensed subtype subsection, or report that BA is not available and that the main text must switch to AUC.
