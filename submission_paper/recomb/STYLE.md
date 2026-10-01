# STYLE: rules for the section writers (RECOMB manuscript)

Read `STORY_MAP.md` first: it holds the claims, the evidence for each and the page budgets. Each file in `sections/` starts with its outline and budget as a comment; replace the `\todo{...}` line with the text.

## Files and build
- One file per unit: `sections/00_abstract.tex` … `sections/04_discussion.tex`; the supplement is in `supplement/S1_…tex` … `S9_…tex`.
- `main.tex` holds the `\section{Experiments}` heading, so each `03_*.tex` file starts with a `\subsection`.
- Build both PDFs with `./build.sh`. It prints the main-text page count (limit **10**, floats included; the title page and the bibliography don't count), the undefined references, and any label defined in both documents.
- Figures: `\includegraphics{fig_name}`. The graphics path is `../aistats/figures/`, so don't write `figures/` before the name.
- Generated tables: `\input{\tablesdir/name}` (`\tablesdir` = `../aistats/tables/generated`). Don't edit the generated files: they belong to the generators.
- Bibliography: `../aistats/references.bib` is used through a relative path, not copied. Add entries there.
- Code link: `\codelink` (prints "[URL to be added]").
- Editorial macros: `\pending{...}` (results still to come from runs that are running), `\todo{...}`, `\needsource`. `\flag` and `\readorder` do nothing here, so don't use them.

## Vocabulary (the same everywhere, main and supplement)
| Term | Meaning | Don't write |
|---|---|---|
| **spill-over** | transcripts assigned to a cell that belong to its neighbours | "leakage", "leak", "contamination" as a term |
| **spill-over fraction κ** | the share of a cell's transcripts that are spill-over (fixed, swept) | "leak fraction" |
| **residual niche signal** | what a held-out probe can still read about the neighbourhood from z, beyond chance | "residual leak", "leak" |
| **breakdown point κ\*** | the smallest spill-over fraction that explains a finding away | — |
| **relocation** (counterfactual) | predicting how a cell type's expression shifts when it is moved to another niche | "transport" |
| **intrinsic state** z / **response** w | the two latents | — |

Internal names (Read A, twin margin, mirror, transport) stay out of the main text, or are defined in one line where they are used. Mirror R² is defined once (§2.3 or §3.5) if it is used.

The copied supplement still says "leakage" and "transport": rename those words when the supplement is edited, never in the generated tables (they come from the generators).

## Notation (copied from the AISTATS method; macros in `macros.tex`)
| Symbol | Macro | Meaning |
|---|---|---|
| $\mathbf{x}_i$, $\ell_i$ | `\vx_i`, `\ell_i` | count vector over $G$ genes; its total (conditioned on, never modelled) |
| $\mathbf{z}_i \in \mathbb{R}^{d_z}$ | `\vz_i`, `\dz` | intrinsic state; prior $\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| $\mathbf{w}_i \in \mathbb{R}^{d_w}$ | `\vw_i`, `\dw` | response; prior $\mathcal{N}(m_\psi(\mathbf{c}_i,t_i), \sigma_w^2\mathbf{I})$, $\sigma_w = 1$ |
| $\mathbf{c}_i$ | `\vc_i` | context descriptor of the niche: deterministic, shared by co-located cells (attention over neighbour types ⊕ masked image embedding $\boldsymbol{\Phi}_i$ ⊕ isolation flag) |
| $t_i \in \{1,\dots,K\}$ | `t_i` | cell-type (lineage) label |
| $\kappa \in [0,1)$ | `\kappa` | spill-over fraction, fixed per fit and swept over a grid |
| $\boldsymbol{\rho}_i$ | `\vrho_i` | clean composition: $\boldsymbol{\rho}_i = \mathrm{softmax}(a(\mathbf{z}_i) + \mathbf{B}\mathbf{w}_i)$ |
| $\bar{\boldsymbol{\rho}}_i$ | `\rhobar_i` | spill-over influx: $\sum_{j\in\mathcal{N}_i}\beta_{ij}\boldsymbol{\rho}_j$ (AISTATS: "foreign influx") |
| $\mathbf{p}_i$ | `\vp_i` | fitted multinomial probability $(1-\kappa)\boldsymbol{\rho}_i + \kappa\bar{\boldsymbol{\rho}}_i$; $\mathbf{x}_i \mid \ell_i \sim \mathrm{Multinomial}(\ell_i, \mathbf{p}_i)$ |
| $\mathbf{B} \in \mathbb{R}^{G\times d_w}$ | `\vB` | loadings: column $k$ is a gene programme |
| $a(\cdot)$ | `a` | decoder: the cell's baseline programme in log space (no type input) |
| $m_\psi(\mathbf{c}, t)$ | `\mpsi` | learned prior-mean network of the response |
| $\beta_{ij}$, $\mathcal{N}_i$ | `\beta_{ij}`, `\nbr{i}` | fixed spill-over kernel on the pruned Delaunay graph; neighbour set |
| $\mathbf{y}_i$, $\boldsymbol{\Phi}_i$ | `\vy_i`, `\vPhi_i` | neighbour-type composition; masked image embedding |
| $\alpha_z$, $\alpha_w$, $\alpha_a$ | `\alphaz`, `\alphaw`, `\alphaa` | weights of the z-divergence, the w-divergence and the invariance term (adversary) |
| $\omega$ | `\omega` | weight of the intrinsic path (the second bound, with w drawn from its prior) |
| $\boldsymbol{\mu}_{z,i}$, $\boldsymbol{\mu}_{w,i}$ | `\vmu_{z,i}` | posterior means (what the probes and readouts use) |

Use these symbols only; don't introduce new ones for the same objects. Vectors are bold through the macros (`\vz`, not `z`).

## Writing rules
- Zero to two numbers per claim. The full numbers go in tables and the supplement.
- Nothing is said twice: each strength where it holds, each limit once.
- The experiment subsections describe; the discussion claims and gives references and reasons.
- No references to code, scripts, file paths, run names or config keys in the text.
- Every number comes from a generated table or a result file. A number still waiting on runs goes in `\pending{...}`; a statement that needs a citation gets `\needsource`.
- The main text must stand alone: the supplement may not be read.
- Math is kept minimal; the full derivations are in the supplement.
- Respect the page budget at the top of your file.

## Labels
- Prefixes: `sec:` (sections), `fig:`, `tab:`, `eq:`, `prop:`, `def:`. Supplement sections use `app:`.
- Both documents import each other's labels **without a prefix** (xr-hyper), because the shared generated tables refer across documents (e.g. `headline.tex` → `tab:headline-full`). The two label namespaces must therefore stay **disjoint**; `build.sh` warns if they aren't.
- In the main text, refer to the supplement as `\suppref{app:...}` ("Supplement S3.2"), and to its floats with `\cref{fig:...}` ("Figure S4") or `\cref{tab:...}`. Supplement numbering is S1, S2, … for sections, figures, tables and equations.
- A generated table may be `\input` in only one of the two documents. The supplement currently inputs: battery, breakdown, breakdown_traj, cellina_cf, context, headline_full, kappa_sweep, planted_percell, probe, sensitivity, synthetic, timing, transport_heldout. If the main text needs one of these (for example `breakdown` for the κ\* table), move it out of the supplement file.
- New labels written in the supplement take an `S-` marker after the prefix (`fig:S-…`, `tab:S-…`, `eq:S-…`). The copied labels keep their names.

### Labels the supplement already defines (reserved: don't reuse them in the main text)
Sections: `app:derivations app:bound app:pathb app:posterior-reg app:gaussian-mi app:symmetries app:amplification app:rationale app:mirror app:graph app:image app:qw app:conditional app:kappa app:variances app:implementation app:architecture app:deviations app:tiles app:probe app:experimental app:sections app:lineage app:baselines app:assets app:inputs app:readouts-section app:readouts app:subtype app:full-results app:response app:moran app:cellina-cf app:latents app:kl-maps app:timing app:simulation app:synthetic app:planted-percell app:planted app:sensitivity-section app:sweep app:sensitivity app:recon-modes app:limitations`

Floats and equations: `fig:batching fig:breakdown fig:contact-kernel fig:graph-compare fig:kappa-bound fig:kappa-sweep fig:kl-seeds fig:kronos-umap fig:latents-main fig:latents-umap fig:mask-radius fig:planted-percell fig:probe fig:sections fig:sensitivity fig:synthetic-misspec fig:transport-heldout fig:voronoi-face tab:architecture tab:baselines tab:battery tab:breakdown tab:breakdown-traj tab:cellina-cf tab:contact tab:context tab:deviations tab:deviations2 tab:headline-full tab:inputs tab:kappa-sweep tab:kl-summary tab:lineage-clusters tab:lineage-ovarian tab:masking tab:moran tab:planted-gap tab:planted-percell tab:probe tab:recon-modes tab:sections tab:sensitivity tab:synthetic tab:timing tab:transport-heldout eq:jensen eq:gauss-kl eq:context-closed`

(`fig:latents-main` is in S6. If the latent figure moves into §3.3 as STORY_MAP plans, take the float out of S6 and keep the label in one place.)

### Labels the supplement expects from the main text
Define these where they fit, or retarget the reference in the supplement when you edit it.
| Label | Proposed home |
|---|---|
| `sec:intro`, `sec:setup`, `sec:method` | already defined (01, 03_1, 02) |
| `sec:generative`, `sec:setting`, `eq:gen-w`, `eq:gen-rho`, `eq:gen-p` | 02_model §2.1 |
| `sec:inference`, `sec:objective`, `sec:batching`, `eq:context-gat` | 02_model §2.2 |
| `sec:invariance`, `sec:selection`, `eq:probe`, `eq:penalty` | 02_model §2.3 (`eq:penalty` is the closed-form penalty; if it isn't in the main text, retarget it to the supplement) |
| `sec:sweep`, `prop:kappa-bound`, `def:breakdown` | 02_model §2.4 |
| `prop:symmetries` | AISTATS planned to move it to the supplement (`app:symmetries`); retarget |
| `fig:overview` | 02_model (Fig 1) |
| `tab:headline` | 03_3 (`\input{\tablesdir/headline}` defines it) |
| `sec:results-z`, `sec:results-w` | 03_3 (and 03_4 for the response) |
| `sec:results-sweep` | 03_7 |
| `fig:battery`, `fig:probe-data`, `fig:breakdown-data` | AISTATS main-text figures that STORY_MAP sends to the supplement or drops; retarget |
