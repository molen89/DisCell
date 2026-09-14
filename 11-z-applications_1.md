# 11 — z applications and their validation (handover)

Six applications of the clean intrinsic state `z`, each with a pre-registered
validation design and pass/fail criteria. Conventions inherit from `08` §1 (posterior
means, within type, spatial-block folds, reference rows, multi-seed). Validation
principle: never validate against a competing *method* — validate against sources that
cannot share our failure mode: **planted truth** (simulation on the real scaffold),
**orthogonal physics** (DAPI, nuclear masks, per-transcript nuclear flags, genome
contiguity), **artifact fingerprints** (dependences only an artifact can have), and
**leak-immune references** (dissociated scRNA-seq has no spatial neighbours). The
claims are about deltas — the dissolved/rejected calls must carry fingerprints, the
kept ones must not.

Context from `ablation_gat_type_only_s1`: atlas and transport are working (transport:
beats-both in 100/155 panels, median slope 0.92; atlas: 6/6 active, driver R² .92–.98).
One terminology fix carried forward: cycle-z R² 0.50 vs the "50-PC ceiling" 0.24 shows
that row is a **linear expression reference**, not a ceiling — rename it everywhere.
One review flag for the atlas: hallmark labels like SPERMATOGENESIS/ADIPOGENESIS on
tumour programs smell like a background problem — hypergeometric enrichment must use
the **5.1k panel as background**, never the genome; verify before any label ships.

## A4 — Decontaminated cycle call (run first: cheapest, hardest ground truth)

**Claim**: the z-based phase call rejects leak-induced false positives that the raw
Tirosh score makes near cycling neighbours.

Method: per-cell calls from the existing probe axes (`z·β_S`, `z·β_G2M`) vs raw-x
Tirosh scores; the contrast group is **raw-positive / z-negative** cells (suspected
contamination victims), vs both-positive and both-negative.

1. **DAPI ground truth (orthogonal physics)**: integrated nuclear DAPI intensity and
   nuclear area per cell (morphology image × nuclear mask — one extraction script).
   S/G2M cells carry ~2× DNA. Pass: both-positive cells shifted high; **raw-only
   positives distributionally ≈ G1** (KS vs both-negative, n.s.) while both-positive
   vs G1 is strongly separated; AUROC(call → DNA-content) higher for z than raw.
2. **Fingerprint**: within type, raw score regresses on neighbour-cycle exposure
   (β-weighted neighbours' cycle score), partialled on own z-call → positive slope =
   contamination; the z score must be flat (within its permutation band).
3. **Nuclear-only recomputation**: Tirosh on nucleus-flagged transcripts only (Prime
   per-transcript flags) — raw-only positives should lose their score; z-calls persist.
4. **Planted world**: cycle-like program + leak on the scaffold; phase-label recovery
   z vs raw at planted κ.

Deliverable: the DAPI violin figure (three groups), the exposure-slope table, planted
recovery. Fail state to report honestly: raw-only positives with S/G2-like DAPI would
mean z *under-calls* — that is a sensitivity loss, not a decontamination win.

**Redesign after the first HGSOC run (inconclusive — instrument, not z):**
- **Contrast population**: NOT the MKI67-high types (global Tirosh thresholds saturate
  there — 92% positive — and contamination is least distinguishable). Run in
  **post-mitotic types stratified by neighbour-cycle exposure**: the leak victims are
  stromal/immune cells at proliferative-niche edges; base rates unsaturated, DAPI
  contrast maximal.
- **Leg 2 null replaced — gene-split design (contamination-specific by physics)**:
  contamination transfers transcripts, niche co-clustering transfers state. Disjoint
  halves A/B of the cycle set: Δ = corr(own_A, nbr_A) − corr(own_A, nbr_B), averaged
  over random splits. Homophily ⇒ Δ ≈ 0 (state is set-independent); leakage inflates
  only same-set ⇒ Δ > 0. Check: ring-1 vs ring-2 exposure contrast (leak is one-hop).
- **Planted world promoted to primary adjudicator for the mechanism claim**; the
  gene-split fingerprint is the real-data leg. DAPI retained, group-level only (FFPE
  sectioning truncates nuclei — integrated intensity is a weak ploidy proxy; the
  split-half reliability S = 0.22 warned of the target's own noise).
- **Pre-registered honest outcome**: if the victim-population test also shows nothing,
  report "leak-induced cycle false-positives are rare at κ = 0.1 on this slide" — a
  finding consistent with 09's aggregate-small picture, not a failure. No new slide
  until the redesign has run — the flaw was design, not tissue.

## A1 — State discovery: dissolving mirage states

**Claim**: clustering raw x within type yields states that are leak/niche artifacts;
z keeps the real ones and dissolves the fakes.

Method: per type ≥ 5k cells: Leiden on within-type PCA(raw x) vs on `mu_z`, resolution
chosen per side by a fixed stability procedure (no matching by hand); cross-side
matching by Jaccard; **dissolved** = raw cluster with max Jaccard < 0.3 to any z
cluster.

Per dissolved state, the fingerprint battery (each vs the kept-state distribution):
1. **y-predictability**: held-out ridge of the state's signature score on neighbour
   composition — artifact states are composition-predictable.
2. **Source match**: defining genes vs adjacent types' marker sets (hypergeometric,
   panel background).
3. **Distance decay**: membership rate vs distance to nearest source-type cell —
   artifact signature is decay on the ≤ 20–30 µm contamination scale.
4. **Segmentation perturbation**: re-derive raw counts nucleus-only (per-transcript
   flags) and recluster — artifact prevalence tracks the expansion; real states are
   invariant.
5. **Leak-immune reference**: public ovarian carcinoma scRNA-seq (pin dataset + DOI):
   signature scoring — dissolved states absent, kept states present.
6. **Planted world**: planted within-type substates + leak; z recovers the planted
   count (ARI), raw inflates it.

Pass: majority of dissolved states carry ≥ 2 fingerprints; kept states carry ≈ none;
planted world clean. Deliverable: per-type kept/dissolved table with the fingerprint
checklist, before/after UMAP pair, and the headline count ("N of M raw states are
artifacts").

## A3 — Intrinsic trajectories (mostly built — formalise)

**Claim**: per-type z axes are real intrinsic continua (tumour-state, interferon,
contractile↔synthetic per the devlog), separable from tissue gradients.

The report already shows the machinery (principal curves; type-partialled niche R²
0.03, coherence 0.05). Remaining validation: (1) seed reproducibility via
gene-signature correlation of axis loadings; (2) hallmark/curated-program monotonicity
along each axis (panel background); (3) presence in the dissociated reference (score
reference cells with axis top-genes; the axis must exist there — leak-immune); (4) the
*then map it* step: Moran's I of each axis on the tissue, reported as selection/history
(§7.8 reading), never as context effect. Pass: reproducible, enriched, present in
reference.

## A2 — Clones and clone × niche (heaviest; tumour slide justifies it)

**Claim**: CNV/clone calling on decontaminated profiles beats raw-x calling.

Method: inferCNVpy two arms — raw x vs decoded clean rates `softmax(a(z))` (leak and
niche channels removed), reference = non-malignant types; clone = clustering of CNV
profiles within tumour types. HGSOC aneuploidy makes this viable at 5.1k genes
(arm-level only; say so).

Validation (genome physics + fingerprints):
1. **Arm contiguity**: CNV segments must respect chromosome arms — contiguity score
   per arm; clean arm ≥ raw arm.
2. **Dosage consistency**: genes in called gains up / losses down — held-out dosage
   correlation.
3. **Chimera fingerprint**: fraction of intermediate/chimeric CNV profiles at clone
   territory boundaries (leak mixes profiles there) — should drop in the clean arm.
4. **Planted world**: planted clones with dosage effects + leak; clone-assignment ARI.
Then, and only then, clone × niche cross-tabulation (clones from z-side, niches from
data) as a *finding* figure. Pass: clean arm ≥ raw on 1–2 and 4, chimera rate down.

## A5 — Atlas transfer (the practitioner pitch)

**Claim**: the decoded clean profile `softmax(a(z))` transfers to dissociated
references better than raw spatial counts, because leakage is precisely what breaks
spatial↔atlas mapping.

Method: pinned public ovarian reference; one label-transfer tool (scANVI or ingest —
pin it), applied identically to raw `x/ℓ` and to clean rates. Circularity guard:
evaluate at **finer granularity than t** (reference subtypes), since t entered
training; state this in the figure caption.

Validation: (1) transfer confidence/entropy distributions; (2) impossible-assignment
rate (lineage-mixing score: cells assigned to lineages contradicting their t
compartment); (3) targeted known-mirage cases (CD3E⁺ "B cells" and equivalents in this
tissue) — raw maps them to doublet/wrong states, clean maps them home; (4) held-out
marker AUC per transferred label. Pass: clean ≥ raw on 1–2–4 and fixes the case
studies in 3.

## A6 — QC instrument (cheap, ship as a table)

**Claim**: the model's residuals are a usable per-cell QC score.

Two scores: within-type Mahalanobis on `mu_z` (mislabel candidate); per-cell held-out
log-lik deficit vs type average (unexplained cell — segmentation failure / doublet the
mixture cannot absorb).

Validation by physical-anomaly enrichment (odds ratios of flagged vs matched controls):
ovrlpy-style vertical signal integrity where computable, nuclear-fraction extremes,
cell-area outliers, negative-probe counts; mislabel candidates additionally checked by
marker-set disagreement with their own label. Pass: OR ≫ 1 with CIs; report precision
on one manually audited subsample (~100 cells).

## Sequencing and shared dependencies

Order: **A4 → A1 → A3 → A6 → A5 → A2** (cost-ordered; A4's DAPI extraction and
A1/A4's nuclear-only counts are the two new data dependencies — build once, reuse).
Planted worlds reuse the gate scaffold; reference atlas pinned once with provenance
(dataset, version, DOI, download date) for A1/A3/A5. Every pass/fail is pre-registered
above; a failed leg is reported as a finding, not tuned away. Multi-seed envelopes
throughout; per-claim verdict lines in the run report, same style as the quadrant.
