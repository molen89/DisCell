# 09 — Communication experiment (w ↔ cell–cell signalling)

Does `w` carry ligand-induced response — connected to genes through `B` — while `z` does
not, and can the pipeline separate communication from leakage? The mimic problem is the
point: contamination decays to ~zero beyond ~20 µm (MisTIC), the same range as juxtacrine
signalling, and cellAdmix found LR inference collapses to ~2 robust pairs after cleaning.
Naive colocalization-style CCC and leakage are near-perfect mimics; DisCell's κ-channel
is the instrument that tells them apart. Conventions inherit from `08` §1 (posterior
means, within receiver type, spatial-block CV, floor / ℓ-baseline / y-baseline rows,
per-κ multi-seed). Contrast with SIMVI's downstream: there the treatment is an *inferred*
archetype coordinate; here the treatment is **observed** — neighbour ligand expression.

## 1. Pair inventory

- Databases: CellChat LR pairs (keep the contact-dependent vs secreted annotation — the
  two classes carry different range logic, §5d) + NicheNet ligand→target regulatory
  potentials for the gene-level validation.
- Filters: L and R in the panel; NicheNet targets ∩ panel ≥ 20; sender types = types with
  mean ligand expression above the panel-wide top quartile for that gene; receiver types
  = receptor-positive types with ≥ 2,000 cells and non-degenerate exposure (below).
- Rank candidates by Var(exposure) × receiver prevalence; keep the top ~20 pairs plus any
  pair whose ligand has an orthogonal stimulation signature available (§5e).
- **Gate zero (run before building anything)**: count pairs surviving panel ∩ CellChat ∩
  (NicheNet targets ≥ 20). Prime 5K is broad so likely plenty, but it is an empirical
  gate; if thin, fall back to targets ≥ 15 or CellChat pathway target sets.
- Provenance: pin and log versions/DOIs of CellChatDB and the NicheNet potential matrices
  with the download date.

## 2. Exposure (the observed treatment)

- One-hop: `E_i = Σ_j β_ij · x_jL / ℓ_j` — the model's own graph weights, neighbours
  only, never cell i's own counts. Data, same status as y.
- Mid-band (secreted pairs only): `E_i^mid` = Gaussian-kernel average of `x_L/ℓ` over
  cells at 50–150 µm (beyond the graph; small new helper). Motivation in §5d.
- **Residualisation — the load-bearing step**: `Ẽ = E − Ê(y)` (ridge of E on sender
  composition, within receiver type). A ligand marks its sender type, so raw E ≈ sender
  proximity; the claim rides on ligand variation *given* who the neighbours are.
  Pairs where `Var(Ẽ)/Var(E)` < ~0.1 have no usable within-composition variation — drop
  them (report the dropped list; this is the "non-degenerate exposure" filter).

## 3. Allegiance and gene program (per pair × receiver type)

1. Ridge `Ẽ ~ mu_w` vs `mu_z` vs {floor, ℓ-baseline, y-baseline}. Asymmetric read,
   updated for type-only GAT sources (07 §4.1): `z` cold is allegiance evidence, and
   `w` hot is now *partially* evidential — the prior can no longer echo
   within-composition sender state (neighbour `μ_z` is not an input), so beating the
   y-baseline on `Ẽ` requires either the deviation channel (receiver's own x — the
   response, i.e. evidence) or an echo through `Φ` (morphology tracking ligand state —
   bounded, but real). The decoy control (§5c) is the certificate that separates those
   two; the program (step 3) carries the gene-level claim.
2. `θ_L` = the w-probe coefficients (unit-normalised); program `B θ̂_L` ∈ R^G = the
   expression response the model attributes to that ligand exposure.
3. Gene-level validation: AUROC of NicheNet targets of L in the `B θ̂_L` ranking, null =
   matched random ligands (matched on target-set size and mean panel expression — this
   absorbs the "targets are well-studied genes" bias).
4. Receptor placement check: receptor *level* may legitimately sit in z (competence is
   intrinsic); the response *program* must sit in w. Flag pairs where `B θ̂_L` top genes
   are dominated by the receptor itself.
5. Optional (SIMVI's s′×z analog): interaction — response strength stratified by receiver
   receptor level (terciles of receptor expression); expect monotone modulation.
6. **Operating point — RESOLVED (doc-10 grid, 2026-09-12)**: run at the standard
   operating point. The open-channel premise is retired: the α_w grid showed opening the
   deviation channel never improves NMI, costs B recovery (0.76 → 0.57), and surfaces no
   dose signal — the 0.1-vs-0.03 ablation is thereby delivered, with the answer "open
   buys nothing". Consequence: both routes to communication-in-w are closed for this
   architecture (prior blind to dose by construction, deviation empirically empty); the
   §5 arms stand as certified negatives, and the only remaining route is the fork's
   designed channel (§7), gated on the MDE titration.

## 4. Leak reattribution (the DisCell-only deliverable)

1. Naive arm: within receiver type, per-gene association `x_g ~ Ẽ` (partial, controlling
   log ℓ) — the squidpy/CellChat-colocalization-style analysis everyone runs.
2. Label the naive hits: *artifact candidates* = the ligand gene itself + sender-exclusive
   markers (expressed in sender; ≈ 0 in far-band receivers — slide-internal definition,
   no atlas needed); *signal candidates* = NicheNet targets.
3. Reattribution test per naive hit: does the leak channel predict it? Compare the naive
   coefficient against the coefficient computed on the model's decontaminated rates ρ
   (equivalently: is the association explained by `κ ρ̄`?). Genes whose association
   vanishes on ρ are leak-attributed; genes retained in `B θ̂_L` are w-attributed.
4. Deliverables: per pair, a stacked bar — % of naive hits leak-attributed / w-retained /
   unexplained — with the retained set's target-AUROC printed on it; and a per-pair
   κ-survival curve (which associations drain as κ rises). Headline sentence of the
   experiment: "X% of naive exposure-associated genes are reattributed to leakage; the
   retained set is target-enriched." Neither SIMVI nor any colocalization method can
   produce this split.

## 5. Validation / benchmark

**(a) Planted worlds on the real scaffold** — the controlled benchmark; reuses the
synthetic-gate machinery. Keep real positions, types, graph, and sender ligand levels;
simulate receiver counts in four worlds:

| world | communication | leakage | correct outcome |
|---|---|---|---|
| A | planted | none | program recovered |
| B | none | planted κ | **nothing detected** (naive arm fires — that's the point) |
| C | planted | planted κ | program recovered at planted size |
| D | none | none | nothing detected |

Planted program: ~30 random non-marker genes, log-fold amplitude 0.3–0.7, response
∝ standardised E. Metrics: planted-target AUROC and effect-size error in A/C; false-
positive program rate in B/D; the 4-world × detected-channel confusion table. World B is
the exam question — a method that "finds communication" there is measuring contamination.

**(b) Real-data baseline race**: same slide, same pairs, score every method on two axes —
curated-target AUROC (sensitivity proxy) vs sender-marker rate in top-k hits
(specificity-against-leak proxy; measurable without ground truth). Methods: naive
exposure-DE (§4.1), SIMVI run on the slide (pip package; its per-type spatial-effect
genes), squidpy nhood-enrichment/CellChat-style colocalization, DisCell. Deliverable: the
frontier plot; the claim is that DisCell dominates the frontier, not that it wins one
axis. Sequencing: SIMVI on 407k cells is a nontrivial training job — run the frontier
race **after** the core pipeline (§2–§4, §5a/c) works; it gates nothing upstream.

**(c) Negative controls**: (i) receptor-negative receivers — run the full pipeline on
types lacking R; any "program" is artifact, gives an empirical FPR; (ii) decoy exposure —
recompute E from a sender-expressed non-ligand gene; target enrichment must fall to the
y-baseline (if it doesn't, the pipeline is measuring sender proximity, not the ligand).

**(d) Range logic as internal validation**: leakage is dead beyond ~20 µm, so any
mid-band (`E^mid`) response on *secreted* pairs is leak-proof by physics — a
self-validating subclass that needs no model trust. Contact-dependent pairs are the hard
case and lean on (a) and the κ decomposition; say so explicitly in the writeup.
**Pre-registered reading for mid-band**: w's one-hop prior physically cannot see a
50–150 µm sender, and the invariance penalty does not target ligand fields — so a
genuine mid-band response reaches w only through the deviation channel (via the
receiver's own x), or lands in z. "w hot, z cold" may legitimately fail here; z-hot on
`E^mid` is a finding about the one-hop architecture, not a bug. This makes the
0.1-vs-0.03 α_w contrast (§3.6) diagnostic on this subclass: mid-band w-hot at 0.03 but
not 0.1 = deviation-mediated capture; z-hot at both = the blind spot, worth reporting.

**(e) Orthogonal stimulation signatures** (where available): score `B θ̂_L` against
published per-type ligand/cytokine response signatures (e.g. the Immune Dictionary for
immune receivers — mouse-derived, so only with cross-species mapping; NicheNet's own
held-out stimulation datasets otherwise). Strongest external evidence when types match.

**(f) Stability**: multi-seed envelopes per pair (B is seed-bistable under κ > 0); report
only pairs stable across seeds; θ/B comparisons across seeds via gene signatures, never
raw dims (08 §2.6).

## 6. Caveats to carry into the writeup

- Association, not causation: exposure is not randomised; receivers can induce ligand in
  senders (reverse path). The causal reading needs the exposure-mapping framing of
  `03` §causal — here we claim allegiance and reattribution, not effects.
- Autocrine signalling is out of scope by construction (own counts never enter E).
- ℓ-baseline stays mandatory: dense regions have both high exposure and depth structure.
- Order: run after 08's precondition checks; κ at the operating point, survival curves
  across the sweep.

## 7. Results addendum and fork decision (2026-09-11, post type-only)

Status after the first full pass: the prior path to communication is closed **by
construction** (type-only sources see composition + Φ, never neighbour expression);
whether the deviation path is also empirically null is settled only by arms run at the
open-channel operating point (α_w study + doc-10 guard) — nulls at α_w = 0.1 are
architecturally guaranteed and count as no evidence. Record which applies.

**MDE titration (required before any "dead" claim ships).** World A with planted
response at log-fold {0.1 … 0.7}: recovery vs amplitude for (i) the pooled naive
detector and (ii) the w-mediated detector at open channel. Deliverable: the
minimum-detectable-effect curve — converts the negatives into "effects below X are
undetectable per cell at ~100 tx/cell", a statement about the modality, not the model.

**Fork rule (pre-registered).** If the *pooled* floor sits above realistic amplitudes:
option 1 is final — w is the composition/context summary, 09 is a debunking instrument,
and no architecture at this depth would have found discovery-grade communication. If
pooled detects but w does not: option 2 earns one guarded spike run — a curated
ligand-exposure panel as **designed context features** (`E` is data, same status as y;
"2× ligand in this niche" is a well-posed `do(c′)`), with the ego-echo-via-leakage
caveat and the §5c controls attached, certified on worlds A/B. Either way option 2 is
the designated successor model, not a patch to the current paper.

**Reattribution calibration.** The 9/28 planted-fake miss makes every "X% reattributed"
figure non-quotable until diagnosed: classify the misses by amplitude, expression level,
and marker class; if the detector is at fault, replace coefficient-vanishing-on-ρ with a
κρ̄-prediction / likelihood-ratio test (the model absorbing the fake defeats the former).
External concordance worth one line: the sole survivor (PDGFB→PDGFRB) matches
cellAdmix's "~2 robust pairs", and it is the textbook paracrine pair.

**Seed discipline.** Battery-clean seed selection (pre-registered gates: w-mirror, NMI
floor, recon envelope) picks the *artifact*; all reported findings remain multi-seed
envelopes. Never select seeds on the analyses' own outcome metrics.

**Anomaly claim — RESOLVED (doc-10 parked, 2026-09-12).** The anomaly score is cut from
the claims; 03's goal list is trimmed at writing time. Fork-record addition from the
same grid: under maximal z-purge with the channel open, w specialised to planted
exposure dose (dose-R² 0.40, AUROC 0.65) — the receiver-side dose signal is extractable
when nothing competes for w, i.e. option 2 (a designed communication channel) is
feasible in principle at planted amplitudes; the MDE titration remains its gate.

## 8. LR co-occurrence map (SIMVI-fig-6h style, controlled)

Purpose: simple, descriptive proof that w carries organised, nameable context signal —
in the field's standard visual language — with the control panel that keeps it honest.
No communication claim; this is a co-occurrence figure. Priority is that it works, not
that it is fancy. All inputs exist (B, w, doc-09 exposure, CellChatDB annotations).

### 8.1 Panel A — the co-occurrence map
- **Rows (genes, stated rule, no hand-picking)**: top ±10 B-loadings per *active*
  program (effective-rank programs only, V10), deduplicated.
- **Row value per cell**: the model's context effect `⟨w_i, B_g⟩`.
- **Columns (LR pairs)**: gate-zero survivors ranked by Var(exposure) × receiver
  prevalence, top ~30, grouped and labelled by CellChatDB pathway family (ECM,
  MHC, …).
- **Column value per cell (receiver side)**: `LR_i = R_g(i) × E_i` — depth-normalised
  log1p receptor expression × one-hop exposure (§2).
- **Entry**: Spearman within receiver type (types with ≥ 500 receptor-positive cells),
  Fisher-z pooled, validation tiles. Significance: within-type permutation of the LR
  score (~200 perms), BH per map, n.s. greyed.

### 8.2 Panel B — the control (same map, composition-partialled)
Partial Spearman given y: rank-transform, ridge-residualise both row and column
variables on y within type, correlate the residuals. Expectation, pre-registered:
Panel A dense (composition-mediated co-occurrence — exactly what fig-6h-style maps
show), Panel B sparse to empty (doc-09 §4.6), survivors highlighted and
cross-referenced against the §4.6 tally (PDGFB→PDGFRB if it holds).

### 8.2b Revision after the first render (2026-09-15)
Three processing fixes and one keeper:
1. **Rows**: at effective rank 2, per-gene rows are duplicates within program-sign
   blocks — collapse to the program coordinates (or one stated representative gene per
   block).
2. **The ladder**: add the missing rung. Three panels — A′ *uncontrolled* pooled
   Spearman (the SIMVI-comparable view; expected dense with large |ρ|), A within-type
   (expected collapse), B composition-partialled (expected empty). The ladder shows the
   published-map structure is type composition.
3. **LR score smoothing**: raw receptor counts starve the Spearman (0–2 counts/cell).
   Use model clean rates ρ_g for the receptor side (note the mild circularity;
   descriptive map) or kNN-spatial smoothing.
3b. **Column rule v2 (coder, ratified)**: the v1 rule produced the redundancy and the
   grey columns by construction (shared-ligand pairs have identical exposure; raw
   Var(E) is mostly sender proximity). New rule, still no hand-picking: rank by
   **Var(Ẽ)** (the §2 composition-residualised exposure), keep **one column per
   ligand** (its best-ranked receptor), require **≥ 2 eligible receiver types**.
4. **Keeper**: the spatial-shift null (coder addition) stays as the printed verdict
   line — zero survivors even in co-occurrence at autocorrelation-aware significance is
   a finding; SIMVI-style maps carry no such null, and the caption says so.

### 8.3 Reading (caption text, fixed in advance)
"Left: w's programs co-occur with curated ligand–receptor axes — the context field
carries recognisable biology. Right: after composition control, the density collapses —
the map measures *where* pairs and programs co-locate, not signalling. Methods that
show only the left panel are reporting composition."

### 8.4 Optional supplement — attention vs LR connectivity
The type_only attention collapses to a K×K "who listens to whom" matrix; correlate it
(one Spearman) against curated LR connectivity per ordered type pair
(Σ_pairs mean ligand in sender × mean receptor in receiver). One heatmap, descriptive
only — attention is weakly constrained at rank-2 w; supplementary grade.

### 8.5 Mechanics
One script over existing modules (exposure from `communication.py`, basis from
`atlas.py`, folds from `validate.py`); outputs `lr_map.{json,png}` under the run.
Operating κ only; one line in the JSON joins survivors against the sweep runs' metrics
for a stability note. Multi-seed: entries compared across seeds by sign agreement of
significant cells, nothing fancier.
