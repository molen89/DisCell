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

1. Ridge `Ẽ ~ mu_w` vs `mu_z` vs {floor, ℓ-baseline, y-baseline}. **What this row
   certifies is asymmetric**: `z` cold is allegiance evidence; `w` hot is NOT response
   evidence — the GAT's source features include neighbour `sg z_j`, which carries the
   sender's ligand state, so w's prior can echo `Ẽ` with no receiver response at all,
   and the y-baseline (composition only) does not catch that richer echo. Read w-hot as
   "exposure visibility" (expected), z-cold as the claim. The induced-response claim is
   carried by the program (step 3) plus the decoy control (§5c), not by this R².
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
6. **Operating point**: this experiment needs the deviation channel open. At α_w = 0.1
   the receiver-side counts path is closed and the idiosyncratic response has nowhere to
   live in w. Run at the α_w-study operating point (0.03 pending seed confirmation), and
   keep a 0.1-vs-0.03 comparison as a deliverable ablation — the cleanest demonstration
   that the channel matters.

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
