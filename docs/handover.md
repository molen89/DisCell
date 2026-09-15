# DisCell — project handover (state as of 2026-09-14)

Written to crystallise everything done so far for whoever picks the project
up cold (architect or engineer). Every number carries its artefact pointer;
narrative lives in [docs/devlog.md](devlog.md), bugs in
[docs/issues.md](issues.md), spec decisions in
[docs/spec_deviations.md](spec_deviations.md). A null or failed leg is a
finding here, never softened.

## 1. What this is

**Question.** In imaging spatial transcriptomics (10x Xenium Prime 5K), a
cell's measured expression mixes three things: what the cell is doing on its
own, how its neighbourhood modulates it, and transcripts that physically
belong to neighbours (segmentation spill-over). DisCell-simple
([07-simple-spec_7.md](../07-simple-spec_7.md)) is a VAE that splits a cell's
counts into an **intrinsic state z** (d_z = 20; encoder sees only the cell's
own counts, depth and type), a **spatial response w** (d_w = 6; posterior
`q(w | c, t, z, x)` around a context-conditional prior `m_ψ(c, t)`, decoded
through a gene-programme matrix B), and a **fixed leakage channel**
(`p = (1 − κ) ρ + κ ρ̄`, ρ̄ = β-weighted neighbour rates, κ = 0.1). Context
`c` = GATv2 over the neighbours' one-hot types (`gat_sources = type_only`,
the ratified default) ⊕ image embedding Φ ⊕ isolated flag. Invariance of z
to context is enforced by an adversary (α_a = 0.3). The claim is a division
of labour that can be *tested*: z must carry intrinsic dynamics and fail
spatial tests; w the reverse.

**Slides.** `xenium_prime_ovarian_cancer_ffpe` (HGSOC, 407,120 cells × 5,101
genes, curated 18-class labels) is the development slide; `xenium_prime_
human_lung_cancer_ffpe` (same size, no curated labels → 10x `graphclust`, 33
clusters) is the transfer slide.

## 2. Operating point, reference model, seeds

| knob | value | where set |
|---|---|---|
| κ (leak) | 0.1 | `TrainConfig.kappa` |
| α_z, α_w, α_a, ω | 0.007 (1/ℓ̄; lung 0.004), 0.1, 0.3, 1 | `TrainConfig` |
| invariance | adversary, 6 steps, lr 2e-3, hidden 64 | `TrainConfig.invariance` |
| GAT sources | `type_only` (era guard reloads old checkpoints as `type_z`) | `networks.DisCell`, `validate.load_run` |
| d_z / d_w / hidden / gat_dim / heads | 20 / 6 / 256 / 32 / 4 | `TrainConfig` |
| tiles / v_pcs / Φ | 4096 cells, 12 PCs in the invariance block, Φ full 384-d | `TrainConfig` |
| budget | sweeps: 200 epochs, patience 20; **reference runs: 500 / 40** | CLI |

**Pinned reference: `ablation_gat_type_only_s1`** (seed 1, 500-epoch budget,
best epoch 64 of 104, 22 min). Selected from the type_only seed triple by
the full battery, not by likelihood alone
([devlog](devlog.md) "type_only seed triple complete"):

| run | best recon (held-out nats/count) | NMI(z, t) | cycle_z R² pooled | cycle_w | mirror R² (perm 0.016) | probe ΔCE (floor ≈ −0.05) |
|---|---|---|---|---|---|---|
| `ablation_gat_type_only` (s0) | −7.2527 | 0.667 | 0.440 | 0.007 | 0.044 | −0.009 |
| **`ablation_gat_type_only_s1`** | **−7.1924** | 0.654 | **0.499** | 0.008 | 0.046 | 0.002 |
| `ablation_gat_type_only_s2` | −7.2310 | 0.666 | 0.465 | 0.010 | 0.045 | −0.011 |
| `reference_best` (type_z, previous reference) | −7.2586 | 0.658 | 0.475 | 0.034 | 0.056 | 0.004 |

(`runs/<run>/metrics.json`: `best.recon_val`, `best.nmi`,
`final.cycle.z.r2_pooled`, `final.cycle.w.r2_pooled`, `final.mirror.r2`,
`final.probe.delta_ce`; `final.cycle.ceiling` is the pre-rename key for the
50-PC linear reference, 0.204–0.235.) The seed spread in recon (0.06) is
optimisation variance over a rugged landscape — all three seeds are clean on
every instrument; s1 is the "strong family" optimum that 200-epoch sweep
seeds never reach (sweep3 κ = 0.1: −7.2562 ± 0.0011).

Reproduce the reference:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m discell.model.train \
  --dataset xenium_prime_ovarian_cancer_ffpe --run-name ablation_gat_type_only_s1 \
  --epochs 500 --patience 40 --seed 1          # all other knobs are the defaults
```

## 3. How to run everything

All paths are keyed by dataset id under `data/` (`discell/paths.py`;
`DISCELL_DATA` relocates the tree). `DS=xenium_prime_ovarian_cancer_ffpe`,
`RUN=ablation_gat_type_only_s1` below. Long jobs: launch detached
(`setsid nohup … > log 2>&1 < /dev/null & disown`) — session-tied background
tasks die with the session.

| step | command | output |
|---|---|---|
| preprocess (bundle, graphs, image embeddings, figures) | `uv run python -m discell.preprocess --sample <sample> [--only bundle\|graph\|embed\|figures]` | `data/datasets/$DS/bundle/full.h5ad`, `embeddings/*.pt` |
| train one run | `uv run python -m discell.model.train --dataset $DS --run-name $RUN --epochs 500 --patience 40 [--seed N] [--label-key graphclust] [--gat-sources type_z] [--subtract-leak] [--d-w 6]` | `runs/$RUN/{best.pt,config.json,history.jsonl,metrics.json,events.*}` |
| run report (quadrant, ‖w‖, B, + atlas/transport/A4 sections when present) | `uv run python -m discell.model.report --dataset $DS --run $RUN` | `runs/$RUN/report/report.md` + `figures/` |
| doc-08 §§3–5 battery (Moran, niche, landmarks, matrix) | `uv run python -m discell.model.validate --dataset $DS --run $RUN [--analyses morans,niche,landmarks,matrix] [--n-perms 1000]` | `runs/$RUN/validation/validation.json` + png |
| doc-08 §6 atlas | `uv run python -m discell.model.atlas --dataset $DS --run $RUN` | `runs/$RUN/atlas/atlas.json`, `program_*.png` |
| doc-08 §7 transport | `uv run python -m discell.model.transport --dataset $DS --run $RUN [--niches 10]` | `runs/$RUN/transport/transport.json`, `transport_summary.png`, `pair*.png` |
| doc-09 communication | `uv run python -m discell.model.communication --dataset $DS --run $RUN [--pairs 20] [--stages …]` | `runs/$RUN/communication/communication.json` |
| doc-09 §8 LR co-occurrence ladder | `uv run python -m discell.model.lr_map --dataset $DS --run $RUN [--pairs 30] [--n-perms 200]` | `runs/$RUN/communication/lr_map.{json,png}` (+ report section) |
| κ sweep (6 κ × 3 seeds, idempotent via `metrics.json`) | `uv run python -m discell.model.sweep --dataset $DS --tag sweep3` | `runs/sweep3_k<κ>_s<seed>/`, `experiments/kappa_sweep_sweep3.json` |
| sweep companion (κ-survival, sensitivity) | `uv run python -m discell.model.validate --dataset $DS --sweep-tag sweep3` | `experiments/atlas_kappa_survival*.json`, `transport_kappa_sensitivity.json` |
| doc-11 shared data | `uv run python -m discell.applications.shared --dataset $DS --stage dapi` / `--stage nuclear-counts` | `qc/nuclear_dapi.parquet`, `qc/nuclear_counts.npz` |
| doc-11 A4 | `uv run python -m discell.applications.a4_cycle --dataset $DS --run $RUN` | `runs/$RUN/applications/a4_cycle.{json,png}` |
| lung transfer | same `train` with `--dataset xenium_prime_human_lung_cancer_ffpe --label-key graphclust --alpha-z 0.004` | `…lung…/runs/reference_graphclust/` |
| tests | `uv run pytest -q` (139 passed, 1 skipped, 5½ min on 2026-09-15) | — |

Experiment-level artefacts (ovarian): `data/datasets/$DS/experiments/`
— `kappa_sweep{,_sweep2,_sweep3}.json`, `validation_sweep2.json`,
`alphaw_study.json`, `w_mirror_certification.json`,
`graphclust_comparison.json`, `atlas_kappa_survival{,_internal}.json`,
`transport_kappa_sensitivity.json`, `calibration*.json`,
`convergence*.json`, `ego_masking.json`; synthetic:
`data/experiments_synthetic/guard_gate.{json,png}`. External data (with
provenance in the devlog): `data/external/CellChatDB.human.rda`,
`nichenet_ligand_target_matrix_nsga2r_final.rds`,
`msigdb_hallmarks_h.all.v2023.2.Hs.symbols.gmt`.

Code map: `discell/model/` — `networks.py` (DisCell, GATv2), `elbo.py`
(loss + Weights), `equations.py` (leak mixture, invariance penalty,
TypeCovariances), `prepare.py` (graph, tiles, ModelData), `train.py`
(Trainer/TrainConfig/CLI), `metrics.py` (NMI, probe, mirror, cycle R²,
w-mirror ΔR²), `cell_cycle.py` (Tirosh scores, MKI67 ranking),
`report.py`, `validate.py` (doc-08 §§3–5 + `load_run`/`collect_latents`),
`atlas.py` (§6), `transport.py` (§7), `communication.py` (doc-09),
`lr_map.py` (doc-09 §8 ladder), `sweep.py`, `calibrate.py`, `synthetic.py`
(gate scaffold);
`discell/applications/` — `shared.py` (DAPI / nuclear counts), `planted.py`
(`fit_synthetic`), `a4_cycle.py`; `discell/preprocess/`, `discell/data/`
(bundle, loader, embeddings); `tests/` (17 files).

## 4. Results, by claim (pinned reference unless stated)

Reading conventions (doc-08 §1): posterior means, within type, spatial-block
folds (tiles → folds), every probe judged against a within-type permutation
**floor**, a log-depth **ℓ-baseline**, and where relevant the 50-PC **linear
expression reference** (renamed from "ceiling": z exceeds it ~2×, so it is a
reference line, not a bound — issues M6).

### 4.1 The disentanglement quadrant — z is intrinsic, w is context

`runs/$RUN/report/report.md`, `validation/validation.json` (`cycle_row`,
`pseudotime`, `matrix`):

| target | z | w | floor / ℓ / linear ref | expect |
|---|---|---|---|---|
| S/G2M cycle score, within cycling types (R²) | **0.50** (report, val split) / 0.44 (battery, block-CV subsample) | 0.006 | −0.001 / −0.000 / 0.215 | z |
| niche label K = 10 (macro AUC, block-CV logistic) | 0.654 † | **0.767** | 0.501 / 0.585 | w |
| pseudotime tissue gradient, type-partialled (niche R², coherence) | 0.005, 0.045 | **0.566, 0.489** | (raw: z 0.60/0.63, w 0.90/0.99) | w |
| Moran's I, mean |I| over dims (perm null ±0.002) | 0.061 | **0.628** (0.36–0.70 per dim) | — | w |
| mid-band landmark distance (R²) | −0.011 | 0.023 | −0.002 / −0.011 | w (near-null) |

† the one flagged cell: z above max(floor, ℓ) on the discrete niche label.
Triage (`morans.mu_z.triage`): the hot z dims are not y-explainable (≤ 3.6%
R²) → benign intrinsic spatial structure (clone patches / programme
territories) read through spatial contiguity, not context leakage; it is
κ-reducible (+0.062 → +0.039 over the grid). Standing watch item.

Training-time guards (all seeds): probe ΔCE at zero with the floor below it;
mirror R² 0.044–0.046 vs permuted 0.016 (attention does not reconstruct the
cell through look-alike neighbours); KL_w ≈ 0.002/dim — w rides its
context prior `m_ψ(c, t)` (prior-R² 0.9998): **w is a context field
evaluated at the cell, not a per-cell measurement** (α_w study, §5).

Per-type ‖w‖ ranking (report): VEGFA⁺ tumour on top, proliferative /
inflammatory tumour next, stereotyped epithelia at the bottom; stable over
18 independent fits and the graphclust relabelling (Spearman 0.577,
`experiments/graphclust_comparison.json: w_spearman`).

### 4.2 κ envelope (sweep3: 6 κ × 3 seeds, 200 epochs, type_only)

`experiments/kappa_sweep_sweep3.json` and the runs' `metrics.json`
(pooled cycle R²):

| κ | recon | NMI | cycle_z | cycle_w | mirror | probe ΔCE | B cosine across seeds (mean/min) |
|---|---|---|---|---|---|---|---|
| 0 | −7.2526 ± 0.003 | 0.665 | 0.458 ± 0.005 | 0.006 | 0.050 | 0.001 | 0.70 / 0.65 |
| 0.05 | −7.2567 | 0.653 | 0.445 ± 0.02 | 0.007 | 0.051 | 0.006 | 0.52 / 0.42 |
| **0.1** | −7.2562 ± 0.001 | 0.659 | 0.450 ± 0.03 | 0.006 | 0.048 | 0.006 | 0.49 / 0.43 |
| 0.2 | −7.2645 | 0.664 | 0.420 ± 0.05 | 0.005 | 0.044 | 0.001 | 0.45 / 0.38 |
| 0.3 | −7.2729 | 0.648 | 0.421 ± 0.05 | 0.010 | 0.040 | −0.007 | 0.47 / 0.38 |
| 0.4 | −7.2847 | 0.642 | 0.425 ± 0.03 | 0.007 | 0.037 | −0.016 | 0.43 / 0.40 |

Reading: the disentanglement reads are κ-robust (cycle_z 0.42–0.46, cycle_w
≤ 0.01, probe at zero); likelihood falls monotonically past κ = 0.1. The
matched-column B cosine (0.43–0.52 across seeds at κ > 0 vs 0.70 at κ = 0)
is **mostly a metric artefact** (issues V11): w uses 1–2 of its 6 dims, so
four B columns are unconstrained; the realised shift μ_w·B agrees across
seeds (shift-space overlap 0.97–0.99, per-cell shift correlation
0.90–0.93). The κ = 0 → κ > 0 drop is still unexplained.
Partial-isolation strata (`recon_strata`): cells that lost
edges to the prune reconstruct *better* (−7.04 vs −7.26) at every κ.

### 4.3 Landmarks and the allegiance matrix (doc-08 §2, §5)

`validation.json: landmarks, matrix`; figures `landmark_map.png`,
`landmark_bands.png`, `allegiance_matrix.png`. Inventory after the rewrite:
vasculature = endothelial ∪ pericytes (206 instances, 8,387 cells),
compact smooth muscle 20–500 cells (238 / 18,578), kNN-smoothed
tumour/stroma interface (36,062 cells, 8.9% of the slide). Per-band fits
with a **y-baseline** (probe from raw neighbour composition) because the
sets are type-defined: interface mid-band w 0.054 vs y-baseline 0.036 →
~60% of the w signal is definitional, the residual reads as
compositional allegiance; **vasculature is the only geometry-grade test and
is a certified null** (mid-band y −0.004, w 0.007). θ cross-type cosine
0.05–0.37. The matrix renders on one absolute scale with "n.s." greying and
a dagger on unexpected-column signal (the niche-z † above).

### 4.4 The w-program atlas (doc-08 §6)

`runs/$RUN/atlas/atlas.json`, `program_*.png`. Canonical basis = varimax on
B weighted by realised variance. The atlas reports **6/6 dimensions
active** — **superseded 2026-09-14 (issues V10)**: cov(μ_w) has
eigen-fractions [0.83, 0.17, 0, 0, 0, 0], i.e. w is rank 2 and the six
rotated coordinates are collinear; the honest read is **two programs**
(one dominant, one secondary) and four null directions. Per rotated
coordinate: Moran's I 0.36–0.70; **joint context-driver R²
0.92–0.98** (composition + image PCs + landmark distances) — the programs
are almost entirely context-explained (composition-level effect, no
communication claim needed). Hallmarks (expressed-panel background, BH q ≤
0.05): program 0 EMT q = 2.6e-12, program 1 EMT q = 7.6e-3, programs 2–5
**none significant** (the pre-gate labels SPERMATOGENESIS / ADIPOGENESIS /
HEDGEHOG / G2M were p ≈ 0.08–0.37 and are withdrawn). Top types: SOX2-OT⁺
tumour (0, 2, 3), macrophages (1, 4, 5).

κ-survival (`experiments/atlas_kappa_survival{,_internal}.json`): vs the
pinned s1, signature correlation 0.29–0.43, flat in κ — confounded with the
long-budget optimum gap; sweep-internal (reference `sweep3_k0.1_s0`):
**along κ within seed 0.62–0.64, across seeds 0.25–0.34** → programs are
κ-stable and seed-variable, like B. Report consensus programs over seeds
and carry identity by hallmark label, never raw correlation.

### 4.5 Counterfactual transport (doc-08 §7)

`runs/$RUN/transport/transport.json: summary`, `transport_summary.png`.
Niches = k-means on neighbour composition (K = 10); panel = type × niche
pair; model quantities from training tiles, observations from held-out
tiles. **Counterfactual = Δ̂program + Δ̂leak with z held fixed** (κ never
changes; the leak content comes from the *new* neighbours). Supported tier
is near-empty by construction (3/158 panels, mean R² 0.009); the
informative regime is **extrapolation, named as such**: 155 panels,
counterfactual mean held-out R² **0.099** (program-only 0.068, leak-only
0.063), median calibration slope **0.92**, counterfactual beats both
single channels in **100/155**. The all-vary "model account" reaches
0.146; the gap (≈ 0.05) is the measured **selection share** of observed
niche differences. Best panel: macrophages niche 0 → 1, R² 0.39 at slope
0.90 (program 0.36, leak 0.15).

κ-sensitivity (`experiments/transport_kappa_sensitivity.json`, seed 0,
**pre-correction "full" = model account, not rerun on the counterfactual
object**): leak-channel R² 0.000 (κ = 0, sanity) → 0.084 (κ = 0.3–0.4),
program 0.056 → 0.042, total κ-robust for κ ≤ 0.2 (0.10–0.11) and degrading
past it (0.084 at 0.4, beats-both 128 → 75/155, slope 1.05 → 0.66). Quote
the split as a κ-range.

### 4.6 Communication (doc-09) — a debunking instrument

`runs/{reference_best,ablation_gat_type_only_s1,ablation_gat_type_only}/
communication/communication.json`. Gate zero: 618 CellChat pairs with ≥ 20
in-panel NicheNet targets. Exposure through the model's own β,
composition-residualised (Var ratio ≥ 0.1 gate), allegiance rows {w, z,
floor, ℓ, y} on block-CV, programs B·θ̂ scored by NicheNet-target AUROC
against 50 matched-null ligands, decoy-exposure control.

- **Exposure visibility is null in every arm** (mean w-R² −0.010 in four
  arms: reference, s1, s0, α_w = 0.05) and under `type_only` this is
  **architectural**: c carries no channel for neighbour expression detail.
- Programs: at most 1–2 fragile candidates — PDGFB→PDGFRB (s1: AUROC
  0.583, pct 1.00, decoy 0.497; clears in 2/3 arms), POSTN→ITGAV/B5 (2/3
  arms, loses its decoy margin in s1), CD99 decoy-rejected; **nothing
  clears in 3/3**. The paper reports the tally, not a winner.
- Reattribution (61% of naive exposure-associated genes leak-attributed) is
  **uncalibrated** — the §5a planted worlds never ran; do not quote.

### 4.7 A4 — decontaminated cycle call (doc-11), parked

`runs/$RUN/applications/a4_cycle.{json,png,log}`; report section "A4".
Post-mitotic population (12 types, 342,743 cells, raw-positive 45.5%).
Raw calls track neighbour-cycle exposure (Q4 − Q1 gap **0.159**, band
[−0.009, 0.008]); rate-matched z calls less so (**0.068**). But the
contamination-specific **gene-split fingerprint is null** (Δ ring 1 0.006,
CI [−0.006, 0.020]; ring 2 0.003) → the gradient is state, not transcript
transfer, and the pre-registered honest outcome applies: **leak-induced
cycle false positives are rare at κ = 0.1 on this slide.** The flatter z
line is closer to the *sensitivity-loss* fail state than a win. DAPI
(group level): both-positive median −0.17 vs neither (KS 0.14), raw-only
+0.04 — supports neither caller. Planted world **0/3 seeds pass** (victim
FPR raw 0.21 vs z 0.32; AUROC 0.68 vs 0.64) — but the adjudicator is
defective (issues V9: global threshold, control FPR 0.02–0.45, under-
powered plant) and the result does not adjudicate. A4 v1 (cycling-type
population) was inconclusive for instrument reasons (92% raw-positive,
DAPI at chance for both callers).

### 4.8 Transfer: annotation-free control and the lung slide

Ovarian with `graphclust` labels (`runs/reference_graphclust`,
`experiments/graphclust_comparison.json`): recon −7.2556 (vs −7.2586),
NMI vs curated labels 0.646, cycle_z 0.323 vs linear reference 0.116
(ratio 2.8 holds; absolute lower with coarser types), cycle_w 0.004,
**but the invariance probe re-fitted against the curated labels finds
ΔCE 0.061** (reference 0.004, floor −0.056): with coarser types the
adversary guards a coarser target and finer-label niche information stays
in z — the label-granularity cost of annotation-free operation, ‖w‖ ranking
Spearman 0.577 vs curated. Lung (`…lung…/runs/reference_graphclust`, α_z
0.004, 33 clusters, 72 min): recon −7.2577, NMI 0.653, cycle_z 0.347 vs
linear ref 0.166, cycle_w −0.000, probe ΔCE −0.010 (floor −0.021), mirror
0.036, Moran |I| w 0.428 / z 0.059, niche AUC w 0.731 / z 0.617 / ℓ 0.544.
**The full allegiance structure transfers with one knob (α_z) changed.**
Only §§3–4 of the battery ran on lung; landmarks/atlas/transport did not.

### 4.9 Findings of 2026-09-14/15 (after the handover was first written)

- **x̃ = x − κℓρ̄ as the encoder input** (`--subtract-leak`, architect-
  approved spec change, gated on results; `runs/xtilde_s{0,1,2}`): recon
  identical (±0.001), niche-z residual −0.006 mean (−0.011 / −0.013 /
  +0.005 by seed), cycle_z inside envelope, w-side rows unmoved, NMI
  −0.017 and Moran-w down in 2/3. **Near-neutral; default stays off.** The
  per-cell benefit it was built for is A4's (parked) planted-world test.
- **w's effective rank is an optimiser property**: cov(μ_w) is rank 1–2 at
  α_w = 0.1 (d_w = 6), rank 3 at 0.05, rank 4 at ≤ 0.03 — in the prior
  field itself, not the deviation channel; a second axis reproduces across
  seeds at 0.05 (cosine 0.77–0.90) but not at 0.10 (0.50–0.77). At the
  sweep budget d_w = 6 is rank ~1, d_w = 3 rank 2 (second axis 3–27%,
  seed-variable), d_w = 2 rank 1, d_w = 8 rank 2 (9–14%, seed-stable, at
  a small cycle_w cost 0.008–0.020). **d_w stays 6**; ablation table
  d_w ∈ {2, 3, 6, 8} × 3 seeds on disk (`runs/dw{2,3,8}_s*`, sweep3 for 6). The atlas' "6 programs" is superseded by "one, sometimes
  two" (issues V10); B stability by shift-space overlap (V11: 0.97–0.99
  across seeds) replaces the matched-column cosine. **Candidate
  re-calibration α_w = 0.05 under type_only** — architect's call.
- **KL_w is a snapshot of a chase**: it oscillates 20–100× between
  evaluations five epochs apart in every seed; s0's `best.pt` sits on a
  spike. Per-cell KL_w at the strong optimum is 0.01–0.1% of w's variance
  per dim with no heavy tail (no anomaly channel).
- **Doc-09 §8 ladder** (`lr_map.py`, three seeds): uncontrolled A′ |ρ| to
  0.7 with 20–29 of 60 cells surviving a Moran-preserving null (type
  composition); within type max |ρ| 0.26–0.34, 1–10 survive; composition-
  partialled max |ρ| 0.14–0.16, **0 survive in every seed**. Descriptive;
  no communication claim; the permutation-null counts barely move down the
  ladder, which is the figure's own lesson about that null.
- New TensorBoard figures for runs trained from 2026-09-14: `kl_spatial`
  (per-cell KL_z / KL_w on the tissue, per type, colour bars),
  `zw_std_trajectories_{umap,pca}` (principal curve on standardised
  [z, w]); within-type panel titles now say the colour is the dominant
  *neighbour* type. Rendered for the pinned reference under
  `runs/ablation_gat_type_only_s1/{report/figures,figures_render}/`.

## 5. Retractions, falsified hypotheses, parked work

| item | what was claimed | what killed it | what remains |
|---|---|---|---|
| **w-mirror mechanism** (devlog 2026-09-10/11) | the high-recon runs at α_w < 0.1 were `m_ψ` reading neighbours' μ_z through the GAT | two pre-registered detectors failed certification (`experiments/w_mirror_certification.json`: ΔR² of negatives 0.068–0.080 > positives 0.036–0.050; recon survives replacing w by a linear function of composition) + channel decomposition; then s1 reached −7.19 with neighbour z structurally absent | the **instability** at α_w < 0.1 is real (recon spread 0.08 vs ±0.005); the high-recon family is a **better optimum**; `type_only`'s case is empirical only |
| **α_w re-calibration to 0.03 / 0.05** | a per-cell w channel opens and pays | seed-bistable at both (0.03: s1/s2 −7.17/−7.23; 0.05 likewise); 0.03_s2 drained cycle_z to 0.354 | α_w = 0.1 stands; w is context-only at every tested α_w (prior-R² ≥ 0.996); selection by battery, never recon |
| **Doc-10 z–w guard** | a Gaussian-MI guard between μ_z and μ_w rescues low-α_w | arm 0 (`data/experiments_synthetic/guard_gate.json`, 20 configs): guard-on never holds what guard-off loses; B recovery worse at every α_w; KL_w closes with α_zw (backdoor α_w) | PARKED per §4, **code removed** (user); a second attempt must first reproduce the historical M3 cliff |
| **A4 v1 → v2** | z rejects leak-induced cycle false positives | v1: instrument (saturation, DAPI at chance); v2: no contamination fingerprint on this slide; planted adjudicator defective | honest outcome recorded; **parked** with issues V9 + the amortisation-gap hypothesis open |
| **Hallmark labels** on atlas programs 2–5 | SPERMATOGENESIS, ADIPOGENESIS, HEDGEHOG, G2M | shipped ungated (p 0.08–0.37); BH gate added | 2 of 6 programs labelled (EMT ×2) |
| **"Ceiling"** for the cycle probe | 50-PC ridge as an upper bound | z beats it ~2× | renamed linear expression reference everywhere |
| **M6 mean-of-types cycle R²** | headline | dominated by non-cycling types' noise | headline = pooled, cell-weighted |

## 6. Limitations and threats to validity (ranked)

1. **Type labels enter everywhere** (enc_z, enc_w, m_ψ, the adversary, the
   GAT sources, and the doc-08 landmark sets). z is type-informed by
   design; every "within type" read is conditional on labels that are
   themselves expression-derived. Mitigations in place: y-baseline rows,
   the graphclust control (structure survives relabelling). Not mitigated:
   validation at finer-than-t granularity (doc-11 A5's guard) has not run.
2. **Amortised per-cell z ignores the cell's own neighbours** (UNVERIFIED,
   issues): `enc_z(x_i, ℓ_i, t_i)` cannot know what leaked into x_i; the
   leak model acts on the decoder side. Any per-cell decontamination claim
   (A1/A2/A4/A5) is at risk until a powered planted world settles it. The
   counts-level correction x̃ = x − κ ℓ ρ̄ is the model-licensed
   alternative.
3. **Single development slide; one transfer slide with no curated labels.**
   All absolute numbers are HGSOC-specific; only the *shape* of the
   allegiance structure has been shown to transfer.
4. **w is over-provisioned and B's column metrics read its null space**:
   w occupies 1–2 of 6 dims; matched-column B cosine 0.43–0.52 and program-
   signature correlation 0.25–0.34 across seeds are dominated by the four
   unconstrained directions, while the realised shifts agree (0.97–0.99).
   The optimum family is reached in 1–2 of 3 seeds at 500 epochs; the
   pinned model is a *selected* optimum. Report programs by effective rank.
5. **w is a context field, not a per-cell measurement** (KL_w ≈ 0): the
   spec's deviation/anomaly channel is closed at the operating point; the
   doc-11 A6 QC score built on w would be reading `m_ψ` noise.
6. **The niche-z residual** (AUC 0.654 vs ℓ 0.585) stands as a flagged,
   triaged-benign watch item; a future operating point pushing it toward
   w's level must rerun the triage.
7. **Communication readout is architecturally blind** to
   within-composition ligand variation (a feature for the debunking
   story, a limit for any positive communication claim); reattribution
   uncalibrated.
8. **Landmark "zero circularity" is true of the ruler, not the sets**;
   only vasculature is geometry-grade (and null). Image-derived landmarks
   would fix this (second-slide upgrade).
9. **Transport supported tier is empty by construction** with data-defined
   niches; the 0.099 R² is extrapolation; κ-sensitivity was run on the
   pre-correction object.
10. **No external baseline yet** (doc-08 §8 SIMVI / resolVI deliberately
    last): every comparison so far is against the model's own references
    (floor / ℓ / linear ref / y), not a competing method.
11. **Cycle target noise**: split-half reliability S 0.22 / G2M 0.52 caps
    what any latent can retain; lung G2M 0.28.
12. **Xenium depth** (~50–300 transcripts/cell): the hard phase label is
    mostly "no signal → G1"; A4's population-level reads inherit this.

## 7. Quality verdict, open questions, next steps

**What the evidence supports at publishable grade** (multi-seed, κ-robust,
reference-cleared, certified instruments): the z/w division of labour on
HGSOC (cycle in z not w; tissue gradient / niche / Moran in w not z, all
type-partialled and against floor/ℓ); its transfer in shape to lung with
one knob; the κ-robustness of those reads; the training-time guards
(probe, mirror, NMI) as a battery; the atlas' "few, territorial, almost
entirely context-explained programs" (with programs over seeds); the
transport counterfactual at the type level (0.099, slope 0.92, 100/155)
*as extrapolation*; the doc-09 nulls (exposure invisibility, at most 1–2
fragile pairs) as a debunking result; the A4 honest outcome.

**Suggestive / single-instance**: the selected optimum's cycle_z 0.50 (0.44
on the battery subsample; 0.42–0.46 over sweep seeds); the selection-share
0.05; the ‖w‖ biological ranking; the interface landmark residual.

**Unsupported or falsified**: any per-cell w reading; the w-mirror
mechanism; the reattribution 61%; a positive communication claim; per-cell
decontamination through μ_z (A4 planted 0/3, adjudicator defective).

**Open questions for the architect**
- Doc-08 §5 matrix: split the mid-band row (interface + vessel-null) —
  decision pending; optional "selection share" row for §7.
- Amortisation gap (limitation 2): accept the counts-level correction as
  the per-cell route, or invest in semi-amortised z refinement?
- Reachability of the strong-family optimum (1–2 / 3 seeds at 500
  epochs): seed ensemble + battery selection as the standard protocol?
- Doc-11 A6 (QC via w residual) given w ≈ m_ψ — redefine on z-side
  Mahalanobis + likelihood deficit only?

**Recommended next steps, cost-ordered**
1. **Doc-08 §8 baselines** (SIMVI, resolVI) on the same folds — the one
   missing external comparison; explicitly last per instruction, now due.
2. **A4 resume**: fix V9 (within-type thresholds, excess FPR, power gate),
   add the leak-subtracted-counts arm, settle the amortisation-gap
   hypothesis. This gates A1/A2/A5's per-cell claims.
3. Doc-09 §5a planted worlds (calibrates the reattribution figure).
4. A1 (mirage states) → A3 (formalise trajectories) → A6 → A5 → A2, per
   doc-11's order, with the reference atlas pinned once.
5. Transport κ-sensitivity rerun on the counterfactual object; lung
   landmarks/atlas/transport for the transfer story.
6. α_w = 0.05 under type_only, 3 seeds at the 500-epoch budget, battery
   selection — the operating-point question reopened by the rank finding
   (pre-registered acceptance in the devlog 2026-09-14).

## 8. Registers — where things are written down

- [devlog.md](devlog.md): chronological narrative, motivations written
  before runs, results after; last entries "A4 v2 — redesign" and "A4 v2
  results — parked".
- [issues.md](issues.md): P/E/M/T/A/V tables + the watch list (updated
  2026-09-14: V1–V9 instrument issues, α_w settled, amortisation-gap and
  A4 items added; the duplicate mirror item struck).
- [spec_deviations.md](spec_deviations.md): doc-07 rows (now incl. the
  `type_only` departure, current defaults), doc-08 rows (§2, §5 rendering,
  §6.1/6.3/6.5, §7.2–7.4), doc-09 rows, doc-10 park record, doc-11 A4 rows.
- Architect documents at the repo root are received as-is:
  `07-simple-spec_7.md`, `08-validation-analyses_1.md`,
  `09-communication-experiment_1.md`, `11-z-applications_1.md`
  (`11-z-applications.md` is the pre-redesign copy). Doc-10 is not in the
  root; its record is the devlog + `data/experiments_synthetic/`.

## 9. Repository state

- Uncommitted on `main` (last commit `fa9e09f`): modified
  `discell/model/{metrics,networks,report,sweep,train,validate}.py`,
  `docs/{devlog,issues,spec_deviations}.md`, `tests/test_model_equations.py`;
  untracked `discell/applications/`, `discell/model/{atlas,transport}.py`,
  `tests/test_model_atlas.py`, `11-z-applications_1.md`, `docs/handover.md`.
  Nothing has been committed since doc-08 §6; commit before handing over.
- Tests: `uv run pytest -q` → 139 passed, 1 skipped (2026-09-15; incl. the x̃ two-pass and lr_map tests).
- Data dependencies present: `qc/nuclear_dapi.parquet` (407,120 rows,
  5,720 without nucleus), `qc/nuclear_counts.npz` (55.8 M nuclear q20
  transcripts, 37.8% of all), `data/external/` (CellChatDB, NicheNet v2
  matrix, MSigDB hallmarks 2023.2), image embeddings
  `embeddings/egomask_ego_v1.pt` (default) among 8 variants.
- Conventions: long jobs detached (`setsid nohup`); sweeps idempotent;
  Bash chaining with `&&` (the shell is fish); GPU 0/1 by
  `CUDA_VISIBLE_DEVICES`.
