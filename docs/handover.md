# DisCell — project handover (state as of 2026-09-21)

Written to crystallise everything done so far for whoever picks the project up
cold (architect or engineer). Every number carries its artefact pointer;
narrative lives in [docs/devlog.md](devlog.md), bugs and instrument defects in
[docs/issues.md](issues.md), spec decisions in
[docs/spec_deviations.md](spec_deviations.md), the work queue in
[docs/todo.md](todo.md). **A null or failed leg is a finding here, never
softened**, and several of the most informative results below are negatives.

Replaces the 2026-09-14 version, which predated `type_only` being pinned, the
atlas rewrite, the transport rewrite, the four-dataset sweeps with the held-out
section, the two leak-measurement programmes, the cell-cycle target decision and
the x̃ verdict. Claims that version carried and that are now withdrawn are listed
in §5, not silently dropped.

## 1. What this is

**Question.** In imaging spatial transcriptomics (10x Xenium Prime 5K) a cell's
measured expression mixes three things: what the cell is doing on its own, how
its neighbourhood modulates it, and transcripts that physically belong to
neighbours (segmentation spill-over). DisCell-simple
([07-simple-spec_7.md](../07-simple-spec_7.md)) is a VAE that splits a cell's
counts into

- an **intrinsic state z** (d_z = 20), encoder `q(z | x_i, ℓ_i, t_i)` — the
  cell's own counts, depth and type, never its neighbours;
- a **spatial response w** (d_w = 6), posterior `q(w | c, t, z, x)` around a
  context-conditional prior `m_ψ(c, t)`, decoded through a gene-programme
  matrix B as `softmax(a(z) + B·w)`;
- a **fixed leakage channel**, `p_i = (1 − κ) ρ_i + κ ρ̄_i` with ρ̄ the
  β-weighted neighbour rates and κ = 0.1 swept, never fitted.

The context is

```
c_i  =  Σ_j α_ij · W_src · onehot(t_j)   ⊕   Φ_i   ⊕   isolated_i
```

— a GATv2 over the neighbours' **types only** (`gat_sources = "type_only"`, the
pinned default since 2026-09-12; `type_z`, which also passed the neighbour's
`sg μ_z`, is kept only to reload old checkpoints), the 384-d KRONOS image
embedding Φ, and an isolated flag. Invariance of z to context is enforced by an
adversary (α_a = 0.3) on the block `v = [y^(−K), 12 PCs(Φ)]`.

The `type_only` case is **empirical, not mechanistic**: the mechanism spec 9
gives for it (m_ψ reconstructing the cell through neighbours' μ_z) was falsified
twice by purpose-built detectors (`experiments/w_mirror_certification.json`);
what stands is that the three-seed ablation improved recon, NMI, mirror and
cycle_w with w's spatial rows unchanged. Spec 9 and the devlog disagree here;
flagged to the architect and recorded in `spec_deviations.md`.

The claim is a division of labour that can be *tested*: z must carry intrinsic
dynamics and fail spatial tests; w the reverse.

**Datasets (four, five slides).**

| dataset id | tissue / prep | cells | labels | α_z | role |
|---|---|---|---|---|---|
| `xenium_prime_ovarian_cancer_ffpe` | HGSOC, FFPE | 407,120 | curated K = 18 | 0.007 | development slide, pinned reference, all batteries |
| `xenium_prime_human_lung_cancer_ffpe` | lung ca., FFPE | 278,324 | graphclust 33 | 0.004 | transfer slide |
| `xenium_prime_human_ovary_ff` | ovary, **fresh frozen**, 8× depth | 1,157,637 | graphclust 39 | 0.0007 | depth contrast; strongest transport read |
| `gse315411_pdltma06_11_prime_solo` / `…_10_prime_dual`, variant `pdl018d` | pediatric-lung TMA, two serial sections of one fibrotic core | 69,422 / 70,757 | shared curated K = 35 | 0.0036 | **held-out-section protocol** |

Nuclear share of assigned q20 counts is 0.45–0.58 pooled on all five slides
(`qc/nuclear_summary.json`): roughly half of every cell's counts lie outside its
nucleus on every slide — the material the leak channel acts on.

## 2. Operating point, pinned reference, seeds

| knob | value | where set |
|---|---|---|
| κ (leak) | **0.1**, swept {0, 0.05, 0.1, 0.2, 0.3, 0.4} | `TrainConfig.kappa` |
| α_z | **1/ℓ̄ per dataset** — 0.007 ovarian, 0.004 lung, 0.0007 FF, 0.0036 GSE core | CLI `--alpha-z` |
| α_w, α_a, ω | 0.1, 0.3, 1 | `TrainConfig` |
| invariance | adversary, 6 steps, lr 2e-3, hidden 64 | `TrainConfig.invariance` |
| GAT sources | **`type_only`** (era guard reloads old checkpoints as `type_z`) | `networks.DisCell`, `validate.load_run` |
| d_z / d_w / hidden / gat_dim / heads | 20 / **6** / 256 / 32 / 4 | `TrainConfig` |
| tiles / v_pcs / Φ | 4096 cells (**2048** on the 69k GSE core), 12 PCs in the invariance block, Φ full 384-d | `TrainConfig` |
| budget | **reference fits 500 / patience 40**; **sweeps 200 / 20** (FF swept at 500/40) | CLI |
| resident counts | **int16** on device, cast per tile (`< 32768` assert) | `Trainer._to_device` |
| `subtract_leak` (x̃) | **off** — final, see §4.9 | `TrainConfig` |
| `gat_sink` | **off** — option only, see §5 | `TrainConfig` |
| `weight_decay` | **0** — see §5 | `TrainConfig` |

**Pinned reference: `ablation_gat_type_only_s1`** (ovarian, seed 1, 500/40, best
epoch 64 of 104, 22 min), selected from the type_only seed triple by the full
battery, never by likelihood alone:

| run | best recon (held-out nats/count) | NMI(z, t) | cycle_z R² pooled | cycle_w | mirror R² (perm 0.016) | probe ΔCE (floor ≈ −0.05) |
|---|---|---|---|---|---|---|
| `ablation_gat_type_only` (s0) | −7.2527 | 0.667 | 0.440 | 0.007 | 0.044 | −0.009 |
| **`ablation_gat_type_only_s1`** | **−7.1924** | 0.654 | **0.499** | 0.008 | 0.046 | 0.002 |
| `ablation_gat_type_only_s2` | −7.2310 | 0.666 | 0.465 | 0.010 | 0.045 | −0.011 |

(`runs/<run>/metrics.json`: `best.recon_val`, `best.nmi`,
`final.cycle.z.r2_pooled`, `final.cycle.w.r2_pooled`, `final.mirror.r2`,
`final.probe.delta_ce`; `final.cycle.ceiling` is the pre-rename key for the
50-PC linear reference.) The 0.06 recon spread is optimisation variance over a
rugged landscape — all three seeds are clean on every instrument. s1 sits in the
"strong family" optimum that 200-epoch sweep seeds never reach
(`sweep3_k0.1_s*`: −7.2562 ± 0.0011), which is why **the reference budget is
500/40 and selection is by battery**: one member of the family exists that buys
recon and drains cycle_z (`alphaw_0.03_s2`, 0.354).

`gat_type_only_wd0_s1` reproduces the pinned run under the current tree (same
best epoch, recon within 0.003, same per-type offsets): the int16 change is
inert and the run is reproducible.

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
tasks die with the session. The shell is fish; chain with `&&`.

| step | command | output |
|---|---|---|
| preprocess (bundle, graphs, embeddings, figures) | `uv run python -m discell.preprocess --sample <sample> [--only bundle\|graph\|embed\|figures] [--donors PDL018D --variant pdl018d]` | `bundle/full.h5ad`, `embeddings/*.pt` |
| train one run | `uv run python -m discell.model.train --dataset $DS --run-name $RUN --epochs 500 --patience 40 [--seed N] [--label-key graphclust] [--alpha-z …] [--tile-cells 2048] [--variant …] [--figures-every 100]` | `runs/$RUN/{best.pt,config.json,history.jsonl,metrics.json,events.*}` |
| any sweep, any dataset | `uv run python -m discell.model.sweep --dataset $DS --param {kappa,d_w,alpha_w} --values … --seeds 0 1 2 --tag sweep3 [--report-only]` | `runs/<tag>_{k,dw,aw}<v>_s<seed>/`, `experiments/kappa_sweep_<tag>.json` |
| **held-out section** | `uv run python -m discell.model.crossslide --dataset <A> --run $RUN --eval-dataset <B> [--eval-variant …]` | `runs/$RUN/crossslide/<B>.json` |
| §7.10 degeneracy, post hoc | `uv run python -m discell.model.degeneracy --dataset $DS --run $RUN` | `runs/$RUN/degeneracy.json` |
| report | `uv run python -m discell.model.report --dataset $DS --run $RUN` | `runs/$RUN/report/report.md` + `figures/` |
| doc-08 §§3–5 battery | `uv run python -m discell.model.validate --dataset $DS --run $RUN [--analyses morans,niche,landmarks,matrix] [--n-perms 1000] [--sweep-tag sweep3]` | `runs/$RUN/validation/validation.json` |
| doc-08 §6 atlas (rewritten) | `uv run python -m discell.model.atlas --dataset $DS --run $RUN [--compare-runs …] [--compare-atlas …]` | `runs/$RUN/atlas/`, recurrence JSON |
| doc-08 §7 transport (rewritten) | `uv run python -m discell.model.transport --dataset $DS --run $RUN [--niches 10]` | `runs/$RUN/transport/{transport.json,transport_table.md,*_summary.png}` |
| doc-09 communication / §8 ladder | `…model.communication` / `…model.lr_map --dataset $DS --run $RUN` | `runs/$RUN/communication/{communication,lr_map}.json` |
| doc-11 shared data | `…applications.shared --dataset $DS --stage {dapi,nuclear-counts,nuclear-summary}` | `qc/nuclear_dapi.parquet`, `qc/nuclear_counts.npz`, `qc/nuclear_summary.json` |
| doc-11 A4 (parked) | `…applications.a4_cycle --dataset $DS --run $RUN` | `runs/$RUN/applications/a4_cycle.{json,png}` |
| x̃ planted gate (closed) | `…applications.xtilde_gate` | `data/experiments_synthetic/xtilde_gate*.{json,png}` |
| leak meter (rejected) | `…experiments.leak_meter --dataset $DS` | `experiments/leak_meter*.json` |
| transcript-flux β^T / κ_i (instrument) | `…experiments.transcript_flux --dataset $DS [--power] [--decisive]` | `experiments/transcript_flux{,_decisive}_*.json` |
| DAPI cycle gates (closed negative) | `…experiments.dapi_cycle --dataset $DS` | `experiments/dapi_cycle.{json,png}` |
| cycle-target before/after | `…model.cycle_target --dataset $DS --run $RUN` | `experiments/cycle_target_<run>.json` |
| neighbour dose | `…experiments.neighbour_dose --dataset $DS --run $RUN` | `experiments/neighbour_dose_<run>.{json,png}` |
| tests | `uv run pytest -q` → **239 passed, 1 skipped** (2026-09-21) | — |

Queue scripts that produced the current grids: `scripts/queue_2026-09-17.sh`
(two lanes, one per GPU) and `scripts/queue_2026-09-21_gse.sh`. Sweeps are
idempotent per run name (a run with a `metrics.json` is skipped before
`assemble()`), so a queue is resumable by relaunch.

External data: `data/external/CellChatDB.human.rda`,
`nichenet_ligand_target_matrix_nsga2r_final.rds`,
`msigdb_hallmarks_h.all.v2023.2.Hs.symbols.gmt`.

Code map: `discell/model/` — `networks.py` (DisCell, GATv2 incl. the optional
sink), `elbo.py`, `equations.py` (leak mixture, invariance penalty,
TypeCovariances), `prepare.py`, `train.py` (Trainer/TrainConfig/CLI),
`metrics.py`, `degeneracy.py`, `cell_cycle.py` + `cycle_target.py`, `report.py`,
`validate.py`, `atlas.py`, `transport.py`, `communication.py`, `lr_map.py`,
`crossslide.py`, `sweep.py`, `calibrate.py`, `synthetic.py`;
`discell/applications/` — `shared.py`, `planted.py`, `a4_cycle.py`,
`xtilde_gate.py`; `discell/experiments/` — `leak_meter.py`,
`transcript_flux.py`, `dapi_cycle.py`, `neighbour_dose.py`;
`discell/preprocess/`, `discell/data/`; `tests/` (24 files).

## 4. Results, by claim

Reading conventions (doc-08 §1): posterior means, within type, spatial-block
folds, every probe judged against a within-type permutation **floor**, a
log-depth **ℓ-baseline**, and where relevant the 50-PC **linear expression
reference** — a reference line, not a bound.

> **Depth qualifier on the cycle ratio (todo 3.3).** The *absolute* read is the
> claim: cycle R² from z far above the permuted control and the ℓ-baseline
> (both ≈ 0 ± 0.001 on every slide), w at zero. The **ratio** z / linear
> reference is a property of the target's reliability and the frame's strength,
> not of the model: **2.1 on ovarian FFPE (178 tx/cell), 2.1 on lung (242),
> 0.96 on the GSE315411 core (279, curated 35 classes), 0.90 on the
> fresh-frozen slide (1,401)** (devlog 2026-09-17, FF results). "z beats the
> 50-PC linear reference ~2×" is an ovarian/lung observation and must never be
> quoted as a model property. The report text and issues M6 carry the
> qualifier.

### 4.1 The disentanglement quadrant — z is intrinsic, w is context

`runs/$RUN/report/report.md`, `validation/validation.json`:

| target | z | w | floor / ℓ / linear ref | expect |
|---|---|---|---|---|
| S/G2M cycle score, within cycling types (R² pooled) | **0.50** (report, val split) / 0.44 (battery subsample) | 0.006 | −0.001 / −0.000 / 0.235 | z |
| niche label K = 10 (macro AUC, block-CV logistic) | 0.654 † | **0.767** | 0.501 / 0.585 | w |
| pseudotime tissue gradient, type-partialled (niche R², coherence) | 0.005, 0.045 | **0.566, 0.489** | (raw: z 0.60/0.63, w 0.90/0.99) | w |
| Moran's I, mean \|I\| over dims (perm null ±0.002) | 0.061 | **0.628** | — | w |
| mid-band landmark distance (R²) | −0.011 | 0.023 | −0.002 / −0.011 | w (near-null) |

† the one flagged cell: z above max(floor, ℓ) on the discrete niche label.
Triage (`morans.mu_z.triage`): the hot z dims are not y-explainable (≤ 3.6 %
R²) → benign intrinsic spatial structure read through spatial contiguity, not
context leakage; κ-reducible (+0.062 → +0.039 over the grid). Standing watch
item.

Training-time guards (all seeds): probe ΔCE at zero with the floor below it;
mirror R² 0.044–0.046 vs permuted 0.016; KL_w ≈ 0.002/dim — **w rides its
context prior; it is a context field evaluated at the cell, not a per-cell
measurement** (prior-R² 0.9998).

**z is not just t (spec §7.10, built 2026-09-17; `runs/<run>/degeneracy.json`).**
On the ovarian seed triple and `reference_best`: I(z;t)/H(t) **0.80–0.82** (a
linear-probe lower bound; high by design — the decoder gets no t), within-type
variance fraction **0.68–0.70**, and replacing every cell's z by its type mean
costs **0.118–0.148 nats/count** of held-out reconstruction with w, ρ̄ and κ
untouched. The pre-registered degenerate pattern (all three at their degenerate
ends) does not occur; the seed triple is tight (spreads 0.019 / 0.021 / 0.022).
Per dimension the within-type fraction runs 0.24 → 0.98: the scalar is a
mixture of near-pure type axes and near-pure state axes, not a uniform property.
Caveats: linear probe, `H(t)` from training frequencies, three seeds scored on
three different held-out sets, and a *substitution* rather than the spec's
one-hot-t retrain (registered as a deviation). The diagnostics now run in
`Trainer.evaluate` on every fit and were back-filled on the 18 sweep3 κ runs.

**What w buys once z is type-averaged (todo 2.3, `experiments/w_contribution.json`).**
Ten runs, six decodes each, ρ̄ and κ fixed: the context-varying part of w is
worth **0.0087–0.0097 nats/count at α_w = 0.1** and grows monotonically as α_w
falls (0.0122 / 0.0126 / 0.0166 / 0.0215 / 0.0244 at 0.1 / 0.07 / 0.05 / 0.03 /
0.02, type_z era) against **0.108–0.118 for per-cell z**. The gauge offset plus
decoder nonlinearity plus the leak mixture buy **exactly nothing** over an
empirical per-type profile lookup ((d)−(e) = −0.004…+0.003, no α_w trend).
Total recon does not move across α_w (0.008 span against a 0.06 seed spread) —
**α_w decides which channel carries the likelihood, not how much there is**, so
recon cannot adjudicate an α_w choice.

### 4.2 The κ envelope — four slides

Sweeps under tag `sweep3`, three seeds per value, 200/20 (FF 500/40);
`runs/sweep3_k*/metrics.json`, aggregate `experiments/kappa_sweep_sweep3.json`
per dataset.

| slide | recon along κ | NMI | cycle_z | mirror | type-mean-z recon gap |
|---|---|---|---|---|---|
| ovarian | plateau to 0.1, then → −7.285 (0.4) | 0.64–0.67 | 0.42–0.46 | 0.050 → 0.037 | 0.10–0.15 |
| lung | −7.254 (0) → −7.300 (0.4) | 0.660–0.665 | 0.38–0.43 | 0.031 → 0.024 | 0.149 → 0.105 |
| FF | −7.315 (0.05–0.1) → −7.326 (0.4) | 0.59 → 0.61, flat after | 0.77–0.78, 0.749 at 0.4 | — | 0.059 → 0.035 |
| GSE core | −7.210 (0) → −7.215 (0.1) → −7.256 (0.4) | 0.60–0.62 | 0.49–0.51 | 0.029 → 0.022 | 0.162 → 0.109 |

**The envelope shape replicates on four slides**: a likelihood plateau then a
decline past κ = 0.1–0.2, disentanglement reads flat, mirror falling with κ, and
the type-mean-z gap falling with κ on all four (the leak channel absorbs part of
what per-cell z carried). Guards hold at every grid point of every grid: probe
ΔCE at floor, cycle_w ≤ 0.02, I(z;t)/H(t) 0.73–0.85, within-type variance
fraction 0.59–0.72 — **z is not degenerate at any point of any grid**.

**α_w grid** (new, 0.02–0.3). Ovarian NMI rises monotonically with α_w
(0.604 → 0.657 → 0.676), FF the same and steeper (0.534 → 0.607 → 0.619), GSE
core the same (0.594 → 0.618 → 0.630), lung flat (0.659–0.667); cycle_z flat
everywhere; KL_w max closes from ~0.05 to ~0.0002 as α_w rises. **0.1 sits where
the w channel is just closed and NMI has plateaued.** Above 0.1 nothing improves
but NMI by 0.01–0.02 while the channel shuts entirely.

**α_w = 0.05 under type_only, three seeds — rejected (todo 2.2, 2026-09-21).**
`runs/alphaw0.05_type_only_s{0,1,2}`, 500/40, full battery. The w side did
exactly what 2.3 predicted: KL_w up 5–10×, effective rank 3–4 against 2,
axis-2 share 0.25–0.39 (bar 0.15), niche AUC w 0.781–0.815 against 0.768. The
veto fired on the z side: **NMI 0.627 / 0.611 against the 0.63 floor on two of
three seeds**, cycle_z 0.438 on one against a 0.44 bar. Recorded, not tuned;
**α_w stays 0.1**. The 2026-09-14 "α_w = 0.05 is a live candidate" entry is
thereby closed.

**d_w** flat within seed spread across {2, 3, 6, 8} on lung, FF (and ovarian
from 2026-09-15); **d_w = 6 stands on three slides**. Effective rank is a
(d_w, α_w, budget) property of the optimiser — fill is 1 of 2, 2 of 3, 1–2 of 6,
2 of 8 — never a tissue property, so any "N programmes" statement must name all
three. d_w = 8 costs cycle_w (0.008–0.020, the highest on record).

**How much leak is there really? Two independent attempts, both instructive.**

- **Leak meter (doc 16, `discell/experiments/leak_meter.py`) — rejected.** A
  type-pooled table from sender-specific genes and the nuclear/extranuclear
  split. The estimator is correct (planted worlds recover the table to 5 %,
  ζ to 0.02, per-cell κ to RMSE < 0.02), but on real slides **every substantive
  check fails**: 0 of 17 ovarian senders reach 30 specific genes (curated labels
  are lineage-nested, so the dominant tumour compartment has none); ζ median
  0.746 against measured nuclear fractions 0.41–0.59, i.e. the wrong side; 20 of
  66 table entries negative, which no leak model can produce; factorisation
  R² = −4.12; per-cell κ median 2.32 at 18 types; a **50× spread over five
  learn/fit tile splits**; and the two sections of one TMA correlate at
  **Spearman −0.085**. Kept as a 6 s/slide diagnostic that asks whether a
  pooled leak table is measurable at all. Answer on these slides: no.
- **Transcript-flux β^T and per-cell κ_i (`transcript_flux.py`) — instrument
  only, adoption refused.** Geometry, no labels, no specific genes: the signed
  nucleus-bisector offset of extranuclear transcripts. Replication across the
  two GSE sections **Spearman 0.843**; β^T vs β_face Spearman 0.485–0.495
  (correlated, not identical); κ_i is a share — median **0.126–0.130**, p95
  0.30–0.32, no tail at 1 on three slides; the nuclear-fraction anchor agrees in
  sign (TAFs low nuclear fraction, high κ; stromal fibroblasts the reverse), the
  anchor the leak meter could not reproduce. Then the three decisive tests
  (todo 5.4–5.7) **all fail**: the cosine-excess content statistic provably
  cannot carry a crossing bar (the p = 0.5 mixture is invariant under swapping
  the arms, so the crossing is at 0.5 for *any* gene subset); κ_i's level is
  insensitive to the transcript-to-own-nucleus relationship it is built from
  (polygon-centroid arm moves it only 0.130 → 0.117); and **the per-edge
  geometric flux share does not predict the content it stands for** (count-matched
  Spearman −0.539 / +0.685 / +0.539, slope 0.23–0.29 against a [0.5, 2] bar),
  while the same machinery recovers a planted per-edge admixture cleanly. The
  two content moments disagree by a factor 1.8–2.5 and the difference-in-differences
  is significantly negative — wrong sign for influx, logged as open anomaly W-tf10.
  **Todo 5.8 is not built**; `equations.leakage_mix` keeps its scalar κ and
  `prepare` keeps β_face.

  **What can be said about κ = 0.1:** the geometric median is 0.126–0.130
  (upper bound by construction, and band-dependent 0.076–0.171 over the 3 × 3
  sensitivity grid) and the independent gene-content estimate brackets the
  admixture at **0.11–0.26**. Combined bracket **0.09–0.25** behind κ = 0.1:
  order-of-magnitude consistency, enough to say the operating point is not off
  by a factor of five, **not enough to call κ = 0.1 measured**.

Partial-isolation strata (`recon_strata`): cells that lost edges to the 40 µm
prune reconstruct *better* (−7.04 vs −7.26) at every κ — unexplained, still on
the watch list.

### 4.3 Landmarks and the allegiance matrix (doc-08 §2, §5)

`validation.json: landmarks, matrix`. Inventory: vasculature = endothelial ∪
pericytes (206 instances, 8,387 cells), compact smooth muscle 20–500 cells
(238 / 18,578), kNN-smoothed tumour/stroma interface (36,062 cells, 8.9 % of the
slide). Per-band fits carry a **y-baseline** because the sets are type-defined:
interface mid-band w 0.054 vs y-baseline 0.036 → ~60 % of the w signal is
definitional; **vasculature is the only geometry-grade test and is a certified
null** (mid-band y −0.004, w 0.007). θ cross-type cosine 0.05–0.37. On
cluster-labelled slides `landmark_inventory` returns empty (type *names* are
matched) — guarded since 2026-09-21 so the atlas no longer crashes, but §2
landmarks and the §5 matrix remain **unrunnable on lung, FF and any graphclust
slide**; naming the clusters would unlock them.

### 4.4 The w-programme atlas (doc-08 §6) — rewritten 2026-09-21

`runs/<run>/atlas/`. The rewrite closes issues V10/V11/V12 and todo 1.4/1.5; the
old atlas would have misread every sweep run on disk. What changed: activity by
the **effective rank r of cov(μ_w)** on **within-type gauge-centred** w
(components ≥ 1 % of variance), varimax *within* the r-dim subspace, each
programme's variance share reported; stability by **shift-space overlap of
μ_w·B** plus per-axis cross-seed cosines (matched-column correlation dropped);
enrichment by a Mann–Whitney rank test on the full loading vector against the
expressed-panel background, BH per programme, q ≤ 0.05; the landmark driver
block skipped when the inventory is empty; label recurrence across seeds and
across slides; **a programme whose label does not recur ships as *unlabelled*,
never named**. 12 planted tests certify it (rank-2 w in 6 dims reads r = 2 where
the old read said 6/6; a permuted sign-flipped basis reads cosine 1.0).

- **Ovarian seed triple (α_w = 0.1):** r = **2 of 6** on all three seeds, spectra
  [0.95, 0.05] / [0.62, 0.38] / [0.94, 0.06]. Axis-1 cross-seed |cos| 0.93–0.98,
  axis-2 0.40–0.83; shift-space overlap 0.67–0.93. Dominant programme = the
  macrophage/stromal axis (F13A1, MRC1, TNXB, KLF4) on every seed, Moran I
  0.37–0.47, **joint context-driver R² 0.92** with Φ the largest partial
  (0.25–0.31) and landmarks ≈ 0; second = the matrix axis (COMP, SFRP4, COL10A1,
  COL11A1), Moran 0.61–0.69. Labels recurring in ≥ 2 of 3 seeds: **EMT and
  HYPOXIA (3/3)**, E2F_TARGETS and G2M_CHECKPOINT (2/3).
- **Cluster-labelled slides:** lung r = 1 ([0.999, 0.001]), Moran 0.42, joint
  0.87, EMT (q < 1e-4). FF r = 3 ([0.55, 0.27, 0.18]), labels EMT,
  MYC_TARGETS_V1, EMT/HYPOXIA; its second programme carries 27 % of w's variance
  but **71 % of the realised shift** — the two shares are reported separately
  for this reason.
- **Cross-slide recurrence** (ovarian s1, lung, FF): EMT 3/3, HYPOXIA 2/3,
  KRAS_SIGNALING_UP 2/3.
- **α_w = 0.05 triple** supplies the cosines the 2.2 verdict lacked: r = 4 / 3 /
  3, and a **third programme with the same signature in all three fits**
  (FOXL2, SFRP4, POSTN, GRIA2, WNT4, GREB1), Moran I 0.84–0.88 — the most
  territorial programme seen — most modulated in stromal fibroblasts and smooth
  muscle, Φ-driven (partial 0.50–0.57 vs composition 0.02–0.09), and
  **unlabelled** (no hallmark at q ≤ 0.05 on 2 of 3 seeds). The extra axes do
  reproduce; it does not reopen the α_w decision, whose veto was on the z side.

κ-survival (`experiments/atlas_kappa_survival*.json`) is unchanged and
**sweep-internal** (0.29–0.43, flat over κ); it has not been regenerated under
the new basis and is now the one w-stability read not in shift space — the
weakest of the three concordance reads (todo 1.6).

### 4.5 Counterfactual transport (doc-08 §7) — rewritten 2026-09-21

`runs/<run>/transport/{transport.json,transport_table.md,*_summary.png}`,
`experiments/transport_kappa_sensitivity_v2.json`. This is the one experiment
that exercises the whole system — z held fixed, the response channel through
m_ψ and B, the leak channel through β and κ — where every other read isolates a
subsystem.

**What is predicted, plainly.** For one cell type and two neighbourhoods A and
B: the per-gene log-rate shift a cell of that type undergoes going from A to B,
from two channels added — the **programme** channel (m_ψ at B's mean context
minus at A's, through B) and the **leak** channel (κ times the difference in
mean foreign influx; κ never changes inside a prediction). Held fixed: the
cell's intrinsic z. Scored on held-out tiles against the observed
depth-normalised mean shift, both sides mean-centred, as R² against the
zero-prediction null plus a calibration slope. The "model account" additionally
lets the type's intrinsic mix differ between niches; its excess is the
**selection share**.

- **Headline, pinned reference:** counterfactual beats both single channels in
  **100/155 (65 %)**, median slope **0.92**, mean R² **0.099** — reproduces the
  record exactly after a substantial rewrite. Seed triple 0.085 / 0.099 / 0.092,
  slope 0.91–0.97; selection share 0.029–0.047 (about a third of an observed
  niche difference is *which cells live there*).
- **The noise ceiling — the main clarity gain, and it changes the reading.**
  Every panel now carries the Spearman–Brown split-half reliability of the
  *observed* shift: the largest R² any predictor could reach. On ovarian the
  mean ceiling is **0.123** — 141 of 155 composition panels are essentially
  unmeasurable, and the 0.099 headline is a mean over mostly noise. On the **14
  trusted panels (ceiling ≥ 0.5 on ≥ 100 genes) the counterfactual reads 0.205
  at slope 1.08, beating both channels in 13/14**, taking 34 % of what is
  reachable. The ceiling is identical at all six κ, as it must be. **The
  transport R² was never small because the model is weak; it was small because
  most panels contain almost nothing measurable. Never quote 0.099 without
  0.205 beside it.**
- **Annotation niches.** Six ordered bands of the kNN-smoothed tumour fraction
  (cuts 0.1/0.3/0.5/0.7/0.9 fixed before any result). Being nested they share
  composition support, which k-means niches cannot: **supported tier 71 panels**
  (73 / 75 on the other seeds), 13 types, R² 0.068, slope 0.79 (0.88 / 0.88
  elsewhere — the one marginal miss, recorded), beats both 42/71; best panel
  Tumor Cells rim → core R² 0.322. **Handover limitation "supported tier empty
  by construction" closes for annotated slides.** Slides whose type names name
  no tumour fall back to composition niches, named as such.
- **The Φ question, answered against expectation.** Freezing Φ at the receiver
  type's mean bites (the programme channel moves by > 0.02 in 47/155 panels) yet
  leaves the total counterfactual unchanged on every slide: **interventionable
  share ≈ 1** (1.04; per-panel median 0.99, IQR 0.94–1.06; envelope 0.86–1.13).
  Neighbour-dose measured Φ's share of *cell-to-cell* variation within a type;
  transport asks about differences of *niche means*, and Φ's cell-to-cell part
  averages out inside a niche. The pre-registered worry that a composition
  counterfactual carrying real Φ mixes intervention with description comes back
  **negative**.
- **κ as a range, on the corrected object** (six sweep3 κ seeds): leak 0.000
  (κ = 0, sanity) → 0.084 (0.3), programme 0.056 → 0.042, total peaking at 0.086
  (κ = 0.2), slope falling 0.99 → 0.60 and leaving the [0.8, 1.2] band at
  κ ≥ 0.3. **Quotable range κ ∈ [0.05, 0.2].** The pre-correction
  `transport_kappa_sensitivity.json` (which scored the *model account*, not the
  counterfactual) is superseded and must not be quoted.
- **Slides.** Lung 158 panels, 0.082, slope 0.84, 92/158. **FF now runs** — the
  exit-137 kill was the instrument holding per-cell rate matrices (23 GB on
  1.16 M cells); it accumulates per-(niche, type) means in the forward pass now,
  ~3 min — and is the strongest read in the programme: **248 panels, R² 0.280,
  beats both 223/248 (90 %), 140 trusted panels at 0.391**, ceiling 0.51; slope
  1.25, just outside the band. **GSE core: ceiling 0.049, zero trusted panels —
  it does not reach the noise floor, and its 0.062 is not a transport result.**

Stale pre-correction `transport.json` files remain on unrelated runs
(`gat_sink_*`, `xtilde_*`, `alphaw0.05_*`, `wd0`) and **must not be compared to
the new numbers**.

A distribution-level companion (MMD, pairwise and leave-one-niche-out) is
**pre-registered and not yet run** (devlog 2026-09-21).

### 4.6 Communication (doc-09) — a debunking instrument

`runs/<run>/communication/{communication,lr_map}.json`. Gate zero: 618 CellChat
pairs with ≥ 20 in-panel NicheNet targets.

- **Exposure visibility is null in every arm** (mean w-R² −0.010 across
  reference, s1, s0 and α_w = 0.05) and under `type_only` this is
  **architectural**: c carries no channel for neighbour expression detail.
- Programmes: at most 1–2 fragile candidates — PDGFB→PDGFRB, POSTN→ITGAV/B5 —
  **nothing clears in 3/3 arms**. The paper reports the tally, not a winner.
- **Doc-09 §8 ladder** (`lr_map.py`, three seeds, three rungs kept): uncontrolled
  A′ |ρ| up to 0.7 with 20–29 of 60 entries surviving a Moran-preserving null
  (that is the SIMVI-comparable view, and it is type composition); within
  receiver type max |ρ| 0.26–0.34 with 1–10 surviving; composition-partialled
  max |ρ| 0.14–0.16 and **0 survivors in every seed**. Side lesson the figure
  demonstrates on itself: a plain permutation null on ~60k spatially smooth
  cells is not a null — its counts barely fall down the ladder while effect size
  collapses 0.7 → 0.3 → 0.15.
- Reattribution (61 % of naive exposure-associated genes leak-attributed) is
  **uncalibrated** — the §5a planted worlds never ran; **do not quote**.

### 4.7 The cell-cycle target — decided 2026-09-21, after two negative programmes

**Decision: the plain Scanpy S/G2M marker score stays the target**, used
continuously, never as a hard phase call, and stated as an imperfect reference
rather than a gold standard. The claim is the *asymmetry* (z predicts it, w does
not) against the permuted floor, the depth baseline and the split-half
reliability as ceiling.

- **No DNA-content label exists on either slide.** Ovarian
  (`experiments/cell_cycle_dapi_analysis.json`): nucleus overlap is 0.179 % of
  nuclei and is *not* the mechanism; integrated DAPI is nuclear footprint area
  (log–log slope 1.035, R² 0.734) with ~12 % background, a 2.65× tile drift and
  a segmentation-route swing 0.17 → 0.86; no variant clears AUROC ≥ 0.70 or
  MKI67 ratio ≥ 2 (median AUROC 0.45–0.49). Fresh frozen, the only remaining
  candidate, through six pre-registered gates
  (`experiments/dapi_cycle.{json,png}`): **every thresholded gate fails, the
  first at gate 1** — flat-field residual 1.55× (bar < 1.2), background 5.0 %
  (< 3), R² on nuclear area **0.853** (< 0.1), bimodality 0 of 38 types (dip
  test 0/38 while BIC alone would have licensed 36), AUROC 0.543 and MKI67 ratio
  1.59. The blocker is physics — a single projected focus plane through a ~5 µm
  section leaves the sectioned fraction of each nucleus unobserved and varying by
  more than the twofold that *is* the 2N/4N signal. Fourth confirmation after
  A4 leg 1, A4 v2 and the ovarian gating analysis. Todo 3.5 closed negative.
- **The "joint" DAPI × Scanpy consensus label of
  `docs/cell_cycle_comparison_report.md` is not defensible and is not adopted**:
  its marker enrichment is the depth axis of the DAPI gate (the 4N gate is 49 %
  top-depth-decile cells against the 2N gate's 7 %); **within depth deciles the
  two MKI67 rates are equal**, depth-matched ratio 1.13. Its niche-border
  enrichment claim for the "leakage candidate" class is unsupported.
- **The depth-neutral target was built and not adopted** (todo 3.6,
  `discell/model/cycle_target.py`, `experiments/cycle_target_<run>.json`). Depth
  neutrality passes cleanly (worst within-type |Spearman(score, log counts)|
  0.356 → 0.016 ovarian, 0.252 → 0.009 lung, zero violations), but split-half
  reliability **falls to 0.082 / 0.160 (ovarian S / G2M) and 0.078 / 0.111
  (lung)**: most of the old score's agreement with itself was depth. cycle_z
  moves −0.08 (ovarian) and −0.16 (lung), attributed cleanly — depth
  conditioning itself costs 0.022 / 0.003, the rest is the rank transform's
  tie-breaking at 50–300 transcripts per cell. Kept as a **robustness column**,
  not a method change; a transformed target would look like a workaround for an
  issue the controls already show is absent.
- **Carried forward as a caveat beside every cycle R²: the honest reliability
  ceiling is ~0.16 for G2M on ovarian, not 0.52** — the old figure was inflated
  by depth. FF is the reliable slide (S 0.52 / G2M 0.69).

### 4.8 Transfer, and the held-out section

**Annotation-free control** (ovarian with graphclust labels,
`experiments/graphclust_comparison.json`): recon −7.2556, NMI vs curated 0.646,
cycle_w 0.004, **but the invariance probe re-fitted against the *curated* labels
reads ΔCE 0.061** (reference 0.004, floor −0.056): with coarser training labels
the adversary guards a coarser target and finer-label niche information stays in
z — the label-granularity cost of annotation-free operation.

**Lung** (`…lung…/runs/reference_graphclust`, α_z 0.004, 33 clusters): recon
−7.2577, NMI 0.653, cycle_z 0.347 vs linear ref 0.166, cycle_w −0.000, probe
ΔCE −0.010 (floor −0.021), mirror 0.036, Moran |I| w 0.428 / z 0.059, niche AUC
w 0.731 / z 0.617 / ℓ 0.544. **The allegiance structure transfers with one knob
(α_z) changed.**

**Fresh frozen** (`…ovary_ff…/runs/reference_graphclust`, α_z 0.0007, 39
classes, 409 epochs / best 369, 183 min): recon −7.3138, NMI 0.595 (drifting
down from 0.643 while recon improves — the §7.10 guard *engaged* and blocked two
recon-improving checkpoints), KL_z **1.22 nats/dim** (the most open z of any
slide), probe ΔCE 0.0054 vs floor −0.0133, mirror 0.055 / 0.016, cycle_z
**0.766** / cycle_w 0.0027, KL_w ≤ 0.0001/dim at the checkpoint (12–30× below
the FFPE slides — α_w sits 140× above 1/ℓ̄ here). **Pass; the α_z bracket does
not fire; nothing is tuned.** Single seed — nothing bounds FF's envelope.
Practical note: **69 % of that run's wall time was TensorBoard figures**;
`--figures-every 100` makes a 500-epoch FF fit ≈ 1.2 h.

**The held-out section (GSE315411, `runs/sweep3_*/crossslide/`) — the strongest
stability statement the programme has.** Train on the `pdl018d` core of section
11, evaluate every swept checkpoint on section 10 under the same 35-class
vocabulary (`discell/model/crossslide.py` asserts vocabulary equality; it
refuses a mismatch rather than remapping). At **every grid point of all three
grids**: reconstruction over all 64 dual tiles is **0.015–0.017 nats/count worse**
than the same-section best (a quarter of the ovarian seed envelope); NMI −0.02
to −0.03; **cycle_z on the held-out section 0.505–0.527, slightly *above* the
same-section 0.49–0.51**; cycle_w ≤ 0.009; probe ΔCE at its floor; mirror
0.033–0.043; I(z;t)/H(t) 0.76–0.79. **The section-to-section generalisation cost
is a constant, independent of κ, d_w and α_w** — the disentanglement reads
survive a change of section at every point of three grids.

The shared-label pipeline behind it is itself a finding
(`scripts/annotate_gse315411/`): **scANVI/scArches transfer is a failed
instrument at this depth** — 60 % of cells called "Alveolar fibroblasts" at
median max-probability 1.000, the majority HLCA label in 26 of 44 clusters, and
every one would have passed the pre-registered gates; the CellRef leg collapses
harder (97 % → CAP1) and the two label sets agree at ARI 0.026. Replaced by
pseudobulk correlation plus marker curation; 35 classes, vocabulary identical on
both slides, cross-slide composition JSD 0.0005, ~19 % Unassigned on the full
slides (core-level low-depth tissue, characterised, kept in the graph as
neighbours) and 3.7–4.8 % on the `pdl018d` core.

### 4.9 x̃, the amortisation gap, and what replaced it

**x̃ = x − κℓρ̄ as the encoder input (`--subtract-leak`): option-only, final
(todo 2.1, closed 2026-09-21).**

- On the slide (three seeds, 2026-09-14): recon identical (±0.001), niche-z
  residual −0.006 mean (−0.011 / −0.013 / **+0.005**), cycle_z inside envelope,
  w-side rows unmoved, **NMI −0.017 consistent**, Moran-w down in 2/3. Near-neutral.
- In a **powered planted world** (`xtilde_gate.py`, V9's defects fixed:
  within-type thresholds, excess FPR = victim − control, power gate raw AUROC
  ≥ 0.9 and raw excess ≥ 0.1; `data/experiments_synthetic/xtilde_gate*.json`):
  the gate passes 6/6 seeds; z beats raw 3/3 (excess 0.265/0.370/0.379 →
  0.106/0.177/0.118); **x̃-z beats plain z on 1 of 3, then on 3 of 12** with
  **3 reversals** over the decisive six-seed follow-up.
- At a **weakened (unsaturated) plant**, x̃ is **worse than plain z on 3 of 6
  seeds with the CI above zero**, and the z probe's own AUROC collapses to
  0.61–0.92 against raw's 0.82–0.96: z is blunting, not decontaminating. The
  saturated-plant benefit was an artefact of a plant too strong to lose anything to.
- With **genuinely cycling victims**, the doc-11 sensitivity-loss fail state
  fires: recall Δ(x̃-z − z) +0.014 / −0.060 / **−0.221** — up to 22 points of
  recall on real cycling cells.
- **Verdict: `subtract_leak` stays default off; spec-07 §7.13 stays parked; no
  κ-sweep rerun.** The author's condition ("if it improves trust in z") is not met.
- Two findings survive the exercise. (i) **The amortisation-gap hypothesis as
  stated is refuted**: per-cell z carries only 31–48 % of raw's excess, so the
  amortised posterior mean does *not* simply inherit what leaked into x_i — the
  penalty and the population-level z law remove most of it. (ii) **The
  counts-level route sits on its own ceiling**: the model's ρ̄ and the *true* ρ̄
  give the same excess to three decimals, so subtracting a mean from a
  multinomial draw removes about a quarter of the excess and no more.
- **New watch item, independent of x̃ and more important than it: the z probe
  itself loses 0.25–0.38 of raw's recall on genuinely planted cycling cells at
  an unsaturated plant.** That bears on every per-cell z claim (doc-11
  A1/A2/A5) and is the finding to carry forward.

### 4.10 Baselines — surveyed, not yet run (todo 4.1)

Devlog 2026-09-21; no installs, no runs; five `uv pip install --dry-run`
resolutions against the live env.

- **The bracketing is confirmed and sharper than expected.** SIMVI = our split
  *without* a leak channel (intrinsic + spatial-induced, annotation-free;
  largest published dataset ≈ 33k cells). resolVI = our leak channel *without* a
  split (one latent, true/diffusion/background mixture with per-cell mixture
  proportions; 1.4 M cells in < 6 h on a 3090). MintFlow sits between (three
  latents, in-silico microenvironment perturbation — a counterfactual analogue
  for the *programme half* of our transport — but no contamination model, and
  labels required). scVIVA and NicheCompass are w-side only. **No published
  method occupies both axes, which is the paper's claim.**
- **Install plan.** `scvi-tools` 1.5.1 and `scviva-tools` 0.1.7 resolve into the
  existing env with zero downgrades (resolVI, scVIVA need only an extra).
  MintFlow would downgrade zarr 3.3 → 2.18 and break the tifffile image path →
  own venv, mandatory. SIMVI pins `scvi-tools ≤ 0.16.2` → own venv on python
  3.10, behind a tutorial-reproduction gate. NicheCompass drags mlflow → isolate.
- **Order and cost:** resolVI first (~12 GPU-h; it owns the contamination row
  nothing else fills), SIMVI second (~10–18), MintFlow third (~20–26), scVIVA /
  NicheCompass last (~7–9). **≈ 50–65 GPU-h total; stages 1–2 (~25 GPU-h)
  already deliver the bracketing claim.** Fairness pre-registered: same tile
  split and fold map, same labels, our graph (and theirs where a method
  insists), d_z 20 / d_w 6 matched, matched wall clock, three seeds, every number
  through `validate.py` against floor / ℓ / linear reference, empty cells read
  "n/a by construction", baseline spatial latents gauge-centred and compared in
  shift space.
- Caveats: `08-validation-analyses_1.md` has no §8 in this revision (the
  bracketing framing was reconstructed from the handover and the register); no
  runtime figure is published for four of the five methods; dry-runs prove
  resolvability, not importability.

## 5. Retractions, falsified hypotheses, parked work

| item | what was claimed | what killed it | what remains |
|---|---|---|---|
| **Per-type ‖w‖ ranking** (report, old handover §4.1, paper app:w-norm) | VEGFA⁺ tumour most context-responsive, stable over 18 fits | issues **V12**: ‖w‖ is an unidentified per-type gauge — `(a(z) − Bμ_t, w + μ_t)` is the same model. Offsets are 4–10× the within-type spread; Spearman(raw rank, centred rank) **−0.41**. Coupled weight decay makes it *worse* (global offset 23, ‖B·mean_t w‖ 81–92) because L2 decays parameters, not a network output | **withdrawn in place.** The read is the **within-type-centred** spread (macrophages, SOX2-OT⁺ tumour, T/NK most modulated; VEGFA⁺ tumour least), or a programme coordinate's within-type variance. Any "clean profile" decodes at `softmax(a(z) + B·m_ψ(c̄_t, t))`, never at w = 0 |
| **"6 programmes, 0 spare"** (atlas) | six active w dimensions | issues **V10**: activity was judged on varimax-rotated variance; a rank-2 w rotated onto 6 axes gives six collinear coordinates that all pass | **superseded.** Programmes = effective rank of cov(μ_w) on gauge-centred w; ovarian r = 2, lung 1, FF 3. Rank is a (d_w, α_w, budget) property — name all three |
| **Matched-column B stability** | κ > 0 destabilises B (cosine 0.43–0.52 vs 0.70 at κ = 0) | issues **V11**: the metric reads the four *null* w-directions, which carry no variance and whose B columns are therefore arbitrary | **replaced** by shift-space overlap of μ_w·B (0.97–0.99 across seeds) + per-axis cosines. **The κ = 0 → κ > 0 drop is still unexplained** and must be said so |
| **Old transport κ-sensitivity** (`transport_kappa_sensitivity.json`) | κ-robustness of the counterfactual | it was run on the *pre-correction* object (the model account, not the counterfactual) | **superseded** by `transport_kappa_sensitivity_v2.json`; quotable range κ ∈ [0.05, 0.2] |
| **α_w = 0.05 as a live candidate** (2026-09-14) | a second context axis becomes seed-stable and pays | run under type_only at 500/40, three seeds: **NMI 0.627 / 0.611 against a 0.63 floor** on two seeds (todo 2.2) | **rejected**; α_w = 0.1. The w-side prediction was *correct* (channel opens, rank 3–4, extra axes reproduce) — the price is on the z side |
| **w-mirror mechanism** | the high-recon runs at α_w < 0.1 were m_ψ reading neighbours' μ_z | two pre-registered detectors failed certification; recon survives replacing w by a linear function of composition; s1 reached −7.19 with neighbour z structurally absent | the **instability** at α_w < 0.1 is real; the high-recon family is a **better optimum**; `type_only`'s case is **empirical only** |
| **Doc-10 z–w guard** | a Gaussian-MI guard rescues low α_w | arm 0 (20 configs): guard-on never holds what guard-off loses | PARKED, **code removed**; a second attempt must first reproduce the historical M3 cliff |
| **A4 decontaminated cycle call** | z rejects leak-induced cycle false positives | v1 instrument-limited; v2's contamination-specific gene-split fingerprint is **null** (Δ ring 1 0.006, CI [−0.006, 0.020]) | honest outcome: **leak-induced cycle false positives are rare at κ = 0.1 on this slide**; parked. Its planted leg has been **retired into `xtilde_gate.py`**, which fixes V9 |
| **x̃ as a default** | a bias fix that improves trust in per-cell z | 12-seed powered world: every clause of the rule fails; reverses at an unsaturated plant; −22 pts recall on real cycling cells | **option-only, final** |
| **`gat_sink` as a default** | letting c count neighbours should help | 3 paired seeds: transport ≥ reference in 1/3; the fitted half-point sits **below one neighbour** (Voronoi degree ≈ 6 gives no dose variation to learn from); in 2/3 seeds m_ψ reads context *worse* (posterior displacement 6×, 64 % context-predictable, zero reconstruction gain) | **option, off by default.** Earns a second look only with a variable-degree graph *plus* a degree-aware invariance target, and with m_ψ fed both the softmax composition and the sink mass |
| **Weight decay as a gauge fix** | Adam L2 would collapse the per-type offsets | offsets grew ~4× and became one global vector; the decoder's gene bias migrated into `B_5 · w_5` | `--weight-decay` stays at 0. A training-time fix needs a penalty on `E[m_ψ]` per type, not on parameters |
| **Doc-16 leak meter** | a per-cell κ_i from a type-pooled leak table | every substantive check fails on three slides (§4.2) | instrument only |
| **Transcript-flux κ_i adoption** | a measured per-cell leak vector replacing the κ grid | the three decisive tests all fail; per-edge flux share does not predict content (slope 0.23–0.29 against [0.5, 2]) | instrument + reporting statistic; the 0.09–0.25 bracket behind κ = 0.1 |
| **DNA-content cycle label** | an independent cycle target from DAPI | ovarian: the integral is nuclear area; FF: all six gates fail, the first at gate 1 | the marker score, with per-slide reliability stated |
| **"Joint" DAPI × Scanpy label** | a consensus cycle label | its enrichment is the DAPI gate's depth axis; depth-matched MKI67 ratio 1.13 | not adopted |
| **Hallmark labels on atlas programmes 2–5** | SPERMATOGENESIS, ADIPOGENESIS, HEDGEHOG, G2M | shipped ungated (p 0.08–0.37) | BH gate; a non-recurring label ships as *unlabelled* |
| **"Ceiling" for the cycle probe** | the 50-PC ridge is an upper bound | z exceeds it on the shallow slides | renamed **linear expression reference**; and the *ratio* now carries the depth qualifier (§4) |
| **M6 mean-of-types cycle R²** | headline | dominated by non-cycling types' noise | headline = **pooled**, cell-weighted. A sweep-report defect that resurrected `r2_mean_types` was fixed 2026-09-21 |

## 6. Limitations and threats to validity (ranked)

1. **Type labels enter everywhere** — `enc_z`, `enc_w`, `m_ψ`, the adversary, the
   GAT sources, the landmark sets. z is type-informed by design and every
   "within type" read is conditional on labels that are themselves
   expression-derived. Mitigations in place: y-baseline rows, the graphclust
   control (structure survives relabelling), and now a *second* vocabulary
   (GSE315411's curated 35 classes) with a genuine held-out section. Not
   mitigated: validation at finer-than-t granularity (doc-11 A5's guard).
   Measured cost of coarse labels: probe ΔCE 0.061 against curated labels.
2. **Per-cell z claims are not supported by a powered test, and one powered test
   argues against them.** The amortisation-gap hypothesis is refuted as stated,
   but **the z probe loses 0.25–0.38 of raw's recall on genuinely planted cycling
   cells at an unsaturated plant** (§4.9). Every doc-11 A1/A2/A5 per-cell claim
   is at risk until that is settled, and the counts-level correction is at its
   own oracle ceiling, so it is not the escape route either.
3. **κ is swept, not measured.** Two independent measurement programmes failed
   (§4.2). What exists is an order-of-magnitude bracket, **0.09–0.25**, and a
   geometric per-cell statistic whose level does not depend on the nucleus it is
   built from and whose per-edge value does not predict the content it stands
   for. κ = 0.1 is a defensible grid point, not a measurement.
4. **w is a context field, not a per-cell measurement** (KL_w ≈ 0.002/dim,
   prior-R² 0.9998; the context-varying part is worth ~0.009 nats/count against
   ~0.115 for z). The spec's deviation/anomaly channel is closed at the
   operating point, so doc-11 A6 (QC via a w residual) would be reading m_ψ
   noise. Opening it costs NMI (§4.2).
5. **w is over-provisioned and any column-wise B metric reads its null space.**
   Fill is 1–2 of 6. The strong-family optimum is reached in 1–2 of 3 seeds at
   500 epochs and never at 200, so **the pinned model is a *selected* optimum**;
   report by effective rank and shift space, and always with the seed envelope.
6. **KL_w is a snapshot of a chase**, not a property of a solution: it oscillates
   20–100× between evaluations five epochs apart in every seed, and one seed's
   `best.pt` sits on a spike. Quote it over the trajectory.
7. **The cycle target is weak and its ceiling was overstated.** Honest split-half
   reliability is ~0.08–0.16 (ovarian S / G2M) once depth is removed, against
   0.22 / 0.52 as previously quoted; FF is the only reliable slide (0.52 / 0.69).
   Every cycle R² must carry its ceiling. There is **no** independent DNA-content
   target on any slide.
8. **Most transport panels are unmeasurable.** Mean noise ceiling 0.123 on
   ovarian (141 of 155 panels essentially noise); the GSE core has **zero**
   trusted panels. Conclusions rest on the trusted tier (ovarian 14 panels, FF
   140) and on the annotation-niche supported tier (71 panels), not on the
   all-panel mean.
9. **The niche-z residual** (AUC 0.654 vs ℓ 0.585) stands as a flagged,
   triaged-benign watch item; x̃ moved it by ~9 % of the gap and cost NMI, so it
   is not the fix. A future operating point pushing it toward w's level must
   rerun the triage.
10. **Communication readout is architecturally blind** to within-composition
    ligand variation — a feature for the debunking story, a limit for any
    positive claim; reattribution uncalibrated and unquotable.
11. **Landmark "zero circularity" is true of the ruler, not the sets**; only
    vasculature is geometry-grade, and it is null. On cluster-labelled slides
    §2 and §5 do not run at all.
12. **No external baseline has been run.** The survey exists and is costed
    (§4.10); every comparison so far is against the model's own references.
13. **Absolute numbers are slide-specific.** Held-out reconstruction in nats per
    count is **not comparable across slides**; only shapes, envelopes and
    asymmetries transfer. The one genuine out-of-sample read is the GSE
    held-out section, on a single 69k core of one donor.
14. **Xenium depth** (~100–300 transcripts/cell on four of five slides): marker
    scores are noise-dominated, hard phase calls are mostly "no signal → G1",
    and several instruments (scANVI transfer, per-gene content statistics) fail
    for this reason alone.
15. **Ring-2 encoder work is wasted under `type_only`** (`posterior_z` still runs
    on seeds ∪ ring1 ∪ ring2 though ring-2 codes are unused), so the measured
    halo overhead is an upper bound on what the design needs (todo 1.6).

## 7. Quality verdict, open questions, next steps

**Publishable grade** (multi-seed, κ-robust, reference-cleared, certified
instruments, and now replicated on four slides):

- the z/w division of labour on HGSOC — cycle in z not w; tissue gradient,
  niche and Moran in w not z, type-partialled and against floor/ℓ;
- its transfer in *shape* to lung, fresh frozen and a pediatric-lung TMA core
  with one knob (α_z) changed;
- **the κ / d_w / α_w envelopes on four slides**, with the same envelope shape
  everywhere and every guard at floor at every grid point;
- **the held-out-section result**: a constant 0.015 nats/count and 0.02–0.03 NMI
  cost at every point of three grids, cycle_z not degraded;
- **z is not a re-encoding of t** (§7.10 battery, seed-tight, now on every fit);
- the training-time guards (probe, mirror, NMI, degeneracy pair) as a battery;
- the atlas's "few, territorial, almost entirely context-explained programmes"
  reported at effective rank with cross-seed and cross-slide label recurrence;
- **the transport counterfactual on its trusted tier** (ovarian 0.205 at slope
  1.08, 13/14; FF 0.391 on 140 panels) and the interventionable share ≈ 1;
- the doc-09 nulls (exposure invisibility, at most 1–2 fragile pairs, zero
  survivors of a Moran-preserving null after composition control) as a
  **debunking** result;
- the negatives as negatives: the leak-measurement programmes, the DAPI target,
  x̃, the sink, weight decay.

**Suggestive / single-instance**: the selected optimum's cycle_z 0.50; the
selection share ≈ 0.03–0.05; the FF slide's everything (one seed); the interface
landmark residual; the α_w = 0.05 third programme (FOXL2/SFRP4/POSTN, Φ-driven,
Moran 0.84–0.88, unlabelled) — reproduced across three seeds but at a rejected
operating point.

**Unsupported or falsified**: any per-cell w reading; the w-mirror mechanism;
the reattribution 61 %; a positive communication claim; per-cell decontamination
through μ_z; a *measured* κ; per-type ‖w‖ as a response ranking; "six
programmes"; an independent DNA-content cycle label.

**Open questions for the architect**

- The **z-probe recall loss** (0.25–0.38 of raw's recall on genuinely planted
  cycling cells at an unsaturated plant) — does this sink doc-11 A1/A2/A5, or
  does it call for a different per-cell caller? This is the most consequential
  open item in the project.
- Spec 9 §4.1/§7.2 motivates `type_only` with a mechanism the devlog falsified
  twice. The paper states the empirical case; the spec should be reconciled.
- Spec §7.13 (x̃) stays parked with the flag off — confirm, or close the section.
- Spec §7.10's literal one-hot-t decoder retrain: the post-hoc substitution reads
  0.118–0.148 nats/count; is the retrain worth a GPU job?
- Reachability of the strong-family optimum (1–2 of 3 seeds at 500 epochs): is
  seed ensemble + battery selection the standard protocol to write down?
- Doc-11 A6 (QC via a w residual) given w ≈ m_ψ — redefine on z-side
  Mahalanobis + likelihood deficit only?
- Doc-08 §5 matrix: split the mid-band row (interface + vessel-null)?

**Recommended next steps, cost-ordered**

1. **resolVI on the ovarian tile split** (~12 GPU-h, needs the `scvi-tools`
   extra approved) — the one row no other method fills, and stage 1 of the
   bracketing claim (todo 4.2).
2. **Todo 1.6 (code)**: skip the ring-2 encoder pass under `type_only` (prove
   `metrics.json` identical on a fixed-seed smoke fit), and move κ-survival from
   matched signature correlation to shift-space overlap so all three concordance
   reads live in the same space.
3. Read what the sweeps have not yet been read for: per-value B / shift-space
   stability and the per-type rows in the regenerated reports.
4. The **distribution-level transport read** (MMD, pairwise and
   leave-one-niche-out), pre-registered 2026-09-21 and not yet run — the
   leave-one-out version is a direct test of whether z is context-free.
5. Paper section B and C items of `submission_paper/aistats/revision_2026-09-17.md`
   (numbers return only after the sweeps are fully read).
6. SIMVI behind its tutorial gate (~10–18 GPU-h), then MintFlow for the
   counterfactual row.
7. Doc-09 §5a planted worlds, if the reattribution figure is ever to be quoted.

## 8. Registers — where things are written down

- [devlog.md](devlog.md): chronological narrative, motivations written *before*
  runs and results after. Current tail: the atlas rewrite, transport clarity,
  the x̃ decisive follow-up, the baselines survey, the MMD motivation, and the
  three-package motivation of 2026-09-21 that commissioned this document.
- [issues.md](issues.md): P / E / M / T / A / V tables plus the watch list.
  Instrument defects V1–V12; **V9 closed** (fixed inside `xtilde_gate.py`),
  **V10 / V11 closed by the atlas rewrite**, **V12 read-time fix in place**
  (training-time fix not attempted). M6 carries the depth qualifier.
- [spec_deviations.md](spec_deviations.md): doc-07 rows (incl. the `type_only`
  departure with its empirical-only case, `subtract_leak`, `gat_sink`, the
  §7.10 type-mean-z substitution, current defaults), doc-08 rows, doc-09 rows,
  doc-10 park record, doc-11 A4 rows.
- [todo.md](todo.md): the work queue with §0 decisions already taken (four
  datasets; GSE sweeps on the core with the full slides held out; compute is not
  a constraint; FF gets its own 500/40 budget; α_w grid extended above 0.1; the
  LR ladder keeps all three rungs; the leak meter is a diagnostic, not an input;
  the article carries no empirical numbers until evaluation starts).
- [sweep_programme.md](sweep_programme.md): the four-dataset grid plan with
  measured per-fit hours.
- `submission_paper/aistats/revision_2026-09-17.md`: paper-vs-reality revision
  list. **A1–A7 applied**; sections B (stale/retracted numbers) and C (results
  with no home in the paper yet) pending.
- Architect documents at the repo root are received as-is: `07-simple-spec_7.md`,
  `08-validation-analyses_1.md`, `09-communication-experiment_1.md`,
  `11-z-applications_1.md`. Doc-10 and doc-16 are not in the root; their record
  is the devlog.
- `docs/cell_cycle_comparison_report.{md,pdf}`: **superseded** — its joint-label
  recommendation is not defensible (§4.7) and its label-based correlation map is
  replaced by the score-based one.

## 9. Repository state

- **Nothing is committed since `096ae47`.** The working tree carries the atlas
  rewrite, the transport rewrite, the cycle-target module, the degeneracy
  module, the sweep tool, `crossslide.py`, the three experiment modules
  (`leak_meter`, `transcript_flux`, `dapi_cycle`), `xtilde_gate.py`, the paper
  edits, and this document. Untracked: `discell/experiments/{dapi_cycle,
  transcript_flux}.py`, `discell/model/cycle_target.py`, `scripts/`,
  `docs/{sweep_programme,todo}.md`, `docs/cell_cycle_comparison_report.*`,
  `submission_paper/{aistats/revision_2026-09-17.md,articles/}`, and eight new
  test files. **Commit before handing over.**
- Tests: `uv run pytest -q` → **239 passed, 1 skipped** (2026-09-21).
- Dependency drift to fix: `diptest` 0.11.0 was added to the venv for the DAPI
  gates and is **not in the dependency file**.
- Data present per dataset: `bundle/full.h5ad`, `embeddings/egomask_ego_v1.pt`
  (the default arm, KRONOS v1, 256 px at 0.5 µm/px, 25 µm ego disk) for all five
  slides, `qc/nuclear_counts.npz` + `qc/nuclear_summary.json` on all five,
  `qc/nuclear_dapi.parquet` on ovarian and FF.
- Known stale artefacts on disk: pre-correction `transport.json` on
  `gat_sink_*`, `xtilde_*`, `alphaw0.05_*`, `wd0`; `atlas_kappa_survival*.json`
  not regenerated under the new basis; `transport_kappa_sensitivity.json`
  (v1) superseded by `_v2`.
- Conventions: long jobs detached (`setsid nohup … & disown`); sweeps idempotent
  per run name; the shell is fish, so chain with `&&`; one battery process per
  card (every reload builds the resident Trainer); GPU 0/1 by
  `CUDA_VISIBLE_DEVICES`; `--figures-every` matters — on the FF slide figures
  were 69 % of wall time.
