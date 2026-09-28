# DisCell — project handover (state as of 2026-09-28)

This document is for an engineer who picks the project up cold. It says where
things stand and names the file that holds each number. **It does not copy
numbers that already live in a table: "see X" means the number is in X, so
copy it from X, not from here.** Where no table exists, a number is quoted
with the devlog entry it comes from. Anything I could not check against a file
is marked *(unverified)*.

The other registers:

- Narrative and motivations: [devlog.md](devlog.md). Every decision since the
  freeze is in the entries from "Phase change: model frozen; validation and
  analysis begin (author, 2026-09-21)" to the end.
- Bugs and watch items: [issues.md](issues.md).
- Spec departures: [spec_deviations.md](spec_deviations.md).
- Work queue: [todo.md](todo.md) §8.
- Manuscript changes: [paperlog.md](paperlog.md).

**A null or failed leg is a finding here, never softened.**

This replaces the 2026-09-21 version, which is in git at `ea44931`
(`git show ea44931:docs/handover.md`). That version described the old pins
(`ablation_gat_type_only_s1`, `reference_graphclust`, `reference`) at curated or
graphclust labels. Since then:

- The model was frozen, then unfrozen for exactly three changes: KL warm-up on
  w, α_z at ½, and adversary composition weight 3.
- The labels became lineage-level.
- All four datasets were re-pinned twice: `final_s*` on 2026-09-24 and
  `finalL_s*` on 2026-09-25/26.
- The invariance probe was rebuilt per block.
- Every decision table is now read at the accepted checkpoint.
- A tile-bootstrap layer wraps the headline reads.

The 2026-09-21 version's §4 results were computed on the old pins and are now
history. §9 below says which of its findings still stand.

## 0. Status in one paragraph

The model and its configuration are final (`finalL`, §2). The final queue (fits,
κ sweep, baselines, tables) finished on 2026-09-26 at 21:07. Two read-out families
that had failed were repaired on 2026-09-27, the run archive was applied, and 6b.2
(marker pairs) and the cycle 2×2 were scored. **Two additions are in progress as of
2026-09-28:** todo 8.21 (a noise-ceiling column with a tile-level split) and 8.22
(subtype recovery). When this was written, no module or output for either was in
the working tree. Two decisions await the author: the cycling set for cycle_z, and
which ceiling column is primary. After those, what remains is paper work (numbers
for sections B/C) and two baseline-comparison figures (6b.7, 6b.9). **Nothing since
`6a21936` (2026-09-27 18:43) is committed** (§13).

## 1. What this is

**Question.** In imaging spatial transcriptomics (10x Xenium Prime 5K), a cell's
measured expression mixes three things: what the cell does on its own, how its
neighbourhood modulates it, and transcripts that belong to its neighbours
(segmentation spill-over). DisCell-simple ([07-simple-spec_7.md](../07-simple-spec_7.md))
is a VAE that splits a cell's counts into three parts:

- **An intrinsic state z** (d_z 20). Encoder `q(z | x_i, ℓ_i, t_i)`: the cell's
  own counts, depth and type, never its neighbours.
- **A spatial response w** (d_w 6). Posterior `q(w | c, t, z, x)` around a
  context-conditional prior `m_ψ(c, t)`, decoded through B as
  `softmax(a(z) + B·w)`.
- **A fixed leak channel**, `p_i = (1 − κ) ρ_i + κ ρ̄_i`. Here ρ̄ is the
  β-weighted neighbour rate (Voronoi-face kernel on the model's 40 µm-pruned
  graph). κ = 0.1 is swept, never fitted.

The context `c_i` concatenates three parts:

- a GATv2 over the neighbours' **types only** (`type_only`);
- the ego-masked 384-d KRONOS image embedding Φ;
- an isolated flag.

The invariance of z to context comes from an adversary (α_a 0.3). It has two
heads: a categorical cross-entropy on composition and a soft cross-entropy on
image-niche membership. The composition head is now weighted ×3 (§2.3).

**What w is at the operating point (R19 reframing, adopted 2026-09-25).** At
α_w = 0.1 the posterior sits on the prior, so every delivered w read is the
context regression `m_ψ(c, t)`. The per-cell deviation channel is closed by
design. The reconstruction modes show this directly: the full-posterior and
intrinsic-only decodes agree (`runs/<run>/recon_modes.json`). w's value lies in
predicting held-out between-niche shifts (transport), not in fitting counts.

Low KL_w is therefore *not* a closed channel. A dead channel is the case where
prior and posterior both ignore c (devlog 2026-09-23, "Interpretive correction").
The I(niche; w) guard decides that, not KL_w.

**The claim** is a testable division of labour. z carries intrinsic dynamics and
fails spatial tests; w does the reverse; the leak channel absorbs spill-over, so
neither latent has to.

**Datasets (four datasets, five sections).** The roles come from the phase-change
entry (devlog 2026-09-21):

- **Primary:** GSE315411 (train on the section-11 core, evaluate on the
  section-10 core; the only genuinely out-of-sample read) and the fresh-frozen
  ovary (the deepest slide, with the most cells).
- **Secondary:** ovarian FFPE (the development slide) and lung FFPE.

| dataset id | tissue / prep | cells | lineage classes (from) | α_z | tile | role |
|---|---|---|---|---|---|---|
| `gse315411_pdltma06_11_prime_solo`, variant `pdl018d` (`…_10_prime_dual` held out) | pediatric-lung TMA, two serial sections | 69,422 / 70,757 (cores) | 30 (curated 35) | 0.0018 | 2048 | primary; held-out section |
| `xenium_prime_human_ovary_ff` | ovary, fresh frozen | 1,157,637 | 10 (38 graphclust clusters) | 0.00035 | 4096 | primary |
| `xenium_prime_ovarian_cancer_ffpe` | HGSOC, FFPE | 407,120 | 12 (curated 18) | 0.0035 | 4096 | secondary; development slide |
| `xenium_prime_human_lung_cancer_ffpe` | lung cancer, FFPE | 278,324 | 20 (32 graphclust clusters) | 0.002 | 4096 | secondary |

Sources:

- cell counts: 2026-09-21 handover §1 and `scripts/logs/final_prep_2026-09-25/AGENT_REPORT.md`;
- class counts: the same report, and each dataset's `labels/lineage_map_applied.csv`;
- α_z and tile size: `runs/finalL_s0/config.json`.

The scope decisions of 2026-09-17 (todo §0) still hold:

- all four datasets are in scope;
- GSE sweeps run on the core, with the full slides held out;
- compute is not a constraint, and every long job runs detached;
- the LR ladder keeps all three rungs.

## 2. The final configuration, and why each element

Checked against each dataset's `runs/finalL_s0/config.json`:

| knob | value | set by |
|---|---|---|
| architecture | type_only GATv2 ⊕ Φ (384) ⊕ flag; d_z 20, d_w 6; query = type; prior m_ψ(c, t) | frozen 2026-09-21 |
| κ and its form | 0.1, `kappa_mode global` | §2.6 |
| α_z | ½ × 1/ℓ̄ per dataset (values in §1) | §2.2 |
| α_w | 0.1 | §2.4 |
| adversary | α_a 0.3; 6 head steps, width 64, lr 2e-3, ensemble 1, input μ_z; **composition weight 3** | §2.3 |
| KL warm-up | **30 epochs on KL_w only**; no checkpoint and no patience inside the ramp | §2.1 |
| labels | **`lineage`** | §2.5 |
| budget | 500 epochs, patience 40, figures every 100 | frozen |
| off | `subtract_leak`, `gat_sink`, `w_penalty`/`lambda_w`, free bits, `kl_warmup_epochs`, `fp_floor`, `phi_proj`, `class_mean_prior`, `prior_type_free` | §2.8 |

**Trap: the `TrainConfig` defaults are not the final configuration.** In
`discell/model/train.py` the defaults are `adv_comp_weight 1.0`,
`w_warmup_epochs 0`, `alpha_z 0.007` and `label_key None`. The final configuration
exists only as CLI flags, set in the `COMMON` and `FLAGS` blocks of
`scripts/queue_2026-09-25_final_lineage.sh`. To reproduce a final fit (the queue's
command with its variables expanded; I have not re-run it):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m discell.model.train \
  --dataset xenium_prime_ovarian_cancer_ffpe --run-name finalL_s0 --seed 0 \
  --alpha-z 0.0035 --gat-sources type_only --kappa 0.1 --d-w 6 --alpha-w 0.1 \
  --alpha-a 0.3 --epochs 500 --patience 40 --figures-every 100 \
  --w-warmup-epochs 30 --adv-comp-weight 3 --label-key lineage
# GSE:  --alpha-z 0.0018 --variant pdl018d --tile-cells 2048
# lung: --alpha-z 0.002      FF: --alpha-z 0.00035
```

Run it detached for anything larger than GSE (§8).

### 2.1 KL warm-up of 30 epochs on w

**Problem.** FF fits failed with a dead context channel: w constant within type,
and I(niche; w) exactly zero, at the pinned α_w. It showed up in three places:

- FF `best_s2`;
- a survey of 329 runs on disk;
- the control arm of the collapse grid.

**Why warm-up.** Warm-up ramps α_w from 0 to 0.1 over 30 epochs, and the converged
objective is unchanged. It is the only remedy whose channel carried niche
information on every FF seed. The others failed:

- Free bits opened the channel in KL but left I(niche; w) at zero on FF: open in
  divergence, dead in information.
- Warm-up on both KL terms (8.9b) failed its rule.
- The warm-up + free-bits combination failed its rule.

**Cost.** Stated in the paper: NMI falls by 0.01–0.02 at the checkpoint (devlog
2026-09-24, "Re-read at the accepted checkpoint").

**Evidence:**

- tables: `data/datasets/xenium_prime_human_ovary_ff/experiments/wcollapse{,_at_best}.md` and `wcollapse_b{,_at_best}.md`;
- verdicts: `scripts/logs/wcollapse_2026-09-23/DECISION{,_at_best}.json` and `scripts/logs/wcollapse_b_2026-09-24/DECISION_B{,_at_best}.json`;
- decision: devlog "Final configuration frozen; re-pin launched (author's decision, 2026-09-24 ~10:00)".

**Rule history, disclosed.** Two amendments were made after seeing data, and they
pushed in opposite directions. Both the original and the amended verdicts are
reported (devlog 2026-09-23, "8.9 rule amendment" and "amendment 2"). The FF-margin
amendment is withdrawn for any future use.

### 2.2 α_z = ½ × 1/ℓ̄

**The ladder.** The pre-registered α_z ladder {¼, ½, 1, 2} adopted ½ on last-epoch
reads (2026-09-22).

**The R26 re-read.** Read at the accepted checkpoint, **no rung qualifies**. The
cycle_z clause misses by 0.003–0.004, because one control seed widened the control
range.

**The decision.** The author kept ½ as a disclosed **near miss, not a rule pass**:
every rung seed beats every control seed but one, with the guards intact.

**Evidence:** `data/datasets/xenium_prime_ovarian_cancer_ffpe/experiments/alpha_z_decision{,_at_best}.json`
and `scripts/logs/at_best_2026-09-24/AGENT_REPORT.md` §1.

**The ℓ̄ caveat (R18, open).** The pinned α_z values are not ½ × the mean count of
the training cells. They match the all-cell median on GSE, lung and FF, and match
neither on ovarian (`scripts/logs/awladder_2026-09-24/AGENT_REPORT.md`, section
"ℓ̄"). The paper must define ℓ̄ explicitly (todo 8.12, "Weights in units of 1/ℓ̄").

### 2.3 Adversary composition weight 3

**Why it was needed.** The per-block probe (R20/R22, §5) showed that z still
carries within-type composition beyond type. A planted world where z ⟂ niche by
construction reproduces the same-sized residual, so the cause is adversary
capacity or weighting, not biology (`scripts/logs/planted_probe_2026-09-24/AGENT_REPORT.md`).

**The adversary ladder (8.17), on ovarian:**

- More head steps did nothing.
- Wider heads and a 3-head ensemble helped a little.
- Weighting the composition head ×3 lowered every probe block, lowered the
  mirror and raised the transport gap closed, at no extra head time.

**The confirmation** on GSE, FF and a third ovarian seed repeated the pattern.
The pre-stated rule still reads "not confirmed":

- on GSE, recon and NMI miss a two-seed envelope by a hair;
- on FF, NMI and cycle_z fall outside the envelope.

The author adopted weight 3 on the trade, and the FF cost is stated in the paper.
Weight 5 cuts leakage further but starts to cost NMI and cycle_z.

**Evidence:**

- tables: `…ovarian_cancer_ffpe/experiments/adv_ladder.md` and `adv_confirm.md` (GSE, FF and ovarian sections);
- verdicts: `scripts/logs/adv_ladder_2026-09-24/DECISION_ADV.json` and `scripts/logs/adv_confirm_2026-09-25/DECISION_ADVC.json`;
- decision: devlog "Final launch (author's decisions, 2026-09-25)".

### 2.4 α_w = 0.1, with the R19 reframing

**The issue (R19).** α_w = 0.1 is 14–143× the bound-equivalent 1/ℓ̄, so KL_w ≈ 0
by construction.

**The ladder.** A pre-registered multiplier ladder tested α_w = m × 1/ℓ̄ for
m ∈ {1, 2, 5, 10, 25}. As α_w falls, the per-cell deviation channel opens and does
real held-out work, but no rung passes every guard:

- On GSE and ovarian, only NMI blocks m5–m10, by 0.004–0.005 under the fresh
  reference.
- On FF (stage 2), m5 and m10 cost NMI heavily.

Lung was dropped from the ladder by a scope amendment made before any fit counted.

**Verdict,** the same under all three reference variants: α_w stays at 0.1, and
the paper adopts R19's reframing. w is the context regression m_ψ(c, t), the
per-cell channel is closed by design, and the ladder is reported as the
sensitivity analysis of what opening that channel costs and buys.

**Evidence:**

- `…ovarian_cancer_ffpe/experiments/awladder.md` (sections for GSE, ovarian and FF, plus the rule under each reference envelope);
- `probe_regrade_awladder.md`;
- `scripts/logs/awladder_2026-09-24/DECISION_AW.json`;
- decision: devlog "Overnight results, 2026-09-25 morning", decision (1).

### 2.5 Lineage-level labels (R9)

**Why.** Every part of the model conditions on t. Where a label encodes a state
(proliferative, VEGFA⁺, inflammatory, activated) or a location (tumour- or
stroma-associated, lining a cyst), that part of the response is handed to the
label, where neither w nor the invariance can see it. **The rule:** merge states
and locations into their lineage, and keep developmentally distinct subtypes
(devlog 2026-09-25, "Lineage-level labels (R9, motivation)").

**The author's choices** (devlog "Final launch"):

- **SOX2-OT⁺ Tumor Cells (ovarian, 9.7 % of cells) → Unassigned.**
  - The pre-registered keep rule (≥ 50 DE genes, plus contiguity) *passes*.
  - But the class is mostly low-depth non-tumour cells. The depth, marker,
    NNLS-decomposition and depth-matched DE reads are in
    `scripts/logs/lineage_2026-09-25/AGENT_REPORT.md`.
  - Lesson recorded: at these cell counts every merged tumour *state* also passes
    DE plus contiguity, so that rule cannot separate a clone from a state.
  - Ovarian Unassigned becomes 10.3 % of cells.
- **"Malignant Cells Lining Cyst" → its own lineage, "Mesothelial-like cyst
  lining".**
  - Mesothelial markers are on; epithelial and Müllerian markers are off.
  - It does not count as tumour in any tumour-band or interface read, even under
    the old labels (`spec_deviations.md`, "Tumour-band definition (2026-09-25)").
- **Grey merges, as proposed:**
  - TAF/SAF → Fibroblasts; TAEC/SAEC → Endothelial.
  - GSE Proliferating, EC activated, Inflammatory, Activated and Myofibroblasts
    are reassigned cell by cell. Expected accuracy for the Proliferating calls is
    ~0.75.
- **FF's mixed immune-near-tumour clusters** keep their lineage call, with the flag
  recorded. The tumour signal in them is the leakage the model exists to handle.

**Where the tables are:**

- `data/datasets/<ds>/labels/lineage_map_{proposed,applied}.csv`;
- `lineage_map_applied.json`, which holds the sha256 of every input;
- the GSE per-cell tables `*_reassignment_proposed.csv`.

`discell/experiments/apply_lineage.py` writes the obs column `lineage` into every
bundle variant and leaves the bundle's `default_label` untouched. **These tables
sit under the gitignored `/data/`, so they are not in git**, although todo 8.13
planned to commit them.

**Label matchers.** Name-matched reads use `discell/model/labels.py` (`is_tumour`,
`is_smooth_muscle`, `is_endothelial`, all case-insensitive). One latent bug: the
`t.?cell` exclusion also matches "Malignan**t C**ells". No current class triggers
it (`scripts/logs/final_prep_2026-09-25/AGENT_REPORT.md`, Addendum 2).

**Consequences, stated in advance:**

- NMI is not comparable with pre-lineage values, because there are fewer classes.
- cycle_z is not comparable either (§5, cycle 2×2).
- The pre-lineage NMI floor of 0.63 is void. The 0.9 × running-max guard still
  applies.

### 2.6 κ = 0.1 in the global form, with the κ-form sensitivity rows (R12)

κ stays swept, not measured: both measurement programmes failed (2026-09-21
handover §4.2). R12 asked what the global form assumes: (a) the same κ for every
gene, and (b) a leaked amount that scales with the receiver's depth.

**Test 1 and 1b (read-only, from transcript positions).**

- Test 1 concluded "donor scaling ruled out"; test 1b withdrew that. At fixed
  receiver and donor area, the donor-depth elasticity is practically positive on
  FFPE and small but not zero on FF.
- What stands: position counts bound leakage from above, and identify neither
  scaling.
- Evidence: `…/experiments/r12_depth_test.{json,md,png}` (ovarian and FF); devlog
  "R12 test 1b — donor scaling is NOT ruled out".

**Test 2 (sensitivity arms).**

- Three arms beside `global`: `--kappa-mode depth`, `gene` and `density`. Each is
  normalised so that the post-clip mean of κ_i equals κ exactly; only the shape
  of the leak differs.
- Every response finding is unchanged: transport on both reads, the response
  channel's LR share, and the surface lean of B. The leak channel's own LR share
  collapses under the gene arm, by construction.
- Verdict: "stable across κ and its form". The global form stands.
- Evidence: `…ovarian_cancer_ffpe/experiments/r12_arms.md` and
  `probe_regrade_r12.md`; devlog "R12 test 2 — the leak's form does not move the
  response findings".

**Caveat.** These arms ran at the *pre-lineage* configuration: ovarian, 200/20,
old labels, composition weight 1, with control `wfix_warmup30_aw0.1_s{0,1,2}`
(checked in `runs/_archive/r12_gene_s0/config.json`). They have not been re-run at
`finalL`, and I found no decision on whether they need to be.

`report.py` and `sweep.py` do not pass `kappa_mode`, so they fail on a gene-form
checkpoint (issue W-qp1 addendum).

### 2.7 The false-positive floor: a sensitivity row only

**The model.** `p_i = (1 − κ − η_i) ρ_i + κ ρ̄_i + η_i·u`, with η_i taken from the
section's negative-control and genomic-control probe counts (`--fp-floor`,
`--fp-area`; `discell/model/fp_floor.py`).

**What happened:**

- "Small changes" held on GSE.
- On ovarian it did not hold: the floor moves about one percentage point of the
  between-niche shift from the leak channel to the response channel, and moves
  the invariance reads by 2–4 seed-sd.
- Almost all of λ rests on an even-spread assumption for genomic-DNA binding.

**Decision (author, 2026-09-25):** not adopted; reported as a sensitivity row
beside the κ-form rows.

**Evidence:**

- `…ovarian_cancer_ffpe/experiments/fp_floor.md` (ovarian and GSE sections) and `probe_regrade_fp.md`;
- runs `fp_s*` and `fp_area_s*`, now in `_archive/`;
- decision: devlog 2026-09-25 morning, decision (2).

The same caveat as §2.6 applies: these ran at the pre-lineage configuration
(`runs/_archive/fp_s0/config.json`: 200/20, composition weight 1, old labels). The
paper text is todo 8.18 (writer, open).

### 2.8 Considered since 2026-09-21 and not adopted

| alternative | outcome | where |
|---|---|---|
| free bits on KL_w (λ 0.05), alone or with warm-up | open by KL, dead by I(niche; w) on FF; the combination fails the ovarian cycle_w guard | `wcollapse*.md` |
| warm-up on both KL_z and KL_w (8.9b) | fails "nothing worse than control" (NMI, cycle_w) | `wcollapse_b*.md` |
| type-free GAT query, image query, type-free prior (8.10) | no read improved; query and prior kept; reported as supporting ablations, run at the w-only warm-up objective | `…ovarian_cancer_ffpe/experiments/queryprior.md` |
| learned projection of Φ to 32 dims (S53) | buys invariance at the cost of type structure in z; not adopted; reported in the implementation table | `…ovarian_cancer_ffpe/experiments/phi_projection{,_lineage}.md` |
| head steps ×2, width ×2, 3-head ensemble (8.17) | no or small effect; composition weight chosen instead | `adv_ladder.md` |
| composition weight 5 | more leakage removed, at a cost in NMI and cycle_z | `adv_confirm.md` |
| lower α_w (m × 1/ℓ̄, m from 1 to 25) | §2.4 | `awladder.md` |
| depth, density or gene-tilted κ as the default | §2.6 | `r12_arms.md` |
| false-positive floor as the default | §2.7 | `fp_floor.md` |

Everything rejected before 2026-09-21 stays rejected: the x̃ input, L2 on w and
weight decay, per-cell κ from the leak meter or transcript flux, the class-mean
prior, the adversary on x̂, DAPI cycle labels, the depth-transformed cycle
target, `gat_sink`, and the doc-10 guard. See §9 and §5 of the 2026-09-21
handover.

## 3. Run inventory (after the 2026-09-27 archive)

The runs kept under `data/datasets/<ds>/runs/` on each dataset (listing checked
2026-09-28):

| family | what it is | notes |
|---|---|---|
| `finalL_s{0,1,2}` | the final triple: 500/40, final configuration | No dead fit (`READOUT.md`). Each run holds `metrics.json`, `history.jsonl`, `best.pt`, `degeneracy.json`, `validation/`, `atlas/`, `transport/`, `recon_modes*.json`, `bootstrap_ci.json`, `marker_pairs.json` and `report/`; GSE runs also hold `crossslide/`. |
| `best` → `finalL_s0` | symlink; "read on best" means this run | re-pins logged in `scripts/logs/final_lineage_2026-09-25/REPIN.tsv` |
| `uncontrolledL_s{0,1}` | α_a = 0 references, otherwise the final configuration; the denominators of every probe fraction | **Both FF references have a dead w channel**: without the adversary, z carries the niche. Recorded in `DEAD_RUNS.tsv`, not refitted. The probe denominators are z-based, so they stand. |
| `sweepL_k{0,0.05,0.2,0.3,0.4}_s{0,1,2}` | κ sweep at the reference budget (500/40, R28), with light reads | a dead sweep fit is recorded, never refitted; none occurred |
| `sweepL_k0.1_s{0,1,2}` | **symlinks** to `finalL_s{0,1,2}` | listed in `SWEEP_LINKS.tsv`; never refitted, never re-read |
| `reference*`, lung `best_pre_az0.5`, ovarian `reference_best` | old pins from before the freeze, kept for history | no read-out table names them |
| ovarian `dw{2,3,8}_s*` | the only ovarian d_w envelope | kept; no read-out table |
| ovarian `phiproj32_s*`, `projL{32,384}_s*` | projection test (S53), on old and on lineage labels | 200/20 |
| `timing_phi{,_dropped,_zeroed}`, ovarian `timing_phiproj32` | timing-mode outputs (only `timing.json`) | |

**`runs/_archive/`: moved, never deleted.**

- Size: 81 runs on GSE, 60 on lung, 106 on FF and 144 on ovarian, about
  15.8 GiB in all.
- Record: every move, with the read-out table that covers it, is listed in
  `runs/_archive/ARCHIVE_2026-09-27.tsv` (columns run, size_bytes, family,
  tables).
- Contents include:
  - the 2026-09-24 re-pin `final_s*` (pre-lineage);
  - the α_z/2 generation `best_s*` / `best_az0.5_s0`;
  - `uncontrolled{,500}_s*`;
  - every decision arm: `wfix*`, `wfixb*`, `qp_*`, `aw_*`, `adv_*`, `advc_*`,
    `r12_*`, `fp_*`, `ladder_az_*`, `sweep3_*`.
- Symlinks moved together with their targets and resolve inside `_archive/`.
- `experiments/` was not touched.
- *(unverified)* I did not check whether every CLI accepts an archived run as
  `--run _archive/<name>`.

**Baselines live outside this repo,** in `/home/rmolen/github/DisCell-baselines/`:

- Exports are under `data/`; the lineage exports are in `data/lineage/`.
- Latents are under `results/{resolvi,resolvi_lineage,simvi,mintflow}`, 7.2 GB in
  all.

At lineage labels:

- **resolVI was refit** on all five bundles (`results/resolvi_lineage/`).
- **SIMVI and MintFlow were not refit.** Their latents come from training on the
  old labels (MintFlow conditions on the label) and were only re-scored on the
  lineage battery (`scripts/logs/final_prep_2026-09-25/AGENT_REPORT.md` §2,
  "Needs a decision", item 3).
- SIMVI on FF is a 100k-cell window.
- MintFlow on FF is infeasible on this host: host OOM before training. The failed
  MintFlow FF directory was deleted.

Coverage per section:

- GSE solo and dual: all three baselines.
- FF: resolVI and the SIMVI window.
- ovarian and lung: resolVI only.

**The final queue's records** are in `scripts/logs/final_lineage_2026-09-25/`:

- `READOUT.md`: the per-run table, dead fits, links, re-pins, failed steps and
  output checklist. It was written 2026-09-26 21:07, *before* the repairs, so its
  MISSING and failed entries for `validation_sweepL.json`, κ-survival and the
  bootstrap are stale. See `scripts/logs/final_repair_2026-09-27/AGENT_REPORT.md`.
- `code_hashes.tsv`: the same four model-code hashes (train, networks, equations,
  elbo) at all 86 fit launches.
- `DEAD_RUNS.tsv`, `SWEEP_LINKS.tsv`, `REPIN.tsv`.
- `timing_all_lineage.md`.
- `envelope_tables_finalL_at_best_ci.md`: the combined envelope tables.

## 4. How to run

All paths are keyed by dataset id under `data/` (`discell/paths.py`). Below,
`DS=<dataset id>`, `RUN=finalL_s0` and
`REFS="--references uncontrolledL_s0 uncontrolledL_s1 --reference-200 none --baseline-tag _lineage"`.
The invocations are taken from `scripts/queue_2026-09-25_final_lineage.sh`.

| step | command | output |
|---|---|---|
| relabel (done) | `uv run python -m discell.experiments.apply_lineage …` (`--sox2ot`, `--cyst` required; `--dry-run`) | obs `lineage`; `labels/lineage_map_applied.{csv,json}` |
| fit | §2 | `runs/$RUN/` |
| dead-channel guard + §7.10 degeneracy | `uv run python -m discell.model.degeneracy --dataset $DS --run $RUN` | `runs/$RUN/degeneracy.json` (block `w_channel`) |
| battery | `uv run python -m discell.model.validate --dataset $DS --run $RUN --analyses morans,niche,probe` | `runs/$RUN/validation/validation.json` (see the probe trap in §5) |
| per-block probe vs the lineage references | `uv run python -m discell.experiments.probe_regrade --dataset $DS --run $RUN --force $REFS` (GSE: repeat with `--dataset gse315411_pdltma06_10_prime_dual --config-from gse315411_pdltma06_11_prime_solo`) | `runs/$RUN/validation/probe_blocks*.json` |
| probe tables | `uv run python -m discell.experiments.probe_regrade --dataset $DS --table lineage_final --baselines --runs 'finalL_s*' 'uncontrolledL_s*' $REFS` | `experiments/probe_regrade_lineage_final.md` (and `…_sweep.md`) |
| atlas | `uv run python -m discell.model.atlas --dataset $DS --run $RUN [--compare-runs finalL_s1 finalL_s2]` | `runs/$RUN/atlas/` |
| transport | `uv run python -m discell.model.transport --dataset $DS --run $RUN --read both --hvg 1000 [--niche-source tumour-band]` | `runs/$RUN/transport/` |
| held-out section (GSE) | `uv run python -m discell.model.crossslide --dataset gse315411_pdltma06_11_prime_solo --run $RUN --eval-dataset gse315411_pdltma06_10_prime_dual` | `runs/$RUN/crossslide/` |
| reconstruction modes | `uv run python -m discell.experiments.recon_modes --dataset $DS --run $RUN [--eval-dataset …]` | `runs/$RUN/recon_modes*.json` |
| tile bootstrap | `uv run python -m discell.experiments.bootstrap --dataset $DS --run $RUN --n 1000 [--reads transport_mean]` | `runs/$RUN/bootstrap_ci.json` |
| envelope tables | `uv run python scripts/envelope_tables.py --datasets <ids> --runs finalL_s0 finalL_s1 finalL_s2 --at best --ci --combined <file>` | `experiments/envelope_table_ci_at_best.md` |
| κ sweep report | `uv run python -m discell.model.sweep --dataset $DS --report-only --tag sweepL --param kappa --values 0 0.05 0.1 0.2 0.3 0.4 --seeds 0 1 2 --label-key lineage --alpha-z … --epochs 500 --patience 40` | `experiments/kappa_sweep_sweepL.json` (caveats in §11) |
| κ-survival | `uv run python -m discell.model.validate --dataset $DS --sweep-tag sweepL --kappas 0 0.05 0.1 0.2 0.3 0.4 --seeds 0 1 2 --analyses kappa_survival,morans,niche --survival-reference finalL_s0` | `experiments/{validation_sweepL,atlas_kappa_survival{,_internal}}.json`, `validation_sweepL_kappa.png` |
| baseline battery | `uv run python -m discell.experiments.baseline_battery --dataset $DS --discell-run $RUN --config-run finalL_s0 --tag _lineage`; one baseline: `--method "<label>" --latents <h5ad>` | `experiments/baseline_battery_lineage.{json,md}` |
| external criteria | `uv run python -m discell.experiments.external_criteria {signalling-share,mi-quadrant,axis-test} --dataset $DS --run $RUN` | `experiments/external_*_<run>.{json,png}` |
| GO localisation | `uv run python -m discell.experiments.go_localisation --dataset $DS --run finalL_s0 --run finalL_s1 --run finalL_s2 --out-stem go_localisation_lineage` | `experiments/go_localisation_lineage.{json,md,png}` |
| cycle 2×2 | `uv run python -m discell.experiments.cycle_2x2 --dataset $DS` | `experiments/cycle_2x2.{json,md}` |
| marker pairs (6b.2) | `uv run python -m discell.experiments.marker_pairs --dataset $DS --run <runs…> --kappa` | `runs/<run>/marker_pairs.json`, `experiments/marker_pairs_kappa.{json,md,png}` |
| timing | `uv run python scripts/timing_mode.py …` (wraps `train --time-only N`) | `experiments/timing*/`, `timing*.md` |
| report | `uv run python -m discell.model.report --dataset $DS --run $RUN` | `runs/$RUN/report/` |
| tests | `uv run pytest -q` | not re-run for this handover (§13) |

The queue scripts that produced the current state:

| queue | what it ran |
|---|---|
| `scripts/queue_2026-09-25_final_lineage.sh` | everything under §3 |
| `scripts/queue_2026-09-25_metrics.sh` | bootstrap, recon modes, projection test and timing on the pre-lineage finals |
| `scripts/queue_2026-09-27_marker_pairs.sh` | 6b.2 scoring |
| `scripts/logs/final_repair_2026-09-27/rerun*.sh` | the repairs |

Code missing from the 2026-09-21 handover's code map. All of it was added on or
after 2026-09-21; `cycle_2x2` and `marker_pairs` are still untracked:

- `discell/experiments/`: `apply_lineage`, `at_best`, `baseline_battery`,
  `bootstrap`, `cycle_2x2`, `export_for_baselines`, `external_criteria`,
  `go_localisation`, `marker_pairs`, `planted_posterior`, `planted_probe`,
  `probe_regrade`, `r12_depth_test`, `recon_modes`, `w_deviation`.
- `discell/model/`: `labels.py`, `attention_read.py`, `fp_floor.py`.
- `scripts/*_table.py`: the decision read-outs.

The rest of the code map is unchanged from §3 of the 2026-09-21 handover.

## 5. Evaluation instruments

Reading conventions are unchanged (doc-08 §1): posterior means, within type,
spatial-block folds, and a within-type permutation floor. **Every decision and
envelope table is read at the accepted checkpoint.** In practice:

- Runs fitted after the R26 trainer fix carry `final_epoch` = the accepted epoch,
  so their `final` block describes `best.pt`.
- Older runs lack `final_epoch` and `last_epoch`. For those,
  `discell/experiments/at_best.py` and the `--at best` flag of the decision
  scripts read the `history.jsonl` row at `best.epoch`.
- Post-hoc tools (validate, degeneracy, transport, atlas, baseline battery) load
  `best.pt` and were never affected.

### 5.1 The battery at the accepted checkpoint

- **What it reads:** NMI(z, t); mirror R²; cycle_z and cycle_w (pooled,
  per-type-centred, with permuted control, log-depth baseline and 50-PC linear
  reference); the §7.10 degeneracy pair (I(z;t)/H(t), within-type variance
  fraction of z); the type-mean-z reconstruction gap; KL_w.
- **Code:** in-trainer `Trainer.evaluate` plus `discell.model.degeneracy`.
- **Where:** `metrics.json` and `history.jsonl`; aggregated per dataset in
  `experiments/envelope_table_ci_at_best.md`.
- **Caveats:** NMI re-evaluated on identical weights varies by up to ~0.003
  (k-means); this is issue T-r26.

### 5.2 Per-block invariance probe vs the uncontrolled fit (R20, R22)

- **What it reads:** per column, the gain ½·log(MSE of the type-only baseline /
  MSE of the probe), in two blocks: composition, and 12 Φ PCs. It is graded by a
  ridge and by an MLP (one network per block), each against its own within-type
  permutation floor.
- **How it is reported:** excess in nats per column; the implied within-type
  variance exp(2·excess) − 1; and fractions of the α_a = 0 fit (·u) and of
  resolVI.
- **The guard:** all four blocks at ≤ 25 % of uncontrolled. The 25 % is a
  judgement, stated as such.
- **Code:** `metrics.probe_gain_per_block{,_mlp}`; `discell/experiments/probe_regrade.py`.
- **Where:** `runs/<run>/validation/probe_blocks.json`,
  `experiments/probe_regrade_lineage_{final,sweep}.md`,
  `experiments/probe_regrade_lineage/` (baselines), and the envelope rows
  "probe excess …" and "invariance guard …".
- **Caveats:**
  - The legacy pooled ΔCE was > 99 % image block. It is kept as a row only.
  - The 2-sd rule was withdrawn the same day as the wrong scale.
  - The denominators come from 2 reference seeds, which is about 10–40 % noise.
  - **Trap:** `validate --analyses probe` grades against `uncontrolled500_s*`
    (old labels, now in `_archive/`). Always follow it with
    `probe_regrade --force $REFS`. The final queue did exactly this
    (`final_prep` report, "Other points to know").

### 5.3 I(niche; w) guard (dead-channel guard)

- **What it reads:** kNN MI between the niche label (K = 10 k-means on
  composition) and μ_w on held-out cells, against a 5-fold within-type
  permutation floor, plus the across-cell variance fraction of w. An excess
  ≤ 0 is flagged "failed fit: dead context channel".
- **Code:** `degeneracy.w_channel_guard`.
- **Where:** `runs/<run>/degeneracy.json["w_channel"]`; envelope row
  "I(niche; w) excess".
- **Caveats:** this guard, not KL_w, decides whether the channel is alive.

### 5.4 Dead-channel detector (training time)

- **What it reads:** summed KL_w < 1e-5 in any of the first 20 epochs. It logs a
  WARNING and sets a flag; training continues.
- **Code:** `train.kl_w_is_dead`.
- **Where:** `metrics.json["dead_w_channel"]`.
- **Caveats:**
  - It can disagree with the guard in the thin-but-live regime.
  - Queues refit a dead `finalL` slot with the next seed (3–8) and log it in
    `DEAD_RUNS.tsv`. Dead sweep fits are recorded, not refitted.

### 5.5 Transport, mean read

- **What it reads:** per (type, niche pair) panel, the predicted log-rate shift.
  The programme part comes via m_ψ and B, the leak part via κ, with z held
  fixed. It is scored against the observed held-out shift as a fraction of the
  split-half noise ceiling. The trusted tier is ceiling ≥ 0.5.
- **Code:** `discell/model/transport.py`, `--read mean|both`.
- **Where:** `runs/<run>/transport/transport{,_table}.{json,md}`; envelope rows
  "transport, mean read: fraction of ceiling (trusted / all panels)".
- **Caveats:**
  - The ceiling splits *cells* at random, so shared tile noise counts as signal.
    That inflates the ceiling and biases the fraction low. A tile-split column is
    being added (8.21).
  - On GSE the trusted tier exists on 2 of 3 seeds.
  - The "all panels" fraction can exceed 1: it is a ratio of means over panels
    whose ceilings can be near 0.

### 5.6 Transport, Read A (distribution level)

- **What it reads:** the MMD gap closed, pairwise and leave-one-niche-out, with
  two target sides:
  - **group-w:** the target is decoded at its group's mean w;
  - **own-target:** each target cell is decoded at its own posterior μ_z, μ_w,
    real context and influx (keys `*_own*`; 6a.6).

  A type-mean predictor is the reference.
- **Code:** `transport.py --read both --hvg 1000` (HVG-1000 companion).
- **Where:** `transport_distribution.json`; envelope rows "Read A …".
- **Caveats:** quote the own-target numbers. The group-w target carries a
  shared-w circularity.

### 5.7 Transport, Read B (matched twin)

- **What it reads:** the nearest-z source cell is transported and compared cell
  to cell with the target.
- **Code:** same module.
- **Where:** `transport_twins.json`; envelope rows "Read B …".

### 5.8 Transport on tumour bands (ovarian)

- **What it reads:** six nested bands of the kNN-smoothed tumour fraction.
- **Code:** `--niche-source tumour-band`.
- **Where:** `transport_tumour-band*.json`.
- **Caveats:** the cyst lining is excluded from "tumour" (§2.5), so old-label
  tumour-band numbers are not bit-reproducible.

### 5.9 Tile-bootstrap CIs (R33)

- **What it does:** resamples 200 µm tiles of held-out cells, 1000 draws. Each
  adapter replays its metric's random stream, so the unit-weight value *is* the
  stored number (`reproduces` is recorded per read). Intervals are conditional on
  the fitted probes, clustering and model. The seed envelope is
  [min lo, max hi] over the per-seed intervals.
- **Code:** `discell/experiments/bootstrap.py`.
- **Where:** `runs/<run>/bootstrap_ci.json`; the CI column of the envelope tables.
- **Caveats:**
  - The two transport fraction-of-ceiling rows use **half-tile subsampling
    without replacement** instead: centred on the draws' median, scaled by
    √(m/(n−m)).
  - Planted coverage of those rows is ~0.95 at high ceilings but 0.78–0.88 at
    ceilings of 0.5–0.6. Real panels sit at or below that, so **read those
    intervals as roughly 0.8 coverage.**
  - Most of each envelope's width is seed spread.
  - Holm over the κ grid exists (`bootstrap.paired_comparisons`) but has no CLI
    entry.
  - The κ* breakdown layer (R33 Definition 1) is not built (todo 8.19,
    "Evaluation layer for Definition 1").

### 5.10 Reconstruction modes (R24)

- **What it reads:** per count on held-out cells:
  - full (own posterior);
  - intrinsic-only (posterior z, w = m_ψ);
  - context-only (type-mean z, w = m_ψ);
  - type-profile reference.

  GSE is also scored on the dual section.
- **Code:** `discell/experiments/recon_modes.py`.
- **Where:** `runs/<run>/recon_modes{,_<dual>}.json`; envelope rows
  "held-out recon, …".
- **Caveats:** checkpoint selection used the full (autoencoding) score. The paper
  must say so.

### 5.11 Cycle 2×2

- **What it reads:** old vs new model × old vs new cycling set (top-4 types of
  each label set, centred per type or lineage), on the same held-out cells. It
  reproduces the in-trainer battery to 1e-8.
- **Code:** `discell/experiments/cycle_2x2.py` (untracked).
- **Where:** `experiments/cycle_2x2.md`; `scripts/logs/cycle_2x2_2026-09-27/AGENT_REPORT.md`.
- **The cycling-set caveat:**
  - The FF and lung cycle_z drop is a **read change**, not a model change.
  - FF's cycle_w exceeds the 0.02 guard on 2 of 3 seeds under the registered
    read. That is the pre-registered over-merging mechanism, and the paper
    states it rather than re-reading the guard to a pass.
  - The devlog's "first reading" of 2026-09-27 was backwards, and is corrected
    in the same entry.
  - Decision pending (§11).

### 5.12 Marker pairs (6b.2)

- **What it reads:** resolVI's double-positive metric on raw counts, x̃, the
  decode and the counts-corrected decode, across κ, on the pairs approved on
  2026-09-23.
- **Code:** `discell/experiments/marker_pairs.py` (untracked).
- **Where:** `experiments/marker_pairs_kappa.{md,json,png}`,
  `runs/<run>/marker_pairs.json`; pairs in
  `experiments/marker_pairs_proposed.csv` and `scripts/logs/marker_pairs_2026-09-23/`.
- **Caveats:**
  - The pre-registered wish is not met as stated by any arm.
  - The ratio is the read.
  - The metric does not identify κ.
  - FF keeps RGS5/EPCAM, as the approved list is written.

### 5.13 GO localisation of B (6b.10)

- **What it reads:** Mann–Whitney tests of |loading| against GO
  cellular-component closures.
- **Code:** `discell/experiments/go_localisation.py`.
- **Where:** `experiments/go_localisation_lineage.{md,json,png}` (finalL);
  `go_localisation.md` (older pins).
- **Caveats:**
  - The FF ovary was the inconsistent slide on the older pins.
  - **The lineage file has not been read into the devlog.**

### 5.14 External criteria (6b.1 signalling share, 6b.3 MIG/MIC, 6b.4 axis test)

- **What it reads:** MintFlow's signalling-gene share, DisCoVR's MIG/MIC with a
  within-type floor, and SIMVI's true-axis / false-axis test.
- **Code:** `discell/experiments/external_criteria.py`.
- **Where:** `experiments/external_{signalling_share,mi_quadrant,axis_test}_finalL_s*.json`.
  6b.1 covers ovarian and FF, 6b.3 covers ovarian, GSE and FF, and 6b.4 covers
  ovarian.
- **Caveats:** **none of these has been read into the devlog.**

### 5.15 κ-survival

- **What it reads:** shift-space overlap of μ_w·B across κ at fixed seed,
  against the seed-to-seed yardstick at κ = 0.1, plus axis-1 |cos| and label
  recurrence.
- **Code:** `validate --sweep-tag`.
- **Where:** `experiments/atlas_kappa_survival{,_internal}.json`,
  `validation_sweepL.json`; numbers in `scripts/logs/final_repair_2026-09-27/AGENT_REPORT.md` §1.
- **Caveats:** there is no pre-registered threshold; the verdict is relative to
  the seed spread.

### 5.16 Atlas

- **What it reads:** effective rank and programmes on gauge-centred w;
  cross-seed axis-1 cosine; hallmark labels (BH).
- **Code:** `discell/model/atlas.py`.
- **Where:** `runs/<run>/atlas/`; envelope rows "atlas …".
- **Caveats:** some seed pairs carry null cosines, which are skipped and
  footnoted in the envelope table.

### 5.17 Timing

- **What it reads:** pure training time: s/epoch, s/evaluation and peak memory,
  with Φ, zeroed Φ and dropped Φ, beside the baselines' fit times.
- **Code:** `train --time-only`, `scripts/timing_mode.py`.
- **Where:** `scripts/logs/final_lineage_2026-09-25/timing_all_lineage.md`,
  `experiments/timing_lineage.md`.
- **Caveats:** runs only on an idle card.

### 5.18 Projection test (S53)

- **What it reads:** full Φ (384) against a learned 32-dim projection, ovarian,
  3 seeds, 200/20.
- **Code:** `--phi-proj 32`, `scripts/phi_projection_table.py`.
- **Where:** `experiments/phi_projection_lineage.md` (lineage labels) and
  `phi_projection.md` (old labels).
- **Caveats:** not adopted.

### 5.19 Held-out section (GSE)

- **What it reads:** the whole battery with weights fitted on section 11,
  evaluated on section 10.
- **Code:** `crossslide.py`, `recon_modes --eval-dataset`,
  `probe_regrade --config-from`.
- **Where:** the envelope column "held-out mean";
  `gse315411_pdltma06_10_prime_dual/experiments/baseline_battery_lineage.md`.

### 5.20 Baseline battery

- **What it reads:** one column per method on the same held-out cells, label key
  and cycle scores.
- **Code:** `discell/experiments/baseline_battery.py`.
- **Where:** `experiments/baseline_battery_lineage.{md,json}` per section.
- **Caveats:**
  - The MintFlow recon cell must not be quoted (B-mf1).
  - SIMVI and MintFlow were trained on the old labels (§3).
  - The baselines see held-out counts unlabelled, which makes a DisCell win
    conservative.

### 5.21 Planted instruments

- **What they read:** the amortisation gap against an exact posterior (6b.8), and
  whether the probe residual is adversary capacity or biology.
- **Code:** `planted_posterior.py`, `planted_probe.py`.
- **Where:** `data/datasets/synthetic_smoke/experiments/planted_posterior.{json,md,png}`;
  `scripts/logs/planted_probe_2026-09-24/`.
- **Caveats:** the claims are comparative. The oracles are regression
  references, not information ceilings.

### 5.22 Subtype recovery (8.22) — being added

- **What it reads:** whether the old state and location sublabels can be
  recovered from μ_z, μ_w, m_ψ(c, t) and both combined, within each lineage, on
  held-out cells, with floors.
- **Code:** not in the working tree on 2026-09-28.
- **Design:** devlog 2026-09-28, "Two additions before the paper numbers", B.

### 5.23 Tile-split noise ceiling (8.21) — being added

- **What it adds:** a second column, "fraction of ceiling, tile-split ceiling",
  beside the current one, never as a silent replacement.
- **Code:** not in the working tree on 2026-09-28.
- **Where:** illustration at `docs/figures/noise_ceiling_split.png`.
- **Design:** devlog 2026-09-28, A.

Instruments that were **not re-run on `finalL`:**

- landmarks and the §5 allegiance matrix (`validate --analyses landmarks,matrix`);
- doc-09 communication and the LR ladder;
- leak meter, transcript flux, DAPI gates and the x̃ gate.

Their last results are on the old pins (2026-09-21 handover §§4.3, 4.6, 4.2,
4.7, 4.9). Landmark classes now exist on lung and FF too, through
`labels.py`.

## 6. Where the current numbers are, by claim

| claim | read it from |
|---|---|
| z intrinsic, w context (NMI, cycle_z vs cycle_w, degeneracy pair, I(niche; w)) | `data/datasets/<ds>/experiments/envelope_table_ci_at_best.md`, four datasets; combined `scripts/logs/final_lineage_2026-09-25/envelope_tables_finalL_at_best_ci.md` |
| held-out section generalisation | GSE envelope table, "held-out mean" column; `gse315411_pdltma06_10_prime_dual/experiments/baseline_battery_lineage.md` |
| invariance: what the adversary removes, what remains | envelope rows "probe excess …", "÷ uncontrolled …", "invariance guard"; `probe_regrade_lineage_final.md`; planted-world attribution in `scripts/logs/planted_probe_2026-09-24/AGENT_REPORT.md` |
| DisCell vs resolVI / SIMVI / MintFlow | `experiments/baseline_battery_lineage.md` per section (five sections); training times in `timing_all_lineage.md` |
| w predicts held-out shifts (transport) | envelope rows "transport …", "Read A …", "Read B …"; per run `transport/` |
| w does no likelihood work (R19, R24) | envelope rows "held-out recon, intrinsic-only / context-only / type-profile" |
| the atlas is stable across κ and seeds | envelope row "atlas cross-seed axis-1 cosine"; `atlas_kappa_survival*.json`; final_repair report §1 |
| the κ envelope at the reference budget | `experiments/kappa_sweep_sweepL.json`, `probe_regrade_lineage_sweep.md`, `validation_sweepL.json`, and `READOUT.md` per-run rows. **Not yet read into the devlog**, apart from κ-survival. |
| the leak form does not matter (κ-form rows) | `…ovarian_cancer_ffpe/experiments/r12_arms.md` (pre-lineage configuration) |
| the false-positive floor changes little | `…ovarian_cancer_ffpe/experiments/fp_floor.md` (pre-lineage configuration) |
| leak-induced false positives are rare; the corrected decode removes them specifically | `experiments/marker_pairs_kappa.md` per dataset |
| the surface lean of B | `experiments/go_localisation_lineage.md` (unread); older pins `go_localisation.md` |
| cycle_z depends on the label set | `experiments/cycle_2x2.md` per dataset |
| the amortisation gap | `synthetic_smoke/experiments/planted_posterior.md` |
| what opening the per-cell w channel would buy (α_w ladder) | `…ovarian_cancer_ffpe/experiments/awladder.md` (pre-lineage configuration) |
| design ablations (query/prior, collapse remedies, adversary ladder, projection) | `queryprior.md`, `wcollapse*.md`, `adv_ladder.md`, `adv_confirm.md`, `phi_projection_lineage.md` |

Label-set comparability, for anyone tabling old and new together:

- **NMI and cycle_z at lineage labels are not comparable with any pre-lineage
  number.**
- Pre-R26 `final` blocks describe the last epoch, not the accepted model.

## 7. Review-driven corrections: done vs open

The review is `submission_paper/aistats/review_2026-09-23.md`. Its checkboxes
track the *paper text*; the experiment and code side is tracked in the devlog and
todo §8.

| item | the defect | experiment / code | paper text (review checkbox) |
|---|---|---|---|
| **R26** | the closing evaluation scored the last-epoch model (`metrics.json["final"]` was `patience` epochs past `best.pt`) | **Done.** The trainer reloads `best.pt` before the closing evaluation; `metrics.json` gains `final_epoch` / `last_epoch`; `at_best.py` and `--at best` re-read older runs. Two decisions flipped on the re-read: α_z (§2.2) and the KL-defined clause of 8.9 (§2.1). Test `test_closing_evaluation_scores_the_accepted_checkpoint` is flaky (T-r26). | open |
| **R20 / R22** | the probe was blind to composition (pooled MSE, > 99 % image); the nonlinear probe was run once | **Done.** Per-block V-information gain, ridge and MLP, uncontrolled denominators, every decision input and baseline re-graded (§5.2). Composition leakage was flat across every swept knob, which led to 8.17 (§2.3). | open (numbers pending) |
| **R12** | the leak term's assumptions (per-gene κ; receiver vs donor depth) | **Done.** Tests 1/1b and 2 (§2.6). Donor scaling is *not* ruled out on FFPE; the response findings are stable across κ's form. The sensitivity rows are at the pre-lineage configuration. | wording applied [x]; sensitivity-row text is todo 8.18 (open) |
| **R17** | what the objective is (coupled model, "M-estimation") | Wording only: the two paths are bounds, J is a criterion; a non-independence note is in app:bound; the convergence diagnostic is held in reserve (`paperlog.md` 2026-09-24). | applied [x] |
| **R19** | the response is priced out: every w read is m_ψ | **Done.** The α_w ladder, the decision and the reframing (§2.4). Recon modes confirm it in likelihood terms. | open; text is todo 8.18 |
| **R9** | labels encode state and location | **Done.** Lineage relabel and re-pin (§2.5). Subtype recovery (8.22) is running. | text adopted [x] |
| **R24** | held-out reconstruction is an autoencoding score and drove selection | **Done.** Recon modes (§5.10), in every envelope table. | open (the paper must name the selection score) |
| **R33** | the finding rule needs a sampling layer | **Partly done.** Tile-bootstrap CIs on every headline read; half-tile subsampling for the transport ratios, with the coverage caveat (§5.9). **Open:** the κ* breakdown table with Bonferroni over the primary readouts (todo 8.19, "Evaluation layer for Definition 1"); Holm over the κ grid has no CLI. | Definition 1 text applied [x] |
| **S53** | the projection test named in the table was never run | **Done** at both label sets; not adopted (§2.8). | open |
| R18 | α_z is not ≈ 1/ℓ̄ | ℓ̄ discrepancy documented (§2.2) | open (todo 8.12, units) |
| R27 | for reference fits the seed also redraws the held-out split | not changed: each `finalL` seed has its own held-out tiles, so the seed envelope mixes model and split variance | open (todo 8.13) |
| R28 | sweep fits do not match the reference protocol | **Done:** `sweepL` runs at the reference budget and configuration | open *(checkbox not updated)* |

## 8. Queues, locks, markers, and the gitignore change

**Launching.**

- Every job longer than a minute runs detached:
  `setsid nohup <cmd> >> <log> 2>&1 < /dev/null & disown`. Session-tied
  background jobs die with the session.
- The user's shell is fish. Chain commands with `&&`, and prefix inline
  environment variables with `env` (e.g. `env SOX2OT=unassigned … setsid nohup …`).
- Queues export `OMP_NUM_THREADS=8`, `NUMBA_NUM_THREADS=8` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Idempotence.**

- A fit is skipped when its `metrics.json` exists.
- Every other step leaves a DONE marker in `<queue root>/done/`, named
  `<step>_<dataset>_<run>`.
- **Markers must be keyed by dataset** (issue W-q1): FF and ovarian once shared
  run names, and a marker keyed without the dataset would have silently skipped
  12 FF fits.
- Relaunching a queue resumes it. Failed steps go to `failed/`, with their logs in
  `logs/<marker>.log`.
- Retries: 10 on a CUDA OOM, 3 on anything else.

**GPU picker** (from `scripts/queue_2026-09-23_wcollapse.sh`, refined through
`…_2026-09-25_final_lineage.sh`):

- One of our jobs per GPU, through an atomic `mkdir` lock at `<root>/locks/gpu<N>`.
- A job starts only above a memory gate: FF 18,500 MiB, everything else
  9,500 MiB, resolVI 4,000 MiB.
- The picker settles 45 s after taking a lock and then re-reads free memory.
- A waiting FF step announces itself in `<root>/waiting/ff.<pid>`; smaller jobs
  then leave FF-sized cards alone.
- Another queue's fresh lock is respected for `FRESH=600` s (the courtesy window).
- MintFlow is charged at its 9 GB peak.
- Timing runs only on a card with no compute process.
- Stale locks are cleared at start and exit.
- Memory sizes: an FF fit needs most of a 24 GB card (about 17 GB resident at
  200/20). Two Trainers on the FF slide do not fit on one card: one battery
  process per card.

**Per-slot bookkeeping** in the final queues:

- `slots/` (the live run per slot), `seeds/` (atomically claimed replacement
  seeds 3–8) and `accepted/`;
- `DEAD_RUNS.tsv`, `REPIN.tsv` and `SWEEP_LINKS.tsv`;
- `code_hashes.tsv`, the sha256 of train/networks/equations/elbo at every fit
  start;
- a preflight that re-derives every arm's `TrainConfig` and stops unless it
  differs from the reference only where the arm says it should.

**Gitignore (G-1).** Until commit `6a21936` (2026-09-27; the author decided it on
2026-09-25), `scripts/` was ignored, so no queue, decision or table script was in
git. That commit changed `.gitignore` to
`/scripts/*` plus `!/scripts/*.py`, `!/scripts/queue_*.sh` and
`!/scripts/cell_cycle/`. Scripts are now tracked; **`scripts/logs/` stays
ignored**, and so does `/data/`. As a result, every `AGENT_REPORT.md`,
`READOUT.md`, `DECISION*.json`, `ARCHIVE_*.tsv` and lineage map exists **only on
this disk**. issues.md still lists G-1 as open; it is resolved in `.gitignore`,
and the register needs the update.

## 9. Retractions and superseded readings

**New since 2026-09-21:**

| item | what was said | what replaced it | where |
|---|---|---|---|
| α_z ½ "adopted by the rule" | the ladder qualified ½ | at the accepted checkpoint no rung qualifies; kept as a disclosed near miss | §2.2 |
| low KL_w = closed or weak w channel (2026-09-21 sweep note) | "the w channel closes entirely" | the posterior sits on an informative prior; the I(niche; w) guard decides | devlog 2026-09-23, "Interpretive correction" |
| free bits keep FF's channel open | open by KL | dead by I(niche; w) on FF | `wcollapse.md` |
| the FF-margin amendment (8.9) | prefer the arm with the largest KL_w margin | it measured divergence, not information; withdrawn for future use | devlog 2026-09-24, "Collapse remedy" |
| type-free query adopted for the re-pin | 8.10 read | the author kept query and prior unchanged | devlog 2026-09-24, "Decision (author)" |
| R12 test 1: donor scaling "ruled out" | negative donor term | confounded by geometry; test 1b: not ruled out on FFPE | `r12_depth_test.md` |
| the 2-sd probe guard | pre-registered 2026-09-24 morning | the wrong scale (floor sd is estimation noise); withdrawn before use | devlog "Per-block probe, first read" |
| legacy pooled probe at floor ⇒ z invariant | every earlier guard | > 99 % image block; composition ungraded (R20) | §5.2 |
| SOX2-OT⁺ tumour cells are a lineage | passes the DE + contiguity rule | mostly low-depth non-tumour cells → Unassigned; the rule cannot tell a clone from a state | §2.5 |
| "the relabel removed a Proliferative-type circularity in cycle_z" (2026-09-27 first reading) | cycle_z fell because a circularity went away | backwards: the old read was per-type centred; the lineage read *introduces* between-cluster variance; the drop is dilution | `cycle_2x2.md`, devlog 2026-09-27 |
| transport fraction-of-ceiling tile-bootstrap CI | rounds 1–2 | invalid (duplicated cells in both halves, then resampling bias); replaced by half-tile subsampling with a coverage caveat | final_repair report §§2, 4, 5 |
| the κ = 0 → κ > 0 drop in B stability (V11) | "still unexplained" | does not appear in shift space at the reference budget | κ-survival, §5.15 |
| the model's neighbourhood radius is 60 µm (bundle clip) | Voronoi-face validation entry | the model prunes at 40 µm (`prepare.DEFAULT_MAX_EDGE_UM`) | devlog 2026-09-23, "Correction" |
| "under nuclear-expansion segmentation" (manuscript) | the reason given against a contact kernel | 69–87 % interior-stain, 1–2.5 % nuclear expansion (R11); the contact-kernel table replaces the argument | devlog 2026-09-24, contact kernel |
| `metrics.json["final"]` = the accepted model | every run before 2026-09-24 | the last epoch (R26) | §5 |
| the MintFlow recon cell | −5.0 per count | constant across different fits; not a likelihood read (B-mf1) | issues.md |

**Carried over from the 2026-09-21 handover §5, still standing** (details there):

- per-type ‖w‖ ranking: withdrawn, gauge (V12);
- "6 programmes": superseded by effective rank (V10);
- matched-column B stability: replaced by shift space (V11);
- the v1 transport κ-sensitivity: superseded;
- α_w = 0.05 as a candidate: rejected, and now also covered by the α_w ladder;
- the w-mirror mechanism: falsified;
- the doc-10 guard: parked and removed;
- A4: parked;
- x̃ as a default: option only;
- `gat_sink` and weight decay: off;
- leak meter and transcript-flux κ_i adoption: instruments only;
- the DNA-content and "joint" DAPI cycle labels: not adopted;
- ungated hallmark labels: BH-gated;
- the "ceiling" for the cycle probe: renamed the linear reference, with the depth
  qualifier;
- M6 mean-of-types: pooled is the headline.

## 10. Limitations and threats to validity (ranked)

1. **Labels enter everywhere, and they are curated.** `enc_z`, `enc_w`, m_ψ, the
   adversary and the GAT sources all read t.
   - The lineage relabel is a curation with grey calls.
   - Per-cell reassignments have an expected accuracy of ~0.75, and some GSE
     classes come mostly from one donor.
   - The cycle 2×2 shows that a headline read moves with the label set alone.
   - Subtype recovery (8.22) is the test of what the merge handed to z and what
     it handed to w.
2. **Invariance is partial.** The adversary removes most, not all, of the
   within-type composition information in z.
   - The ≤ 25 %-of-uncontrolled guard is a judgement. It passes on FF and on few
     seeds elsewhere (envelope row "invariance guard").
   - The planted world attributes the residual to adversary capacity and
     weighting, but the planted model is smaller than the real one.
   - The denominators rest on two reference seeds, and both FF references have a
     dead w channel.
3. **κ is swept, not measured.** Donor scaling of the leak is not ruled out on
   FFPE. The κ-form and false-positive-floor sensitivity rows exist only at the
   pre-lineage configuration and 200/20.
4. **w is the context regression (R19).** No per-cell w claim is supported. w's
   value is shift prediction (transport), and its likelihood contribution is
   negligible (recon modes).
5. **Per-cell z claims remain at risk.** Two findings pull in different
   directions, and neither settles it:
   - The 2026-09-21 watch item (the z probe loses 0.25–0.38 of raw's recall on
     planted cycling cells at an unsaturated plant) has not been revisited.
   - 6b.8 measures the amortisation gap as below a supervised amortiser of the
     same inputs, which is a comparative statement only.
6. **The cycle target is weak and label-dependent.** Cycle_z depends on which
   cycling set is read. FF's cycle_w exceeds the guard on 2 of 3 seeds under the
   registered read. The honest reliability ceilings are the ones in the
   2026-09-21 handover §4.7.
7. **The transport scale is biased and its intervals under-cover.** The
   random-cell-split ceiling inflates the ceiling, so the fraction reads low.
   The half-tile intervals are about 0.8 coverage. GSE's trusted tier is thin.
8. **The dead-channel failure mode exists.** Warm-up removed it in the grid, and
   no `finalL` fit died, but the detector, the guard and the refit rule stay in
   force.
9. **The baselines are not symmetric.**
   - SIMVI and MintFlow were trained on the old labels.
   - SIMVI's FF run is a window, and MintFlow cannot run on FF on this host.
   - MintFlow's recon cell is unusable.
   - The baselines see held-out counts unlabelled, which is conservative for
     DisCell.
10. **The sensitivity and ablation rows predate the final configuration:**
    κ-form, false-positive floor, the α_w ladder, query/prior, the adversary
    ladder and the old-label projection test.
11. **Absolute numbers are slide-specific**, and NMI is not comparable across
    label sets. The one genuinely out-of-sample read is the GSE held-out section:
    a single ~70k-cell core from one donor.
12. **Instruments not re-run on `finalL`:** landmarks and the §5 matrix, doc-09
    communication and the LR ladder (§5).
13. **The seed also redraws the held-out split (R27),** so seed envelopes mix
    model variance with split variance.

## 11. Open issues, pending decisions, next steps

**Open issues (register: issues.md):**

| id | what | state |
|---|---|---|
| **G-1** | `scripts/` was gitignored | resolved in `.gitignore` (commit `6a21936`, §8); **issues.md not yet updated** |
| **T-r26** | `test_closing_evaluation_scores_the_accepted_checkpoint` is flaky: NMI re-evaluated on identical weights varies by ~0.003 (k-means) against a tight tolerance. It also fails on a clean HEAD worktree (`scripts/logs/metrics_2026-09-25/AGENT_REPORT.md`). | open; the fix is a tolerance that allows the re-evaluation noise |
| **W-qp1** (+ addendum) | `report.py` and `sweep.py` rebuild `DisCell` field by field and omit `class_mean_prior`, `query`, `prior_type_free` and `kappa_mode`, so ablation and κ-form runs cannot be reloaded by them. Only `validate.load_run` is complete. *(unverified whether `phi_proj` is also missing there)* | open; route both through `load_run` |
| *(not in issues.md)* | `sweep.py` has three limits. It cannot pass `--w-warmup-epochs` or `--adv-comp-weight`, so `sweepL` was fitted with one `train.py` call per fit. In `--report-only`, its w-guard, recon strata and per-type \|w\| use seed 0's split for every seed, which overlaps train and test for seeds 1–2 (the metric rows and B-stability are unaffected). `sweep.py` also still reads `final` for runs made before the R26 fix. | open; source: `scripts/logs/final_prep_2026-09-25/AGENT_REPORT.md` and `scripts/logs/at_best_2026-09-24/AGENT_REPORT.md` |
| **B-mf1** | the MintFlow recon cell in `baseline_battery*.md` is constant across different fits (−5.035 / −5.018); likely not the fit's decoded rates | open; **do not quote**; still present in the lineage tables |

**Decisions pending (author):**

1. **The cycling set for cycle_z.** Report cycle_z on both the old-type and the
   lineage reads, or register a label-independent set (e.g. MKI67 > 0, or the
   top decile of the S+G2M score) before use. Separately: whether the cycle_w
   guard should in future be read centred per the finest label available.
   Either choice must be registered before it is applied. Sources:
   `cycle_2x2.md`; devlog 2026-09-27 "Decision needed (author)".
2. **The primary ceiling column (8.21).** Both columns go on every transport
   table; the author chooses after seeing both, and the choice is recorded in
   the devlog. The tile-split correction *raises* our headline number, so it is
   an added column, never a silent replacement.
3. *(not raised anywhere I found)* Whether the κ-form, false-positive-floor and
   α_w-ladder sensitivity rows must be re-run at `finalL` (lineage labels,
   composition weight 3, 500/40) before they go in the paper.

**Next steps, in order:**

1. Land 8.21 (tile-split ceiling column) and 8.22 (subtype recovery) on the
   `finalL` triples; take decisions 1–2.
2. Read what is on disk and not yet in the devlog:
   - external criteria at `finalL` (§5.14);
   - `go_localisation_lineage.md`;
   - the `sweepL` κ-envelope tables (`kappa_sweep_sweepL.json`,
     `probe_regrade_lineage_sweep.md`).
3. **Paper sections B and C** of `submission_paper/aistats/revision_2026-09-17.md`:
   - B, stale or retracted numbers; C, results with no home yet;
   - numbers only from the `finalL` tables in §6;
   - todo 8.18 text (the false-positive-floor sentence, the κ-form rows, the R19
     reframing, the α_w ladder as sensitivity);
   - close the review checkboxes of §7.
4. **6b.7, the three-way per-cell contamination figure:** our κ grid, resolVI's
   per-cell mixture proportion (`results/resolvi_lineage/`) and MintFlow's
   microenvironment score, against the transcript-flux share, on the same cells.
   MintFlow exists only on GSE and on old labels.
5. **6b.9:** MintFlow's in-silico perturbation, scored with our noise ceiling and
   gap-closed statistic (GSE only).
6. R33's κ* breakdown table (todo 8.19, "Evaluation layer for Definition 1"),
   and a CLI for `bootstrap.paired_comparisons` (Holm over the κ grid).
7. Housekeeping:
   - commit (§13);
   - W-qp1 and T-r26;
   - add `diptest` to `pyproject.toml`;
   - bring issues.md (G-1) and todo §8 up to date. Todo §8 has duplicate item
     numbers (8.11, 8.12, 8.14, 8.15b and 8.19 each appear twice) and stale
     statuses (8.1 "reopened", 8.14 "open", 8.7 "open").
8. Still open from before, and not blocking the paper:
   - todo 8.11, "Invariance escalation": all runs use the adversary, while the
     implementation text says the rule "decides between them on each section";
   - todo 8.12 (weights in units of 1/ℓ̄);
   - todo 8.13 R27.

## 12. Registers

| register | what it holds |
|---|---|
| [devlog.md](devlog.md) | chronological narrative; the motivation is written before each run and results after. The current tail is 2026-09-28, "Two additions before the paper numbers". |
| [issues.md](issues.md) | P / E / M / T / A / V tables and the watch list; the latest items are W-qp1, W-q1, T-r26, B-mf1 and G-1 |
| [spec_deviations.md](spec_deviations.md) | doc-07/08/09/10/11 rows; the tail adds the optional flags (L2 on w, the 6b.5 ablations) and the tumour-band definition of 2026-09-25 |
| [todo.md](todo.md) | §0 decisions; §6a/6b article-derived instruments (6b.7 and 6b.9 open); §8 finalisation |
| [paperlog.md](paperlog.md) | manuscript changes, append-only |
| `submission_paper/aistats/review_2026-09-23.md` | the review items (R*, S*), each with a checkbox for the paper text |
| `submission_paper/aistats/revision_2026-09-17.md` | the paper-vs-reality list; A applied, B and C pending |
| `scripts/logs/*/AGENT_REPORT.md` | one report per delegated work package, with code, tests, commands and caveats. **Local only** (gitignored). The ones behind this document are the `*_2026-09-2{3,4,5,7}` folders. |
| [sweep_programme.md](sweep_programme.md) | the 2026-09-17 grid plan; historical |

The architect documents (07/08/09/11-*.md) sit at the repo root and are received
as-is. Departures from them go in `spec_deviations.md`.

## 13. Repository state (2026-09-28)

**Last commit.** `6a21936` (2026-09-27 18:43, "settled for default and added
better type selection"). It added the `.gitignore` change and 33 previously
ignored scripts. The lineage-era model and evaluation code that the final queue
ran with (`labels.py`, `apply_lineage.py`, `bootstrap.py`, `recon_modes.py`,
`fp_floor.py`, …) was committed earlier, in `99fdb97` (2026-09-25 18:40) and the
commits before it.

**Uncommitted.** The 2026-09-27 repairs and additions:

- modified: `discell/experiments/bootstrap.py` (the replay-precision fix, the
  distinct-cell split and half-tile subsampling), `discell/model/validate.py`
  (the `sweep_companion` fix), `scripts/archive_2026-09-25.py` (`--apply`),
  `scripts/envelope_tables.py`, and their tests;
- untracked: `discell/experiments/{cycle_2x2,marker_pairs}.py`, their tests,
  `tests/test_scripts_archive_apply.py`, `scripts/queue_2026-09-27_marker_pairs.sh`,
  `docs/figures/`, `submission_paper/articles/litterature_review.md`;
- `docs/devlog.md`, `docs/todo.md`, and this file.

**Commit before handing over.**

**Tests.** 53 test files (526 `test_*` functions by grep). The suite was **not
re-run** for this handover. T-r26 is a known failure, also on a clean tree. The
targeted suites named in each agent report passed when those reports were
written.

**Dependency drift.** `diptest` is used by the DAPI gates and is still not in
`pyproject.toml`.

**Data outside git.**

- Everything under `data/`: bundles, runs, experiments and lineage maps.
- `scripts/logs/`.
- `/home/rmolen/github/DisCell-baselines/` (a separate tree).

Losing this disk loses the decision records.

**Stale artefacts to know about:**

- `READOUT.md`'s MISSING and failed lines (§3).
- `experiments/envelope_table_at_best.md`, `envelope_table_ci.md` and
  `baseline_battery.md` (without `_lineage`) describe the pre-lineage `final_s*`
  or older pins. Superseded by the `_ci_at_best` and `_lineage` files.
- Old-label tumour-band reads differ from any re-read by the cyst-lining
  exclusion.
- The pre-correction `transport.json` files on runs now in `_archive/` must not
  be compared with current transport numbers.

## 14. What I could not verify for this handover

- That the reproduce command in §2 produces `finalL_s0` bit for bit. I expanded
  it from the queue script and did not run it.
- Whether every CLI accepts `--run _archive/<name>` for archived runs.
- Whether `report.py` and `sweep.py` also drop `phi_proj` (W-qp1 scope).
- The current pass count of the test suite (not run).
- What state todo 8.21 and 8.22 are in. The todo says "running 2026-09-28", but
  I found no code or output for either in the working tree. They may be in
  another session.
- Whether `validate --analyses probe` still runs now that its
  `uncontrolled500_s*` references are in `_archive/`.
