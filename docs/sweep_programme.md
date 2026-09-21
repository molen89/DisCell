# The sweep programme across the four datasets (plan, 2026-09-17)

Three ablations — κ, d_w, α_w — on all four datasets, so that "stable where
it is stable" is a measured claim on more than the development slide. This
file is the inventory, the schedule and the launcher pattern. Nothing here
has been run: it is the pre-registration.

**How.** One CLI, `discell/model/sweep.py`, fits one model per (value, seed)
of one swept knob, everything else held at the operating point, and reads the
finished runs back into one comparison
(`experiments/<param>_sweep[_<tag>].json`). Per-dataset knobs — label column,
α_z, tile size, bundle variant, budget — are CLI arguments, so the same three
commands run on any dataset. Runs with a `metrics.json` are skipped, so a
relaunch of the same command resumes an interrupted grid.

**Evaluated by.** Per value: recon, NMI(z,t), cycle R² in z and w against the
50-PC linear reference, probe ΔCE, mirror R², and (when the producing run
writes them) the §7.10 degeneracy pair and the type-mean recon gap. Across
seeds within a value and along the value axis: matched-column |corr| of B,
plus ‖w‖ per type and reconstruction stratified by the edges-lost QC column.

**What is wished for.** That the reads which are flat along κ on the ovarian
slide (§4.2 of the handover) are flat along κ on three further slides; that
d_w = 6 survives on tissues the choice was not made on; and that the α_w
picture — context field at 0.1, a second reproducible context axis at 0.05,
identity theft below ~0.02 — is a property of the model rather than of one
ovarian FFPE section. A grid that disagrees across slides is the finding, not
a failure, and is reported as such.

---

## 1. The grids

| ablation | values | seeds | fits |
|---|---|---|---|
| κ | 0, 0.05, 0.1, 0.2, 0.3, 0.4 | 0, 1, 2 | 18 |
| d_w | 2, 3, 6, 8 | 0, 1, 2 | 12 |
| α_w | 0.03, 0.05, 0.07, 0.1, **0.2, 0.3** | 0, 1, 2 | 18 |
| | | | **48** |

The α_w values above 0.1 are kept on purpose: everything measured so far
(the α_w study, `experiments/alphaw_study.json`) walks *down* from 0.1 and
only ever sees the channel open. Whether the channel closes further, and what
that costs, has never been measured under `type_only`; `cal2_aw_0.2` is a
calibration-era single fit under the closed-form invariance and is not
evidence.

The three grids share one point: κ = 0.1, d_w = 6, α_w = 0.1 *is* the
operating point and *is* the centre of all three. Fitting it once per dataset
and reusing it in all three tables gives **42 distinct fits per dataset**.
The cost of not doing so is 6 extra fits per dataset (two duplicate triples);
the benefit would be that each grid is self-contained under its own run-name
prefix. Open decision Q2 below.

## 2. Per-dataset knobs

| dataset | cells (variant) | label key | α_z | tile cells | note |
|---|---|---|---|---|---|
| `xenium_prime_ovarian_cancer_ffpe` | 407,120 | default (curated 18) | 0.007 | 4096 | development slide |
| `xenium_prime_human_lung_cancer_ffpe` | 278,324 | `graphclust` (32+Unassigned) | 0.004 | 4096 | |
| `xenium_prime_human_ovary_ff` | 1,157,637 | `graphclust` (38+Unassigned) | 0.0007 | 4096 | fresh frozen, 8× depth |
| `gse315411_pdltma06_11_prime_solo` | 69,422 (`pdl018d`) | default (curated 35) | 0.0036 | 2048 | `--variant pdl018d` |

α_z is the 1/mean-count rule, not a tuned number; every other knob is the
calibrated operating point (`type_only`, α_a 0.3 / 6 adversary steps / lr
2e-3, ω 1, d_z 20, hidden 256, gat_dim 32, heads 4).

**GSE315411 (decided by the author).** The sweeps run on the **core
`pdl018d` of the solo section** — 69k cells, curated 35-class vocabulary, the
configuration `runs/reference` already validated (recon −7.2201, NMI 0.625,
cycle_z 0.478 / cycle_w 0.002). The two **full slides (1.10M / 1.12M cells)
are not swept**: they are the held-out-section test, which is an evaluation
of a fitted checkpoint (`discell.model.crossslide`), not a fit. The optional
extra leg in §5 applies that evaluation to every swept core run.

**Fresh-frozen slide: full grids, no shrinking.** All 42 fits, same values,
same seeds. Memory is the only real constraint and it does not bite: with
int16 resident counts the measured footprint is ~18–20 GB of a 24 GB card
(512 tiles × 4096 cells, 1.37M nodes including rings), so **one FF fit per
GPU, two in flight machine-wide**. No `--max-cells` window is needed — and
none exists as a flag; the int16 change already bought the headroom. Tile
size does not change residency (every tile is resident), so the 4096 tiles of
the FF reference stay.

## 3. What already exists and can be reused

Ovarian only. Everything below is `type_only`, 200 epochs / patience 20,
defaults otherwise — the same protocol this programme uses.

| grid | on disk | reusable | new fits needed |
|---|---|---|---|
| κ | `sweep3_k{0,0.05,0.1,0.2,0.3,0.4}_s{0,1,2}` | all 18 | **0** |
| d_w | `dw{2,3,8}_s{0,1,2}` (9) + `sweep3_k0.1_s{0,1,2}` as d_w = 6 | all 12 | **0** |
| α_w | `sweep3_k0.1_s{0,1,2}` as α_w = 0.1 | 3 | **15** |

Not reusable, and a trap to avoid: `alphaw_0.02`, `alphaw_0.03{,_s1,_s2}`,
`alphaw_0.05{,_s1,_s2}`, `alphaw_0.07` are **`type_z`-era** runs (the α_w
study and its seed check predate the 2026-09-12 ratification of
`type_only`; their `config.json` has no `gat_sources` key at all), and they
were fitted at 500/40 rather than the sweep budget. They must not be mixed
into a `type_only` α_w table; the sub-0.1 seeds are re-fitted here.
`cal2_aw_*` are closed-form-era calibration fits (40 epochs, α_a 0.03, no
`invariance` key) and are likewise out.

No runs exist for any ablation on lung, FF or GSE: those three datasets have
one reference fit each and nothing else.

## 4. Hours

Per-fit wall clock at the sweep budget, extrapolated from the four measured
reference/sweep runs (`metrics.json: minutes`, and the epoch at which each
stopped). These are single-job-per-GPU numbers; the measured contention
factor when two fits share the 32 cores during the figure step is 2–7×,
which is what `--figures-every` and the thread caps exist to avoid.

| dataset | measured anchor | s/epoch | epochs to stop | per fit (plan) |
|---|---|---|---|---|
| ovarian | `sweep3_k0.1_s0` 6.3 min, stopped ep. 79 | ~4.8 | ~80 | **10 min** |
| GSE core | `reference` 11.7 min (500/40), stopped ep. 94 | ~7.5 | ~75–95 | **12 min** |
| lung | `reference_graphclust` 71.9 min (500/40), stopped ep. 99 | 43 contended, ~15–20 alone | ~80 | **25 min** (60 contended) |
| FF | `reference_graphclust` 182.7 min (500/40), stopped ep. **409** | 26.8 | **200 cap** | **90 min** |

| leg | new fits | GPU-hours |
|---|---|---|
| ovarian α_w | 15 | 2.5 |
| GSE core, all three | 42 | 8.4 |
| lung, all three | 42 | 17.5 |
| FF, all three | 42 | 63 |
| **total** | **141** | **91.4** |

On 2 GPUs, one fit per GPU: **≈ 46 h of wall clock**, call it **50–60 h**
with contention and the report passes — a bit over two days of detached
running. Split by leg: ovarian 1.5 h, GSE 4 h, lung 9 h, FF 32 h.

**The FF budget is the one honest wrinkle.** FF's own reference did not stop
until epoch 409 (best 369), where ovarian/lung/GSE stop at 94–104. A 200-epoch
cap therefore truncates *every* FF fit, and FF sweep numbers would be
comparable to each other but not to FF's reference nor to the other slides'
converged optima. Running FF at its reference budget (500/40) costs
183 min × 42 = **128 GPU-h for FF alone** (total 156 GPU-h, ≈ 78 h wall, ~3.3
days), which the author's "compute is not a constraint" allows. Open decision
Q1.

## 5. Recommended order — cheapest and most informative first

1. **Ovarian α_w, 15 fits, 2.5 h.** Closes the only dataset where all three
   grids then exist in one era, and the 0.2/0.3 arm is genuinely new: it is
   the first measurement of what over-tightening α_w costs. Uses runs already
   on disk for the rest of the table.
2. **GSE core, all three grids, 42 fits, 8.4 h.** The cheapest complete
   dataset, a different tissue and organism context (pediatric lung), a
   curated 35-class vocabulary, and the only slide with a true held-out
   section. If the κ envelope is going to break somewhere, this is where it
   is cheapest to find out.
3. **Lung, all three, 42 fits, 17.5 h.** Annotation-free labels on a second
   FFPE tumour slide; the closest replicate of the ovarian protocol.
4. **FF, all three, 42 fits, 63 h (or 128 h at 500/40).** Last: most
   expensive, one fit per GPU, and the most likely to need its own budget
   decision. Detached, over two to three days.

*Optional leg, after 2:* cross-slide evaluation of every swept GSE core run
on the dual section (`discell.model.crossslide`, evaluation only, ~1–2 min
per run) — 42 evaluations, ~1.5 h, turning each ablation into "does the
stability survive a section change?". Open decision Q6.

## 6. The launcher pattern

The sweep walks its grid serially in one process, so parallelism comes from
splitting disjoint work across the two GPUs; idempotence is per run name, so
two processes with disjoint `--seeds` (or disjoint `--param`) never collide.

```bash
DS=gse315411_pdltma06_11_prime_solo
mkdir -p logs

# GPU 0: seeds 0 and 1
CUDA_VISIBLE_DEVICES=0 setsid nohup env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  uv run python -m discell.model.sweep --dataset $DS --variant pdl018d \
  --param kappa --tag sweep3 --seeds 0 1 \
  --alpha-z 0.0036 --tile-cells 2048 --epochs 200 --patience 20 \
  --figures-every 200 \
  > logs/${DS}_kappa_s01.log 2>&1 < /dev/null & disown

sleep 120        # stagger: the two assemble() passes must not collide

# GPU 1: seed 2
CUDA_VISIBLE_DEVICES=1 setsid nohup env OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  uv run python -m discell.model.sweep --dataset $DS --variant pdl018d \
  --param kappa --tag sweep3 --seeds 2 \
  --alpha-z 0.0036 --tile-cells 2048 --epochs 200 --patience 20 \
  --figures-every 200 \
  > logs/${DS}_kappa_s2.log 2>&1 < /dev/null & disown

# once both are done: one report pass over the full grid
CUDA_VISIBLE_DEVICES=0 uv run python -m discell.model.sweep --dataset $DS \
  --variant pdl018d --param kappa --tag sweep3 --tile-cells 2048 --report-only
```

Rules that make this work, each from a measured lesson:

- **Detached, always.** `setsid nohup … < /dev/null & disown`; a session-tied
  background job dies with the session.
- **`--figures-every 200`** (= `--epochs`), so the ×18-panel figure step runs
  at most once per fit. The figure step is what oversubscribes the CPU when
  two fits share the machine; `figures_every` must stay a multiple of
  `eval_every` (5). Metrics are logged every eval regardless.
- **Thread caps** `OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8` on every process.
- **Stagger by ~2 minutes** so the two `assemble()` passes (graph build,
  embeddings load) do not peak together.
- **One fit per GPU on FF**; two elsewhere is possible but CPU-bound, so one
  per GPU is the plan everywhere.
- **Resume by relaunching the same command.** A finished run has a
  `metrics.json` and is skipped; an interrupted one has none and is refitted
  from scratch. `--force` overrides.
- **Skip the centre where it exists.** With the reuse plan (Q2), the d_w
  leg is launched as `--param d_w --values 2 3 8` and the α_w leg as
  `--param alpha_w --values 0.03 0.05 0.07 0.2 0.3`; the centre triple comes
  from the κ leg's `k0.1` runs. The report pass is then run with the full
  grid (`--values 2 3 6 8`) only if the centre was fitted under that name.
- **Run names.** `<tag>_<abbrev><value>_s<seed>` with abbreviations `k`,
  `dw`, `aw`; `--tag ''` drops the prefix. `--tag sweep3 --param kappa`
  reproduces the existing `sweep3_k0.1_s0` names exactly, and
  `--tag '' --param d_w` reproduces `dw2_s0`; use `--tag '' --param alpha_w`
  for `aw0.05_s0`. The report lands in `experiments/<param>_sweep_<tag>.json`
  (`kappa_sweep_sweep3.json` unchanged; an empty tag gives
  `<param>_sweep_untagged.json`).

## 7. Open decisions for the author

1. **FF budget.** 200/20 like the others (63 h, but every FF fit truncated —
   FF's reference needed 409 epochs) or 500/40 (128 h for FF, comparable to
   FF's own reference but a different budget from the other three slides)?
   Recommendation: 500/40 for FF, with the budget difference stated in the
   paper as a per-dataset protocol note.
2. **The shared centre point.** Reuse κ = 0.1 / d_w = 6 / α_w = 0.1 across
   all three tables (42 fits/dataset, the plan above), or fit
   `dw6_s*` and `aw0.1_s*` separately so each grid is self-contained under
   its own name (48 fits/dataset, +6 duplicates)?
3. **Degeneracy metrics first?** Another agent is adding the §7.10 degeneracy
   metrics to the per-run evaluation. The sweep report reads them if present
   and reports `null` if not. If the programme launches first, all 141 runs
   will lack them and will need a re-evaluation pass. Recommendation: land
   those metrics, then launch. This is the one hard scheduling dependency.
4. **α_w = 0.02?** The grid starts at 0.03. 0.02 is where the measured
   knife-edge sits (NMI 0.612, sliding at the guard floor) and would put a
   failure point inside every dataset's curve — informative, 3 fits/dataset.
5. **d_w above 8?** The d_w = 8 result was "fill is still 2, but the second
   axis's share is 3× larger and seed-stable" — not monotone, unexplained.
   Adding 12 to the grid on the cheap datasets (ovarian, GSE) would say
   whether the share keeps growing with the box. 3 fits/dataset.
6. **Cross-slide leg on GSE.** Evaluate all 42 swept core runs on the dual
   section (~1.5 h), or only the centre triple?
7. **Priority if something has to give.** If the FF leg is too long to sit
   in the queue, the cheapest honest cut is d_w on FF (12 fits, 18–37 h) —
   d_w was answered on ovarian and closed at 6 — rather than shrinking any
   grid.
