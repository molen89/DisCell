#!/usr/bin/env python3
"""The read-out of the alpha_w multiplier ladder (review R19).

Pre-registration: devlog "alpha_w in units of 1/l-bar: the multiplier ladder
(R19, motivation, 2026-09-24 13:50)". alpha_w = m / l-bar, m in {1, 2, 5, 10,
25}, plus the reference rung alpha_w = 0.1; four datasets, seeds 0 1 2, every
read at the accepted checkpoint.

    uv run python scripts/awladder_table.py --init-ladder   # once, before any fit
    uv run python scripts/awladder_table.py --at best       # the read-out

``--init-ladder`` fixes l-bar once per dataset and writes the rungs' alpha_w
to ``LADDER.json``, which the queue reads. The read-out writes
``<ovarian>/experiments/awladder.{json,md}`` and, only once every rung of
every dataset has all three seeds and every read, ``DECISION_AW.json``.

**l-bar.** The pre-registration defines l-bar as the mean total count of the
training cells and says it equals 1/(2 alpha_z) at the pinned alpha_z. On the
data it does not: the training-cell mean is 1.2-1.8x 1/(2 alpha_z) (the
alpha_z rule's count scale sits at the median on GSE, lung and FF). The ladder
uses l-bar = 1/(2 alpha_z) (``LBAR_CONVENTION``): it is the scale behind every
number the pre-registration states (reference rung m = 14.3 / 25 / 27.8 / 143)
and it keeps alpha_z and alpha_w in one unit (alpha_z = 1/2 x 1/l-bar exactly).
The training-cell means and the ladder they would give are recorded beside it.

**Guards** (pre-registered; every seed, every dataset; a rung against the
reference rung of its own dataset; "one-seed-sd" = the reference's sample sd,
the convention of scripts/wcollapse_table.py; a value better than the band in
the field's named direction passes, as there):

1. recon: the rung's seed spread (max - min) is within the width of the
   reference's band widened by one sd on each side, (max - min) + 2 sd.
   [reading of "recon seed spread within the reference rung's band"; the
   level reading -- every seed at or above the reference min - sd -- is
   reported beside it, not used]
2. NMI >= 0.9 x its running max (history, post-warm-up evaluations up to the
   accepted epoch), and not below the reference mean by more than 0.02.
3. probe: per seed, ``invariance_pass`` of ``validation/probe_blocks.json``
   (the per-block invariance probe, composition vs image, built by another
   lane, coordinator 2026-09-24) when that file exists; else the legacy
   pooled probe at its floor, |delta_ce| <= |noise_floor|. The legacy pooled
   delta_ce is always reported, labelled legacy.
4. cycle_z inside the reference band widened by one sd, or above it.
5. cycle_w <= 0.02.
6. I(niche; w) guard alive (degeneracy.json).
7. w-side mirror (w_deviation.json ``w_mirror.r2``) not above the reference
   mean by more than one reference sd.

**Reference envelopes** (coordinator, 2026-09-24, after stage 1): every guard
and the gain clause are applied three times, against the fresh reference fits
(``aw_ref0.1_s<k>``, the rungs' own trainer and budget), against the reused
w-collapse-grid fits (``aw_ref0.1wfix_s<k>`` -> ``wfix_warmup30_aw0.1_s<k>``),
and against all of them pooled. A dataset with only one kind (GSE: fresh only)
uses it in every variant. The variants are reported side by side; none is
promoted over the others here.

**Rule.** Adopt the smallest m that passes every guard on every dataset AND
whose held-out gain (mean over seeds) exceeds the reference rung's mean by
more than the reference's seed range (max - min) on GSE and on FF. Otherwise
alpha_w stays at 0.1 and w is the context regression m_psi (R19's reframing).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from discell.experiments.at_best import AT_CHOICES, HEADER, battery_at_best

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "datasets"
GS = "gse315411_pdltma06_11_prime_solo"
LU = "xenium_prime_human_lung_cancer_ffpe"
OV = "xenium_prime_ovarian_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
DATASETS = (GS, LU, OV, FF)
SHORT = {GS: "GSE", LU: "lung", OV: "ovarian", FF: "FF"}
PRIMARY = (GS, FF)
#: what the pinned fits read the dataset with (scripts/queue_2026-09-24_final.sh)
SPECS = {GS: ("pdl018d", 2048, None), LU: ("full", 4096, "graphclust"),
         OV: ("full", 4096, None), FF: ("full", 4096, "graphclust")}
MULTIPLIERS = (1, 2, 5, 10, 25)
REFERENCE_AW = 0.1
REFERENCE = "ref0.1"
#: the reused reference fits' alias (a link to wfix_warmup30_aw0.1_s<k>)
REUSED = "ref0.1wfix"
VARIANTS = ("fresh", "reused", "pooled")
VARIANT_LABEL = {"fresh": "fresh reference", "reused": "reused reference (wfix)",
                 "pooled": "fresh + reused pooled"}
RUNGS = (REFERENCE,) + tuple(f"m{m}" for m in MULTIPLIERS)
SEEDS = (0, 1, 2)
#: the reference rung's existing fits (same configuration, preflighted by the
#: queue): ovarian and FF reuse the w-collapse grid's warm-up arm
REUSED_REFERENCE = {OV: "wfix_warmup30_aw0.1_s", FF: "wfix_warmup30_aw0.1_s"}
#: l-bar := 1 / (2 alpha_z); see the module docstring
LBAR_CONVENTION = "alpha_z"
LOGS = REPO / "scripts" / "logs" / "awladder_2026-09-24"
LADDER_PATH = LOGS / "LADDER.json"
DECISION_PATH = LOGS / "DECISION_AW.json"
OUT_STEM = ROOT / OV / "experiments" / "awladder"
#: The author's scope change (2026-09-24, before any fit counted): stage 1 =
#: GSE and ovarian, seeds 0 1, all six rungs; stage 2 = FF, seeds 0 1, the
#: reference plus the rungs named then; lung dropped. The queue records the
#: scope in SCOPE.json and :func:`set_scope` narrows the globals to it; the
#: defaults below are the pre-registered full grid.
SCOPE_PATH = LOGS / "SCOPE.json"
RUNGS_OF: dict = {ds: RUNGS for ds in DATASETS}


def set_scope(scope: dict) -> None:
    """Narrow DATASETS, SEEDS and the rungs per dataset to *scope*
    (``{"seeds": [...], "datasets": {dataset: [rung, ...]}}``)."""
    global DATASETS, SEEDS, RUNGS_OF
    DATASETS = tuple(ds for ds in (GS, LU, OV, FF) if ds in scope["datasets"])
    SEEDS = tuple(int(s) for s in scope["seeds"])
    RUNGS_OF = {ds: tuple(r for r in RUNGS if r in scope["datasets"][ds])
                for ds in DATASETS}
CYCLE_W_GUARD = 0.02
NMI_GUARD = 0.9
NMI_TOLERANCE = 0.02
GUARDS = ("recon_spread", "nmi_running_max", "nmi_vs_reference",
          "probe", "cycle_z_band", "cycle_w", "niche_guard",
          "w_mirror")


def run_name(rung: str, seed: int) -> str:
    return f"aw_{rung}_s{seed}"


def _write(path: Path, text: str) -> None:
    """Atomic: two queue workers may refresh the partial table at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# -- the ladder ---------------------------------------------------------------

def training_count_scale(dataset: str) -> dict:
    """Mean total count of the training cells (each seed's split) and the
    all-cell mean and median, from the bundle, as ``prepare.assemble`` splits."""
    from discell import paths
    from discell.data.loader import CellGraphDataset
    from discell.model.prepare import spatial_tiles

    variant, tile_cells, label_key = SPECS[dataset]
    opened = CellGraphDataset.from_dataset(paths.dataset(dataset), variant,
                                           graph="voronoi", label_key=label_key)
    totals = np.asarray(opened.counts.sum(axis=1)).ravel().astype(np.float64)
    tiles = spatial_tiles(opened.positions_um, tile_cells)
    by_seed = {}
    for seed in SEEDS:           # prepare.assemble's split stream
        order = np.random.default_rng([seed, 2]).permutation(len(tiles))
        n_val = max(1, int(round(0.15 * len(tiles))))
        train = np.concatenate([tiles[i] for i in order[n_val:]])
        by_seed[str(seed)] = float(totals[train].mean())
    return {"train_mean_by_seed": by_seed, "all_mean": float(totals.mean()),
            "all_median": float(np.median(totals)), "n_cells": int(len(totals))}


def init_ladder(force: bool = False) -> dict:
    if LADDER_PATH.exists() and not force:
        raise SystemExit(f"{LADDER_PATH} exists: l-bar is fixed once")
    out = {"convention": LBAR_CONVENTION, "multipliers": list(MULTIPLIERS),
           "reference_alpha_w": REFERENCE_AW, "datasets": {}}
    for ds in DATASETS:
        alpha_z = json.loads((ROOT / ds / "runs" / "best_s1" / "config.json")
                             .read_text())["alpha_z"]
        scale = training_count_scale(ds)
        lbar_az = 1.0 / (2.0 * alpha_z)
        lbar_mean = scale["train_mean_by_seed"]["0"]
        # alpha_w = m / l-bar = 2 m alpha_z, rounded off float dust
        rungs = {f"m{m}": round(2.0 * m * alpha_z, 10) for m in MULTIPLIERS}
        rungs[REFERENCE] = REFERENCE_AW
        links = {}
        if ds in REUSED_REFERENCE:
            links.update({run_name(REFERENCE, s): f"{REUSED_REFERENCE[ds]}{s}"
                          for s in SEEDS})
        for rung, aw in rungs.items():   # a rung that IS the reference config
            if rung != REFERENCE and aw == REFERENCE_AW:
                links.update({run_name(rung, s): run_name(REFERENCE, s)
                              for s in SEEDS})
        out["datasets"][ds] = {
            "alpha_z": alpha_z, "lbar": lbar_az,
            "lbar_from_alpha_z": lbar_az,
            "lbar_train_mean": lbar_mean, "count_scale": scale,
            "train_mean_over_lbar": lbar_mean / lbar_az,
            "alpha_w": rungs,
            "reference_m": REFERENCE_AW * lbar_az,
            "alpha_w_if_train_mean": {f"m{m}": round(m / lbar_mean, 7)
                                      for m in MULTIPLIERS},
            "reference_m_if_train_mean": REFERENCE_AW * lbar_mean,
            "links": links}
    _write(LADDER_PATH, json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# -- one run ------------------------------------------------------------------

def read_run(dataset: str, run: str, ladder_aw: float | None,
             at: str = "best") -> dict | None:
    """Every number one fit contributes, or None before its fit has landed."""
    d = ROOT / dataset / "runs" / run
    metrics = _load(d / "metrics.json")
    if metrics is None:
        return None
    config = _load(d / "config.json") or {}
    battery = battery_at_best(d) if at == "best" else metrics["final"]
    history = [json.loads(line) for line in
               (d / "history.jsonl").read_text().splitlines() if line.strip()]
    best_epoch = int(metrics["best"]["epoch"])
    warmup = int(config.get("w_warmup_epochs", 0) or 0)
    eligible = [r["nmi"] for r in history
                if warmup <= int(r["epoch"]) <= best_epoch]
    nmi = battery["nmi"]
    out = {
        "run": run, "target": d.resolve().name,
        "alpha_w": config.get("alpha_w"), "alpha_w_ladder": ladder_aw,
        "alpha_w_ok": bool(ladder_aw is not None and np.isclose(
            config.get("alpha_w", np.nan), ladder_aw)),
        "best_epoch": best_epoch,
        "at_best": bool(battery.get("at_best", at != "best")),
        "dead_w_channel_trainer": bool(metrics.get("dead_w_channel")),
        "recon": battery["recon_val"], "nmi": nmi,
        "nmi_running_max": max(eligible) if eligible else float("nan"),
        "probe_delta_ce": battery["probe"]["delta_ce"],
        "probe_noise_floor": battery["probe"]["noise_floor"],
        "mirror": battery["mirror"]["r2"],
        "cycle_z": (battery.get("cycle") or {}).get("z", {}).get(
            "r2_pooled", float("nan")),
        "cycle_w": (battery.get("cycle") or {}).get("w", {}).get(
            "r2_pooled", float("nan")),
        "kl_w": float(np.sum(battery["kl_w_per_dim"])),
        "kl_w_per_dim": battery["kl_w_per_dim"],
    }
    out["nmi_running_max_ok"] = bool(
        eligible and nmi >= NMI_GUARD * out["nmi_running_max"])
    out["probe_at_floor"] = bool(abs(out["probe_delta_ce"])
                                 <= abs(out["probe_noise_floor"]))
    blocks = probe_blocks(d)
    out["invariance_pass"] = None if blocks is None else blocks["invariance_pass"]
    out["probe_source"] = "legacy" if out["invariance_pass"] is None else "blocks"
    out["probe_ok"] = (out["probe_at_floor"] if out["invariance_pass"] is None
                       else bool(out["invariance_pass"]))
    for name, block in (blocks or {}).get("blocks", {}).items():
        out[f"{BLOCK_PREFIX}{name}"] = block["excess"]
    out["cycle_w_ok"] = bool(out["cycle_w"] <= CYCLE_W_GUARD)

    guard = (_load(d / "degeneracy.json") or {}).get("w_channel")
    out["niche_excess"] = guard["w_niche_mi_excess"] if guard else float("nan")
    out["niche_alive"] = (None if guard is None
                          else not guard["dead_context_channel"])

    out.update(transport_reads(d))
    dev = _load(d / "w_deviation.json")
    if dev is not None:
        cycle = (dev["deviation"].get("cycle") or {})
        out.update({
            "w_mirror": dev["w_mirror"]["r2"],
            "w_mirror_floor": dev["w_mirror"]["r2_floor"],
            "d_mirror": dev["d_mirror"]["r2"],
            "d_mirror_floor": dev["d_mirror"]["r2_floor"],
            "dev_nmi": dev["deviation"]["nmi_type"],
            "dev_cycle": cycle.get("r2_pooled", float("nan")),
            "dev_var_share": dev["deviation"]["var_share_of_mu_w"],
            "gain": dev["recon"]["gain"],
            "recon_mu_w": dev["recon"]["mu_w"],
            "recon_prior_w": dev["recon"]["prior_w"],
            # the read and the trainer saw one checkpoint
            "dev_recon_check": dev["recon"]["mu_w"] - battery["recon_val"]})
    return out


BLOCK_PREFIX = "block_excess:"
#: the transport tier read, all panels (the 8.x / r12_table convention)
TIER = "extrapolation"


def transport_reads(run_dir: Path) -> dict:
    """Read A (k-means niches) and, where run, Read B (tumour bands): the
    own-target median gap closed of the distribution read, and the mean
    read's fraction of the noise ceiling and beats-both share of panels.
    All from best.pt reads (transport loads the checkpoint)."""
    out = {}
    for tag, stem in (("readA", "transport"), ("readB", "transport_tumour-band")):
        mean = _load(run_dir / "transport" / f"{stem}.json")
        if mean:
            block = (mean.get("summary") or {}).get(TIER) or {}
            n = block.get("n_panels", 0)
            ceiling = block.get("counterfactual_of_ceiling")
            out[f"{tag}_of_ceiling"] = (float(ceiling) if ceiling is not None
                                        and np.isfinite(ceiling) else float("nan"))
            out[f"{tag}_beats_both"] = (block["full_beats_both"] / n if n
                                        else float("nan"))
        dist = _load(run_dir / "transport" / f"{stem}_distribution.json")
        if dist:
            gap = ((dist.get("summary_model_own") or {}).get("pairwise")
                   or {}).get("median_gap_closed")
            out[f"{tag}_own_gap"] = float("nan") if gap is None else float(gap)
    return out



def probe_blocks(run_dir: Path) -> dict | None:
    """``validation/probe_blocks.json`` (per-block invariance probe), if any:
    its ``invariance_pass`` and every nested entry carrying ``excess`` and
    ``pass``, keyed by its path."""
    raw = _load(run_dir / "validation" / "probe_blocks.json")
    if raw is None:
        return None
    blocks: dict = {}

    def walk(obj: dict, prefix: str = "") -> None:
        for key, value in obj.items():
            if not isinstance(value, dict):
                continue
            if "excess" in value and "pass" in value:
                blocks[prefix + str(key)] = {"excess": value["excess"],
                                             "pass": value["pass"]}
            else:
                walk(value, prefix + str(key) + "/")

    walk(raw)
    return {"invariance_pass": raw.get("invariance_pass"), "blocks": blocks}


FIELDS = ("best_epoch", "recon", "nmi", "probe_delta_ce", "probe_noise_floor",
          "mirror", "cycle_z", "cycle_w", "kl_w", "niche_excess", "w_mirror",
          "d_mirror", "dev_nmi", "dev_cycle", "dev_var_share", "gain",
          "dev_recon_check", "readA_own_gap", "readA_of_ceiling",
          "readA_beats_both", "readB_own_gap", "readB_of_ceiling",
          "readB_beats_both")


def summarise(runs: list[dict]) -> dict:
    out = {"n_seeds": len(runs), "run_names": [r["run"] for r in runs],
           "n_probe_blocks": sum(r.get("probe_source") == "blocks" for r in runs),
           "n_invariance_pass": sum(bool(r.get("invariance_pass")) for r in runs)}
    block_fields = sorted({k for r in runs for k in r if k.startswith(BLOCK_PREFIX)})
    for field in FIELDS + tuple(block_fields):
        vals = [r[field] for r in runs
                if field in r and r[field] is not None and r[field] == r[field]]
        if vals:
            out[field] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                          "max": float(max(vals)), "n": len(vals),
                          "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                          else 0.0}
    return out


def reference_names(ladder: dict, ds: str) -> dict:
    """The fresh and the reused reference runs of *ds* in the seed scope.

    Reused = every link onto a ``wfix_`` fit; fresh = ``aw_ref0.1_s<k>`` that
    is not such a link (on FF, whose ``aw_ref0.1_s<k>`` are still links, there
    is no fresh pair)."""
    links = ladder["datasets"][ds]["links"]
    wfix = {n for n, t in links.items() if t.startswith("wfix_")}
    seed_of = lambda name: int(name.rsplit("_s", 1)[1])
    return {"fresh": [run_name(REFERENCE, s) for s in SEEDS
                      if run_name(REFERENCE, s) not in wfix],
            "reused": sorted((n for n in wfix if seed_of(n) in SEEDS),
                             key=seed_of)}


def _cell(ds: str, names: list, aw: float, at: str) -> dict:
    runs = [r for r in (read_run(ds, n, aw, at) for n in names) if r is not None]
    return {"alpha_w": aw, "runs": runs, "n_expected": len(names),
            "names": list(names), **summarise(runs)}


def collect(ladder: dict, at: str = "best") -> dict:
    cells = {}
    for ds in DATASETS:
        rungs = ladder["datasets"][ds]["alpha_w"]
        refs = reference_names(ladder, ds)
        for rung in RUNGS_OF[ds]:
            names = (refs["fresh"] if rung == REFERENCE
                     else [run_name(rung, s) for s in SEEDS])
            cells[(ds, rung)] = _cell(ds, names, rungs[rung], at)
        if REFERENCE in RUNGS_OF[ds] and refs["reused"]:
            cells[(ds, REUSED)] = _cell(ds, refs["reused"], rungs[REFERENCE], at)
    return cells


def reference_variant(cells: dict, ds: str, variant: str) -> dict:
    """The reference cell of *ds* under *variant*; with only one kind of
    reference fit on *ds*, that kind in every variant."""
    fresh = cells[(ds, REFERENCE)]
    reused = cells.get((ds, REUSED))
    if reused is None or not reused.get("n_expected", 1):
        return fresh
    if not fresh.get("n_expected", len(SEEDS)):
        return reused
    if variant == "fresh":
        return fresh
    if variant == "reused":
        return reused
    runs = fresh["runs"] + reused["runs"]
    return {"alpha_w": fresh["alpha_w"], "runs": runs,
            "n_expected": fresh.get("n_expected", len(SEEDS))
            + reused.get("n_expected", len(SEEDS)),
            "names": fresh.get("names", []) + reused.get("names", []),
            **summarise(runs)}


# -- guards and the rule --------------------------------------------------------

def _complete(cell: dict, field: str) -> bool:
    n = cell.get("n_expected", len(SEEDS))
    return (n > 0 and cell["n_seeds"] == n and field in cell
            and cell[field]["n"] == n)


def guards(cell: dict, ref: dict) -> dict:
    """Clause -> True / False / None (None: a seed or read is still missing)."""
    runs = cell["runs"]

    def per_seed(field, test, needs_ref=True):
        if not _complete(cell, field) or (needs_ref and not _complete(ref, field)):
            return None
        return bool(all(test(r[field]) for r in runs))

    out = {}
    if _complete(cell, "recon") and _complete(ref, "recon"):
        spread = cell["recon"]["max"] - cell["recon"]["min"]
        width = ref["recon"]["max"] - ref["recon"]["min"] + 2 * ref["recon"]["sd"]
        out["recon_spread"] = bool(spread <= width)
    else:
        out["recon_spread"] = None
    out["nmi_running_max"] = (
        bool(all(r["nmi_running_max_ok"] for r in runs))
        if cell["n_seeds"] == len(SEEDS) else None)
    out["nmi_vs_reference"] = per_seed(
        "nmi", lambda v: v >= ref["nmi"]["mean"] - NMI_TOLERANCE)
    out["probe"] = (bool(all(r.get("probe_ok", r["probe_at_floor"])
                             for r in runs))
                             if cell["n_seeds"] == len(SEEDS) else None)
    out["cycle_z_band"] = per_seed(
        "cycle_z", lambda v: v >= ref["cycle_z"]["min"] - ref["cycle_z"]["sd"])
    out["cycle_w"] = per_seed("cycle_w", lambda v: v <= CYCLE_W_GUARD,
                              needs_ref=False)
    alive = [r["niche_alive"] for r in runs]
    out["niche_guard"] = (None if cell["n_seeds"] < len(SEEDS)
                          or any(a is None for a in alive) else all(alive))
    out["w_mirror"] = per_seed(
        "w_mirror", lambda v: v <= ref["w_mirror"]["mean"] + ref["w_mirror"]["sd"])
    # reported, not a clause: the level reading of guard 1
    if _complete(cell, "recon") and _complete(ref, "recon"):
        out["_recon_level_info"] = bool(
            cell["recon"]["min"] >= ref["recon"]["min"] - ref["recon"]["sd"])
    return out


def gain_clause(cell: dict, ref: dict) -> dict:
    if not (_complete(cell, "gain") and _complete(ref, "gain")):
        return {"pass": None}
    bar = ref["gain"]["max"] - ref["gain"]["min"]
    excess = cell["gain"]["mean"] - ref["gain"]["mean"]
    return {"pass": bool(excess > bar), "excess": excess, "bar": bar}


def _tri(values) -> bool | None:
    """All True -> True; any False -> False; else None."""
    values = list(values)
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None


def decide(cells: dict, variant: str = "fresh") -> dict:
    """The rule with every rung measured against *variant*'s reference."""
    refs = {ds: reference_variant(cells, ds, variant) for ds in DATASETS}
    per_rung = {}
    for m in MULTIPLIERS:
        rung = f"m{m}"
        entry = {"guards": {}, "gain": {}}
        for ds in DATASETS:
            if rung not in RUNGS_OF[ds]:           # outside the reduced scope
                entry["guards"][ds] = {**{k: None for k in GUARDS},
                                       "_not_run": True}
                entry["gain"][ds] = {"pass": None, "not_run": True}
                continue
            g = guards(cells[(ds, rung)], refs[ds])
            entry["guards"][ds] = g
            entry["gain"][ds] = gain_clause(cells[(ds, rung)], refs[ds])
        guards_ok = _tri(v for ds in DATASETS
                         for k, v in entry["guards"][ds].items()
                         if not k.startswith("_"))
        gain_ok = _tri(entry["gain"][ds]["pass"] if ds in DATASETS else None
                       for ds in PRIMARY)
        entry["guards_every_dataset"] = guards_ok
        entry["gain_on_primary"] = gain_ok
        entry["status"] = ("pass" if guards_ok is True and gain_ok is True
                           else "fail" if False in (guards_ok, gain_ok)
                           else "undecided")
        per_rung[rung] = entry
    adopted, undecided_before = None, None
    for m in MULTIPLIERS:              # the smallest m that passes
        status = per_rung[f"m{m}"]["status"]
        if status == "undecided":
            undecided_before = m
            break
        if status == "pass":
            adopted = m
            break
    def done(c):
        return (all(_complete(c, f) for f in ("gain", "w_mirror"))
                and all(x["niche_alive"] is not None for x in c["runs"]))

    complete = (all(done(cells[(ds, r)]) for ds in DATASETS
                    for r in RUNGS_OF[ds] if r != REFERENCE)
                and all(done(refs[ds]) for ds in DATASETS))

    def first(test):
        """The smallest m for which *test* is True, if no smaller m is open."""
        for m in MULTIPLIERS:
            value = test(per_rung[f"m{m}"])
            if value is None:
                return None
            if value:
                return m
        return None

    def with_neighbour(m):
        if m is None:
            return []
        above = [k for k in MULTIPLIERS if k > m]
        return [m] + above[:1]

    # the stage-2 hand-off: the smallest rung passing every guard on the
    # datasets in scope, plus its neighbour above; and the same with GSE's
    # gain clause required as well (reported, the coordinator picks)
    by_guards = first(lambda e: e["guards_every_dataset"])
    by_guards_gain = first(lambda e: _tri(
        [e["guards_every_dataset"], e["gain"].get(GS, {}).get("pass")]))
    candidates = {"guards_only": with_neighbour(by_guards),
                  "guards_and_gse_gain": with_neighbour(by_guards_gain)}
    not_run = [ds for ds in DATASETS if undecided_before is not None and
               entry_not_run(per_rung[f"m{undecided_before}"], ds)]
    if undecided_before is not None and not_run:
        verdict = (f"undecided: m = {undecided_before} was not run on "
                   + ", ".join(SHORT[d] for d in not_run)
                   + " (reduced scope)")
    elif undecided_before is not None and FF not in DATASETS:
        verdict = (f"undecided: m = {undecided_before} needs FF (the gain "
                   "clause's second primary dataset; stage 2)")
    elif undecided_before is not None:
        verdict = f"undecided: m = {undecided_before} still incomplete"
    elif adopted is not None:
        verdict = f"adopt m = {adopted} (alpha_w = {adopted} / l-bar)"
    else:
        verdict = ("alpha_w stays at 0.1: no rung passes; w is the context "
                   "regression m_psi and the per-cell channel is closed by "
                   "design (R19's reframing)")
    return {"per_rung": per_rung, "adopted_m": adopted,
            "undecided": undecided_before is not None, "complete": complete,
            "verdict": verdict, "stage2_candidates": candidates,
            "reference": variant,
            "reference_runs": {ds: [r.get("target", r["run"]) for r in refs[ds]["runs"]]
                               for ds in DATASETS},
            "scope": {"datasets": list(DATASETS), "seeds": list(SEEDS),
                      "rungs": {ds: list(RUNGS_OF[ds]) for ds in DATASETS}}}


def entry_not_run(entry: dict, ds: str) -> bool:
    return bool(entry["guards"].get(ds, {}).get("_not_run"))


# -- rendering ------------------------------------------------------------------

def _fmt(cell: dict, field: str, digits: int = 4) -> str:
    if field not in cell:
        return "--"
    e = cell[field]
    text = f"{e['mean']:.{digits}g} [{e['min']:.{digits}g}, {e['max']:.{digits}g}]"
    expected = cell.get("n_expected", cell.get("n_seeds", e["n"]))
    return text + (f" (n={e['n']}/{expected})" if e["n"] < expected else "")


def _mark(v) -> str:
    return "--" if v is None else ("pass" if v else "**FAIL**")


COLUMNS = (("best_epoch", "best epoch"), ("recon", "recon"), ("nmi", "NMI"),
           ("probe_delta_ce", "legacy probe dCE (pooled)"),
           ("probe_noise_floor", "legacy probe floor"),
           ("mirror", "mirror"), ("cycle_z", "cycle_z"), ("cycle_w", "cycle_w"),
           ("kl_w", "KL_w (sum)"), ("niche_excess", "I(niche;w) excess"),
           ("w_mirror", "w-mirror"), ("d_mirror", "d-mirror"),
           ("dev_nmi", "deviation NMI"), ("dev_cycle", "deviation cycle R2"),
           ("dev_var_share", "d var share"), ("gain", "held-out gain"),
           ("readA_own_gap", "Read A own-target gap closed"),
           ("readA_of_ceiling", "Read A mean: fraction of ceiling"),
           ("readA_beats_both", "Read A mean: beats-both"),
           ("readB_own_gap", "Read B (tumour band) own gap"),
           ("readB_of_ceiling", "Read B mean: fraction of ceiling"),
           ("readB_beats_both", "Read B mean: beats-both"))
#: the "w quality" block beside the guards (coordinator, 2026-09-24 evening):
#: informational, not a pre-registered clause
W_QUALITY = (("gain", "held-out gain"), ("readA_own_gap", "Read A own gap closed"),
             ("niche_excess", "I(niche;w) excess"))


def _failures(entry: dict) -> str:
    """The clauses a rung fails, per dataset, for the side-by-side summary."""
    out = []
    for ds in DATASETS:
        g, gc = entry["guards"][ds], entry["gain"][ds]
        if g.get("_not_run"):
            continue
        bad = [k for k in GUARDS if g.get(k) is False]
        if ds in PRIMARY and gc.get("pass") is False:
            bad.append("gain")
        if bad:
            out.append(f"{SHORT[ds]} " + ", ".join(bad))
    return "; ".join(out)


def guard_section(ruling: dict) -> list:
    """The guard table, the rule and the verdict under one reference."""
    L = ["| rung | dataset | " + " | ".join(GUARDS)
         + " | (info) recon level | gain - ref gain | bar (ref range) | gain clause |",
         "|" + "---|" * (len(GUARDS) + 6)]
    for m in MULTIPLIERS:
        entry = ruling["per_rung"][f"m{m}"]
        for ds in DATASETS:
            g, gc = entry["guards"][ds], entry["gain"][ds]
            primary = ds in PRIMARY
            if g.get("_not_run"):
                L.append(f"| m{m} | {SHORT[ds]} | not run (reduced scope)"
                         + " |" * (len(GUARDS) + 4))
                continue
            L.append(f"| m{m} | {SHORT[ds]} | "
                     + " | ".join(_mark(g[k]) for k in GUARDS)
                     + f" | {_mark(g.get('_recon_level_info'))} | "
                     + (f"{gc['excess']:+.5f} | {gc['bar']:.5f} | "
                        if gc.get("pass") is not None else "-- | -- | ")
                     + ((_mark(gc["pass"]) if primary else "(not a clause)")
                        + " |"))
    L += [""]
    for m in MULTIPLIERS:
        e = ruling["per_rung"][f"m{m}"]
        L.append(f"- **m{m}**: {e['status']} (guards on every dataset: "
                 f"{_mark(e['guards_every_dataset'])}; gain on GSE and FF: "
                 f"{_mark(e['gain_on_primary'])})")
    cand = ruling["stage2_candidates"]
    show = lambda ms: ", ".join(f"m{m}" for m in ms) if ms else "none (yet)"
    L += ["", "Stage-2 candidates (smallest rung passing every guard on the "
          "datasets in scope, plus its neighbour above): "
          f"**{show(cand['guards_only'])}**; with GSE's gain clause also "
          f"required: {show(cand['guards_and_gse_gain'])}.",
          "", f"**Verdict: {ruling['verdict']}**", ""]
    return L


def dataset_rows(cells: dict, ds: str) -> list:
    """(label, cell) per row of a dataset: every distinct reference variant
    first, then the rungs."""
    rows = []
    for rung in RUNGS_OF[ds]:
        if rung != REFERENCE:
            rows.append((rung, cells[(ds, rung)]))
            continue
        seen = []
        for v in VARIANTS:
            ref = reference_variant(cells, ds, v)
            if any(ref is c or ref.get("names") == c.get("names") for c in seen):
                continue
            seen.append(ref)
            kind = ("fresh" if ref is cells[(ds, REFERENCE)] else
                    "reused" if ref is cells.get((ds, REUSED)) else "pooled")
            rows.append((f"ref0.1 {kind} ({len(ref.get('names', []))})", ref))
    return rows


def w_quality_block(cells: dict) -> list:
    L = ["## w quality (beside the guards; informational, not a pre-registered "
         "clause)", "",
         "Does lowering alpha_w buy a better context channel? The held-out "
         "gain of the deviation read, Read A's own-target gap closed (k-means "
         "niches, distribution read), and the I(niche;w) excess of the guard.",
         "",
         "| dataset | rung | alpha_w | " + " | ".join(c[1] for c in W_QUALITY) + " |",
         "|" + "---|" * (3 + len(W_QUALITY))]
    for ds in DATASETS:
        for label, cell in dataset_rows(cells, ds):
            L.append(f"| {SHORT[ds]} | {label} | {cell['alpha_w']:g} | "
                     + " | ".join(_fmt(cell, c[0]) for c in W_QUALITY) + " |")
    return L + [""]


def render(cells: dict, rulings: dict, ladder: dict, at: str) -> str:
    L = ["# The alpha_w multiplier ladder (R19, pre-registered 2026-09-24 13:50)"
         + (f" -- {HEADER}" if at == "best" else " -- last-epoch reads"), "",
         "Cells are seed mean [min, max] over seeds "
         + " ".join(str(x) for x in SEEDS) + ". alpha_w = m / l-bar; "
         "`ref0.1` is the current alpha_w = 0.1. In-trainer reads are the "
         "history row at the accepted epoch; I(niche;w) (degeneracy.json) and "
         "the deviation read (w_deviation.json) load best.pt.", "",
         "## l-bar", "",
         f"Convention: **l-bar = 1 / (2 alpha_z)** (`{ladder['convention']}`). "
         "The pre-registration's definition (mean total count of the training "
         "cells) does not equal 1/(2 alpha_z) on the data; both are listed, "
         "and the ladder the training-cell mean would give is recorded in "
         "LADDER.json.", "",
         "**Scope** (author, 2026-09-24, before any fit counted; departs from "
         "the pre-registered 4 datasets x 3 seeds): "
         + "; ".join(f"{SHORT[ds]}: {', '.join(RUNGS_OF[ds])}" for ds in DATASETS)
         + f"; seeds {', '.join(str(x) for x in SEEDS)}. Lung dropped; FF "
         "(stage 2) runs the reference plus the stage-1 candidates only.", "",
         "| dataset | alpha_z | l-bar = 1/(2 alpha_z) | training-cell mean (seed 0) "
         "| all-cell median | reference m | " + " | ".join(
             f"alpha_w m{m}" for m in MULTIPLIERS) + " |",
         "|" + "---|" * (6 + len(MULTIPLIERS))]
    for ds in DATASETS:
        e = ladder["datasets"][ds]
        L.append(f"| {SHORT[ds]} | {e['alpha_z']:g} | {e['lbar']:.1f} | "
                 f"{e['lbar_train_mean']:.1f} | "
                 f"{e['count_scale']['all_median']:.0f} | "
                 f"{e['reference_m']:.1f} | " + " | ".join(
                     f"{e['alpha_w'][f'm{m}']:g}" for m in MULTIPLIERS) + " |")
    links = {ds: ladder["datasets"][ds]["links"] for ds in DATASETS}
    in_scope = {run_name(r, x) for r in RUNGS for x in SEEDS}
    shared = [f"{SHORT[ds]} {a} -> {b}" for ds in DATASETS
              for a, b in links[ds].items()
              if (a in in_scope and a.split("_s")[0][3:] in RUNGS_OF[ds])
              or a in reference_names(ladder, ds)["reused"]]
    L += ["", "Runs read through a link (same configuration, not refitted): "
          + (", ".join(shared) if shared else "none") + ".", ""]
    for ds in DATASETS:
        rows = dataset_rows(cells, ds)
        block_cols = sorted({k for _, cell in rows for k in cell
                             if k.startswith(BLOCK_PREFIX)})
        L += [f"## {SHORT[ds]} ({ds})", "",
              "| rung | alpha_w | n | " + " | ".join(c[1] for c in COLUMNS)
              + " | probe guard source | invariance_pass | "
              + "".join(f"{k[len(BLOCK_PREFIX):]} excess | " for k in block_cols),
              "|" + "---|" * (5 + len(COLUMNS) + len(block_cols))]
        for rung, cell in rows:
            nb = cell["n_probe_blocks"]
            source = ("legacy" if nb == 0 else f"blocks {nb}/{cell['n_seeds']}"
                      + ("" if nb == cell["n_seeds"] else " (rest legacy)"))
            L.append(f"| {rung} | {cell['alpha_w']:g} | {cell['n_seeds']} | "
                     + " | ".join(_fmt(cell, c[0]) for c in COLUMNS)
                     + f" | {source} | "
                     + (f"{cell['n_invariance_pass']}/{nb}" if nb else "--")
                     + " | " + "".join(f"{_fmt(cell, k)} | " for k in block_cols))
        bad = [r["run"] for _, cell in rows for r in cell["runs"]
               if not r["alpha_w_ok"] or not r["at_best"]
               or abs(r.get("dev_recon_check", 0.0)) > 1e-4]
        dead = [r["run"] for _, cell in rows for r in cell["runs"]
                if r["dead_w_channel_trainer"]]
        L += ["", "Checks: " + ("alpha_w, at-best row and deviation-read recon "
                                "agree on every run" if not bad else
                                "**disagreement on " + ", ".join(bad) + "**")
              + "; trainer dead-channel flag: "
              + (", ".join(dead) if dead else "none") + ".", ""]
    L += ["## The rule under each reference envelope", "",
          "| rung | " + " | ".join(VARIANT_LABEL[v] for v in VARIANTS) + " |",
          "|" + "---|" * (1 + len(VARIANTS))]
    for m in MULTIPLIERS:
        cols = []
        for v in VARIANTS:
            e = rulings[v]["per_rung"][f"m{m}"]
            why = _failures(e)
            cols.append(f"{e['status']}" + (f" ({why})" if why else ""))
        L.append(f"| m{m} | " + " | ".join(cols) + " |")
    L += ["| **verdict** | " + " | ".join(rulings[v]["verdict"] for v in VARIANTS)
          + " |", ""]
    L += w_quality_block(cells)
    for v in VARIANTS:
        r = rulings[v]
        L += [f"## Guards against the {VARIANT_LABEL[v]}", "",
              "Reference runs: " + "; ".join(
                  f"{SHORT[ds]} " + ", ".join(r["reference_runs"][ds])
                  for ds in DATASETS) + ".", ""]
        L += guard_section(r)
    complete = all(rulings[v]["complete"] for v in VARIANTS)
    L += ["Grid complete: " + ("yes" if complete else "no")
          + "; DECISION_AW.json is written only with FF in scope and every "
          "variant decided."]
    return "\n".join(L) + "\n"


def write_decision(rulings: dict, ladder: dict, path: Path) -> bool:
    """The hand-off, once every reference variant is complete and decided and
    both primary datasets are in scope. It carries every variant's verdict;
    when they disagree, ``adopted_m`` is None and the disagreement is named."""
    if (not all(ds in DATASETS for ds in PRIMARY)
            or any(not r["complete"] or r["undecided"] for r in rulings.values())):
        print(f"decision withheld ({path}): grid incomplete or undecided")
        return False
    adopted = {v: r["adopted_m"] for v, r in rulings.items()}
    agree = len(set(adopted.values())) == 1
    m = next(iter(adopted.values())) if agree else None
    payload = {
        "adopted_m": m, "variants_agree": agree,
        "verdict": (next(iter(rulings.values()))["verdict"] if agree else
                    "the reference variants disagree: " + ", ".join(
                        f"{v} -> {a}" for v, a in adopted.items())),
        # one multiplier for every section, the dropped lung included
        "alpha_w": ({ds: e["alpha_w"][f"m{m}"]
                     for ds, e in ladder["datasets"].items()} if m is not None
                    else {ds: REFERENCE_AW for ds in ladder["datasets"]}
                    if agree else None),
        "by_reference": {v: {"adopted_m": r["adopted_m"], "verdict": r["verdict"],
                             "status_by_rung": {k: e["status"] for k, e
                                                in r["per_rung"].items()}}
                         for v, r in rulings.items()},
        "lbar": {ds: e["lbar"] for ds, e in ladder["datasets"].items()},
        "lbar_convention": ladder["convention"],
        "scope": next(iter(rulings.values()))["scope"],
        "read_at": HEADER}
    _write(path, json.dumps(payload, indent=1, allow_nan=False))
    print(f"wrote {path}: {json.dumps(payload)}")
    return True


def _jsonable(cells: dict) -> list:
    return [{"dataset": ds, "rung": rung, **cell}
            for (ds, rung), cell in cells.items()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--init-ladder", action="store_true",
                        help="fix l-bar and the rungs' alpha_w (once)")
    parser.add_argument("--force", action="store_true",
                        help="with --init-ladder: overwrite LADDER.json")
    parser.add_argument("--at", choices=AT_CHOICES, default="best")
    args = parser.parse_args(argv)
    if args.init_ladder:
        init_ladder(args.force)
        return 0
    ladder = json.loads(LADDER_PATH.read_text())
    if SCOPE_PATH.exists():
        set_scope(json.loads(SCOPE_PATH.read_text()))
    cells = collect(ladder, args.at)
    rulings = {v: decide(cells, v) for v in VARIANTS}
    text = render(cells, rulings, ladder, args.at)
    print(text)
    stem = OUT_STEM if args.at == "best" else OUT_STEM.with_name(
        OUT_STEM.name + "_at_final")
    _write(stem.with_suffix(".json"), json.dumps(
        {"read_at": HEADER if args.at == "best" else "final", "ladder": ladder,
         "cells": _jsonable(cells), "decisions": rulings}, indent=1,
        default=float))
    _write(stem.with_suffix(".md"), text)
    print(f"wrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")
    if args.at == "best":
        write_decision(rulings, ladder, DECISION_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
