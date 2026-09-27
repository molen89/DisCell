#!/usr/bin/env python3
"""The projection test (S53; metrics package item 3), read off disk.

Ovarian, final configuration at 200/20 with the 30-epoch alpha_w warm-up:
``phiproj32_s{0,1,2}`` (a learned linear map Phi 384 -> 32 in c, ``--phi-proj
32``) against ``wfix_warmup30_aw0.1_s{0,1,2}`` (the full 384-d Phi, same
flags otherwise). Rows: held-out reconstruction (and the recon_modes decodes
where run), NMI, the guards (per-block probe fractions of uncontrolled and
the invariance guard, I(z;t)/H(t), within-type variance, I(niche; w) excess,
the dead-context flag), parameter count, and memory from the timing mode
(``experiments/timing/timing_{phi,phiproj32}.json``).

Usage::

    python scripts/phi_projection_table.py
    python scripts/phi_projection_table.py --full projL384_s --proj projL32_s \
        --tag _lineage --config runs/finalL_s0/config.json \
        --reference uncontrolledL_s{0,1}      # the lineage re-read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from discell import paths

DATASET = "xenium_prime_ovarian_cancer_ffpe"
ARMS = {"full Φ (384)": "wfix_warmup30_aw0.1_s", "projected Φ (32)": "phiproj32_s"}
SEEDS = (0, 1, 2)

ROWS = (
    ("held-out reconstruction (nats/count)", "recon", "{:.4f}"),
    ("recon, intrinsic-only decode", "recon_intrinsic", "{:.4f}"),
    ("recon, context-only decode", "recon_context", "{:.4f}"),
    ("NMI vs type labels", "nmi", "{:.3f}"),
    ("best epoch", "epoch", "{:.0f}"),
    ("probe ÷ uncontrolled, composition / ridge", "ridge_comp", "{:.2f}"),
    ("probe ÷ uncontrolled, image / ridge", "ridge_img", "{:.2f}"),
    ("probe ÷ uncontrolled, composition / MLP", "mlp_comp", "{:.2f}"),
    ("probe ÷ uncontrolled, image / MLP", "mlp_img", "{:.2f}"),
    ("invariance guard passes (fraction of seeds)", "invariance_pass", "{:.2f}"),
    ("I(z;t)/H(t)", "mi_ratio", "{:.3f}"),
    ("within-type variance fraction of z", "within_var", "{:.3f}"),
    ("I(niche; w) excess (nats)", "w_mi_excess", "{:+.4f}"),
    ("dead context channel (fraction of seeds)", "dead_w", "{:.2f}"),
    ("parameters", "n_params", "{:,.0f}"),
)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def record(run: str) -> dict | None:
    run_dir = paths.dataset(DATASET).root / "runs" / run
    metrics = _load(run_dir / "metrics.json")
    if metrics is None:
        return None
    out = {"run": run, "recon": metrics["best"]["recon_val"],
           "nmi": metrics["best"]["nmi"], "epoch": metrics["best"]["epoch"],
           "dead_w": float(bool(metrics.get("dead_w_channel")))}
    blocks = _load(run_dir / "validation" / "probe_blocks.json") or {}
    for family in ("ridge", "mlp"):
        for block in ("comp", "img"):
            out[f"{family}_{block}"] = ((blocks.get(family) or {}).get(block)
                                        or {}).get("fraction_of_uncontrolled")
    if blocks.get("invariance_pass") is not None:
        out["invariance_pass"] = float(blocks["invariance_pass"])
    deg = _load(run_dir / "degeneracy.json") or {}
    out["mi_ratio"] = (deg.get("degeneracy") or {}).get("mi_ratio")
    out["within_var"] = (deg.get("degeneracy") or {}).get("within_var_fraction")
    out["w_mi_excess"] = (deg.get("w_channel") or {}).get("w_niche_mi_excess")
    modes = (_load(run_dir / "recon_modes.json") or {}).get("modes") or {}
    out["recon_intrinsic"] = (modes.get("intrinsic") or {}).get("recon")
    out["recon_context"] = (modes.get("context") or {}).get("recon")
    try:
        import torch

        state = torch.load(run_dir / "best.pt", map_location="cpu",
                           weights_only=False)["model"]
        out["n_params"] = float(sum(v.numel() for k, v in state.items()
                                    if "kappa" not in k))
    except Exception:
        out["n_params"] = None
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full", default=ARMS["full Φ (384)"],
                        help="run-name prefix of the full-Φ arm (+ seed)")
    parser.add_argument("--proj", default=ARMS["projected Φ (32)"],
                        help="run-name prefix of the projected arm (+ seed)")
    parser.add_argument("--tag", default="",
                        help="phi_projection<TAG>.md/.json; timing from "
                             "experiments/timing<TAG>/")
    parser.add_argument("--config", default="runs/final_s1/config.json",
                        help="the configuration the arms were fitted at (text)")
    parser.add_argument("--reference", default="uncontrolled500_s{0,1}",
                        help="the uncontrolled fits the probe was graded against (text)")
    args = parser.parse_args(argv)
    arms = {"full Φ (384)": args.full, "projected Φ (32)": args.proj}
    records = {arm: [r for r in (record(f"{prefix}{s}") for s in SEEDS) if r]
               for arm, prefix in arms.items()}
    head = ["read"] + [f"{arm}: mean [min, max] (n)" for arm in arms]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, fmt in ROWS:
        cells = [label]
        for arm in arms:
            vals = [r[key] for r in records[arm]
                    if r.get(key) is not None and np.isfinite(r[key])]
            cells.append("" if not vals else
                         f"{fmt.format(np.mean(vals))} [{fmt.format(min(vals))}, "
                         f"{fmt.format(max(vals))}] ({len(vals)})")
        lines.append("| " + " | ".join(cells) + " |")
    timing_dir = paths.dataset(DATASET).root / "experiments" / f"timing{args.tag}"
    mem = []
    for arm, name in (("phi", "full Φ (384)"), ("phiproj32", "projected Φ (32)")):
        t = _load(timing_dir / f"timing_{arm}.json")
        if t:
            mem.append(f"{name}: {t['s_per_epoch']:.2f} s/epoch, peak "
                       f"{t['peak_train_mib']:.0f} MiB training / "
                       f"{t['peak_eval_mib']:.0f} MiB evaluation "
                       f"(resident {t['resident_mib']:.0f} MiB)")
    per_seed = ["", "Per seed:", ""] + [
        f"* `{r['run']}`: recon {r['recon']:.4f}, NMI {r['nmi']:.3f}, epoch "
        f"{r['epoch']}, guard {r.get('invariance_pass')}"
        for arm in arms for r in records[arm]]
    text = ("### The projection test (S53): full Φ vs a learned 32-d projection, "
            "ovarian, 200/20\n\n" + "\n".join(lines) + "\n\n"
            "Arms differ only in `--phi-proj 32` (a learned nn.Linear(384→32) "
            "applied to Φ before it enters c); both at the final configuration "
            f"(`{args.config}`) with 200 epochs, patience 20, "
            "`--w-warmup-epochs 30`. Guards from `validation/probe_blocks.json` "
            f"(probe_regrade, graded against {args.reference}) and "
            "`degeneracy.json`. Memory from the timing mode (`scripts/"
            "timing_mode.py`, final configuration, 20 epochs): "
            + ("; ".join(mem) if mem else "not yet measured") + ".\n"
            + "\n".join(per_seed) + "\n")
    out = paths.dataset(DATASET).root / "experiments" / f"phi_projection{args.tag}.md"
    out.write_text(text)
    (out.with_suffix(".json")).write_text(json.dumps(records, indent=2,
                                                     default=float))
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
