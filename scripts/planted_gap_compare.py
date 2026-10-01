#!/usr/bin/env python3
"""tab:planted-gap, pre-final (2026-09-23) against final configuration.

Reads data/datasets/synthetic_smoke/experiments/planted_posterior.json (the
run behind tab:planted-gap) and planted_posterior_final.json (the rerun at
the final configuration) and writes planted_posterior_final_vs_prefinal.md
beside them: the table's rows old / new, the claims of app:planted's
"The result" paragraph re-checked, and the rows the new table needs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXP = Path("data/datasets/synthetic_smoke/experiments")
DEPTHS = (100, 300, 1000)
MISMATCH = (-0.1, -0.05, 0.0, 0.05, 0.1)
ROWS = (("discell_warped", "DISCELL encoder", 2),
        ("mlp_oracle", "Perceptron, the encoder's inputs", 2),
        ("mlp_oracle_rhobar", "Perceptron, the encoder's inputs and rho_bar", 2),
        ("linear_oracle", "Linear regression, the encoder's inputs", 2),
        ("prior_mean", "Type mean", 1))


def cell(table, m, d, arm, field="arms"):
    return table[f"{m:+.2f}|{d}"][field][arm]


def rng(e, n=3):
    return f"{e['mean']:.{n}f} ({e['min']:.{n}f}-{e['max']:.{n}f})"


def main() -> int:
    old = json.loads((EXP / "planted_posterior.json").read_text())
    new = json.loads((EXP / "planted_posterior_final.json").read_text())
    to, tn = old["table"], new["table"]
    L = ["# tab:planted-gap: pre-final protocol against the final configuration", "",
         "Old: `planted_posterior.json` (2026-09-23; `applications.planted.fit_synthetic`, "
         f"{old['design']['epochs']} fixed epochs, closed-form penalty alpha_a 0.02, "
         "alpha_z 0.007, alpha_w 0.1, no warm-up, all tiles trained). "
         "New: `planted_posterior_final.json` (the production Trainer at the TrainConfig "
         "defaults: adversary alpha_a 0.3, composition weight 3, alpha_w 0.1 with the "
         "30-epoch warm-up, 500 epochs / patience 40 on 15 % held-out tiles, best "
         "checkpoint, alpha_z = 1/2 / mean count; widths "
         f"{new['design'].get('widths')}, tiles {new['design'].get('tile_cells')}). "
         "Same worlds (world seed = seed), depths, mismatches and seeds; the exact "
         "posterior and the four references are the same computation, so their rows "
         "must agree to rounding.", "",
         "Gap = ||estimate - E[z|x]|| / ||sd(z|x)||, mean over cells, 3-seed mean (min-max), "
         "true leak fraction (mismatch 0).", "",
         "| row | depth | old | new |", "|---|---|---|---|"]
    for arm, name, _ in ROWS:
        for d in DEPTHS:
            L.append(f"| {name} | {d} | {rng(cell(to, 0.0, d, arm))} | "
                     f"{rng(cell(tn, 0.0, d, arm))} |")
    ref_diff = max(abs(cell(to, 0.0, d, a)["mean"] - cell(tn, 0.0, d, a)["mean"])
                   for a, _, _ in ROWS[1:] for d in DEPTHS)
    L += ["", f"Largest old/new difference over the four reference rows: {ref_diff:.2e} "
          "(they do not depend on the fit).", ""]

    L += ["## The gap in the world's own z units (mismatch 0), and the posterior sd", "",
          "| depth | old abs | new abs | posterior sd |", "|---|---|---|---|"]
    for d in DEPTHS:
        L.append(f"| {d} | {rng(cell(to, 0.0, d, 'discell_warped', 'abs'), 4)} | "
                 f"{rng(cell(tn, 0.0, d, 'discell_warped', 'abs'), 4)} | "
                 f"{tn[f'+0.00|{d}']['posterior_sd']:.4f} |")

    L += ["", "## The leak-fraction mismatch (DISCELL encoder, gauge-free reading)", "",
          "| mismatch | " + " | ".join(f"old d{d} | new d{d}" for d in DEPTHS) + " |",
          "|---|" + "---|---|" * len(DEPTHS)]
    for m in MISMATCH:
        L.append(f"| {m:+.2f} | " + " | ".join(
            f"{cell(to, m, d, 'discell_warped')['mean']:.3f} | "
            f"{cell(tn, m, d, 'discell_warped')['mean']:.3f}" for d in DEPTHS) + " |")
    L += ["", "| depth | verdict | old | new |", "|---|---|---|---|"]
    for d in DEPTHS:
        vo, vn = old["verdict"][str(d)], new["verdict"][str(d)]
        for k in ("min_is_at_matched", "monotone_each_side", "max_wing_over_matched"):
            f = (lambda x: f"{x:.3f}") if k == "max_wing_over_matched" else str
            L.append(f"| {d} | {k} | {f(vo[k])} | {f(vn[k])} |")

    # the claims of app:planted "The result", re-checked on the new run
    enc = {d: cell(tn, 0.0, d, "discell_warped")["mean"] for d in DEPTHS}
    mlp = {d: cell(tn, 0.0, d, "mlp_oracle")["mean"] for d in DEPTHS}
    rho = {d: cell(tn, 0.0, d, "mlp_oracle_rhobar")["mean"] for d in DEPTHS}
    absd = [cell(tn, 0.0, d, "discell_warped", "abs")["mean"] for d in DEPTHS]
    wing = max(new["verdict"][str(d)]["max_wing_over_matched"] for d in DEPTHS)
    L += ["", "## app:planted \"The result\", claim by claim, on the new run", "",
          f"* encoder below the perceptron on the same inputs at every depth: "
          f"{all(enc[d] < mlp[d] for d in DEPTHS)} "
          f"({', '.join(f'{enc[d]:.3f} vs {mlp[d]:.3f}' for d in DEPTHS)})",
          f"* per-seed: encoder below that perceptron in "
          f"{sum(1 for r in new['runs'] if r['mismatch'] == 0.0 and r['gaps']['discell_warped']['mean'] < r['gaps']['mlp_oracle']['mean'])}"
          f"/{sum(1 for r in new['runs'] if r['mismatch'] == 0.0)} (depth, seed) cells",
          f"* encoder minus the rho_bar perceptron: "
          f"{', '.join(f'd{d} {enc[d] - rho[d]:+.3f}' for d in DEPTHS)}",
          f"* gap in z units across depths: {', '.join(f'{a:.4f}' for a in absd)} "
          f"(max/min {max(absd) / min(absd):.2f})",
          f"* worst wing over matched (\"at most a tenth\"): {wing:.3f}",
          f"* smallest at the true leak fraction: "
          f"{', '.join(f'd{d} ' + str(new['verdict'][str(d)]['min_is_at_matched']) for d in DEPTHS)}"]

    fits = [r["fit"] for r in new["runs"] if "fit" in r]
    if fits:
        be = [f["best_epoch"] for f in fits]
        L += ["", "## The new fits", "",
              f"* {len(fits)} fits; best epoch {min(be)}-{max(be)} (median {int(np.median(be))}); "
              f"last epoch {min(f['last_epoch'] for f in fits)}-{max(f['last_epoch'] for f in fits)}; "
              f"dead w channel: {sum(bool(f['dead_w_channel']) for f in fits)}",
              "* alpha_z per depth: " + ", ".join(
                  f"d{d} " + "/".join(sorted({f"{r['fit']['alpha_z']:.5f}" for r in new['runs']
                                               if r['depth'] == d}))
                  for d in DEPTHS)]

    L += ["", "## tab:planted-gap at the final configuration (for the writer; .tex not edited)", "",
          "| Estimate of the posterior mean | 100 | 300 | 1,000 |", "|---|---|---|---|"]
    for arm, name, n in ROWS:
        L.append(f"| {name} | " + " | ".join(
            f"{cell(tn, 0.0, d, arm)['mean']:.{n}f}" for d in DEPTHS) + " |")
    out = EXP / "planted_posterior_final_vs_prefinal.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
