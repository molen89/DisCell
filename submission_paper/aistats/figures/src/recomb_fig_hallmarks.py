"""RECOMB supplement figure: hallmark labels of every response programme (fig:hallmarks).

Rows: the MSigDB hallmark sets significant (BH q <= 0.05, AUC > 0.5) on at least one programme
of any fit, ordered by the number of programmes that carry them. Columns: section, then seed,
then programme (in the fit's own variance order; its share of w's within-type variance above).
Cells: the Mann-Whitney AUC of the set's absolute loadings against the rest of the expressed
panel (0.5 = null), with a dot where q <= 0.05. All 50 sets are tested on every programme
(atlas.py's test, recomputed in full and checked against the atlas's stored top three).

Source, run-internal (programmes are identified only up to rotation):
  data/datasets/<ds>/experiments/hallmark_full_finalL.json   all 50 sets per programme of finalL_s{0,1,2}
Run from the repository root (DISCELL_VOCAB=recomb, as build.sh does). Outputs
../recomb_fig_hallmarks.pdf (+ .png) and prints the row counts.
"""
from pathlib import Path
import sys

import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402
from recomb_fig_programmes import SECTIONS, SEEDS, WIDTH, HALLMARK_NAMES, ds_root  # noqa: E402

SHORT = {**HALLMARK_NAMES,
         "INFLAMMATORY_RESPONSE": "inflammatory response",
         "ESTROGEN_RESPONSE_EARLY": "oestrogen response early",
         "ESTROGEN_RESPONSE_LATE": "oestrogen response late",
         "E2F_TARGETS": "E2F targets",
         "KRAS_SIGNALING_UP": "KRAS signalling up",
         "KRAS_SIGNALING_DN": "KRAS signalling down",
         "MYC_TARGETS_V1": "MYC targets v1",
         "G2M_CHECKPOINT": "G2/M checkpoint",
         "CHOLESTEROL_HOMEOSTASIS": "cholesterol homeostasis",
         "OXIDATIVE_PHOSPHORYLATION": "oxidative phosphorylation",
         "COAGULATION": "coagulation",
         "INTERFERON_ALPHA_RESPONSE": r"IFN$\alpha$ response",
         "INTERFERON_GAMMA_RESPONSE": r"IFN$\gamma$ response",
         "ADIPOGENESIS": "adipogenesis",
         "ALLOGRAFT_REJECTION": "allograft rejection",
         "DNA_REPAIR": "DNA repair",
         "IL2_STAT5_SIGNALING": "IL-2/STAT5 signalling"}
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "recomb_fig_hallmarks"
SPAN = 0.3                       # colour scale: AUC 0.5 +- SPAN
GAP_SEED, GAP_SECTION = 0.35, 0.9


def columns():
    """One entry per programme: (section title, seed, programme index, share, {hallmark: (auc, sig)})."""
    cols = []
    for ds, title in SECTIONS:
        full = json.loads((ds_root(ds) / "experiments" / "hallmark_full_finalL.json").read_text())
        for s, run in enumerate(SEEDS):
            for p in full[run]:
                if not p["active"] or p["variance_share"] <= 0:
                    continue
                cols.append((title, s, p["program"], p["variance_share"],
                             {h["hallmark"]: (h["auc"], h["significant"]) for h in p["hallmarks"]}))
    return cols


def draw():
    S.apply()
    cols = columns()
    count, sections = {}, {}
    for title, _, _, _, hm in cols:
        for h, (_, sig) in hm.items():
            if sig:
                count[h] = count.get(h, 0) + 1
                sections.setdefault(h, set()).add(title)
    rows = sorted(count, key=lambda h: (-count[h], -len(sections[h]), SHORT.get(h, h)))
    # x positions with gaps between seeds and sections
    xs, x, prev = [], 0.0, None
    for title, s, _, _, _ in cols:
        if prev is not None:
            x += 1 + (GAP_SECTION if title != prev[0] else GAP_SEED if s != prev[1] else 0)
        xs.append(x)
        prev = (title, s)
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(WIDTH, 0.25 + 0.115 * len(rows)))
    ax = fig.add_axes([0.235, 0.06, 0.71, 0.86])
    for j, (_, _, _, _, hm) in enumerate(cols):
        for i, h in enumerate(rows):
            y = len(rows) - 1 - i
            if h in hm:
                auc, sig = hm[h]
                ax.add_patch(Rectangle((xs[j] - 0.5, y - 0.5), 1, 1, lw=0,
                                       facecolor=cmap(0.5 + (auc - 0.5) / (2 * SPAN))))
                if sig:
                    ax.plot(xs[j], y, "o", ms=1.6, color=S.INK, mew=0)
            else:
                ax.add_patch(Rectangle((xs[j] - 0.5, y - 0.5), 1, 1, lw=0, facecolor=S.GHOST))
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{SHORT.get(h, h.lower().replace("_", " "))} ({count[h]})" for h in rows[::-1]], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    top = len(rows) - 0.4
    for j, (_, _, _, share, _) in enumerate(cols):          # variance share above each column
        ax.text(xs[j], top + 0.15, f"{100 * share:.0f}", ha="center", va="bottom",
                fontsize=4.6, color=S.INK2, rotation=90)
    for title in dict.fromkeys(c[0] for c in cols):         # section and seed labels
        idx = [j for j, c in enumerate(cols) if c[0] == title]
        ax.text((xs[idx[0]] + xs[idx[-1]]) / 2, top + 1.6, rf"\textbf{{{title}}}",
                ha="center", va="bottom", fontsize=6.5, color=S.INK, clip_on=False)
        for s in range(len(SEEDS)):
            js = [j for j in idx if cols[j][1] == s]
            ax.plot([xs[js[0]] - 0.4, xs[js[-1]] + 0.4], [-0.9, -0.9], color=S.MUTED, lw=0.5,
                    clip_on=False, solid_capstyle="butt")
            ax.text((xs[js[0]] + xs[js[-1]]) / 2, -1.1, str(s), ha="center", va="top",
                    fontsize=5.5, color=S.INK2, clip_on=False)
    ax.text(xs[0] - 0.6, -1.1, "seed", ha="right", va="top", fontsize=5.5, color=S.INK2)
    ax.text(xs[0] - 0.6, top + 0.15, r"\% of $\mathbf{w}$", ha="right", va="bottom",
            fontsize=5.5, color=S.INK2)
    # colour bar
    cax = fig.add_axes([0.958, 0.3, 0.012, 0.4])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5 - SPAN, 0.5 + SPAN))
    cb = fig.colorbar(sm, cax=cax, ticks=[0.2, 0.5, 0.8])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_visible(False)
    cax.set_title("AUC", fontsize=6, pad=3)
    S.save(fig, OUT)
    return rows, count, sections, len(cols)


if __name__ == "__main__":
    rows, count, sections, n = draw()
    print(f"{n} programmes, {len(rows)} hallmark sets significant somewhere")
    for h in rows:
        print(f"  {h}: significant on {count[h]} programmes, sections {sorted(sections[h])}")
