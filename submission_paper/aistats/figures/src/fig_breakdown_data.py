"""Figure fig:breakdown-data: the claimed contrasts across the leakage sweep, per section.

The measured counterpart of the fig:breakdown schematic. One row per claimed contrast,
one column per section. In each panel, at every kappa of the sweep: the pooled estimate
over the three seeds and its interval (spatial block bootstrap over tiles, two-sided,
Bonferroni-adjusted over the m contrasts of that section), the zero line, and the
breakdown point kappa* where there is one. Filled: the contrast holds at that grid point
(every seed has its kappa = 0 sign and the interval excludes 0); hollow: from kappa* on.
A contrast outside a section's set leaves its panel empty, except where the section's
record marks it not applicable (the tumour axis on the TMA sections, which have no tumour
cells): that panel states why. A member of the set not yet computed is marked as such. Rows share their vertical scale across sections.

Sources: scripts/logs/breakdown_2026-09-29/breakdown_all*.json (not the trajectory
table), then each section's data/datasets/<section>/experiments/breakdown.json for any
member the combined file does not hold yet. Status and kappa* are read as stored, not
recomputed. Members and families are those of tab:breakdown (scripts/paper_tables.py:
BD_MEMBERS, BD_SECTIONS, _bd_family). Rerun when the breakdown queue writes the rest.

Outputs ../fig_breakdown_data.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_breakdown_data"
SHORT = {"cycle_asym_q90": r"cycle asymmetry," "\n" r"$R^2(\mathbf{z}) - R^2(\mathbf{w})$",
         "w_niche_mi_excess": r"$I(\mathrm{niche};\mathbf{w})$" "\n" r"$-$ its floor",
         "readA_minus_typemean": "Read A $-$ type-mean\nreference",
         "readB_twin_margin": "twin margin",
         "transport_cf_minus_leak": "transport $-$\nleakage part",
         "signalling_response_lr_vs_other": "signalling share,\nLR $-$ other",
         "axis_tau_true_minus_false": "tumour axis,\ntrue $-$ false"}
assert set(SHORT) == {k for k, _ in PT.BD_MEMBERS}

# ---- read: the combined table, then the per-section files for anything it lacks
sections, used = {}, []
for p in sorted(PT.BREAKDOWN.glob("breakdown_all*.json")):
    if "trajectory" in p.name:
        continue
    used.append(p)
    for sec, s in json.loads(p.read_text()).items():
        sections.setdefault(sec, s)
for sec, ds in PT.BD_SECTIONS:
    q = PT.DATA / ds / "experiments" / "breakdown.json"
    if q.exists():
        s = json.loads(q.read_text())
        base = sections.setdefault(sec, s)
        for k, m in s["members"].items():
            if k not in base["members"]:
                base["members"][k] = m
                used.append(q)
print("read:", *sorted({str(u.relative_to(PT.REPO)) for u in used}), sep="\n  ")

rows = [k for k, _ in PT.BD_MEMBERS]
fam = {sec: PT._bd_family(sec, sections[sec]["m_s"], sections[sec]) for sec, _ in PT.BD_SECTIONS}
has_data = {k: any(k in sections[sec]["members"] for sec, _ in PT.BD_SECTIONS) for k in rows}
H = [1.0 if has_data[k] else 0.38 for k in rows]

S.apply()
fig_h = 0.42 * sum(H) + 0.8
fig = plt.figure(figsize=(S.TEXTWIDTH, fig_h))
gs = fig.add_gridspec(len(rows), len(PT.BD_SECTIONS), height_ratios=H, left=0.215, right=0.99,
                      top=1 - 0.36 / fig_h, bottom=0.52 / fig_h, hspace=0.35, wspace=0.42)
missing = []
zero_named = False
for r, key in enumerate(rows):
    row_axes = []
    for c, (sec, ds) in enumerate(PT.BD_SECTIONS):
        s = sections[sec]
        na = (s.get("not_applicable") or {}).get(key)
        if key not in fam[sec]:
            if na:                         # empty by design: say why, in the panel
                ax = fig.add_subplot(gs[r, c])
                ax.text(0.5, 0.5, "not applicable:\n" + na["short"], transform=ax.transAxes,
                        ha="center", va="center", fontsize=5.8, color=S.MUTED, style="italic",
                        linespacing=1.15)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            continue
        ax = fig.add_subplot(gs[r, c])
        row_axes.append(ax)
        grid = np.array(s["grid"])
        e = s["members"].get(key)
        if e is None:
            missing.append(f"{sec}/{key}")
            ax.text(0.5, 0.5, "being computed", transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, color=S.MUTED, style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.axhline(0.5, xmin=0.02, xmax=0.98, color=S.GHOST, lw=4, zorder=0)
            continue
        t = e["trajectory"]
        nan = {"estimate": np.nan, "ci": [np.nan, np.nan]}   # a grid point without draws
        m = np.array([(t[f"{g:g}"] or nan)["estimate"] for g in grid])
        lo = np.array([(t[f"{g:g}"] or nan)["ci"][0] for g in grid])
        hi = np.array([(t[f"{g:g}"] or nan)["ci"][1] for g in grid])
        ks = e.get("kappa_star")
        if e["status"] == "no finding":
            ok = np.zeros(grid.size, bool)
        elif ks is None:
            ok = np.ones(grid.size, bool)
        else:
            ok = grid < ks
        ax.axhline(0, color=S.MUTED, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        if not zero_named:                     # name the zero line once, in the first panel
            ax.text(0.98, 0, "no contrast", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=5.6, color=S.INK2, style="italic")
            zero_named = True
        ax.vlines(grid, lo, hi, color=S.INK2, lw=0.9, zorder=2)
        ax.scatter(grid[ok], m[ok], s=7, color=S.INK, zorder=4, lw=0)
        ax.scatter(grid[~ok], m[~ok], s=7, facecolor="white", edgecolor=S.INK2, lw=0.8, zorder=4)
        if ks is not None:
            ax.axvline(ks, color=S.BLUE, lw=1.0, ls=(0, (1.2, 1.4)), zorder=3)
            # just above the panel, so it covers no point and stays inside its column
            ax.text(min(ks, 0.36), 1.0, rf"$\kappa^\star = {ks:g}$", transform=ax.get_xaxis_transform(),
                    fontsize=5.8, color=S.INK, ha="center", va="bottom", clip_on=False)
        span_lo, span_hi = min(0.0, np.nanmin(lo)), max(0.0, np.nanmax(hi))
        pad = 0.15 * (span_hi - span_lo)
        ax.set_ylim(span_lo - pad, span_hi + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2, min_n_ticks=2))
        ax.set_xlim(-0.03, 0.43)
        ax.set_xticks(grid)
        ax.set_xticklabels(["0", "", "0.1", "0.2", "", "0.4"] if r == len(rows) - 1 or
                           not any(k in fam[sec] for k in rows[r + 1:]) else [])
        ax.tick_params(labelsize=5.8, length=2, pad=1)
    if not row_axes:
        continue
    if r == 0:                                       # once: every contrast is signed so
        S.ybetter(row_axes[0], up=True, fontsize=5.8)  # that above zero is the claimed side
    y_mid = row_axes[0].get_position().y0 + row_axes[0].get_position().height / 2
    fig.text(0.008, y_mid, SHORT[key], ha="left", va="center", fontsize=6.5, color=S.INK,
             linespacing=1.15)
# column heads
for c, (sec, ds) in enumerate(PT.BD_SECTIONS):
    x0 = gs[0, c].get_position(fig).x0
    x1 = gs[0, c].get_position(fig).x1
    fig.text((x0 + x1) / 2, 1 - 0.1 / fig_h,
             rf"{PT.SHORT[ds]}" "\n" rf"($m = {sections[sec]['m_s']}$)", ha="center", va="top",
             fontsize=7, color=S.INK, linespacing=1.2)
fig.text((gs[0, 0].get_position(fig).x0 + gs[0, -1].get_position(fig).x1) / 2, 0.33 / fig_h,
         r"leak fraction $\kappa$", ha="center", va="center", fontsize=7.5)
keys = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=S.INK, label="holds"),
        plt.Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", mec=S.INK2, label=r"from $\kappa^\star$ on"),
        plt.Line2D([], [], color=S.INK2, lw=0.9, label="interval (Bonferroni over the section's $m$)"),
        plt.Line2D([], [], color=S.BLUE, lw=1.0, ls=(0, (1.2, 1.4)), label=r"breakdown point $\kappa^\star$")]
fig.legend(handles=keys, loc="lower center", ncol=4, frameon=False, fontsize=6.3, handlelength=1.6,
           columnspacing=1.4, bbox_to_anchor=(0.59, -0.005))
S.save(fig, OUT)
print("members in a family but not yet computed:", ", ".join(missing) or "none")
