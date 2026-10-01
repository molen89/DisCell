"""RECOMB figure: the leading response programme of each section (fig:programmes).

Per section (columns), from the seed-0 fit of the final configuration (finalL_s0):
  a  territory: the leading programme's score per cell (within-type centred), on the section;
  b  signature: the five largest positive and negative gene loadings of the programme, with
     its hallmark labels and in how many of the three seeds each label recurs;
  c  drivers: cross-validated R^2 of the programme score from neighbour composition, the image
     context and distances to landmarks, alone (light) and unique given the others (dark).
  d  (last column) within-type Moran's I of mu_z without / with the adversary and of mu_w,
     mean over seeds with the range (the values of tab:moran).

Sources, all run-internal (programmes are identified only up to rotation):
  data/datasets/<ds>/runs/finalL_s{0,1,2}/atlas/atlas.json   signatures, hallmarks, drivers,
                                                             Moran's I, cross-seed matching
  data/datasets/<ds>/runs/finalL_s0/atlas/programs.npy       loadings (G, r)
  data/datasets/<ds>/runs/finalL_s0/best.pt  (B.weight)       to rebuild the per-cell scores
  data/datasets/<ds>/experiments/kl_maps_finalL_s0.npz        posterior means mu_w, positions
  data/datasets/<ds>/runs/{finalL,uncontrolledL}_s*/validation/validation.json   Moran's I
The per-cell score is rebuilt exactly as the atlas builds it (within-type centring, the
effective-rank eigenbasis of cov(mu_w), the rotation solved from programs.npy); the script
checks that the rebuilt variance shares equal the atlas's. Run from the repository root.
Outputs ../recomb_fig_programmes.pdf (+ .png) and prints a verification log.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

SECTIONS = [("xenium_prime_ovarian_cancer_ffpe", "Ovarian FFPE"),
            ("xenium_prime_human_lung_cancer_ffpe", "Lung FFPE"),
            ("xenium_prime_human_ovary_ff", "Ovarian FF"),
            ("gse315411_pdltma06_11_prime_solo", "Lung TMA core")]
SEEDS = ["finalL_s0", "finalL_s1", "finalL_s2"]
WIDTH = 6.5                      # RECOMB text width (letter, 1-inch margins), inches
N_SHOW, N_GENES = 150_000, 5
HALLMARK_NAMES = {
    "EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "TNFA_SIGNALING_VIA_NFKB": r"TNF$\alpha$/NF-$\kappa$B",
    "HYPOXIA": "hypoxia",
    "INFLAMMATORY_RESPONSE": "inflammatory resp.",
    "MYOGENESIS": "myogenesis",
    "ESTROGEN_RESPONSE_EARLY": "oestrogen resp. early",
    "ESTROGEN_RESPONSE_LATE": "oestrogen resp. late",
}
BLOCKS = [("composition_y", "composition"), ("phi_pcs", "image"),
          ("landmark_distances", "landmarks")]
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "recomb_fig_programmes"
RED, BLUE = "#c8433a", S.BLUE


def ds_root(ds):
    return Path("data/datasets") / ds


def atlas(ds, run):
    return json.loads((ds_root(ds) / "runs" / run / "atlas" / "atlas.json").read_text())


def programme_scores(ds):
    """Per-cell score of every programme on the target cells, rebuilt as the atlas does."""
    import torch
    d = np.load(ds_root(ds) / "experiments" / "kl_maps_finalL_s0.npz")
    names = [str(x) for x in d["type_names"]]
    target = np.array([not nm.startswith("Unassigned") for nm in names])[d["t"]]
    mu, t = d["mu_w"][target].astype(np.float64), d["t"][target]
    c = mu.copy()
    for g in np.unique(t):
        c[t == g] -= c[t == g].mean(0)
    c -= c.mean(0)
    ev, evec = np.linalg.eigh(np.cov(c.T))
    order = np.argsort(-ev)
    ev, evec = ev[order].clip(min=0), evec[:, order]
    r = max(1, int((ev / ev.sum() >= 0.01).sum()))
    scale = np.sqrt(ev[:r]) + 1e-12
    run = ds_root(ds) / "runs" / "finalL_s0"
    b = torch.load(run / "best.pt", map_location="cpu",
                   weights_only=False)["model"]["B.weight"].numpy().astype(np.float64)
    programs = np.load(run / "atlas" / "programs.npy").astype(np.float64)
    be = b @ (evec[:, :r] * scale)
    rot = np.linalg.lstsq(be, programs, rcond=None)[0]
    resid = np.linalg.norm(be @ rot - programs) / np.linalg.norm(programs)
    u = (c @ evec[:, :r] / scale) @ rot
    axes = u.T @ c / len(u)
    share = (axes ** 2).sum(1) / c.var(0).sum()
    ref = [p["variance_share"] for p in atlas(ds, "finalL_s0")["programs"]]
    assert resid < 1e-6 and np.allclose(share, ref, atol=1e-4), (ds, resid, share, ref)
    return u[:, 0], d["xy_um"][target]


def moran(ds, prefix, key, n):
    vals = []
    for s in range(n):
        m = json.loads((ds_root(ds) / "runs" / f"{prefix}_s{s}" / "validation"
                        / "validation.json").read_text())["morans"][key]
        i, v = np.array(m["I"]), np.array(m["var_per_dim"])
        vals.append(float((i * v).sum() / v.sum()))
    return np.mean(vals), min(vals), max(vals)


def leading_summary(ds):
    """Programme 0 of seed 0, its hallmark labels and their recurrence over seeds."""
    a0 = atlas(ds, "finalL_s0")
    p0 = a0["programs"][0]
    # the programme matched to programme 0 in the other seeds (one-to-one, by |cosine|)
    matched = {"finalL_s0": p0}
    cos = []
    for run in SEEDS[1:]:
        m = a0["cross_seed"][run]["axis_cosine"][0]
        matched[run] = atlas(ds, run)["programs"][m["program"]]
        cos.append(m["cosine"])
    s1s2 = atlas(ds, "finalL_s1")["cross_seed"]["finalL_s2"]["axis_cosine"]
    j = a0["cross_seed"]["finalL_s1"]["axis_cosine"][0]["program"]
    if s1s2[j] is not None:
        cos.append(s1s2[j]["cosine"])
    labels = []
    for h in p0["hallmarks"]:
        if not h["significant"]:
            continue
        n = sum(any(g["hallmark"] == h["hallmark"] and g["significant"]
                    for g in matched[run]["hallmarks"]) for run in SEEDS)
        top = sum(next((g["hallmark"] for g in matched[run]["hallmarks"]
                        if g["significant"]), None) == h["hallmark"] for run in SEEDS)
        labels.append((h["hallmark"], h["auc"], h["direction"], n, top))
    return dict(p0=p0, labels=labels, cos=(min(cos), max(cos)),
                ranks=[atlas(ds, r)["rank"] for r in SEEDS],
                landmarks=a0["landmark_classes"])


def draw():
    S.apply()
    fig = plt.figure(figsize=(WIDTH, 3.55))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.78],
                          height_ratios=[1.05, 1.0, 0.45],
                          left=0.1, right=0.9, top=0.91, bottom=0.075,
                          wspace=0.3, hspace=0.6)
    rng = np.random.default_rng(0)
    log = []
    for col, (ds, title) in enumerate(SECTIONS):
        info = leading_summary(ds)
        p0 = info["p0"]
        score, xy = programme_scores(ds)
        # --- a: territory
        ax = fig.add_subplot(gs[0, col])
        idx = rng.choice(len(score), size=min(N_SHOW, len(score)), replace=False)
        x, y = xy[idx, 0], xy[idx, 1]
        if ds == "xenium_prime_human_ovary_ff":          # tall section: rotate for display
            x, y = y, -x
        lim = np.percentile(np.abs(score), 98)
        ax.scatter(x, y, c=score[idx], cmap="RdBu_r", vmin=-lim, vmax=lim, s=0.04,
                   linewidths=0, rasterized=True)
        S.tissue_axes(ax)
        y0, y1 = ax.get_ylim()                       # inverted: y0 is the bottom
        ax.set_ylim(y0 + 0.22 * (y0 - y1), y1)       # room for the scale bar
        S.scalebar(ax, 1000 if ds.startswith("gse") else 2000, 1.0,
                   label="1\\,mm" if ds.startswith("gse") else "2\\,mm", lw=1.0)
        ax.set_title(rf"\textbf{{{title}}}" + "\n" +
                     rf"{100 * p0['variance_share']:.0f}\% of $\mathbf{{w}}$, I = {p0['moran_I']:.2f}",
                     fontsize=7, pad=2)
        if col == 0:
            S.panel_label(ax, "a", x=-0.06, y=1.2)
        # --- b: signature
        ax = fig.add_subplot(gs[1, col])
        hi = p0["signature_high"][:N_GENES]
        lo = p0["signature_low"][:N_GENES]
        genes = [g for g, _ in hi] + [g for g, _ in lo]
        vals = np.array([v for _, v in hi] + [v for _, v in lo])
        vals = vals / np.abs(vals).max()
        ypos = np.arange(len(genes))[::-1]
        ax.barh(ypos, vals, color=[RED if v > 0 else BLUE for v in vals], height=0.7)
        for yy, g, v in zip(ypos, genes, vals):
            ax.text(-0.04 if v > 0 else 0.04, yy, rf"\textit{{{g}}}", fontsize=6,
                    ha="right" if v > 0 else "left", va="center", color=S.INK)
        ax.axvline(0, color=S.MUTED, lw=0.4)
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        lab = [rf"{HALLMARK_NAMES.get(h, h.lower())} {n}/3"
               for h, _, _, n, _ in info["labels"][:2]]
        ax.set_title("\n".join(lab), fontsize=6.3, pad=2)
        if col == 0:
            S.panel_label(ax, "b", x=-0.06, y=1.2)
        # --- c: drivers
        ax = fig.add_subplot(gs[2, col])
        drv = p0["drivers"]
        names = [(k, n) for k, n in BLOCKS if k in drv["marginal"]]
        yb = np.arange(len(names))[::-1]
        marg = [max(drv["marginal"][k], 0) for k, _ in names]
        part = [max(drv["partial"][k], 0) for k, _ in names]
        ax.barh(yb, marg, color=S.BLUE_RAMP[1], height=0.72)
        ax.barh(yb, part, color=S.BLUE_RAMP[5], height=0.72)
        ax.set_yticks(yb)
        ax.set_yticklabels([n for _, n in names] if col == 0 else [], fontsize=6)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"], fontsize=6)
        ax.set_title(rf"joint $R^2$ {drv['joint']:.2f}", fontsize=6.3, pad=2)
        S.light_grid(ax, "x")
        if col == 0:
            S.panel_label(ax, "c", x=-0.06, y=1.12)
        log.append((title, info, drv))
    # --- d: Moran's I, z vs w
    ax = fig.add_subplot(gs[0:2, 4])
    short = ["Ov.\\ FFPE", "Lung FFPE", "Ov.\\ FF", "TMA core"]
    for k, (ds, _) in enumerate(SECTIONS):
        yy = len(SECTIONS) - 1 - k
        rows = [(moran(ds, "uncontrolledL", "mu_z", 2), "o", "none", S.INK2),
                (moran(ds, "finalL", "mu_z", 3), "o", S.INK2, S.INK2),
                (moran(ds, "finalL", "mu_w", 3), "o", BLUE, BLUE)]
        for (m, a, b), mk, fc, ec in rows:
            ax.plot([a, b], [yy, yy], color=ec, lw=0.8)
            ax.scatter([m], [yy], marker=mk, facecolor=fc, edgecolor=ec, s=16, lw=0.8, zorder=3)
        log.append(("moran", ds, [r[0] for r in rows]))
    ax.set_yticks(range(len(SECTIONS)))
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.set_yticklabels(short[::-1], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.6, len(SECTIONS) - 0.4)
    ax.set_xlabel("within-type Moran's I", fontsize=6.5, labelpad=1)
    S.light_grid(ax, "x")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec=S.INK2, ms=4),
               Line2D([], [], ls="none", marker="o", mfc=S.INK2, mec=S.INK2, ms=4),
               Line2D([], [], ls="none", marker="o", mfc=BLUE, mec=BLUE, ms=4)]
    ax.legend(handles, [r"$\mathbf{z}$, no adversary", r"$\mathbf{z}$", r"$\mathbf{w}$"],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6, frameon=False,
              handletextpad=0.2, ncol=1)
    S.panel_label(ax, "d", x=0.0, y=1.02)
    S.save(fig, OUT)
    return log


def tumour_gradient(ds, min_cells=3000):
    """Check behind the text's "its territory lies around the tumour" (ovarian FFPE): per type,
    the mean leading-programme score by the share of tumour cells among the 50 nearest cells,
    and the Spearman correlation of the score with the distance to the nearest tumour cell."""
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    d = np.load(ds_root(ds) / "experiments" / "kl_maps_finalL_s0.npz")
    names = [str(x) for x in d["type_names"]]
    target = np.array([not nm.startswith("Unassigned") for nm in names])[d["t"]]
    score, xy = programme_scores(ds)
    t = d["t"][target]
    tumour = np.isin(d["t"], [i for i, nm in enumerate(names) if nm == "Tumour"])
    frac = tumour[cKDTree(d["xy_um"]).query(xy, k=50)[1]].mean(1)
    dist = cKDTree(d["xy_um"][tumour]).query(xy)[0]
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.01]
    for g in np.unique(t):
        m = t == g
        if m.sum() < min_cells or names[g] == "Tumour":
            continue
        row = [round(float(score[m & (frac >= a) & (frac < b)].mean()), 2)
               if (m & (frac >= a) & (frac < b)).sum() > 200 else None
               for a, b in zip(bins[:-1], bins[1:])]
        print(f"   {names[g]}: score by tumour share {row}, "
              f"Spearman with tumour distance {spearmanr(score[m], dist[m])[0]:.2f}")


if __name__ == "__main__":
    for row in draw():
        if row[0] == "moran":
            print("Moran", row[1], [tuple(round(v, 3) for v in r) for r in row[2]])
            continue
        title, info, drv = row
        p0 = info["p0"]
        print(f"{title}: share {p0['variance_share']}, Moran {p0['moran_I']:.3f}, "
              f"ranks {info['ranks']}, leading-axis cosine {info['cos'][0]:.2f}-{info['cos'][1]:.2f}")
        print("   +", [g for g, _ in p0["signature_high"][:N_GENES]],
              "-", [g for g, _ in p0["signature_low"][:N_GENES]])
        for h, auc, dirn, n, top in info["labels"]:
            print(f"   label {h}: AUC {auc:.2f}, direction {dirn:+.3f}, "
                  f"significant in {n}/3 seeds, first label in {top}/3")
        print("   drivers marginal", {k: round(v, 2) for k, v in drv["marginal"].items()},
              "unique", {k: round(v, 2) for k, v in drv["partial"].items()},
              "joint", round(drv["joint"], 2), "landmarks", info["landmarks"])
    print("Ovarian FFPE, leading programme against tumour proximity:")
    tumour_gradient("xenium_prime_ovarian_cancer_ffpe")
