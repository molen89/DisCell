"""Figure: the KRONOS image embedding under three masks, as UMAPs of one subsample.

(a) The whole 128 um patch; (b) the 25 um disc over the cell zeroed -- the
embedding the model consumes as Phi; (c) everything outside the cell's polygon
zeroed -- what the cell alone carries. KRONOS v1, four channels, 256 px at
0.5 um/px, the arms of the ego-masking experiment (embeddings on disk). The same
60,000 randomly drawn cells in every panel (seed 0), PCA to 50 components, then
UMAP (30 neighbours, min_dist 0.3, cosine). Colour: the section's lineage labels
(obs `lineage`, the labels the model trains on), smaller lineages grouped for display.
The layouts are cached in data/fig_kronos_umap.npz. Outputs ../fig_kronos_umap.pdf
(+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

DS = "xenium_prime_ovarian_cancer_ffpe"
EMB = Path(f"data/datasets/{DS}/embeddings")
ARMS = [("egomask_unmasked_v1", r"whole patch"),
        ("egomask_ego_v1", r"disc zeroed: the model's $\bm\Phi_i$"),
        ("egomask_ego_only_v1", r"the cell alone")]
N_CELLS, SEED = 60_000, 0
HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "fig_kronos_umap.npz"
OUT = HERE.parent / "fig_kronos_umap"
# the section's lineage labels (the ones the model trains on), smaller lineages grouped for display
GROUPS = [("Tumour", ["Tumour"]),
          ("Fibroblasts", ["Fibroblasts"]),
          ("Smooth muscle, pericytes", ["Smooth muscle cells", "Pericytes"]),
          ("Endothelium", ["Endothelial cells"]),
          ("Immune", ["Macrophages", "T/NK cells"]),
          ("Other epithelia, cyst lining", ["Fallopian tube epithelium", "Ciliated epithelial cells",
                                            "Urothelial-like cells", "Mesothelial-like cyst lining"])]
# the validated categorical order of the reference palette, fixed; Unassigned in neutral grey
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
UNASSIGNED = "#b5b3ab"


def load(name):
    m = torch.load(EMB / f"{name}.pt", map_location="cpu", weights_only=False)
    return list(m["cell_ids"]), np.asarray(m["embeddings"], dtype=np.float32)


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    ids, layouts = list(z["ids"]), [z[f"umap{k}"] for k in range(len(ARMS))]
else:
    from sklearn.decomposition import PCA
    import umap
    loaded = [load(name) for name, _ in ARMS]
    common = set(loaded[0][0])
    for cid, _ in loaded[1:]:
        common &= set(cid)
    ids = sorted(np.random.default_rng(SEED).choice(sorted(common), N_CELLS, replace=False))
    layouts = []
    for cid, X in loaded:
        pos = {c: k for k, c in enumerate(cid)}
        Xs = X[[pos[c] for c in ids]]
        Xp = PCA(50, random_state=SEED).fit_transform(Xs)
        layouts.append(umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=SEED).fit_transform(Xp))
        print("umap done", Xs.shape)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, ids=np.array(ids, dtype=object), **{f"umap{k}": u for k, u in enumerate(layouts)})

a = ad.read_h5ad(f"data/datasets/{DS}/bundle/full.h5ad", backed="r")
label = a.obs["lineage"].astype(str).reindex(ids).to_numpy()
group_of = {old: k for k, (_, olds) in enumerate(GROUPS) for old in olds}
g = np.array([group_of.get(x, -1) for x in label])
colour = np.array([COLOURS[k] if k >= 0 else UNASSIGNED for k in g])
order = np.random.default_rng(SEED).permutation(len(ids))            # no group always drawn on top

S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.7))
for c, ((_, title), u) in enumerate(zip(ARMS, layouts)):
    ax = fig.add_axes([0.025 + c / 3, 0.2, 1 / 3 - 0.035, 0.7])
    ax.scatter(u[order, 0], u[order, 1], s=0.25, c=colour[order], lw=0, alpha=0.6, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    ax.text(0.02, 1.0, title, transform=ax.transAxes, fontsize=7.5, color=S.INK, ha="left", va="bottom")
    S.panel_label(ax, "abc"[c], x=0.0, y=1.0)
keys = [Line2D([], [], marker="o", ls="", ms=4, color=col, label=name) for (name, _), col in zip(GROUPS, COLOURS)]
keys.append(Line2D([], [], marker="o", ls="", ms=4, color=UNASSIGNED, label="Unassigned"))
fig.legend(handles=keys, loc="lower center", ncol=4, frameon=False, fontsize=6.5, handletextpad=0.2,
           columnspacing=1.1, bbox_to_anchor=(0.5, -0.005))
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"), "| cells", len(ids), "| unassigned", int((g < 0).sum()))
