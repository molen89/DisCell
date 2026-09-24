"""Is the decoded composition rho peaked or spread? Read off a fitted run.

A softmax can be arbitrarily peaked; nothing in its form prevents it. What
forbids it here is the multinomial likelihood, whose optimum for a cell is its
own observed composition x_i / l_i. So rho should sit near the data's own
concentration, nowhere near one-hot.

Concentration is reported as the effective number of genes, exp(entropy):
G for uniform, 1 for one-hot.
"""
import numpy as np, torch
from discell.model.degeneracy import load_trainer

DATASET, RUN = "xenium_prime_ovarian_cancer_ffpe", "best"
trainer, run_dir, epoch = load_trainer(DATASET, RUN, "cpu")
G = trainer.data.x.shape[1]
print(f"{DATASET}/{RUN} @ epoch {epoch}, G = {G}")


def eff_genes(p, axis=-1):
    p = np.clip(p, 1e-30, None)
    return np.exp(-(p * np.log(p)).sum(axis))


rho_all, emp_all, ell_all = [], [], []
with torch.no_grad():
    for b in trainer.val_batches[:12]:
        fwd = trainer.model(**trainer._forward_kwargs(b),
                            kappa=trainer.config.kappa, sample=False)
        n = b["n_seeds"]
        rho_all.append(fwd.log_rho[:n].exp().cpu().numpy())
        x = trainer.data.x[b["nodes"][:n]].toarray()
        ell_all.append(x.sum(1))
        emp_all.append(x / np.maximum(x.sum(1, keepdims=True), 1))

rho = np.vstack(rho_all); emp = np.vstack(emp_all); ell = np.concatenate(ell_all)
print(f"{len(rho)} held-out cells, median library size {np.median(ell):.0f}")

for lab, p in (("rho (model)", rho), ("x/l (observed)", emp)):
    e = eff_genes(p)
    print(f"\n{lab}")
    print(f"  effective genes exp(H): median {np.median(e):8.1f}  "
          f"p5 {np.percentile(e,5):7.1f}  p95 {np.percentile(e,95):7.1f}   (G={G}, one-hot=1)")
    print(f"  max_g p_g            : median {np.median(p.max(1)):.4f}  "
          f"p99 {np.percentile(p.max(1),99):.4f}")
    k = np.sort(p, axis=1)[:, ::-1]
    print(f"  mass in top 10 genes : median {np.median(k[:, :10].sum(1)):.3f}   "
          f"top 100: {np.median(k[:, :100].sum(1)):.3f}")

# how many cells are anywhere near one-hot?
print(f"\ncells with max_g rho_g > 0.5: {(rho.max(1) > 0.5).sum()} / {len(rho)}")
print(f"cells with effective genes < 10: {(eff_genes(rho) < 10).sum()} / {len(rho)}")

# what a one-hot rho would cost, per count, against the fitted model
eps = 1e-12
ll_fit = (emp * np.log(rho + eps)).sum(1)          # per-count log-lik of rho
onehot = np.zeros_like(rho); onehot[np.arange(len(rho)), rho.argmax(1)] = 1.0
ll_oh = (emp * np.log(onehot + eps)).sum(1)
print(f"\nper-count log-lik: fitted rho {np.mean(ll_fit):.3f} nats, "
      f"one-hot at rho's argmax {np.mean(ll_oh):.1f} nats")
