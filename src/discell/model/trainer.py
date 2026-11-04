"""The fit loop and the evaluation it runs.

:class:`Runner` holds a network and one section's tiles on a device and
evaluates; :class:`Trainer` adds the optimizers and the loop:

* per step: the model update on one tile (objective plus ``alpha_a`` times the
  adversary's excess skill), then ``adv_steps`` updates of the adversary heads
  on ``sg mu_z``;
* every ``eval_every`` epochs an evaluation on the held-out tiles; a
  checkpoint is accepted when the held-out reconstruction improves **and** the
  NMI of ``z`` against the types stays at least ``nmi_guard`` times its running
  maximum (a model that explains everything by spillover reconstructs well, so
  reconstruction alone must not decide);
* evaluations inside the ``alpha_w`` warm-up neither supply the checkpoint nor
  start the patience counter nor set the NMI reference;
* the dead-context-channel detector watches the opening evaluations;
* the closing evaluation scores the accepted checkpoint.

Nothing is written to disk here.
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import torch

from discell.evaluate import metrics as M
from discell.evaluate.guard import W_GUARD_NICHES, kl_w_is_dead, niche_labels, w_channel_guard
from discell.evaluate.mask import mask_record, metric_target_mask, nmi_inputs
from discell.graph.tiles import tile_batch
from discell.model.config import Config
from discell.model.data import ModelData
from discell.model.equations import spillover_mix
from discell.model.networks import Adversary, DisCell, soft_cross_entropy
from discell.model.objective import Weights, adversary_terms, discell_loss

log = logging.getLogger("discell.model.trainer")


def resolve_device(device: str) -> torch.device:
    """*device*, or the CPU with a warning when CUDA is asked for but absent."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"device {device!r} requested but CUDA is not available: "
                      "running on the CPU", RuntimeWarning, stacklevel=3)
        return torch.device("cpu")
    return torch.device(device)


def build_network(config: Config, data: ModelData) -> DisCell:
    return DisCell(n_genes=data.x.shape[1], n_types=len(data.type_names),
                   phi_dim=data.phi.shape[1], median_counts=data.median_counts,
                   d_z=config.d_z, d_w=config.d_w, hidden=config.hidden,
                   gat_dim=config.gat_dim, heads=config.heads)


def tile_tensors(data: ModelData, tile: np.ndarray, device: torch.device) -> dict:
    """One tile's tensors on *device*: its seeds, their halo and the edges.

    Counts are small integers held as int16 (cast to float32 per step); only
    the context nodes' counts are needed, since ring 2 enters the GAT
    through its types alone.
    """
    b = tile_batch(data.graph, tile)
    counts = data.x[b.nodes[:b.n_context]].toarray()
    if counts.size and counts.max() >= 32768:
        raise ValueError("counts exceed int16")
    return dict(
        nodes=b.nodes,
        x=torch.tensor(counts.astype(np.int16), device=device),
        t=torch.tensor(data.t[b.nodes], device=device),
        phi=torch.tensor(data.phi[b.nodes], device=device),
        isolated=torch.tensor(data.graph.isolated[b.nodes], device=device),
        gat_src=torch.tensor(b.gat_src, device=device),
        gat_dst=torch.tensor(b.gat_dst, device=device),
        spillover_src=torch.tensor(b.spillover_src, device=device),
        spillover_dst=torch.tensor(b.spillover_dst, device=device),
        spillover_beta=torch.tensor(b.spillover_beta, dtype=torch.float32, device=device),
        y_seed=torch.tensor(data.graph.y[b.nodes[:b.n_seeds]], device=device),
        ephi_seed=None if data.e_phi is None else torch.tensor(
            data.e_phi[b.nodes[:b.n_seeds]], device=device),
        n_seeds=b.n_seeds, n_context=b.n_context,
    )


class Runner:
    """A network over one section's resident tiles: sweeps and evaluation.

    *config* must be resolved (a numeric ``alpha_z``).
    """

    def __init__(self, network: DisCell, data: ModelData, config: Config,
                 device: torch.device):
        self.config, self.data, self.device = config, data, device
        self.network = network.to(device)
        self.train_batches = [self._to_device(t) for t in data.train_tiles]
        self.val_batches = [self._to_device(t) for t in data.val_tiles]
        self.exclude = (config.unassigned_label,)

    def _to_device(self, tile: np.ndarray) -> dict:
        """One tile's tensors, resident for the whole fit (:func:`tile_tensors`)."""
        return tile_tensors(self.data, tile, self.device)

    @staticmethod
    def _forward_kwargs(batch: dict) -> dict:
        keys = ("t", "phi", "isolated", "gat_src", "gat_dst", "spillover_src", "spillover_dst",
                "spillover_beta", "n_seeds", "n_context")
        return {"x": batch["x"].float(), **{k: batch[k] for k in keys}}

    @torch.no_grad()
    def sweep(self, batches: list[dict], recon: bool = False, context: bool = False,
              z_bar: torch.Tensor | None = None) -> dict:
        """Per-seed arrays over *batches*, posterior means throughout.

        *recon*: per-cell held-out reconstruction (``recon``), and with *z_bar*
        ``(K, d_z)`` also with each seed's ``z`` replaced by its type's mean
        (``recon_typemean``; ``w``, the influx and ``kappa`` unchanged).
        *context*: also the context vectors ``c``.
        """
        self.network.eval()
        keys = ("nodes", "mu_z", "mu_w", "prior_w", "c", "kl_w", "recon",
                "recon_typemean")
        out: dict = {k: [] for k in keys}
        kappa = self.config.kappa
        for batch in batches:
            fwd = self.network(**self._forward_kwargs(batch), kappa=kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(batch["nodes"][:n])
            out["mu_z"].append(fwd.mu_z[:n].cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
            if context:
                out["c"].append(fwd.c[:n].cpu().numpy())
            per_dim = 0.5 * (-fwd.logvar_w[:n] + fwd.logvar_w[:n].exp()
                             + (fwd.mu_w[:n] - fwd.prior_mean_w[:n]) ** 2 - 1.0)
            out["kl_w"].append(per_dim.cpu().numpy())
            if recon:
                x = batch["x"][:n].float().cpu().numpy()
                out["recon"].append(M.per_cell_reconstruction(x, fwd.log_p.cpu().numpy()))
                if z_bar is not None:
                    log_rho = self.network.log_rho(z_bar[batch["t"][:n]], fwd.mu_w[:n])
                    log_p = spillover_mix(log_rho.exp(), fwd.rho_bar, kappa)
                    out["recon_typemean"].append(
                        M.per_cell_reconstruction(x, log_p.cpu().numpy()))
        self.network.train()
        return {k: np.concatenate(v) if v else None for k, v in out.items()}

    def evaluate(self, final: bool = False) -> dict:
        """The held-out reads.

        Always: ``recon_val`` (every held-out cell; the checkpoint signal),
        ``recon_val_targets``, ``nmi`` (every cell; the guard signal), the
        pooled probe ``probe`` on target cells and ``kl_w_per_dim``. With
        *final*, also ``nmi_targets``, ``mirror``, ``degeneracy``,
        ``recon_gap``, the w-channel guard ``w_channel`` and ``eval_mask``,
        plus the latents of every cell under ``latents``.
        """
        data, seed, exclude = self.data, self.config.seed, self.exclude
        names = data.type_names
        n_types = len(names)
        train = self.sweep(self.train_batches, context=final)
        t_train = data.t[train["nodes"]]
        z_bar = None
        if final:
            z_bar = torch.tensor(M.type_means(train["mu_z"], t_train, n_types),
                                 device=self.device)
        val = self.sweep(self.val_batches, recon=True, context=final, z_bar=z_bar)

        rows_all = np.concatenate([train["nodes"], val["nodes"]])
        z_all = np.vstack([train["mu_z"], val["mu_z"]])
        t_all = data.t[rows_all]
        train_mask = np.arange(len(rows_all)) < len(train["nodes"])
        targets = metric_target_mask(t_all, names, exclude)
        tr_t = train_mask[targets]
        z_t, t_t = z_all[targets], t_all[targets]
        t_val = data.t[val["nodes"]]
        vt = metric_target_mask(t_val, names, exclude)
        report = {
            "recon_val": float(val["recon"].mean()),
            "recon_val_targets": float(val["recon"][vt].mean()),
            "nmi": M.z_type_nmi(z_all, t_all, seed=seed),
            "probe": M.probe_delta_ce(z_t, t_t, data.v_block[rows_all][targets],
                                      data.vbar_t, tr_t, ~tr_t, seed=seed),
            "kl_w_per_dim": val["kl_w"].mean(axis=0).tolist(),
        }
        if not final:
            return report

        z_n, t_n = nmi_inputs(z_all, t_all, names, exclude)
        c_all = np.vstack([train["c"], val["c"]])
        recon_t = report["recon_val_targets"]
        recon_typemean = float(val["recon_typemean"][vt].mean())
        x_val = data.x[val["nodes"][vt]]
        niches = niche_labels(data.graph.y, data.graph.degrees, W_GUARD_NICHES, seed)
        guard = w_channel_guard(val["mu_w"][vt], niches[val["nodes"]][vt], t_val[vt],
                                seed=seed)
        guard["niches"] = W_GUARD_NICHES
        report.update({
            "nmi_targets": (report["nmi"] if z_n is z_all
                            else M.z_type_nmi(z_n, t_n, seed=seed)),
            "mirror": M.mirror_r2(z_t, c_all[targets], t_t, seed=seed),
            "degeneracy": M.type_degeneracy(z_t, t_t, tr_t, ~tr_t, seed=seed),
            "recon_gap": {
                "recon": recon_t, "recon_typemean_z": recon_typemean,
                "gap": recon_t - recon_typemean,
                "recon_type_profile": M.type_profile_reconstruction(
                    data.x[train["nodes"]], t_train, x_val, t_val[vt])},
            "w_channel": guard,
            "eval_mask": mask_record(names, t_all, exclude),
            "latents": {"nodes": rows_all, "mu_z": z_all,
                        "mu_w": np.vstack([train["mu_w"], val["mu_w"]]),
                        "prior_w": np.vstack([train["prior_w"], val["prior_w"]])},
        })
        return report


class Trainer(Runner):
    """One fit: the network, the adversary, their optimizers and the loop.

    *config* must be resolved (a numeric ``alpha_z``). The torch seed is set
    here, before the networks are built, so a seed fixes the initial weights.
    """

    def __init__(self, config: Config, data: ModelData, device: torch.device):
        if isinstance(config.alpha_z, str):
            raise ValueError("resolve alpha_z before building a Trainer")
        if config.image_free != (data.e_phi is None):
            raise ValueError("data and config disagree on the image descriptor")
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        network = build_network(config, data)
        super().__init__(network, data, config, device)
        self.adversary = self.adversary_optimizer = None
        if config.alpha_a:
            n_ephi = 0 if data.e_phi is None else data.e_phi.shape[1]
            self.adversary = Adversary(config.d_z, len(data.type_names), n_ephi,
                                       hidden=config.adv_hidden).to(device)
            self.adversary_optimizer = torch.optim.Adam(self.adversary.parameters(),
                                                        lr=config.adv_lr)
            self.ybar_t = torch.tensor(data.graph.ybar_t, device=device)
            self.phibar_t = (None if data.phibar_t is None
                             else torch.tensor(data.phibar_t, device=device))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs)
        self.history: list[dict] = []
        self.losses: list[dict] = []

    def step(self, batch: dict, epoch: int) -> dict:
        """One model update, then the adversary's updates. Returns the scalars."""
        config = self.config
        kwargs = self._forward_kwargs(batch)
        fwd = self.network(**kwargs, kappa=config.kappa)
        n = batch["n_seeds"]
        weights = Weights(omega=config.omega, alpha_z=config.alpha_z,
                          alpha_w=config.alpha_w_at(epoch))
        terms = discell_loss(fwd, kwargs["x"][:n], weights)
        scalars = terms.scalars()
        loss = terms.loss
        if self.adversary is not None:
            adv = adversary_terms(self.adversary, fwd.mu_z[:n], batch["t"][:n],
                                  batch["y_seed"], batch["ephi_seed"], self.ybar_t,
                                  self.phibar_t, comp_weight=config.adv_comp_weight)
            loss = loss + config.alpha_a * adv.encoder_term
            scalars.update(adv_encoder=float(adv.encoder_term.detach()),
                           adv_excess_y=adv.excess_y, adv_excess_phi=adv.excess_phi)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.grad_clip)
        self.optimizer.step()
        if self.adversary is not None:
            scalars["adv_head_loss"] = self._head_steps(
                fwd.mu_z[:n].detach(), batch["t"][:n], batch["y_seed"], batch["ephi_seed"])
        return scalars

    def _head_steps(self, z_frozen: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                    e_phi: torch.Tensor | None) -> float:
        """``adv_steps`` updates of the heads on frozen ``mu_z``; the last head loss.

        The first ``zero_grad`` also clears what the model loss left on the heads.
        """
        weight = self.config.adv_comp_weight
        for _ in range(self.config.adv_steps):
            self.adversary_optimizer.zero_grad()
            log_y, log_phi = self.adversary(z_frozen, t)
            head_loss = weight * soft_cross_entropy(y, log_y)
            if e_phi is not None:
                head_loss = head_loss + soft_cross_entropy(e_phi, log_phi)
            head_loss = head_loss.mean()
            head_loss.backward()
            self.adversary_optimizer.step()
        return float(head_loss.detach())

    def fit(self) -> dict:
        """Run the loop; leaves the accepted checkpoint in ``self.network``.

        Returns ``{best, final, final_epoch, last_epoch, dead_w_channel,
        minutes}``; ``final`` is the closing evaluation of the accepted
        checkpoint (of the last epoch when no checkpoint was accepted) and
        carries the latents of every cell under ``latents``.
        """
        config = self.config
        best = {"recon_val": -np.inf, "nmi": 0.0, "epoch": -1}
        best_state = None
        nmi_max, stale, step = 0.0, 0, 0
        dead_w, last_epoch = False, -1
        started = time.time()
        for epoch in range(config.epochs):
            last_epoch = epoch
            for index in self.rng.permutation(len(self.train_batches)):
                self.losses.append({"step": step, "epoch": epoch,
                                    **self.step(self.train_batches[index], epoch)})
                step += 1
            self.schedule.step()
            if (epoch + 1) % config.eval_every:
                continue

            report = self.evaluate()
            warming = epoch < config.w_warmup_epochs
            if not warming:
                nmi_max = max(nmi_max, report["nmi"])
            if kl_w_is_dead(report["kl_w_per_dim"], epoch):
                dead_w = True
                log.warning("dead context channel: KL_w ~ 0 at epoch %d (sum %.3g)",
                            epoch, float(np.sum(report["kl_w_per_dim"])))
            self.history.append({"epoch": epoch, "step": step,
                                 "alpha_w_eff": config.alpha_w_at(epoch), **report})
            log.info("epoch %d  recon %.4f  nmi %.3f  dCE %.4f  [%.0fs]", epoch,
                     report["recon_val"], report["nmi"], report["probe"]["delta_ce"],
                     time.time() - started)
            if warming:
                continue
            improved = report["recon_val"] > best["recon_val"]
            guarded = report["nmi"] >= config.nmi_guard * nmi_max
            if improved and guarded:
                best = {"recon_val": report["recon_val"], "nmi": report["nmi"],
                        "epoch": epoch}
                best_state = {k: v.detach().clone()
                              for k, v in self.network.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= max(1, config.patience // config.eval_every):
                    log.info("early stop at epoch %d (best %d)", epoch, best["epoch"])
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        else:
            log.warning("no checkpoint was accepted in %d epochs (warm-up %d): the "
                        "model is the last epoch's", config.epochs, config.w_warmup_epochs)
        final = self.evaluate(final=True)
        summary = {"best": best, "final": final,
                   "final_epoch": best["epoch"] if best["epoch"] >= 0 else last_epoch,
                   "last_epoch": last_epoch, "dead_w_channel": dead_w,
                   "minutes": (time.time() - started) / 60}
        log.info("best recon %.4f (epoch %d), NMI %.3f, %.1f min", best["recon_val"],
                 best["epoch"], best["nmi"], summary["minutes"])
        return summary
