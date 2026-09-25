#!/usr/bin/env python3
"""One DisCell fit: the loop, its diagnostics, and the TensorBoard record.

One invocation is one point of the kappa sweep. Outputs land under
``data/datasets/<id>/runs/<name>/``: TensorBoard events, ``config.json``,
``best.pt`` (weights + covariance state), ``metrics.json``.

Early stopping is **joint** (spec 7.10): a checkpoint counts as best only when
held-out reconstruction improves *and* z-type NMI has not collapsed below a
fraction of its own running best -- a model explaining everything with leakage
has excellent reconstruction, which is exactly why reconstruction alone must
not decide.

Usage::

    python -m discell.model.train --dataset <id> --embeddings egomask_ego_v1 --kappa 0.2
    python -m discell.model.train --dataset <id> --embeddings egomask_ego_v1 \\
        --kappa 0 --alpha-a 0 --run-name k0_uncontrolled
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from discell import paths
from discell.model import metrics as M
from discell.model.elbo import Weights, discell_loss, ensemble_adversary_terms
from discell.model.equations import TypeCovariances, leakage_mix
from discell.model.fp_floor import fp_floor_eta
from discell.model.networks import (KAPPA_MODES, DisCell, kappa_ratio_stats,
                                    read_density_areas, read_gene_share)
from discell.model.prepare import ModelData, assemble, tile_batch

log = logging.getLogger("discell.model.train")

#: dead-context-channel detector (2026-09-23, FF best_s2). When the prior
#: network m_psi(c, t) ignores c, q(w|.) sits exactly on it and every
#: per-dimension KL_w reads 0.0000 from the first evaluation onward: the run
#: is a type-conditioned intrinsic autoencoder with a leak term, and every
#: other read (recon, NMI, probe, cycle) looks healthy. Healthy fits carry a
#: summed KL_w of 3e-4 to 8e-3 at the first evaluations, so the threshold sits
#: two orders below the smallest live value. Checked only over the opening
#: epochs: a channel that is going to open has opened by then.
DEAD_W_KL_SUM = 1e-5
DEAD_W_MAX_EPOCH = 20


def kl_w_is_dead(kl_w_per_dim, epoch: int) -> bool:
    """True when *epoch* is an opening epoch whose summed KL_w is ~0."""
    return (epoch < DEAD_W_MAX_EPOCH
            and float(np.sum(kl_w_per_dim)) < DEAD_W_KL_SUM)


def dead_w_channel(history: Sequence[dict]) -> bool:
    """The detector over a whole ``history.jsonl`` (one record per evaluation)."""
    return any(kl_w_is_dead(record["kl_w_per_dim"], int(record["epoch"]))
               for record in history if record.get("kl_w_per_dim") is not None)


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines one fit; serialised into the run directory."""

    dataset: str
    variant: str = "full"
    embeddings: str = "egomask_ego_v1"
    #: obs column for t; None = the bundle default (curated cell_group).
    #: "graphclust" runs on unsupervised clusters -- the no-annotation control
    label_key: str | None = None
    run_name: str | None = None
    # the model
    d_z: int = 20
    d_w: int = 6
    hidden: int = 256
    gat_dim: int = 32
    heads: int = 4
    phi_pca: int | None = None          # None: full-dimension Phi into c
    #: the projection test (S53, 2026-09-25): a learned nn.Linear(Phi -> D)
    #: applied to Phi before it enters c. 0 = the full Phi = every run before
    #: it, bit for bit.
    phi_proj: int = 0
    v_pcs: int = 12                     # Phi PCs inside the invariance block
    # the objective -- defaults are the calibrated operating point (2026-09,
    # four calibration rounds + two sweeps; see docs/devlog.md): adversary at
    # alpha_a = 0.3 / 6 steps / lr 2e-3 (MLP-probe leak 14% of uncontrolled at
    # zero NMI cost), kappa = 0.1 (end of the recon plateau, most B-stable).
    # Under invariance = "closed_form" the straight-through-scaled alpha_a
    # equivalent is ~0.02, not 0.3.
    #: GAT source features. "type_only" (default, ratified 2026-09-12 after
    #: the doc-10 PARK + source ablation: better on every measured axis, and
    #: removes the neighbour-z channel entirely); "type_z" = the original
    #: spec-4.1 sources [onehot(t_j), sg mu_z_j], kept for era-reproduction
    gat_sources: str = "type_only"
    subtract_leak: bool = False         # spec 7.13: encoders read x - kappa*l*rho_bar
    gat_sink: bool = False              # attention sink: neighbour dose, saturating
    invariance: str = "adversary"       # "closed_form" before spec 4.6 escalation
    adv_lr: float = 2e-3
    adv_steps: int = 6                  # CLI also --adv-head-steps (8.17)
    adv_hidden: int = 64                # CLI also --adv-head-width (8.17)
    #: the adversary-capacity ladder (8.17 pre-registration, 2026-09-24).
    #: Both defaults are every run before it, bit for bit.
    #: adv_ensemble: K independent head pairs, their head losses summed, the
    #: encoder penalised on the members' mean excess (elbo.
    #: ensemble_adversary_terms). adv_comp_weight: multiplies the composition
    #: excess (encoder term) and the composition CE (head loss).
    adv_ensemble: int = 1
    adv_comp_weight: float = 1.0
    kappa: float = 0.1
    omega: float = 1.0
    alpha_z: float = 0.007
    alpha_w: float = 0.1
    alpha_a: float = 0.3
    #: optional L2 on the w channel (2026-09-21 pre-registration): 0 = off =
    #: every run before it. "w" = lambda_w * E||w||^2 per seed; "type_mean" =
    #: lambda_w * sum_t (n_t/n)||mean_t m_psi||^2 (the V12 proposal).
    lambda_w: float = 0.0
    w_penalty: str = "none"
    #: objective ablations (6b.5, 2026-09-21). All three defaults are the
    #: pinned runs; each is a single term of the objective, nothing else.
    #: (i)  second_kl=False       -- the z-KL weighted 1*alpha_z, not
    #:      (1+omega)*alpha_z: the second bound's copy dropped (spec 6.2).
    #: (ii) is not a flag: omega=0 already drops the intrinsic path (b), and
    #:      the (1+omega) factor then reads 1 on its own.
    #: (iii) class_mean_prior=True -- p(w|t) = N(mu_t, I) with mu_t a learned
    #:      per-type vector in place of m_psi(c, t) (DisCoVR); q(w|.) unchanged.
    #: (iv) adv_input="xhat"      -- the adversary heads read the decoded clean
    #:      composition log rho_i = log_softmax(a(z) + Bw) instead of mu_z.
    second_kl: bool = True
    class_mean_prior: bool = False
    adv_input: str = "mu_z"             # "mu_z" | "xhat"
    #: the two context-collapse remedies (2026-09-23 pre-registration). Both
    #: default off = every run before them, and then the objective is the
    #: pinned one bit for bit.
    #: warm-up: alpha_w_eff(epoch) = alpha_w * min(1, epoch / N), the KL_w term
    #: only, so the converged objective is unchanged. 0 = no warm-up.
    w_warmup_epochs: int = 0
    #: warm-up on BOTH KL terms (8.9b pre-registration, 2026-09-24): alpha_z
    #: and alpha_w each scaled by min(1, epoch / N) -- alpha_z on both copies
    #: of the z-divergence, the (1 + omega) factor unchanged. Adversary, recon
    #: and every other term untouched. Mutually exclusive with
    #: w_warmup_epochs. 0 = off.
    kl_warmup_epochs: int = 0
    #: free bits: per-dimension KL_w charged as max(KL_k - lambda, 0). 0 = off.
    w_free_bits: float = 0.0
    #: the three query/prior ablations (2026-09-23 pre-registration). Both
    #: defaults are the pinned architecture, bit for bit.
    #: query: what the GAT is queried with -- "type" = embed(t_i) (pinned),
    #: "type_free" = one learned vector shared by all cells, "image" = a
    #: linear map of the ego-masked Phi_i. Arms (i) and (ii).
    query: str = "type"
    #: arm (iii): m_psi reads c only, t dropped from the prior; q(w|.) unchanged.
    prior_type_free: bool = False
    #: review R12 Test 2 (2026-09-24): the form of the leak coefficient,
    #: networks.KAPPA_MODES. "global" = one kappa = every run before it, bit for
    #: bit; "depth" = kappa_i = kappa * clip(sum_j beta_ij l_j / l_i, 0, 0.5 /
    #: kappa); "gene" = kappa_g = min(kappa * s_g / mean(s_g), 0.9) with s_g
    #: read from kappa_gene_source -- None resolves to the dataset's
    #: experiments/gene_extranuclear_share.npy, and the resolved path is what
    #: config.json records; "density" = "depth" on l/A, A the cell area
    #: (networks.density_areas; the run dir's kappa_density.json records how).
    kappa_mode: str = "global"
    kappa_gene_source: str | None = None
    #: the depth/density normaliser (amendments 1-2, 2026-09-24): the m with
    #: mean(kappa * clip(r_i / m, 0, 0.5 / kappa)) = kappa over the connected
    #: training cells, so the post-clip mean kappa_i is kappa. None = solved
    #: once at Trainer setup and recorded here (config.json) and in the run
    #: dir's kappa_<mode>.json.
    kappa_ratio_mean: float | None = None
    #: the fixed false-positive floor (todo 8.15b, devlog 2026-09-24 21:20 B):
    #: p_i = (1 - kappa_i - eta_i) rho_i + kappa_i rho_bar_i + eta_i u, u = 1/G,
    #: eta_i = min(lambda_i / l_i, 0.2), lambda from the Xenium cell table's
    #: negative-control and genomic-control counts (discell.model.fp_floor).
    #: False = every run before it, bit for bit; works under every kappa_mode.
    #: fp_area: lambda_i proportional to segmented area, section total fixed
    #: (the data-decided alternative; False = one lambda per section).
    #: fp_lambda / fp_cap_share: None = derived at Trainer setup and recorded
    #: here (config.json); the run dir's fp_floor.json holds the components.
    fp_floor: bool = False
    fp_area: bool = False
    fp_lambda: float | None = None
    fp_cap_share: float | None = None
    # optimisation
    epochs: int = 200
    tile_cells: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0          # Adam's coupled L2; 0 = the pinned runs
    grad_clip: float = 10.0
    val_fraction: float = 0.15
    patience: int = 20
    nmi_guard: float = 0.9              # best must keep NMI >= guard * running max
    cov_ema: float = 0.05
    eval_every: int = 5
    figures_every: int = 25
    panel_types: int = 8            # types shown in per-type figures
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.adv_ensemble < 1:
            raise ValueError(f"adv_ensemble must be >= 1, got {self.adv_ensemble}")
        if self.w_warmup_epochs > 0 and self.kl_warmup_epochs > 0:
            raise ValueError(
                "--w-warmup-epochs and --kl-warmup-epochs are mutually "
                f"exclusive (got {self.w_warmup_epochs} and "
                f"{self.kl_warmup_epochs})")
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {KAPPA_MODES}, "
                             f"got {self.kappa_mode!r}")
        if self.kappa_mode == "gene" and self.kappa_gene_source is None:
            object.__setattr__(self, "kappa_gene_source", str(
                paths.dataset(self.dataset).root / "experiments"
                / "gene_extranuclear_share.npy"))

    @property
    def warmup_epochs(self) -> int:
        """Length of the warm-up era, whichever flag set it; 0 = none."""
        return max(self.w_warmup_epochs, self.kl_warmup_epochs)

    def alpha_z_at(self, epoch: int) -> float:
        """The warmed-up alpha_z at *epoch*; the constant alpha_z unless
        kl_warmup_epochs is on (the w-only warm-up never touches it)."""
        if self.kl_warmup_epochs <= 0:
            return self.alpha_z
        return self.alpha_z * min(1.0, epoch / self.kl_warmup_epochs)

    def alpha_w_at(self, epoch: int) -> float:
        """The warmed-up alpha_w at *epoch* (either warm-up flag); the
        constant alpha_w when both are off."""
        if self.warmup_epochs <= 0:
            return self.alpha_w
        return self.alpha_w * min(1.0, epoch / self.warmup_epochs)

    def weights(self, epoch: int | None = None) -> Weights:
        return Weights(omega=self.omega,
                       alpha_z=(self.alpha_z if epoch is None
                                else self.alpha_z_at(epoch)),
                       alpha_w=(self.alpha_w if epoch is None
                                else self.alpha_w_at(epoch)),
                       alpha_a=self.alpha_a,
                       lambda_w=self.lambda_w, w_penalty=self.w_penalty,
                       second_kl=self.second_kl, w_free_bits=self.w_free_bits)

    def name(self) -> str:
        return self.run_name or f"discell_k{self.kappa:g}_seed{self.seed}"


class Trainer:
    """Owns one fit: tiles to device, steps, evaluation sweeps, the log."""

    def __init__(self, config: TrainConfig, data: ModelData) -> None:
        self.config, self.data = config, data
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu"
            else "cpu")
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        # figures draw their own subsamples: a logging knob must never move
        # the training-shuffle stream
        self.figure_rng = np.random.default_rng([config.seed, 3])
        #: the epoch currently being stepped -- read by the KL warm-ups
        self.epoch = 0
        if config.figures_every % config.eval_every:
            raise ValueError("figures_every must be a multiple of eval_every, "
                             f"got {config.figures_every} / {config.eval_every}")

        # review R12 depth/density forms: the cell areas (density) and the
        # amendment's normaliser mean_train(r), once, over the connected
        # training cells; a config that already carries it keeps its value
        self.cell_area = self.density_report = self.kappa_ratio_report = None
        if config.kappa_mode == "density":
            self.cell_area, self.density_report = read_density_areas(
                config.dataset, config.variant, data.t, len(data.p_t),
                data.type_names)
        if config.kappa_mode in ("depth", "density"):
            self.kappa_ratio_report = kappa_ratio_stats(
                data.graph.in_edges, data.totals,
                np.concatenate(data.train_tiles), config.kappa, self.cell_area)
            if config.kappa_ratio_mean is None:
                config = dataclasses.replace(
                    config, kappa_ratio_mean=self.kappa_ratio_report[
                        "normaliser"])
                self.config = config

        # 8.15b: the false-positive floor's eta_i, once per cell (numpy only:
        # no torch draw moves); a config that carries lambda keeps it
        self.fp_eta = self.fp_report = None
        if config.fp_floor:
            self.fp_eta, self.fp_report = fp_floor_eta(
                config.dataset, config.variant, data.totals, data.x.shape[1],
                config.fp_area, config.fp_lambda)
            config = dataclasses.replace(
                config, fp_lambda=self.fp_report["lambda_used"],
                fp_cap_share=self.fp_report["cap_share"])
            self.config = config

        self.model = DisCell(
            n_genes=data.x.shape[1], n_types=len(data.p_t),
            phi_dim=data.phi.shape[1], median_counts=data.median_counts,
            d_z=config.d_z, d_w=config.d_w, hidden=config.hidden,
            gat_dim=config.gat_dim, heads=config.heads,
            gat_sources=config.gat_sources,
            subtract_leak=config.subtract_leak,
            gat_sink=config.gat_sink,
            class_mean_prior=config.class_mean_prior,
            query=config.query,
            prior_type_free=config.prior_type_free,
            kappa_mode=config.kappa_mode,
            kappa_gene_share=(read_gene_share(config.kappa_gene_source,
                                              data.gene_names)
                              if config.kappa_mode == "gene" else None),
            kappa_ratio_mean=config.kappa_ratio_mean,
            phi_proj=config.phi_proj,
        ).to(self.device)
        self.covariances = None
        self.adversary = self.adversary_optimiser = None
        if config.alpha_a and config.invariance == "closed_form":
            self.covariances = TypeCovariances(
                len(data.p_t), config.d_z, data.v_block.shape[1],
                ema=config.cov_ema).to(self.device)
        elif config.alpha_a and config.invariance == "adversary":
            from discell.model.networks import Adversary

            if data.e_phi is None or data.phibar_t is None:
                raise ValueError("invariance='adversary' needs e_phi/phibar_t "
                                 "on the ModelData (assemble builds them)")
            # ablation (iv): the heads read the decoded composition rho_i
            # (G columns) instead of mu_z (d_z columns).
            adv_in = (config.d_z if config.adv_input == "mu_z"
                      else data.x.shape[1])
            # 8.17: K = adv_ensemble independent pairs, built in order from
            # the same RNG stream; K = 1 is the single Adversary as before
            members = [Adversary(adv_in, len(data.p_t), data.e_phi.shape[1],
                                 hidden=config.adv_hidden).to(self.device)
                       for _ in range(config.adv_ensemble)]
            self.adversary = (members[0] if len(members) == 1
                              else torch.nn.ModuleList(members))
            self.adversary_optimiser = torch.optim.Adam(
                self.adversary.parameters(), lr=config.adv_lr)
            self.ybar_t = torch.tensor(data.graph.ybar_t, device=self.device)
            self.phibar_t = torch.tensor(data.phibar_t, device=self.device)
        self.p_t = torch.tensor(data.p_t, device=self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=config.epochs)

        self.train_batches = [self._to_device(t) for t in data.train_tiles]
        self.val_batches = [self._to_device(t) for t in data.val_tiles]

        run_root = paths.dataset(config.dataset).root / "runs"
        self.run_dir = run_root / config.name()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.kappa_ratio_report is not None:
            (self.run_dir / f"kappa_{config.kappa_mode}.json").write_text(
                json.dumps({"ratio_mean_used": config.kappa_ratio_mean,
                            "ratio": self.kappa_ratio_report,
                            "area": self.density_report}, indent=1))
        if self.fp_report is not None:
            (self.run_dir / "fp_floor.json").write_text(
                json.dumps(self.fp_report, indent=1))

    def _to_device(self, tile: np.ndarray) -> dict:
        """One tile's tensors, resident on the device for the whole fit."""
        b = tile_batch(self.data.graph, tile)
        # Counts are small integers (max 856 on the deepest slide), so int16 is
        # exact and halves the resident footprint -- what lets a 1.16M-cell
        # slide fit a 24 GB card. _forward_kwargs casts a tile back to float32.
        # Under type_only the encoder never touches ring 2 (spec 4.5), so its
        # counts are not gathered to the device at all; type_z needs them as
        # GAT sources.
        n_resident = (len(b.nodes) if self.config.gat_sources == "type_z"
                      else b.n_context)
        counts = self.data.x[b.nodes[:n_resident]].toarray()
        assert counts.max() < 32768, "counts exceed int16; widen the resident dtype"
        dense = torch.tensor(counts.astype(np.int16), device=self.device)
        batch = dict(
            nodes=b.nodes,
            x=dense,
            t=torch.tensor(self.data.t[b.nodes], device=self.device),
            phi=torch.tensor(self.data.phi[b.nodes], device=self.device),
            isolated=torch.tensor(self.data.graph.isolated[b.nodes],
                                  device=self.device),
            gat_src=torch.tensor(b.gat_src, device=self.device),
            gat_dst=torch.tensor(b.gat_dst, device=self.device),
            leak_src=torch.tensor(b.leak_src, device=self.device),
            leak_dst=torch.tensor(b.leak_dst, device=self.device),
            leak_beta=torch.tensor(b.leak_beta, dtype=torch.float32,
                                   device=self.device),
            v=torch.tensor(self.data.v_block[b.nodes[:b.n_seeds]],
                           device=self.device),
            y_seed=torch.tensor(self.data.graph.y[b.nodes[:b.n_seeds]],
                                device=self.device),
            ephi_seed=None if self.data.e_phi is None else torch.tensor(
                self.data.e_phi[b.nodes[:b.n_seeds]], device=self.device),
            n_seeds=b.n_seeds, n_context=b.n_context,
        )
        if self.cell_area is not None:
            batch["area"] = torch.tensor(self.cell_area[b.nodes[:n_resident]],
                                         dtype=torch.float32, device=self.device)
        if self.fp_eta is not None:            # 8.15b: seeds only, (n, 1)
            batch["eta"] = torch.tensor(self.fp_eta[b.nodes[:b.n_seeds]],
                                        dtype=torch.float32,
                                        device=self.device)[:, None]
        return batch

    @staticmethod
    def _forward_kwargs(batch: dict) -> dict:
        keep = ("x", "t", "phi", "isolated", "gat_src", "gat_dst",
                "leak_src", "leak_dst", "leak_beta", "n_seeds", "n_context")
        out = {k: batch[k] for k in keep}
        out["x"] = batch["x"].float()          # resident int16 -> float32 per tile
        if "area" in batch:                    # the R12 density form only
            out["area"] = batch["area"]
        if "eta" in batch:                     # the 8.15b floor only
            out["eta"] = batch["eta"]
        return out

    # -- steps -------------------------------------------------------------

    def _step(self, batch: dict):
        """One model update, then the adversary's update(s) when escalated."""
        import dataclasses as dc

        config = self.config
        kwargs = self._forward_kwargs(batch)
        fwd = self.model(**kwargs, kappa=config.kappa)
        n = batch["n_seeds"]
        weights = config.weights(self.epoch)
        extras: dict = {}
        if self.adversary is not None:
            adv_feat = (fwd.mu_z[:n] if config.adv_input == "mu_z"
                        else fwd.log_rho[:n])
            adv = ensemble_adversary_terms(
                self._adversary_heads(), adv_feat, batch["t"][:n],
                batch["y_seed"], batch["ephi_seed"], self.ybar_t,
                self.phibar_t, comp_weight=config.adv_comp_weight)
            terms = discell_loss(fwd, kwargs["x"][:n], batch["t"][:n],
                                 weights=dc.replace(weights, alpha_a=0.0))
            loss = terms.loss + weights.alpha_a * adv.encoder_term
            terms.penalty = float(adv.encoder_term.detach())
            extras = {"adv_excess_y": adv.excess_y,
                      "adv_excess_phi": adv.excess_phi}
        else:
            terms = discell_loss(fwd, kwargs["x"][:n], batch["t"][:n],
                                 weights=weights, v=batch["v"],
                                 covariances=self.covariances, p_t=self.p_t)
            loss = terms.loss
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       config.grad_clip)
        self.optimiser.step()

        if self.adversary is not None:
            head_loss = self._head_steps(adv_feat.detach(), batch["t"][:n],
                                         batch["y_seed"], batch["ephi_seed"])
            extras["adv_head_loss"] = float(head_loss.detach())
        return terms, extras

    def _adversary_heads(self) -> list:
        """The head pairs: one, or the adv_ensemble members (8.17)."""
        return ([self.adversary] if self.config.adv_ensemble == 1
                else list(self.adversary))

    def _head_steps(self, z_frozen: torch.Tensor, t: torch.Tensor,
                    y: torch.Tensor, e_phi: torch.Tensor) -> torch.Tensor:
        """The adversary's ``adv_steps`` updates on ``sg`` features; returns
        the last head loss -- the members' sum, the composition CE times
        ``adv_comp_weight`` (8.17; K = 1 and weight 1 are the pinned loop)."""
        from discell.model.networks import soft_cross_entropy

        c = self.config.adv_comp_weight
        heads = self._adversary_heads()
        for _ in range(self.config.adv_steps):
            self.adversary_optimiser.zero_grad()
            losses = []
            for head in heads:
                log_y, log_phi = head(z_frozen, t)
                losses.append((c * soft_cross_entropy(y, log_y)
                               + soft_cross_entropy(e_phi, log_phi)).mean())
            head_loss = (losses[0] if len(losses) == 1
                         else torch.stack(losses).sum())
            head_loss.backward()
            self.adversary_optimiser.step()
        return head_loss

    def _decode_seeds(self, fwd, z_seeds: torch.Tensor, n: int,
                      w_seeds: torch.Tensor | None = None) -> torch.Tensor:
        """``log p`` of the seeds with *z_seeds* (and optionally *w_seeds*) in
        place of their own z (and w).

        The foreign influx and kappa stay as the forward pass left them, so at
        ``z_seeds = fwd.mu_z[:n]`` and ``w_seeds`` unset or ``fwd.mu_w[:n]``
        (sample=False) this is ``fwd.log_p``.
        """
        log_rho = self.model.log_rho(
            z_seeds, fwd.mu_w[:n] if w_seeds is None else w_seeds)
        return leakage_mix(log_rho.exp(), fwd.rho_bar, fwd.kappa_eff, fwd.eta)

    @torch.no_grad()
    def _sweep(self, batches: list[dict], want_log_p: bool = False,
               z_bar: torch.Tensor | None = None) -> dict:
        """Collect per-seed arrays over *batches* in eval mode.

        *z_bar* (K, d_z): also decode every seed with its type's mean z in
        place of its own (``log_p_typemean``) -- the spec 7.10 degeneracy gap.
        """
        self.model.eval()
        out = {k: [] for k in ("nodes", "mu_z", "mu_w", "prior_w", "c",
                               "kl_w", "kl_z", "log_p", "log_p_typemean")}
        for batch in batches:
            fwd = self.model(**self._forward_kwargs(batch),
                             kappa=self.config.kappa, sample=False)
            n = batch["n_seeds"]
            out["nodes"].append(batch["nodes"][:n])
            out["mu_z"].append(fwd.mu_z[:n].cpu().numpy())
            out["mu_w"].append(fwd.mu_w[:n].cpu().numpy())
            out["prior_w"].append(fwd.prior_mean_w[:n].cpu().numpy())
            out["c"].append(fwd.c[:n].cpu().numpy())

            per_dim = 0.5 * (-fwd.logvar_w[:n]
                             + fwd.logvar_w[:n].exp()
                             + (fwd.mu_w[:n] - fwd.prior_mean_w[:n]) ** 2 - 1.0)
            out["kl_w"].append(per_dim.cpu().numpy())
            out["kl_z"].append((0.5 * (-fwd.logvar_z[:n] + fwd.logvar_z[:n].exp()
                                       + fwd.mu_z[:n] ** 2 - 1.0)
                                ).sum(dim=-1).cpu().numpy())
            if want_log_p:
                out["log_p"].append(fwd.log_p.cpu().numpy())
            if z_bar is not None:
                out["log_p_typemean"].append(self._decode_seeds(
                    fwd, z_bar[batch["t"][:n]], n).cpu().numpy())
        self.model.train()
        return {k: np.concatenate(v) if v else None for k, v in out.items()}

    # -- evaluation --------------------------------------------------------

    def evaluate(self) -> dict:
        train = self._sweep(self.train_batches)
        t_train = self.data.t[train["nodes"]]
        # type mean of mu_z over TRAINING cells: the "z is just t" decode
        z_bar = M.type_means(train["mu_z"], t_train, len(self.data.p_t))
        val = self._sweep(self.val_batches, want_log_p=True,
                          z_bar=torch.tensor(z_bar, device=self.device))

        x_val = np.vstack([self.data.x[b["nodes"][:b["n_seeds"]]].toarray()
                           for b in self.val_batches])
        rows_all = np.concatenate([train["nodes"], val["nodes"]])
        z_all = np.vstack([train["mu_z"], val["mu_z"]])
        c_all = np.vstack([train["c"], val["c"]])
        t_all = self.data.t[rows_all]

        train_mask = np.zeros(len(rows_all), dtype=bool)
        train_mask[:len(train["nodes"])] = True
        probe = M.probe_delta_ce(z_all, t_all, self.data.v_block[rows_all],
                                 self.data.vbar_t, train_mask, ~train_mask,
                                 seed=self.config.seed)
        cycle = None
        if self.data.cycle is not None:
            cyc = self.data.cycle
            type_names = [str(n) for n in self.data.type_names]
            cycling = np.array([g for g in cyc["cycling_types"]
                                if "nassigned" not in type_names[g]][:4])
            scores = np.stack([cyc["s_score"], cyc["g2m_score"]],
                              axis=1)[rows_all]
            w_all = np.vstack([train["mu_w"], val["mu_w"]])
            cycle = {"types": cycling.tolist(),
                     "reliability": cyc.get("reliability"),
                     "z": M.cycle_r2(z_all, t_all, scores, cycling,
                                     train_mask, ~train_mask,
                                     seed=self.config.seed),
                     "w": M.cycle_r2(w_all, t_all, scores, cycling,
                                     train_mask, ~train_mask,
                                     seed=self.config.seed),
                     # the 50-PC LINEAR EXPRESSION REFERENCE (renamed from
                     # "ceiling", doc-11 flag: z beats it ~2x, so it is a
                     # linear reference line, never a bound)
                     "linear_ref": M.cycle_r2(cyc["x_pcs"][rows_all], t_all,
                                           scores, cycling, train_mask,
                                           ~train_mask,
                                           seed=self.config.seed),
                     # the l-baseline (doc 08 conventions): depth alone --
                     # G2/M cells carry ~2x mRNA, so l is not trivially cold
                     # even for an intrinsic target
                     "lbaseline": M.cycle_r2(
                         np.log(self.data.totals[rows_all].clip(min=1.0)
                                )[:, None].astype(np.float64),
                         t_all, scores, cycling, train_mask, ~train_mask,
                         seed=self.config.seed)}
        # spec 7.10 degeneracy pair: I(z;t)/H(t) + within-type variance, and
        # the held-out recon lost when each cell's z is its type's mean z
        recon = M.held_out_reconstruction(x_val, val["log_p"])
        recon_typemean = M.held_out_reconstruction(x_val, val["log_p_typemean"])
        recon_gap = {
            "recon": recon, "recon_typemean_z": recon_typemean,
            "gap": recon - recon_typemean,
            "recon_type_profile": M.type_profile_reconstruction(
                self.data.x[train["nodes"]], t_train, x_val,
                self.data.t[val["nodes"]])}
        return {
            "cycle": cycle,
            "recon_val": recon,
            "nmi": M.z_type_nmi(z_all, t_all, seed=self.config.seed),
            "mirror": M.mirror_r2(z_all, c_all, t_all, seed=self.config.seed),
            "probe": probe,
            "degeneracy": M.type_degeneracy(z_all, t_all, train_mask,
                                            ~train_mask, seed=self.config.seed),
            "recon_gap": recon_gap,
            "kl_w_per_dim": val["kl_w"].mean(axis=0).tolist(),
            "collected": {"rows": rows_all, "z": z_all,
                          "w": np.vstack([train["mu_w"], val["mu_w"]]),
                          "prior_w": np.vstack([train["prior_w"],
                                                val["prior_w"]]),
                          "kl_w": np.vstack([train["kl_w"], val["kl_w"]]),
                          "kl_z": np.concatenate([train["kl_z"], val["kl_z"]])},
        }

    # -- figures -----------------------------------------------------------

    #: preferred types for the per-type panels; falls back to the most
    #: abundant labelled types when a name is absent from this slide.
    CANONICAL_TYPES = ("Tumor Cells", "Macrophages", "T and NK Cells",
                       "Tumor Associated Fibroblasts")

    def _umap(self, points: np.ndarray) -> np.ndarray:
        import umap

        return umap.UMAP(n_neighbors=15, min_dist=0.1,
                         random_state=self.config.seed).fit_transform(points)

    def _panel_types(self) -> list[int]:
        """Canonical types first, filled to ``panel_types`` by abundance."""
        names = [str(n) for n in self.data.type_names]
        chosen = [names.index(n) for n in self.CANONICAL_TYPES if n in names]
        order = np.argsort(-self.data.p_t)
        chosen += [int(g) for g in order
                   if g not in chosen and "nassigned" not in names[g]]
        return chosen[:self.config.panel_types]

    def figures(self, writer, collected: dict, step: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from discell.model.metrics import principal_curve

        rows, z, w = collected["rows"], collected["z"], collected["w"]
        deviation = w - collected["prior_w"]      # what the cell did beyond
        anomaly = collected["kl_w"].sum(axis=1)   # its niche's expectation
        rng = self.figure_rng
        names = [str(n) for n in self.data.type_names]
        t_rows = self.data.t[rows]
        palette = plt.get_cmap("tab20")
        log_w = np.sign(w) * np.log1p(np.abs(w))     # signed log: w is signed
        connected = self.data.graph.degrees[rows] > 0

        # one global subsample; one member set per panel type; every UMAP
        # fitted at most once per figure event via the cache
        pick = rng.choice(len(rows), min(len(rows), 12_000), replace=False)
        umap_cache: dict = {}

        def type_members(g: int) -> np.ndarray:
            members = np.flatnonzero((t_rows == g) & connected)
            member_rng = np.random.default_rng([self.config.seed, 7, g])
            if len(members) > 8_000:
                members = member_rng.choice(members, 8_000, replace=False)
            return members

        members_of = {g: type_members(g) for g in self._panel_types()}

        def project(points: np.ndarray, method: str, key=None) -> np.ndarray:
            if method == "umap":
                if key is not None and key in umap_cache:
                    return umap_cache[key]
                coords = self._umap(points)
                if key is not None:
                    umap_cache[key] = coords
                return coords
            from sklearn.decomposition import PCA

            return PCA(2, random_state=self.config.seed).fit_transform(points)

        # -- spatial grids, one panel per latent dimension -------------------
        def spatial_grid(values: np.ndarray, tag: str) -> None:
            grid_pick = rng.choice(len(rows), min(len(rows), 40_000),
                                   replace=False)
            position = self.data.positions[rows[grid_pick]]
            dims = values.shape[1]
            cols = min(dims, 5)
            nrows = (dims + cols - 1) // cols
            fig, axes = plt.subplots(nrows, cols,
                                     figsize=(3.0 * cols, 3.0 * nrows))
            for k, ax in enumerate(np.ravel(axes)):
                if k >= dims:
                    ax.axis("off")
                    continue
                lim = np.percentile(np.abs(values[grid_pick, k]), 98) or 1.0
                ax.scatter(position[:, 0], position[:, 1],
                           c=values[grid_pick, k], s=0.5, cmap="RdBu_r",
                           vmin=-lim, vmax=lim, rasterized=True)
                ax.set_title(f"{tag}[{k}]", fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"spatial {tag}, posterior mean", fontsize=10)
            writer.add_figure(f"figures/{tag}_spatial", fig, step)

        # -- per-type ||w||: the sweep's headline ranking, live per run ------
        def w_norm_boxplot() -> None:
            norms = np.linalg.norm(w, axis=1)
            groups = [g for g in range(len(self.p_t))
                      if (t_rows == g).sum() >= 30]
            groups.sort(key=lambda g: -norms[t_rows == g].mean())
            fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(groups)), 4.2))
            box = ax.boxplot([norms[t_rows == g] for g in groups],
                             showmeans=True, showfliers=False,
                             patch_artist=True)
            for patch, g in zip(box["boxes"], groups):
                patch.set_facecolor(palette(g % 20)); patch.set_alpha(0.7)
            ax.set_xticklabels([names[g][:18] for g in groups],
                               rotation=45, ha="right", fontsize=6)
            for k, g in enumerate(groups):
                ax.text(k + 1, norms[t_rows == g].mean(),
                        f"{norms[t_rows == g].mean():.1f}",
                        ha="center", va="bottom", fontsize=5)
            ax.set_ylabel("||w||", fontsize=8)
            ax.set_title("spatial responsiveness by type, ordered by mean",
                         fontsize=9)
            fig.tight_layout()
            writer.add_figure("figures/w_norm_by_type", fig, step)

        # -- one figure per VARIABLE: rows = all cells then each panel type;
        # columns = UMAP | PCA (w additionally carries a log-w UMAP column).
        # The all-cells row is coloured by type; type rows by dominant
        # neighbour (z should mix, w should organise).
        def variable_panels(values: np.ndarray, tag: str,
                            extra: tuple[np.ndarray, str] | None = None) -> None:
            columns = [(values, tag, "umap"), (values, tag, "pca")]
            if extra is not None:
                columns.append((extra[0], extra[1], "umap"))
            groups = list(members_of)
            dominant = self.data.graph.y[rows].argmax(axis=1)
            n_rows = 1 + len(groups)
            fig, axes = plt.subplots(n_rows, len(columns),
                                     figsize=(4.0 * len(columns),
                                              3.4 * n_rows))
            axes = np.atleast_2d(axes)
            for c, (source, source_tag, method) in enumerate(columns):
                coords = project(source[pick], method,
                                 key=(source_tag, "global")
                                 if method == "umap" else None)
                ax = axes[0, c]
                for g in range(len(self.p_t)):
                    sel = t_rows[pick] == g
                    if sel.any():
                        ax.scatter(coords[sel, 0], coords[sel, 1], s=0.6,
                                   color=palette(g % 20),
                                   label=names[g][:16], rasterized=True)
                ax.set_title(f"{source_tag} {method.upper()} | all cells | "
                             "colour = cell type", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                if c == len(columns) - 1:
                    ax.legend(fontsize=4, markerscale=6, ncol=2)
            for r, g in enumerate(groups, start=1):
                members = members_of[g]
                for c, (source, source_tag, method) in enumerate(columns):
                    ax = axes[r, c]
                    if len(members) < 50:
                        ax.axis("off")
                        continue
                    coords = project(source[members], method,
                                     key=(source_tag, g)
                                     if method == "umap" else None)
                    for nb in np.unique(dominant[members]):
                        sel = dominant[members] == nb
                        ax.scatter(coords[sel, 0], coords[sel, 1], s=0.8,
                                   color=palette(int(nb) % 20),
                                   label=names[nb][:14], rasterized=True)
                    # one type per row: the colour is NOT the cell's type
                    # (all cells here share it) but the type that dominates
                    # its neighbourhood -- z should mix, w should organise
                    ax.set_title(f"{source_tag} {method.upper()} | "
                                 f"{names[g][:22]} | colour = dominant "
                                 "NEIGHBOUR type", fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
                    if c == len(columns) - 1:
                        ax.legend(fontsize=4, markerscale=5, ncol=1,
                                  title="dominant neighbour type",
                                  title_fontsize=6)
            fig.suptitle(f"{tag}: all cells by type, then within-type by "
                         "dominant neighbour", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            writer.add_figure(f"figures/{tag}_panels", fig, step)

        # -- z within cycling AND canonical types, coloured by Tirosh phase --
        def z_cycle_panels() -> None:
            """Hard labels default to G1 at Xenium depth, so the quantitative
            read is the continuous-score probe (val/cycle_r2_*); this is the
            picture, in both projections."""
            cyc = self.data.cycle
            phase_colour = {0: "#b8b8b8", 1: "#e6772e", 2: "#7d3ac1"}
            phase_name = {0: "G1", 1: "S", 2: "G2M"}
            wanted = list(dict.fromkeys(
                [g for g in cyc["cycling_types"] if "nassigned" not in names[g]]
                [:4] + list(members_of)))
            groups = [g for g in wanted
                      if ((t_rows == g) & connected).sum() >= 200][:6]
            fig, axes = plt.subplots(len(groups), 2,
                                     figsize=(8.0, 3.4 * len(groups)))
            for r, g in enumerate(groups):
                members = members_of.get(g)
                if members is None:
                    members = type_members(g)
                phase = cyc["phase"][rows[members]]
                for c, method in enumerate(("umap", "pca")):
                    ax = axes[r, c] if len(groups) > 1 else axes[c]
                    coords = project(z[members], method, key=("z", g))
                    for p in (0, 1, 2):
                        sel = phase == p
                        if sel.any():
                            ax.scatter(coords[sel, 0], coords[sel, 1], s=0.8,
                                       color=phase_colour[p],
                                       label=f"{phase_name[p]} ({sel.sum():,})",
                                       rasterized=True)
                    ax.set_title(f"z {method.upper()} | {names[g][:22]}",
                                 fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
                    if c == 1:
                        ax.legend(fontsize=5, markerscale=6)
            fig.suptitle("z within cycling + canonical types, by Tirosh phase "
                         "(hard label defaults to G1 at this depth)",
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure("figures/z_cell_cycle", fig, step)

        # -- pseudotime: curve fitted in FULL latent space, rendered per
        # projection (curve points mapped through their nearest cells), so
        # both projection figures share one pseudotime and differ only in
        # layout. Direction is arbitrary; reads are ordering and spatial
        # organisation, never sign.
        trajectory_cache: dict = {}

        def fitted_trajectory(values: np.ndarray, tag: str, scope) -> tuple:
            key = (tag, scope)
            if key not in trajectory_cache:
                members = pick if scope == "global" else members_of[scope]
                trajectory_cache[key] = principal_curve(values[members])
            return trajectory_cache[key]

        def trajectories(values: np.ndarray, tag: str, method: str) -> None:
            from scipy.spatial import cKDTree

            scopes = [("all cells", "global", pick)]
            scopes += [(names[g], g, members_of[g]) for g in members_of]
            fig, axes = plt.subplots(len(scopes), 2,
                                     figsize=(8.0, 3.4 * len(scopes)))
            for r, (label, scope, members) in enumerate(scopes):
                ax_u, ax_s = axes[r]
                if len(members) < 50:
                    ax_u.axis("off"); ax_s.axis("off")
                    continue
                pseudotime, curve = fitted_trajectory(values, tag, scope)
                coords = project(values[members], method,
                                 key=(tag, scope) if method == "umap" else None)
                _, nearest = cKDTree(values[members]).query(
                    curve, k=min(20, len(members)))
                curve_2d = coords[nearest].mean(axis=1)
                ax_u.scatter(coords[:, 0], coords[:, 1], c=pseudotime, s=0.8,
                             cmap="viridis", rasterized=True)
                ax_u.plot(curve_2d[:, 0], curve_2d[:, 1], color="black", lw=1.2)
                ax_u.set_title(f"{tag} {method.upper()} + curve (fit in "
                               f"{values.shape[1]}-D) | {label[:20]}",
                               fontsize=8)
                position = self.data.positions[rows[members]]
                ax_s.scatter(position[:, 0], position[:, 1], c=pseudotime,
                             s=0.8, cmap="viridis", rasterized=True)
                ax_s.set_aspect("equal")
                ax_s.set_title("pseudotime in tissue coordinates", fontsize=8)
                for ax in (ax_u, ax_s):
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{tag} pseudotime along the full-space principal "
                         "curve (direction arbitrary)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure(f"figures/{tag}_trajectories_{method}", fig, step)

        def w_spatial_extended() -> None:
            """w per dim, its deviation from the prior per dim, then ||w|| and
            the per-cell KL anomaly (spec 4.3: how far this cell deviates from
            its expected response)."""
            grid_pick = rng.choice(len(rows), min(len(rows), 40_000),
                                   replace=False)
            position = self.data.positions[rows[grid_pick]]
            d_w = w.shape[1]
            panels = ([(w[:, k], f"w[{k}]", "RdBu_r", True)
                       for k in range(d_w)]
                      + [(deviation[:, k], f"w[{k}] − m_ψ[{k}]", "RdBu_r", True)
                         for k in range(d_w)]
                      + [(np.linalg.norm(w, axis=1), "‖w‖", "viridis", False),
                         (anomaly, "KL(q(w)‖p)  anomaly", "magma", False)])
            cols = 5
            nrows = (len(panels) + cols - 1) // cols
            fig, axes = plt.subplots(nrows, cols,
                                     figsize=(3.0 * cols, 3.0 * nrows))
            for k, ax in enumerate(np.ravel(axes)):
                if k >= len(panels):
                    ax.axis("off")
                    continue
                values, title, cmap, diverging = panels[k]
                sample = values[grid_pick]
                if diverging:
                    lim = np.percentile(np.abs(sample), 98) or 1.0
                    kwargs = dict(cmap=cmap, vmin=-lim, vmax=lim)
                else:
                    kwargs = dict(cmap=cmap, vmin=np.percentile(sample, 2),
                                  vmax=np.percentile(sample, 98))
                ax.scatter(position[:, 0], position[:, 1], c=sample, s=0.5,
                           rasterized=True, **kwargs)
                ax.set_title(title, fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("spatial w: posterior mean, deviation from the "
                         "prior, magnitude, anomaly", fontsize=10)
            writer.add_figure("figures/w_spatial", fig, step)

        def z_cycle_projection() -> None:
            """The architect's instrument for a low-variance axis: project z
            onto the probe's own directions (z·beta_S vs z·beta_G2M -- expect a
            G1 blob at the origin with an arc through S into G2M), and colour
            the within-type UMAP by kNN-smoothed scores, since the per-cell
            score is mostly noise. UMAP/PCA display dominant variance, not
            retained information; this displays the retained axis directly."""
            from sklearn.neighbors import NearestNeighbors

            cyc = self.data.cycle
            phase_colour = {0: "#b8b8b8", 1: "#e6772e", 2: "#7d3ac1"}
            groups = [g for g in cyc["cycling_types"]
                      if "nassigned" not in names[g]
                      and ((t_rows == g) & connected).sum() >= 300][:4]
            scores = np.stack([cyc["s_score"], cyc["g2m_score"]],
                              axis=1)[rows]
            fig, axes = plt.subplots(len(groups), 3,
                                     figsize=(11.5, 3.4 * len(groups)))
            axes = np.atleast_2d(axes)
            for r, g in enumerate(groups):
                members = members_of.get(g)
                if members is None:
                    members = type_members(g)
                z_c = z[members] - z[members].mean(axis=0)
                target = scores[members] - scores[members].mean(axis=0)
                gram = z_c.T @ z_c + 1e-3 * np.eye(z_c.shape[1])
                beta = np.linalg.solve(gram, z_c.T @ target)   # (d_z, 2)
                projected = z_c @ beta
                phase = cyc["phase"][rows[members]]
                ax = axes[r, 0]
                for p in (0, 1, 2):
                    sel = phase == p
                    if sel.any():
                        ax.scatter(projected[sel, 0], projected[sel, 1],
                                   s=0.9, color=phase_colour[p],
                                   label=("G1", "S", "G2M")[p],
                                   rasterized=True)
                ax.set_title(f"z·β_S vs z·β_G2M | {names[g][:20]}", fontsize=8)
                ax.legend(fontsize=5, markerscale=6)
                neighbours = NearestNeighbors(n_neighbors=min(30, len(members) - 1)
                                              ).fit(z[members])
                nearest = neighbours.kneighbors(return_distance=False)
                smoothed = scores[members][nearest].mean(axis=1)
                coords = project(z[members], "umap", key=("z", g))
                for c, (label, values) in enumerate(
                        (("S (kNN-smoothed)", smoothed[:, 0]),
                         ("G2M (kNN-smoothed)", smoothed[:, 1])), start=1):
                    ax = axes[r, c]
                    lim = (np.percentile(values, 2), np.percentile(values, 98))
                    ax.scatter(coords[:, 0], coords[:, 1], c=values, s=0.9,
                               cmap="magma", vmin=lim[0], vmax=lim[1],
                               rasterized=True)
                    ax.set_title(f"z UMAP, {label}", fontsize=8)
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])
            rel = cyc.get("reliability") or {}
            fig.suptitle("cycle along the probe's own axes (split-half score "
                         f"reliability: S {rel.get('s', float('nan')):.2f}, "
                         f"G2M {rel.get('g2m', float('nan')):.2f})", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            writer.add_figure("figures/z_cycle_projection", fig, step)

        # -- per-cell KL to the priors on the tissue: all cells, then each
        # panel type on its own cells. KL_z is against N(0, I) -- how much
        # the cell's counts pin its state; KL_w is against m_psi(c, t) -- how
        # far the cell deviates from its niche's expected response.
        def kl_spatial() -> None:
            grid_pick = rng.choice(len(rows), min(len(rows), 40_000),
                                   replace=False)
            scopes = [("all cells", grid_pick)]
            scopes += [(names[g], members_of[g]) for g in members_of]
            columns = ((collected["kl_z"], "KL(q(z) ‖ N(0,I))", "viridis"),
                       (anomaly, "KL(q(w) ‖ m_ψ(c,t))", "magma"))
            fig, axes = plt.subplots(len(scopes), 2,
                                     figsize=(6.4, 3.0 * len(scopes)))
            for r, (label, members) in enumerate(scopes):
                for c, (values, title, cmap) in enumerate(columns):
                    ax = axes[r, c]
                    if len(members) < 50:
                        ax.axis("off")
                        continue
                    sample = values[members]
                    position = self.data.positions[rows[members]]
                    points = ax.scatter(position[:, 0], position[:, 1], c=sample,
                                        s=0.5, cmap=cmap,
                                        vmin=np.percentile(sample, 2),
                                        vmax=np.percentile(sample, 98),
                                        rasterized=True)
                    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.02
                                 ).ax.tick_params(labelsize=5)
                    ax.set_title(f"{title} | {label[:20]} | median "
                                 f"{np.median(sample):.3f}", fontsize=7)
                    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("per-cell KL to the priors in tissue coordinates "
                         "(colour scale per panel, 2-98%)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            writer.add_figure("figures/kl_spatial", fig, step)

        w_spatial_extended()
        kl_spatial()
        spatial_grid(z, "z")
        w_norm_boxplot()
        variable_panels(z, "z")
        variable_panels(w, "w", extra=(log_w, "logw"))
        trajectories(w, "w", "umap")
        trajectories(w, "w", "pca")
        trajectories(z, "z", "umap")
        trajectories(z, "z", "pca")
        # the joint state: [z, w] standardised per dimension, so neither
        # block wins by scale (w's top dim has ~25x the variance of a z dim)
        zw = np.hstack([z, w])
        zw = (zw - zw.mean(axis=0)) / (zw.std(axis=0) + 1e-6)
        trajectories(zw, "zw_std", "umap")
        trajectories(zw, "zw_std", "pca")
        if self.data.cycle is not None:
            z_cycle_panels()
            z_cycle_projection()

        loadings = self.model.B.weight.detach().cpu().numpy()   # (G, d_w)
        fig, axes = plt.subplots(1, self.config.d_w,
                                 figsize=(3.0 * self.config.d_w, 2.6))
        for k, ax in enumerate(np.atleast_1d(axes)):
            top = np.argsort(-np.abs(loadings[:, k]))[:12]
            ax.barh(range(len(top)), loadings[top, k], height=0.7)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([str(g) for g in top], fontsize=5)
            ax.set_title(f"B[:, {k}] top genes", fontsize=8)
        writer.add_figure("figures/B_loadings", fig, step)
        plt.close("all")

    # -- the fit -----------------------------------------------------------

    def time_only(self, n_epochs: int) -> dict:
        """The timing mode (``--time-only``); see :func:`_time_only`."""
        return _time_only(self, n_epochs)

    def fit(self) -> dict:
        from torch.utils.tensorboard import SummaryWriter

        config = self.config
        writer = SummaryWriter(self.run_dir)
        (self.run_dir / "config.json").write_text(
            json.dumps({**dataclasses.asdict(config),
                        "git": _git_state()}, indent=2))

        history_path = self.run_dir / "history.jsonl"
        history_path.unlink(missing_ok=True)
        best = {"recon_val": -np.inf, "nmi": 0.0, "epoch": -1}
        nmi_max, stale, step = 0.0, 0, 0
        dead_w = False
        last_epoch = -1
        started = time.time()
        for epoch in range(config.epochs):
            self.epoch = epoch
            last_epoch = epoch
            for index in self.rng.permutation(len(self.train_batches)):
                terms, extras = self._step(self.train_batches[index])
                if step % 20 == 0:
                    for key, value in terms.scalars().items():
                        writer.add_scalar(f"train/{key}", value, step)
                    for key, value in extras.items():
                        writer.add_scalar(f"train/{key}", value, step)
                    if terms.penalty_info:
                        writer.add_scalar(
                            "train/penalty_excluded_fraction",
                            terms.penalty_info["excluded_fraction"], step)
                step += 1
            self.schedule.step()

            if (epoch + 1) % config.eval_every:
                continue
            report = self.evaluate()
            # During the alpha_w warm-up the objective is NOT the pinned one
            # (2026-09-23): the KL_w term is scaled down, so those epochs are
            # optimising a different loss. They must not supply the checkpoint,
            # must not start the patience counter, and must not set the NMI
            # guard's reference -- otherwise `best` comes from an era whose
            # objective the run does not end on. No warm-up = no change. The
            # same gate holds for the warm-up on both KL terms (8.9b).
            warming = epoch < config.warmup_epochs
            if not warming:
                nmi_max = max(nmi_max, report["nmi"])
            writer.add_scalar("val/recon", report["recon_val"], step)
            writer.add_scalar("val/nmi", report["nmi"], step)
            writer.add_scalar("val/mirror_r2", report["mirror"]["r2"], step)
            writer.add_scalar("val/mirror_r2_permuted",
                              report["mirror"]["r2_permuted"], step)
            writer.add_scalar("val/probe_delta_ce",
                              report["probe"]["delta_ce"], step)
            writer.add_scalar("val/probe_noise_floor",
                              report["probe"]["noise_floor"], step)
            writer.add_scalar("val/degeneracy_mi_ratio",
                              report["degeneracy"]["mi_ratio"], step)
            writer.add_scalar("val/degeneracy_within_var_fraction",
                              report["degeneracy"]["within_var_fraction"], step)
            writer.add_scalar("val/recon_gap_typemean_z",
                              report["recon_gap"]["gap"], step)
            if report.get("cycle"):
                type_names = [str(n) for n in self.data.type_names]
                for latent in ("z", "w", "linear_ref", "lbaseline"):
                    entry = report["cycle"][latent]
                    writer.add_scalar(f"val/cycle_r2_{latent}",
                                      entry["r2_mean_types"], step)
                    writer.add_scalar(f"val/cycle_r2_{latent}_pooled",
                                      entry["r2_pooled"], step)
                    writer.add_scalar(f"val/cycle_r2_{latent}_permuted",
                                      entry["r2_permuted"], step)
                    for g, value in entry["by_type"].items():
                        writer.add_scalar(
                            f"val/cycle_r2_{latent}_types/{type_names[g][:24]}",
                            value, step)
            for k, value in enumerate(report["kl_w_per_dim"]):
                writer.add_scalar(f"val/kl_w_dim{k}", value, step)
            if kl_w_is_dead(report["kl_w_per_dim"], epoch):
                dead_w = True
                log.warning("dead context channel: KL_w ≈ 0 at epoch %d "
                            "(sum KL_w = %.3g over %d dims)", epoch,
                            float(np.sum(report["kl_w_per_dim"])),
                            len(report["kl_w_per_dim"]))
            with history_path.open("a") as sink:
                sink.write(json.dumps(
                    {"epoch": epoch, "step": step,
                     "alpha_z_eff": config.alpha_z_at(epoch),
                     "alpha_w_eff": config.alpha_w_at(epoch),
                     **{k: v for k, v in report.items() if k != "collected"}},
                    default=float) + "\n")
            log.info("epoch %d  recon %.4f  nmi %.3f  mirror %.3f  dCE %.4f  [%.0fs]",
                     epoch, report["recon_val"], report["nmi"],
                     report["mirror"]["r2"], report["probe"]["delta_ce"],
                     time.time() - started)

            if (epoch + 1) % config.figures_every == 0:
                self.figures(writer, report["collected"], step)

            if warming:
                log.info("epoch %d inside the KL warm-up (%d epochs): "
                         "not eligible for best, patience not started",
                         epoch, config.warmup_epochs)
                continue
            improved = report["recon_val"] > best["recon_val"]
            guarded = report["nmi"] >= config.nmi_guard * nmi_max
            if improved and guarded:
                best = {"recon_val": report["recon_val"], "nmi": report["nmi"],
                        "epoch": epoch,
                        "degeneracy": report["degeneracy"],
                        "recon_gap": report["recon_gap"]}
                stale = 0
                covariance_state = None
                if self.covariances is not None:
                    covariance_state = {
                        name: getattr(self.covariances, name).cpu()
                        for name in ("mean", "second", "batch_count", "seen")}
                torch.save({"model": self.model.state_dict(),
                            "covariances": covariance_state,
                            "config": dataclasses.asdict(config),
                            "epoch": epoch}, self.run_dir / "best.pt")
            else:
                stale += 1
                if stale >= max(1, config.patience // config.eval_every):
                    log.info("early stop at epoch %d (best %d)", epoch, best["epoch"])
                    break

        # `final` must describe the checkpoint that ships, not the model
        # `patience` epochs past it (review R26, 2026-09-24): until then the
        # closing evaluation scored the in-memory weights, so every read of
        # metrics.json["final"] described a model nobody uses. Restoring into
        # self.model also puts callers that keep using this trainer after fit()
        # (calibrate._short_fit) on the accepted weights.
        if best["epoch"] >= 0:
            payload = torch.load(self.run_dir / "best.pt", map_location=self.device,
                                 weights_only=False)
            self.model.load_state_dict(payload["model"])
            if self.covariances is not None and payload["covariances"] is not None:
                for name, value in payload["covariances"].items():
                    setattr(self.covariances, name,
                            value.to(self.covariances.mean.device))
        final = self.evaluate()
        final.pop("collected")
        summary = {"best": best, "final": final,
                   # which epoch `final` describes; absent from runs made before
                   # the R26 fix, whose `final` is the last trained epoch
                   "final_epoch": best["epoch"] if best["epoch"] >= 0 else last_epoch,
                   "last_epoch": last_epoch,
                   "dead_w_channel": dead_w,
                   "minutes": (time.time() - started) / 60}
        (self.run_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str))
        writer.close()
        log.info("run %s: best recon %.4f (epoch %d), NMI %.3f, %.1f min",
                 config.name(), best["recon_val"], best["epoch"], best["nmi"],
                 summary["minutes"])
        return summary


def _peak_mib(device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def _time_only(trainer: "Trainer", n_epochs: int) -> dict:
    """Pure-training timing (devlog "Metrics package", item 4, 2026-09-25).

    *n_epochs* epochs of ``_step`` over the shuffled training tiles with no
    evaluation between them (the fit loop minus its evaluation, checkpoint
    and figure work), timed per epoch with the device synchronised; then ONE
    ``evaluate()``, timed on its own. Peak allocated GPU memory is read for
    the training phase (reset after model and tiles are resident, so it is
    the step's working set on top of the resident slide) and for the
    evaluation; the resident footprint before the first step is reported
    beside them. Nothing is checkpointed; the result is also written to the
    run directory as ``timing.json``.
    """
    device = trainer.device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device) / 2 ** 20
        torch.cuda.reset_peak_memory_stats(device)
    epoch_s = []
    for epoch in range(n_epochs):
        trainer.epoch = epoch
        if cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for index in trainer.rng.permutation(len(trainer.train_batches)):
            trainer._step(trainer.train_batches[index])
        trainer.schedule.step()
        if cuda:
            torch.cuda.synchronize(device)
        epoch_s.append(time.perf_counter() - t0)
        log.info("time-only epoch %d: %.2f s", epoch, epoch_s[-1])
    train_peak = _peak_mib(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    trainer.evaluate()
    if cuda:
        torch.cuda.synchronize(device)
    eval_s = time.perf_counter() - t0
    # the first epoch carries CUDA/cuDNN warm-up; the steady state excludes it
    steady = epoch_s[1:] if len(epoch_s) > 1 else epoch_s
    out = {"run": trainer.config.name(), "dataset": trainer.config.dataset,
           "n_epochs": n_epochs, "epoch_s": epoch_s,
           "s_per_epoch": float(np.mean(steady)),
           "s_per_epoch_all": float(np.mean(epoch_s)),
           "eval_s": eval_s,
           "resident_mib": resident if cuda else None,
           "peak_train_mib": train_peak,
           "peak_eval_mib": _peak_mib(device),
           "n_train_tiles": len(trainer.train_batches),
           "n_val_tiles": len(trainer.val_batches),
           "n_cells": int(trainer.data.graph.n_cells),
           "phi_dim": int(trainer.data.phi.shape[1]),
           "phi_proj": trainer.config.phi_proj,
           "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if cuda else None}
    (trainer.run_dir / "timing.json").write_text(json.dumps(out, indent=2))
    return out


def _git_state() -> str:
    """Best-effort commit id (+dirty marker) so a run names the code it ran."""
    import subprocess

    try:
        root = Path(__file__).resolve().parents[2]
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout
        return f"{head}{'+dirty' if dirty.strip() else ''}" if head else "unknown"
    except Exception:
        return "unknown"


def run(config: TrainConfig) -> dict:
    data = assemble(config.dataset, config.variant, config.embeddings,
                    tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                    v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                    seed=config.seed, label_key=config.label_key)
    return Trainer(config, data).fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    defaults = TrainConfig(dataset="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default=defaults.variant)
    parser.add_argument("--embeddings", default=defaults.embeddings)
    parser.add_argument("--run-name", default=None)
    for field in ("kappa", "omega", "alpha_z", "alpha_w", "alpha_a", "lr",
                  "weight_decay", "lambda_w", "w_free_bits", "adv_lr",
                  "adv_comp_weight",
                  "val_fraction", "nmi_guard", "cov_ema", "grad_clip"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=float,
                            default=getattr(defaults, field))
    for field in ("d_z", "d_w", "hidden", "gat_dim", "heads", "adv_steps",
                  "adv_hidden", "v_pcs", "epochs", "tile_cells", "patience",
                  "eval_every", "figures_every", "panel_types", "seed",
                  "w_warmup_epochs", "kl_warmup_epochs", "adv_ensemble",
                  "phi_proj"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=int,
                            default=getattr(defaults, field))
    # 8.17's names for the two existing head-capacity knobs (aliases)
    parser.add_argument("--adv-head-steps", dest="adv_steps", type=int,
                        default=argparse.SUPPRESS, help="= --adv-steps")
    parser.add_argument("--adv-head-width", dest="adv_hidden", type=int,
                        default=argparse.SUPPRESS, help="= --adv-hidden")
    parser.add_argument("--phi-pca", type=int, default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--w-penalty", default=defaults.w_penalty,
                        choices=("none", "w", "type_mean"))
    parser.add_argument("--no-second-kl", dest="second_kl", action="store_false",
                        help="ablation (i): charge the z-KL once, not "
                             "(1+omega)-fold (spec 6.2's named mistake)")
    parser.add_argument("--class-mean-prior", action="store_true",
                        help="ablation (iii): p(w|t) = N(mu_t, I), a learned "
                             "per-type vector in place of m_psi(c, t)")
    parser.add_argument("--query", default=defaults.query,
                        choices=("type", "type_free", "image"),
                        help="arms (i)/(ii): the GAT query -- embed(t_i), one "
                             "shared learned vector, or a linear map of Phi_i")
    parser.add_argument("--prior-type-free", action="store_true",
                        help="arm (iii): m_psi(c) only, t dropped from the "
                             "prior; the posterior q(w|.) is unchanged")
    parser.add_argument("--adv-input", default=defaults.adv_input,
                        choices=("mu_z", "xhat"),
                        help="ablation (iv): adversary heads read the decoded "
                             "composition rho_i instead of mu_z")
    parser.add_argument("--invariance", default=defaults.invariance,
                        choices=("closed_form", "adversary"))
    parser.add_argument("--gat-sources", default=defaults.gat_sources,
                        choices=("type_z", "type_only"))
    parser.add_argument("--subtract-leak", action="store_true",
                        help="spec 7.13: encoders read x - kappa*l*rho_bar")
    parser.add_argument("--kappa-mode", default=defaults.kappa_mode,
                        choices=KAPPA_MODES,
                        help="review R12: the leak coefficient's form -- one "
                             "kappa (global), per cell from donor/receiver "
                             "depth (depth), per gene from the extranuclear "
                             "share s_g (gene)")
    parser.add_argument("--kappa-gene-source", default=None,
                        help="s_g file for --kappa-mode gene (default: the "
                             "dataset's experiments/gene_extranuclear_share.npy)")
    parser.add_argument("--fp-floor", action="store_true",
                        help="8.15b: fixed false-positive floor eta_i u, eta_i "
                             "= min(lambda/l_i, 0.2), lambda from the Xenium "
                             "cell table's control counts")
    parser.add_argument("--fp-area", action="store_true",
                        help="with --fp-floor: lambda_i proportional to the "
                             "segmented area (section total fixed)")
    parser.add_argument("--gat-sink", action="store_true",
                        help="attention sink: c grows with neighbour count "
                             "(saturating dose) instead of seeing fractions only")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--time-only", type=int, default=0, metavar="N",
                        help="timing mode (2026-09-25): N training epochs, no "
                             "evaluation inside them, then one timed "
                             "evaluation; prints s/epoch and peak GPU memory "
                             "as JSON and writes timing.json. 0 = a normal fit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = vars(build_parser().parse_args(argv))
    quiet = args.pop("quiet")
    time_only = args.pop("time_only")
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    if time_only:
        config = TrainConfig(**args)
        data = assemble(config.dataset, config.variant, config.embeddings,
                        tile_cells=config.tile_cells, phi_pca=config.phi_pca,
                        v_pcs=config.v_pcs, val_fraction=config.val_fraction,
                        seed=config.seed, label_key=config.label_key)
        print(json.dumps(Trainer(config, data).time_only(time_only)))
        return 0
    run(TrainConfig(**args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
