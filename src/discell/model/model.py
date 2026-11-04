"""A fitted DisCell model, :func:`fit`, and the AnnData outputs.

Written into the AnnData by :func:`fit` and :meth:`Model.encode`::

    .obsm['X_discell_z']      posterior mean of z (the intrinsic state)
    .obsm['X_discell_w']      posterior mean of w (the niche response)
    .obsm['X_discell_mpsi']   prior mean m_psi(c, t) of w
    .varm['discell_B']        response loadings, genes x d_w
    .uns['discell']['fit']    summary scalars, plus the configuration and the
                              fit's metrics as JSON strings
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from discell._version import __version__
from discell.model.config import Config
from discell.model.data import NUM_THREADS, ModelData, assemble
from discell.model.networks import DisCell
from discell.model.trainer import Runner, Trainer, resolve_device

MODEL_FORMAT = "discell-model"
MODEL_VERSION = 1


class Model:
    """A fitted network with the configuration, types and genes it was fitted on.

    ``config`` is resolved (``alpha_z`` is the value used). ``metrics`` is the
    fit's summary (best epoch, closing evaluation, dead-channel flags),
    ``history`` its evaluations and ``losses`` its per-step training terms.
    """

    def __init__(self, config: Config, network: DisCell, type_names, gene_names,
                 metrics: dict | None = None, history: list | None = None,
                 losses: list | None = None):
        self.config = config
        self.network = network
        self.type_names = [str(v) for v in type_names]
        self.gene_names = [str(v) for v in gene_names]
        self.metrics = metrics
        self.history = history or []
        self.losses = losses or []

    # -- data ----------------------------------------------------------------

    def assemble(self, adata) -> ModelData:
        """*adata*'s fit arrays under this model's settings (tiles, split, PCs).

        *adata* must carry a DisCell graph with the same types and the same genes.
        """
        cfg = self.config
        data = assemble(adata, cfg.label_key, cfg.image_key, counts_layer=cfg.counts_layer,
                        tile_cells=cfg.tile_cells, val_fraction=cfg.val_fraction,
                        v_pcs=cfg.v_pcs, seed=cfg.seed)
        if [str(v) for v in data.type_names] != self.type_names:
            raise ValueError("the AnnData's types differ from the model's: "
                             f"{list(data.type_names)} vs {self.type_names}")
        if [str(v) for v in data.gene_names] != self.gene_names:
            raise ValueError("the AnnData's genes differ from the model's (names or order)")
        if data.phi.shape[1] != self.network.phi_dim:
            raise ValueError(f"image descriptor width {data.phi.shape[1]} != the "
                             f"model's {self.network.phi_dim}")
        return data

    def runner(self, data: ModelData) -> Runner:
        return Runner(self.network, data, self.config, resolve_device(self.config.device))

    def latents(self, data: ModelData) -> dict:
        """Posterior means of every cell, in tile order (``nodes``)."""
        runner = self.runner(data)
        sweep = runner.sweep(runner.train_batches + runner.val_batches)
        return {k: sweep[k] for k in ("nodes", "mu_z", "mu_w", "prior_w")}

    def encode(self, adata, *, data: ModelData | None = None) -> None:
        """Write the latents, the loadings and the fit summary into *adata*.

        *data*: an assembly of *adata* already made with :meth:`assemble`.
        """
        data = data if data is not None else self.assemble(adata)
        self._write(adata, self.latents(data))

    def evaluate(self, adata, *, data: ModelData | None = None) -> dict:
        """The closing evaluation of a fit, on *adata*'s held-out tiles.

        NMI, reconstruction, the pooled probe, mirror, degeneracy, the
        type-mean reconstruction gap and the w-channel guard (see
        :meth:`discell.model.trainer.Runner.evaluate`).
        """
        data = data if data is not None else self.assemble(adata)
        report = self.runner(data).evaluate(final=True)
        report.pop("latents")
        return report

    def _write(self, adata, latents: dict) -> None:
        order = np.argsort(latents["nodes"])
        if not np.array_equal(latents["nodes"][order], np.arange(adata.n_obs)):
            raise AssertionError("tiles must cover every cell exactly once")
        adata.obsm["X_discell_z"] = latents["mu_z"][order]
        adata.obsm["X_discell_w"] = latents["mu_w"][order]
        adata.obsm["X_discell_mpsi"] = latents["prior_w"][order]
        adata.varm["discell_B"] = self.network.B.weight.detach().cpu().numpy().copy()
        adata.uns.setdefault("discell", {})["fit"] = self.summary()

    def summary(self) -> dict:
        """The scalars written to ``.uns['discell']['fit']`` (h5ad-writable)."""
        cfg = self.config
        out = {"version": __version__, "kappa": float(cfg.kappa), "seed": int(cfg.seed),
               "alpha_z": float(cfg.alpha_z), "image_free": bool(cfg.image_free),
               "kmeans_threads": int(NUM_THREADS),
               "config": json.dumps(cfg.to_dict())}
        if self.metrics:
            final = self.metrics["final"]
            guard = final.get("w_channel", {})
            out.update(best_epoch=int(self.metrics["best"]["epoch"]),
                       recon_val=float(final["recon_val"]), nmi=float(final["nmi"]),
                       dead_w_channel=bool(self.metrics["dead_w_channel"]),
                       niche_information_check=float(guard.get("niche_information_check", np.nan)),
                       dead_context_channel=bool(guard.get("dead_context_channel", False)),
                       metrics=json.dumps(self.metrics, default=float))
        return out

    # -- persistence -----------------------------------------------------------

    def save(self, directory) -> Path:
        """Write ``model.pt`` (weights, configuration, types, genes) and, when the
        model comes from a fit, ``metrics.json``, ``history.jsonl`` and
        ``losses.jsonl`` into *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        net = self.network
        torch.save({"format": MODEL_FORMAT, "version": MODEL_VERSION,
                    "config": self.config.to_dict(),
                    "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "type_names": self.type_names, "gene_names": self.gene_names,
                    "phi_dim": int(net.phi_dim), "median_counts": float(net.median_counts)},
                   directory / "model.pt")
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=1))
        if self.metrics is not None:
            (directory / "metrics.json").write_text(
                json.dumps(self.metrics, indent=1, default=float))
        for name, rows in (("history", self.history), ("losses", self.losses)):
            if rows:
                (directory / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r, default=float) + "\n" for r in rows))
        return directory

    @classmethod
    def load(cls, directory, device: str | None = None) -> Model:
        """A model written by :meth:`save`; *device* overrides the stored one."""
        directory = Path(directory)
        payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"{directory / 'model.pt'} is not a DisCell model")
        if payload["version"] > MODEL_VERSION:
            raise ValueError(f"model format {payload['version']} is newer than this "
                             f"package reads ({MODEL_VERSION})")
        config = Config.from_dict(payload["config"])
        if device is not None:
            config = dataclasses.replace(config, device=device)
        network = DisCell(n_genes=len(payload["gene_names"]),
                          n_types=len(payload["type_names"]), phi_dim=payload["phi_dim"],
                          median_counts=payload["median_counts"], d_z=config.d_z,
                          d_w=config.d_w, hidden=config.hidden, gat_dim=config.gat_dim,
                          heads=config.heads)
        network.load_state_dict(payload["state_dict"], strict=True)
        metrics = None
        if (directory / "metrics.json").exists():
            metrics = json.loads((directory / "metrics.json").read_text())
        return cls(config, network, payload["type_names"], payload["gene_names"],
                   metrics=metrics)


def fit(adata, config: Config, out_dir=None) -> Model:
    """Fit DisCell to *adata* and write the outputs into it (see module doc).

    *adata* meets the input contract and carries the graph of
    :func:`discell.build_graph`, built with ``config.label_key`` and
    ``config.unassigned_label``. ``config.image_key = None`` fits the image-free
    model. With *out_dir* the model is also saved there (:meth:`Model.save`).
    """
    meta = adata.uns.get("discell", {}).get("graph")
    if meta is None:
        raise KeyError("no DisCell graph in adata: run discell.build_graph first")
    if meta["unassigned_label"] != config.unassigned_label:
        raise ValueError(f"the graph was built with unassigned_label="
                         f"{meta['unassigned_label']!r}, the config has "
                         f"{config.unassigned_label!r}")
    data = assemble(adata, config.label_key, config.image_key,
                    counts_layer=config.counts_layer, tile_cells=config.tile_cells,
                    val_fraction=config.val_fraction, v_pcs=config.v_pcs, seed=config.seed)
    config = config.resolve(data.median_counts)
    trainer = Trainer(config, data, resolve_device(config.device))
    summary = trainer.fit()
    latents = summary["final"].pop("latents")
    model = Model(config, trainer.network, data.type_names, data.gene_names,
                  metrics=summary, history=trainer.history, losses=trainer.losses)
    model._write(adata, latents)
    if out_dir is not None:
        model.save(out_dir)
    return model


__all__ = ["Model", "fit"]
