#!/usr/bin/env python3
"""Turn a raw slide into everything downstream needs, in one command.

Three stages, plus optional figures. Each checks for its own output and skips
when it is already there, so re-running after adding a slide, or after only the
embedding step failed, costs nothing for the parts already done::

    python -m discell.preprocess --sample <outs-dir>
      [1/3] bundle      RUN   -> bundle/full.h5ad + polygons + edges + params
      [2/3] embed       SKIP  embeddings/full_v1.pt exists
      [3/3] figures     SKIP  --figures not given

``--force`` redoes everything; ``--force-from embed`` redoes that stage and the
ones after it.

The heavy read -- counts, boundaries, polygon repair, both graphs -- happens at
most once per invocation and is shared by whichever stages need it. When the
bundle already exists and only embeddings are wanted, it is reloaded instead of
rebuilt, which skips the graph construction the image side has no use for.

Usage::

    python -m discell.preprocess --sample <outs-dir>
    python -m discell.preprocess --sample <outs-dir> --model v2 --embeddings-name kronos2_4ch
    python -m discell.preprocess --sample <outs-dir> --variant dev --max-cells 3000 --figures
    python -m discell.preprocess --sample <outs-dir> --only bundle --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from discell import paths

log = logging.getLogger("discell.preprocess")

STAGES = ("bundle", "embed", "figures")
FIGURE_KINDS = ("graph", "crops", "qc", "compare")
DEFAULT_FIGURES = "graph,crops,qc"


class Run:
    """One invocation's state: which stages run, and the shared AnnData."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset = paths.dataset_for(args.sample).ensure()
        self.variant = args.variant
        self.sample_dir = None
        self._adata = None
        self._has_graphs = False

        start = 0 if args.force else (
            STAGES.index(args.force_from) if args.force_from else len(STAGES))
        self.forced = set(STAGES[start:])
        self.wanted = (set(s.strip() for s in args.only.split(",")) if args.only
                       else set(STAGES))
        unknown = self.wanted - set(STAGES)
        if unknown:
            raise SystemExit(f"unknown stage {sorted(unknown)}; pick from {list(STAGES)}")

    # -- the shared load ---------------------------------------------------

    def adata(self, need_graphs: bool):
        """The sample as AnnData, loaded at most once per run.

        Reloads the bundle when one exists and it carries what is needed -- an
        h5ad read is seconds against minutes to rebuild the graphs.
        """
        from discell.preprocess.xenium import load_sample
        from discell.tiff import find_tissue_image

        if self._adata is not None and (self._has_graphs or not need_graphs):
            return self._adata

        bundle = self.dataset.bundle(self.variant)
        if bundle.exists() and "bundle" not in self.forced:
            from discell.data.bundle import load_bundle

            log.info("Reusing %s", bundle.name)
            self._adata = load_bundle(self.dataset.bundle_dir, self.variant)
            self._has_graphs = True
            directory = self._adata.uns.get("xenium_dir")
            self.sample_dir = Path(directory) if directory else Path(self.args.sample)
        else:
            self._adata, self.sample_dir = load_sample(
                self.args.sample,
                clip_radius_um=self.args.clip_radius_um,
                max_cells=self.args.max_cells,
                contact_tolerance_um=self.args.contact_tolerance_um,
                wall_tolerance_um=self.args.wall_tolerance_um,
                build_graphs=need_graphs,
            )
            self._has_graphs = need_graphs
        self.image_path = find_tissue_image(self.sample_dir)
        return self._adata

    # -- stages ------------------------------------------------------------

    def bundle(self) -> str:
        from discell.preprocess.bundle import save_bundle

        target = self.dataset.bundle(self.variant)
        if target.exists() and "bundle" not in self.forced:
            return f"SKIP  {target.name} exists"

        adata = self.adata(need_graphs=True)
        written = save_bundle(
            adata, self.dataset.bundle_dir, self.variant,
            params={
                "sample": str(self.sample_dir),
                "contact_tolerance_um": self.args.contact_tolerance_um,
                "wall_tolerance_um": self.args.wall_tolerance_um,
                "clip_radius_um": self.args.clip_radius_um,
                "max_cells": self.args.max_cells,
            },
            dataset_id=self.dataset.dataset_id,
        )
        self.dataset.write_manifest(
            bundle=self.variant,
            platform=adata.uns.get("platform", "xenium"),
            source=str(self.sample_dir),
            sample_id=adata.uns.get("sample", {}).get("sample_id", ""),
            n_cells=int(adata.n_obs),
            n_features=int(adata.n_vars),
            microns_per_pixel=float(adata.uns.get("microns_per_pixel", float("nan"))),
            default_label=adata.uns.get("default_label", ""),
        )
        size = sum(p.stat().st_size for p in written.values()) / 1e6
        return (f"RUN   {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
                f"{len(written)} files, {size:.0f} MB")

    def embed(self) -> str:
        from discell.preprocess.kronos import embed_sample, write_embeddings

        name = self.args.embeddings_name or f"{self.variant}_{self.args.model}"
        target = self.dataset.embeddings_dir / f"{name}.pt"
        if target.exists() and "embed" not in self.forced:
            return f"SKIP  {target.name} exists"

        adata = self.adata(need_graphs=False)
        if self.image_path is None:
            raise SystemExit(f"no morphology image under {self.sample_dir}")

        cells = None
        if self.args.limit:
            import numpy as np

            rng = np.random.default_rng(self.args.seed)
            cells = sorted(rng.choice(adata.n_obs, min(self.args.limit, adata.n_obs),
                                      replace=False))
        embeddings, cell_ids, meta = embed_sample(
            adata, self.image_path, model=self.args.model,
            channels=[int(c) for c in self.args.channels.split(",")],
            cells=cells, patch_px=self.args.patch_px, half_um=self.args.half_um,
            batch_size=self.args.batch_size, block_px=self.args.block_px,
            device=self.args.device, cache_dir=self.args.cache_dir,
            hf_token=self.args.hf_token,
            drop_unknown_markers=self.args.drop_unknown_markers,
        )
        write_embeddings(target, embeddings, cell_ids,
                         meta={"dataset": self.dataset.dataset_id, **meta})
        return f"RUN   {embeddings.shape[0]:,} x {embeddings.shape[1]} -> {target.name}"

    def figures(self) -> str:
        kinds = [k.strip() for k in (self.args.figures or "").split(",") if k.strip()]
        if not kinds:
            return "SKIP  --figures not given"
        unknown = [k for k in kinds if k not in FIGURE_KINDS]
        if unknown:
            raise SystemExit(f"unknown figure kind {unknown}; pick from {list(FIGURE_KINDS)}")

        adata = self.adata(need_graphs=True)
        label_key = adata.uns.get("default_label", "cell_group")
        out_dir = self.dataset.figures
        written: list[Path] = []

        if "graph" in kinds:
            from discell.preprocess.plotting.render import render

            written += render(adata, self.sample_dir, out_dir, graph=self.args.graph,
                              label_key=label_key, edge_metric="apposed_wall_um",
                              color_edges_by_metric=True)
        if "crops" in kinds and self.image_path is not None:
            from discell.preprocess.plotting.crop_view import render_crop

            mpp = float(adata.uns["microns_per_pixel"])
            written.append(render_crop(
                adata, self.image_path, 0, out_dir / "kronos_input.png",
                half_um=(self.args.patch_px * mpp) / 2,
                channels=[int(c) for c in self.args.channels.split(",")]))
        if "compare" in kinds:
            from discell.preprocess.plotting.compare_graphs import (
                build_variants, densest_region, render_comparison,
            )

            variants = build_variants(adata)
            written.append(render_comparison(
                adata, self.sample_dir, out_dir, variants,
                densest_region(adata, 400.0), label_key=label_key))
        if "qc" in kinds:
            written += self._figure_qc(adata, label_key, out_dir)

        return f"RUN   {len(written)} figure(s) -> {out_dir}"

    def _figure_qc(self, adata, label_key, out_dir) -> list[Path]:
        """UMAP of the image embeddings, coloured by label."""
        import numpy as np

        from discell.data.embeddings import load_embeddings
        from discell.preprocess.embedding_qc import (
            plot_umap, run_umap, separation_report, stratified_sample,
        )

        name = self.args.embeddings_name or f"{self.variant}_{self.args.model}"
        source = self.dataset.embeddings_dir / f"{name}.pt"
        if not source.exists():
            log.warning("no %s -- skipping the embedding QC figure", source.name)
            return []

        cell_ids = np.asarray(adata.obs_names, dtype=str)
        vectors, mask = load_embeddings(source, cell_ids,
                                        dataset=self.dataset.dataset_id)
        labels = adata.obs[label_key].astype(str).to_numpy()
        rng = np.random.default_rng(self.args.seed)
        picked = stratified_sample(labels[mask], self.args.per_label, rng)
        rows = np.flatnonzero(mask)[picked]

        report = separation_report(vectors[rows], labels[rows], seed=self.args.seed)
        log.info("embedding separation %+.4f, 1-NN %.1f%% (chance %.1f%%)",
                 report["separation"], 100 * report["nn_agreement"],
                 100 * report["chance"])
        coords = run_umap(vectors[rows], seed=self.args.seed)
        return [plot_umap(
            coords, labels[rows], out_dir / f"{name}_umap.png",
            title=f"{self.dataset.dataset_id} -- {name}",
            subtitle=f"separation {report['separation']:+.3f}, "
                     f"1-NN {100 * report['nn_agreement']:.1f}% "
                     f"(chance {100 * report['chance']:.1f}%)")]


def run(args: argparse.Namespace) -> int:
    job = Run(args)
    print(f"DATASET  {job.dataset.dataset_id}  (bundle {job.variant!r})")
    print(f"SOURCE   {args.sample}")
    print()
    for index, stage in enumerate(STAGES, start=1):
        if stage not in job.wanted:
            print(f"  [{index}/{len(STAGES)}] {stage:<10} SKIP  not in --only")
            continue
        started = time.time()
        outcome = getattr(job, stage)()
        elapsed = "" if outcome.startswith("SKIP") else f"  [{time.time() - started:.1f}s]"
        print(f"  [{index}/{len(STAGES)}] {stage:<10} {outcome}{elapsed}")
    print()
    print(f"Ready: CellGraphDataset.from_dataset({job.dataset.dataset_id!r}, "
          f"{job.variant!r}, embeddings=...)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", required=True,
                        help="raw sample directory or *_outs.zip; determines the dataset id")
    parser.add_argument("--variant", default=paths.DEFAULT_VARIANT,
                        help="bundle name within the dataset, for holding several graph "
                             f"settings side by side (default {paths.DEFAULT_VARIANT!r})")

    graph = parser.add_argument_group("graph")
    graph.add_argument("--contact-tolerance-um", type=float, default=None)
    graph.add_argument("--wall-tolerance-um", type=float, default=None)
    graph.add_argument("--clip-radius-um", type=float, default=30.0)
    graph.add_argument("--max-cells", type=int, default=None,
                       help="limit to the N cells nearest the tissue centre")

    image = parser.add_argument_group("embeddings")
    image.add_argument("--model", default="v1", choices=("v1", "v2"),
                       help="v1 = KRONOS (384-d, marker ids); v2 = KRONOS2 (768-d, names)")
    image.add_argument("--embeddings-name", default=None,
                       help="output stem; default <variant>_<model>")
    image.add_argument("--channels", default="0,1,2,3",
                       help="image channels to embed; '0' is a DAPI-only baseline")
    image.add_argument("--patch-px", type=int, default=256,
                       help="input edge in pixels; must be a multiple of 16")
    image.add_argument("--half-um", type=float, default=None,
                       help="fix the field in microns and resample to --patch-px "
                            "instead of cropping natively")
    image.add_argument("--batch-size", type=int, default=128)
    image.add_argument("--block-px", type=int, default=4096,
                       help="image block decoded at once when cropping in bulk")
    image.add_argument("--limit", type=int, default=None, help="embed only N cells")
    image.add_argument("--cache-dir", default=str(paths.MODELS))
    image.add_argument("--hf-token", default=None)
    image.add_argument("--drop-unknown-markers", action="store_true",
                       help="v2: skip channels that are neither in the vocabulary "
                            "nor registrable, instead of failing")

    figures = parser.add_argument_group("figures")
    figures.add_argument("--figures", nargs="?", const=DEFAULT_FIGURES, default=None,
                         help=f"comma list of {list(FIGURE_KINDS)}; bare --figures "
                              f"means {DEFAULT_FIGURES!r}")
    figures.add_argument("--graph", default=None,
                         help="graph to draw; default is the loader's")
    figures.add_argument("--per-label", type=int, default=100,
                         help="cells per label in the embedding QC sample")

    parser.add_argument("--only", default=None,
                        help=f"run only these stages, comma separated: {list(STAGES)}")
    parser.add_argument("--force", action="store_true", help="redo every stage")
    parser.add_argument("--force-from", default=None, choices=STAGES,
                        help="redo this stage and the ones after it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from discell.data.loader import DEFAULT_GRAPH

    args = build_parser().parse_args(argv)
    if args.graph is None:
        args.graph = DEFAULT_GRAPH
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
