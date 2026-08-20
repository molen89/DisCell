#!/usr/bin/env python3
"""Where everything lives, organised by dataset.

    data/
      raw/, interim/   source slides -- SYMLINKS to wherever they were
                       downloaded, never copies
      models/          KRONOS weights, shared by every dataset
      datasets/<id>/   bundle/  embeddings/  figures/  logs/  dataset.json

A ``dataset_id`` is the slug of the source sample directory, so every spelling
of one slide resolves to the same outputs::

    Xenium_Prime_Ovarian_Cancer_FFPE                 -> xenium_prime_ovarian_cancer_ffpe
    Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip  -> xenium_prime_ovarian_cancer_ffpe

This is a convention, and a convention that lives only in a comment gets
violated silently: a flat ``embeddings/`` looks fine right up to the second
slide, when one file overwrites another or a run pairs one slide's bundle with
another's vectors -- neither of which raises. Putting the dataset in the path
is what makes that impossible. Override the root with ``DISCELL_DATA``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger("discell.paths")

DATA_ROOT = Path(os.environ.get("DISCELL_DATA")
                 or Path(__file__).resolve().parent.parent / "data")

RAW = DATA_ROOT / "raw"
INTERIM = DATA_ROOT / "interim"
MODELS = DATA_ROOT / "models"          # shared: one checkpoint embeds every slide
DATASETS = DATA_ROOT / "datasets"

DEFAULT_VARIANT = "full"
EMBEDDING_SUFFIXES = (".pt", ".npz", ".parquet")

#: Dropped from a sample name before slugging. 10x labels its exports
#: inconsistently -- one Xenium zip is ``..._outs.zip``, the other
#: ``..._XRrun_outs.zip`` -- and that must not fork one slide into two ids.
_NOISE = re.compile(r"(\.tar\.gz|\.tgz|\.tar|\.zip)$|_(xrrun|outs)$", re.I)


def slug(value: str | Path) -> str:
    """Canonical ``dataset_id`` for a sample directory or archive."""
    path = Path(str(value))
    name = path.parent.name if path.name == "outs" else path.name
    while (stripped := _NOISE.sub("", name)) != name:
        name = stripped
    text = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not text:
        raise ValueError(f"{value!r} names no dataset; pass the sample directory")
    return text


@dataclass(frozen=True)
class Dataset:
    """Every output location for one slide.

    Pure paths: call :meth:`ensure` before writing, so a mistyped id cannot
    create a directory as a side effect and then be listed as a real dataset.
    """

    dataset_id: str

    root = property(lambda self: DATASETS / self.dataset_id)
    bundle_dir = property(lambda self: self.root / "bundle")
    embeddings_dir = property(lambda self: self.root / "embeddings")
    figures = property(lambda self: self.root / "figures")
    logs = property(lambda self: self.root / "logs")
    manifest_path = property(lambda self: self.root / "dataset.json")

    def ensure(self) -> "Dataset":
        for directory in (self.bundle_dir, self.embeddings_dir, self.figures, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def bundle(self, variant: str = DEFAULT_VARIANT) -> Path:
        """One bundle's ``.h5ad``; its siblings share the stem. A dataset can
        hold several -- different graph tolerances, a small dev subset."""
        if "/" in str(variant):
            raise ValueError(f"variant is a bare name, not a path: {variant!r}")
        return self.bundle_dir / f"{Path(variant).stem}.h5ad"

    def variants(self) -> list[str]:
        return sorted(p.stem for p in self.bundle_dir.glob("*.h5ad"))

    def embeddings_file(self, name: str | Path | None) -> Path | None:
        """Resolve an embeddings argument given as a bare name or a path.

        Only paths inside this dataset are accepted; reaching across datasets
        is the substitution the layout exists to prevent.
        """
        if name is None:
            return None
        for candidate in (Path(name),
                          *(self.embeddings_dir / f"{Path(name).name}{suffix}"
                            for suffix in ("", *EMBEDDING_SUFFIXES))):
            if candidate.exists():
                if self.embeddings_dir.resolve() not in candidate.resolve().parents:
                    raise ValueError(f"{candidate} belongs to another dataset")
                return candidate
        return Path(name)                    # let the caller raise the real error

    def embedding_names(self) -> list[str]:
        return sorted({p.stem for p in self.embeddings_dir.glob("*")
                       if p.suffix in EMBEDDING_SUFFIXES})

    def manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def write_manifest(self, bundle: str | None = None, **fields) -> Path:
        """Merge *fields* into ``dataset.json``.

        Per-bundle facts go under ``bundles[<variant>]``, so a 3k-cell dev
        bundle cannot overwrite what the full one recorded.
        """
        record = self.manifest()
        if bundle:
            record["bundles"] = {**record.get("bundles", {}),
                                 bundle: {**record.get("bundles", {}).get(bundle, {}),
                                          **fields}}
        else:
            record.update(fields)
        record["dataset_id"] = self.dataset_id
        record["updated"] = datetime.now().isoformat(timespec="seconds")
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(record, indent=2) + "\n")
        return self.manifest_path

    def __str__(self) -> str:
        return self.dataset_id


def dataset(value: str | Path | Dataset) -> Dataset:
    """A Dataset from an id, a sample name, or a source path."""
    return value if isinstance(value, Dataset) else Dataset(slug(value))


dataset_for = dataset          # reads better at a --sample call site


def list_datasets() -> list[Dataset]:
    """Datasets on disk, skipping directories whose name is not already an id.

    A hand-made ``<id> copy`` would otherwise be reported as a second slide.
    """
    found = []
    for path in sorted(DATASETS.glob("*")):
        if not path.is_dir():
            continue
        if path.name != slug(path.name):
            log.warning("%s/ is not a dataset id -- ignoring", path.name)
            continue
        found.append(Dataset(path.name))
    return found


def add_dataset_args(parser, variant: bool = True) -> None:
    parser.add_argument("--dataset", default=None,
                        help="dataset id or sample name; inferred from the tree "
                             "when only one dataset has a bundle")
    if variant:
        parser.add_argument("--variant", default=DEFAULT_VARIANT,
                            help=f"bundle within the dataset (default {DEFAULT_VARIANT!r})")


def resolve_bundle(args) -> tuple[Dataset, str]:
    """``(dataset, variant)`` from parsed args, checking the bundle exists.

    Falls back to the only bundled dataset, but never picks between several.
    """
    variant = getattr(args, "variant", None) or DEFAULT_VARIANT
    if getattr(args, "dataset", None):
        ds = dataset(args.dataset)
    else:
        candidates = [d for d in list_datasets() if d.variants()]
        if len(candidates) != 1:
            raise SystemExit(
                "--dataset is required; available:\n"
                + "\n".join(f"  {d} ({', '.join(d.variants())})" for d in candidates)
                if candidates else
                f"No bundles under {DATASETS}. Build one:\n"
                f"  python -m discell.preprocess --sample <sample dir>")
        ds = candidates[0]
    if not ds.bundle(variant).exists():
        raise SystemExit(f"{ds} has no bundle {variant!r}; "
                         f"have: {', '.join(ds.variants()) or 'none'}")
    return ds, variant


def describe() -> str:
    lines = [f"DISCELL_DATA = {DATA_ROOT}", ""]
    for ds in list_datasets():
        bundles = ds.manifest().get("bundles", {})
        counts = ", ".join(
            f"{v} ({bundles[v]['n_cells']:,} cells)"
            if isinstance(bundles.get(v, {}).get("n_cells"), int) else v
            for v in ds.variants())
        lines += [f"  {ds}",
                  f"    bundle     {counts or '-'}",
                  f"    embeddings {', '.join(ds.embedding_names()) or '-'}",
                  f"    figures    {len(list(ds.figures.glob('*')))}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for directory in (RAW, INTERIM, MODELS, DATASETS):
        directory.mkdir(parents=True, exist_ok=True)
    print(describe())
