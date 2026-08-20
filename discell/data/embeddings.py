#!/usr/bin/env python3
"""Read per-cell image embeddings written by :mod:`discell.preprocess.kronos`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.data.embeddings")

#: Width of a KRONOS v1 vector. Only used to size the zero-fill when no
#: embeddings are given, but it has to match what a real file would bring or the
#: model built against it changes shape the moment one arrives.
DEFAULT_EMBEDDING_DIM = 384


def load_embeddings(path: str | Path, cell_ids: Sequence[str],
                    dataset: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load per-cell image embeddings aligned to *cell_ids*.

    Accepts ``.npz`` with ``cell_ids`` and ``embeddings`` arrays, or a parquet
    with a cell-id column plus either an ``embedding`` list column or numeric
    feature columns. Cells with no embedding get zeros; the returned mask marks
    which rows are real, so a model can avoid training on fabricated vectors.

    Two guards stop embeddings from one slide being served against another's
    bundle, which used to fail only as a warning about zero-filled rows:

    * if the file records a ``dataset`` and *dataset* is given, they must match;
    * if **no** cell id matches at all, that is a mismatched file rather than an
      incomplete one, whatever the metadata says.
    """
    path = Path(path)
    recorded: str | None = None
    if path.suffix in (".pt", ".pth"):
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("dataset")
        ids = np.asarray(payload["cell_ids"]).astype(str)
        vectors = np.asarray(payload["embeddings"].float().numpy(), dtype=np.float32)
    elif path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        ids = np.asarray(payload["cell_ids"]).astype(str)
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
    elif path.suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
        id_col = next((c for c in ("cell", "cell_id", "barcode") if c in frame), None)
        if id_col is None:
            raise ValueError(f"{path} has no cell id column")
        ids = frame[id_col].astype(str).to_numpy()
        if "embedding" in frame:
            vectors = np.stack(frame["embedding"].to_numpy()).astype(np.float32)
        else:
            numeric = frame.drop(columns=[id_col]).select_dtypes("number")
            vectors = numeric.to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding format: {path.suffix}")

    if dataset and recorded and recorded != dataset:
        raise ValueError(
            f"{path.name} was computed on dataset {recorded!r}, but this bundle is "
            f"{dataset!r}. Use that dataset's own embeddings:\n"
            f"  data/datasets/{dataset}/embeddings/"
        )

    lookup = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    lookup = lookup[~lookup.index.duplicated()]
    position = lookup.reindex(pd.Index(np.asarray(cell_ids, dtype=str)))
    found = position.notna().to_numpy()

    if not found.any():
        raise ValueError(
            f"{path.name} shares no cell ids with this bundle -- it belongs to a "
            f"different dataset"
            + (f" ({recorded})" if recorded else "")
            + f". Its ids look like {[str(v) for v in ids[:2]]}, the bundle's like "
              f"{[str(v) for v in np.asarray(cell_ids, dtype=str)[:2]]}."
        )

    out = np.zeros((len(cell_ids), vectors.shape[1]), dtype=np.float32)
    out[found] = vectors[position[found].astype(int).to_numpy()]
    if not found.all():
        log.warning("%d of %d cells have no image embedding (zero-filled)",
                    int((~found).sum()), len(cell_ids))
    return out, found
