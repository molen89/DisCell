"""Shared helper: pin the GPU BEFORE torch/scvi are imported anywhere."""
import os
import argparse


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--gpu", type=int, default=0,
                   help="Physical GPU index to use. -1 = CPU. Applied via CUDA_VISIBLE_DEVICES "
                        "before torch is imported, so downstream code always sees device 0.")
    p.add_argument("--root", default="/home/rmolen/cellxgene_all_visium/highres_raw_with_images/GEO_Xenium/GSE315411",
                   help="GSE315411 root directory")
    p.add_argument("--work", default=None,
                   help="Working/output directory (default: <root>/annotate)")
    p.add_argument("--seed", type=int, default=0)
    return p


def pin_gpu(gpu: int) -> str:
    """Must be called before importing torch / scvi / scanpy-with-torch backends."""
    if gpu is None or gpu < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # make cuBLAS deterministic-ish and avoid grabbing all memory up front
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return "gpu"


def resolve_work(args):
    from pathlib import Path
    root = Path(args.root)
    work = Path(args.work) if args.work else root / "annotate"
    work.mkdir(parents=True, exist_ok=True)
    return root, work


def set_seed(seed: int):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
