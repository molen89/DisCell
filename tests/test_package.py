"""The core package imports without any optional extra."""

import subprocess
import sys

EXTRAS = ("scanpy", "matplotlib", "tensorboard", "torch_geometric", "tifffile",
          "kronos", "umap", "zarr")


def test_core_import_pulls_in_no_extra():
    code = ("import sys, discell, discell.io, discell.graph, discell.model, discell.image, "
            "discell.sensitivity, discell.tl, discell.pl; "
            f"print([m for m in {EXTRAS!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"
