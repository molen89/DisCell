"""The documentation runs: the Python snippets of docs/, and the example notebooks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("use_cases.md", "code.md")
NOTEBOOKS = ("01_quickstart_simulated.ipynb", "02_public_section.ipynb")
NOTEBOOK_MAX_BYTES = 2_000_000


def python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", path.read_text(), flags=re.S | re.M)


@pytest.mark.slow
@pytest.mark.parametrize("name", DOCS)
def test_doc_snippets_run(name, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import torch

    blocks = python_blocks(ROOT / "docs" / name)
    assert blocks, f"no python blocks in {name}"
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    namespace: dict = {"__name__": "__docs__"}
    try:
        for block in blocks:
            exec(compile(block, name, "exec"), namespace)
    finally:
        torch.set_num_threads(threads)
        import matplotlib.pyplot as plt

        plt.close("all")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_stored_executed(name):
    path = ROOT / "examples" / name
    assert path.stat().st_size < NOTEBOOK_MAX_BYTES
    text = path.read_text()
    assert "/home/" not in text
    nb = json.loads(text)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["execution_count"] is not None for c in code)
    assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])


@pytest.mark.slow
def test_quickstart_notebook_executes(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    nb = nbformat.read(ROOT / "examples" / NOTEBOOKS[0], as_version=4)
    env_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the notebook is a CPU demo
    try:
        nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path)}}).execute()
    finally:
        if env_before is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_before
