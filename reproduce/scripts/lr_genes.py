"""The ligand-receptor genes of the signaling member, from CellChatDB.

Every ligand and receptor subunit of every CellChatDB human interaction (complexes
expanded into their subunits), as the paper used it. CellChatDB is part of
CellChat (GPL-3.0) and is not redistributed here: download
``CellChatDB.human.rda`` (``download.sh`` fetches it from the CellChat
repository). Reading the R data file needs ``pip install rdata``.

    python reproduce/scripts/lr_genes.py CellChatDB.human.rda -o lr_genes.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cellchat_lr_genes(rda: str | Path) -> list[str]:
    """Sorted union of the ligand and receptor subunits of every interaction."""
    import rdata

    db = rdata.read_rda(str(rda))["CellChatDB.human"]
    interactions, complexes = db["interaction"], db["complex"]

    def subunits(name: str) -> list[str]:
        if name in complexes.index:
            return [s for s in complexes.loc[name] if isinstance(s, str) and s]
        return [name]

    genes: set[str] = set()
    for _, row in interactions.iterrows():
        for name in (row["ligand"], row["receptor"]):
            genes.update(subunits(name))
    return sorted(genes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rda")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)
    genes = cellchat_lr_genes(args.rda)
    Path(args.out).write_text("\n".join(genes) + "\n")
    print(f"{len(genes)} genes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
