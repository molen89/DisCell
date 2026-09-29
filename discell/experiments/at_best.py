"""The in-trainer battery of the accepted checkpoint, read back from history.jsonl.

Review R26. Runs trained before the fix wrote ``metrics.json["final"]`` from
the in-memory model after early stopping, ``patience`` epochs past the
checkpoint that ``best.pt`` holds. ``metrics.json["best"]`` keeps only
reconstruction, NMI, epoch and degeneracy of the accepted checkpoint.
``history.jsonl`` logs the full evaluation at every evaluation epoch, and
``best.pt`` is saved from the same weights that the row at ``best["epoch"]``
evaluated. So that row is the accepted checkpoint's full battery, and it has
the same shape as ``final``. A read-out swaps one for the other and parses it
unchanged.

The post-hoc reads (``degeneracy.json``, transport, atlas, cross-slide)
already load ``best.pt``, so nothing here touches them.

Under the evaluation mask (devlog 2026-09-28, Unassigned never a metric
target) the history row is an unmasked read. The accepted checkpoint's
battery under the mask is ``degeneracy.json``'s ``battery`` -- the same
``Trainer.evaluate`` on ``best.pt``, re-run -- and :func:`battery_at_best`
returns it when it exists, with ``nmi`` and ``recon_val`` the target reads
(the checkpoint-selection scores kept as ``nmi_selection`` /
``recon_val_selection``). Under the switch it returns the history row as
before.
"""

from __future__ import annotations

import json
from pathlib import Path

#: the header mark of every read-out made with ``--at best``
HEADER = "reads at the accepted checkpoint (R26)"
AT_CHOICES = ("final", "best")


def masked_battery(run_dir) -> dict | None:
    """``best.pt``'s battery re-read under the evaluation mask in force
    (``degeneracy.json``), in the shape of ``final``; None under the switch,
    or when no such re-read is on disk."""
    from discell.model import eval_mask as EM

    path = Path(run_dir) / "degeneracy.json"
    if not EM.exclusions() or not path.exists():
        return None
    post_hoc = json.loads(path.read_text())
    battery = post_hoc.get("battery")
    if not battery or not EM.same_mask(post_hoc.get("eval_mask")):
        return None
    return {**battery,
            "nmi_selection": battery.get("nmi"),
            "recon_val_selection": battery.get("recon_val"),
            "nmi": battery.get("nmi_targets"),
            "recon_val": battery.get("recon_val_targets"),
            "at_best": True,
            "source": "degeneracy.json battery (best.pt re-read, eval_mask)"}


def battery_at_best(run_dir) -> dict:
    """The history row at ``metrics["best"]["epoch"]``, in the shape of ``final``.

    Only a row whose epoch equals the best epoch and whose reconstruction and
    NMI equal ``best``'s is taken for the accepted checkpoint. Without one,
    the record is ``final`` with ``at_best`` False and a reason. The exception
    is a run trained after the R26 fix: there ``final`` re-evaluates
    ``best.pt``, as ``metrics["final_epoch"]`` says, so ``at_best`` stays True.
    """
    run_dir = Path(run_dir)
    masked = masked_battery(run_dir)
    if masked is not None:
        return masked
    metrics = json.loads((run_dir / "metrics.json").read_text())
    best = metrics["best"]
    epoch = int(best["epoch"])
    history = run_dir / "history.jsonl"
    rows = ([json.loads(line) for line in history.read_text().splitlines()
             if line.strip()] if history.exists() else [])
    row = {int(r["epoch"]): r for r in rows}.get(epoch)
    if row is None:
        reason = f"no history row at best epoch {epoch}"
    elif (row.get("recon_val"), row.get("nmi")) != (best["recon_val"],
                                                    best["nmi"]):
        reason = (f"history row at epoch {epoch} disagrees with "
                  "metrics['best'] on recon/NMI")
    else:
        assert int(row["epoch"]) == epoch
        return {**row, "at_best": True}
    final_epoch = metrics.get("final_epoch")
    return {**metrics["final"], "epoch": final_epoch,
            "at_best": final_epoch == epoch, "at_best_reason": reason}


def suffixed(path, at: str) -> Path:
    """*path* with ``_at_best`` before its extension when reading at best, so
    a read-out at the accepted checkpoint never overwrites a last-epoch one."""
    path = Path(path)
    if at != "best":
        return path
    return path.with_name(f"{path.stem}_at_best{path.suffix}")
