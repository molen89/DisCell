"""The configuration of one fit. The defaults are the paper's final configuration.

User-facing options come first; the advanced block below them holds values the
paper fixed (network sizes, optimizer and adversary settings, the warm-up and
the early-stopping rule). They are documented and validated, but changing them
leaves the configuration the paper evaluated.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from discell.io.anndata import UNASSIGNED


@dataclass(frozen=True, kw_only=True)
class Config:
    """One fit.

    Parameters
    ----------
    label_key
        ``.obs`` column of cell types (required). Missing values count as
        *unassigned_label*.
    kappa
        Spillover fraction, in ``[0, 1)``: the assumed share of a cell's
        expected counts that comes from its neighbors. Fixed for a fit; the
        sweep refits over a grid of values.
    seed
        Seeds the initialization, the training shuffle, the held-out tile split,
        the image PCA subsample and the image clusters.
    unassigned_label
        Type name of unannotated cells. They stay in training and in the graph
        but are never a metric target.
    image_key
        ``.obsm`` key of a per-cell image descriptor; None fits the image-free
        model.
    alpha_z
        Weight of ``KL(q(z) || N(0, I))``. ``"auto"`` = 0.5 / median total count
        over all cells; a number is used as given.
    alpha_a
        Weight of the adversary term; 0 gives the uncontrolled reference fit.
    tile_cells
        Maximum cells per spatial tile (one training step).
    epochs, patience
        Epoch budget, and epochs without an accepted checkpoint before stopping.
    device
        Torch device. A CUDA device that is not available falls back to the CPU
        with a warning.
    counts_layer
        ``.layers`` key of the raw counts; None reads ``.X``.
    """

    label_key: str
    kappa: float = 0.1
    seed: int = 0
    unassigned_label: str = UNASSIGNED
    image_key: str | None = None
    alpha_z: float | str = "auto"
    alpha_a: float = 0.3
    tile_cells: int = 4096
    epochs: int = 500
    patience: int = 40
    device: str = "cuda"
    counts_layer: str | None = None
    # -- advanced: fixed in the paper -------------------------------------
    d_z: int = 20                   #: intrinsic-state latent size
    d_w: int = 6                    #: niche-response latent size
    hidden: int = 256               #: encoder and decoder width
    gat_dim: int = 32               #: GAT output width
    heads: int = 4                  #: GAT heads
    v_pcs: int = 12                 #: image PCs in the probe block and under e_Phi
    omega: float = 1.0              #: weight of the prior-drawn-w bound
    alpha_w: float = 0.1            #: weight of KL(q(w) || N(m_psi(c, t), I))
    w_warmup_epochs: int = 30       #: alpha_w ramps up linearly over these epochs
    adv_lr: float = 2e-3            #: adversary learning rate
    adv_steps: int = 6              #: adversary updates per model update
    adv_hidden: int = 64            #: adversary head width
    adv_comp_weight: float = 3.0    #: weight of the composition half of the adversary
    lr: float = 1e-3                #: model learning rate (cosine schedule)
    grad_clip: float = 10.0         #: gradient-norm clip of the model update
    val_fraction: float = 0.15      #: share of tiles held out
    nmi_guard: float = 0.9          #: an accepted checkpoint keeps NMI >= this x its running max
    eval_every: int = 5             #: epochs between evaluations

    def __post_init__(self) -> None:
        if not isinstance(self.label_key, str) or not self.label_key:
            raise ValueError("label_key must name an .obs column")
        if not 0.0 <= self.kappa < 1.0:
            raise ValueError(f"kappa must be in [0, 1), got {self.kappa}")
        if isinstance(self.alpha_z, str):
            if self.alpha_z != "auto":
                raise ValueError(f"alpha_z must be 'auto' or a number, got {self.alpha_z!r}")
        elif not self.alpha_z > 0:
            raise ValueError(f"alpha_z must be positive, got {self.alpha_z}")
        for name in ("alpha_a", "alpha_w", "omega", "adv_comp_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("tile_cells", "epochs", "patience", "eval_every", "d_z", "d_w",
                     "gat_dim", "heads", "v_pcs", "adv_steps", "adv_hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.hidden < 4:
            raise ValueError(f"hidden must be >= 4, got {self.hidden}")
        if self.w_warmup_epochs < 0:
            raise ValueError(f"w_warmup_epochs must be >= 0, got {self.w_warmup_epochs}")
        for name in ("lr", "adv_lr", "grad_clip"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if not 0.0 <= self.nmi_guard <= 1.0:
            raise ValueError(f"nmi_guard must be in [0, 1], got {self.nmi_guard}")

    @property
    def image_free(self) -> bool:
        return self.image_key is None

    def alpha_w_at(self, epoch: int) -> float:
        """``alpha_w`` during the warm-up: ``alpha_w * min(1, epoch / w_warmup_epochs)``."""
        if self.w_warmup_epochs <= 0:
            return self.alpha_w
        return self.alpha_w * min(1.0, epoch / self.w_warmup_epochs)

    def resolve(self, median_counts: float) -> Config:
        """This configuration with ``alpha_z = "auto"`` replaced by its value."""
        if self.alpha_z != "auto":
            return self
        return dataclasses.replace(self, alpha_z=0.5 / float(median_counts))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> Config:
        return cls(**record)
