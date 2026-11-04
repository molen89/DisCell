"""Optional per-cell image descriptors (the ``[image]`` extra), from KRONOS2.

KRONOS2's weights are gated and never distributed with DisCell; :func:`embed`
uses the user's own Hugging Face access. Importing this package needs no extra:
``tifffile``, ``zarr``, ``transformers`` and ``huggingface_hub`` load on use.
KRONOS v1, which the paper's image descriptor used, is deprecated by its authors
and closed to new access requests, and is not supported.
"""

from discell.image.crops import DEFAULT_MASK_RADIUS_UM, MASK_MODES, ego_disc, iter_crop_blocks
from discell.image.kronos import ACCESS_MESSAGE, embed
from discell.image.kronos2 import XENIUM_MARKER_NAMES, load_kronos2, require_kronos2
from discell.image.markers import intensity_scaling_factor
from discell.image.tiff import find_morphology, image_shape, read_raw_window

__all__ = ["ACCESS_MESSAGE", "DEFAULT_MASK_RADIUS_UM", "MASK_MODES", "XENIUM_MARKER_NAMES",
           "ego_disc", "embed", "find_morphology", "image_shape", "intensity_scaling_factor",
           "iter_crop_blocks", "load_kronos2", "read_raw_window", "require_kronos2"]
