"""s2_dataset.py
================
PyTorch `Dataset` for Sentinel‑2 patch tensors (DN scale).

Hard‑coded resources
--------------------
LIST_FILE  = "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/valid_list/valid_S2.txt"
STATS_FILE = "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/statistic/s2_stats_from_list.npz"

Assumptions
-----------
* All patches are **already in raw Digital Number (DN) scale**, with nodata
  pixels set to **0** during earlier preprocessing.
* The statistics file stores channel‑wise ``mean`` and ``std`` **in the same DN
  scale**.

The dataset performs *standardisation* in DN space, **then restores nodata
pixels back to 0** so that the network never sees extreme negative values at
those locations.

Example
-------
>>> ds = S2Dataset()
>>> x, mask = ds[0]
>>> x.shape  # (C, H, W)
>>> mask.float().mean()  # valid‑pixel ratio
"""
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Hard‑coded paths
# -----------------------------------------------------------------------------
LIST_FILE: Path  = Path(
    "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/valid_list/valid_S2.txt"
)
STATS_FILE: Path = Path(
    "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/statistic/s2_stats_from_list.npz"
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NODATA_DN: int = 0  # Sentinel‑2 nodata value after preprocessing

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_stats(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load channel‑wise mean/std tensors from ``path`` (DN scale)."""
    stats = np.load(path)
    mean = torch.from_numpy(stats["mean"]).float()  # (C,)
    std  = torch.from_numpy(stats["std"]).float()   # (C,)
    return mean, std


def _read_list(file_path: Path) -> List[Path]:
    """Return list of absolute file paths from a newline‑delimited text file."""
    with open(file_path, "r") as f:
        return [Path(line.strip()) for line in f if line.strip()]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class S2Dataset(Dataset):
    """Sentinel‑2 patch dataset using the hard‑coded file list.

    Parameters
    ----------
    list_file : Path, optional
        Alternative list file path. If ``None``, uses the global ``LIST_FILE``.
    stats_file : Path, optional
        Alternative stats file path. If ``None``, uses ``STATS_FILE``.
    return_mask : bool, default ``True``
        If ``True``, ``__getitem__`` returns ``(tensor, mask)`` where *mask* is
        a boolean (H, W) tensor indicating valid pixels.
    """

    def __init__(
        self,
        list_file: Optional[Path] = None,
        stats_file: Optional[Path] = None,
        return_mask: bool = True,
    ) -> None:
        super().__init__()
        self.paths = _read_list(list_file or LIST_FILE)
        self.mean, self.std = _load_stats(stats_file or STATS_FILE)
        self.return_mask = return_mask

    # --------------------------------------------------------------- PyTorch API
    def __len__(self) -> int:  # noqa: D401
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with rasterio.open(path) as src:
            dn = src.read().astype(np.float32)  # (C, H, W)

        # mask: True where pixel is valid (non‑nodata) across any band
        mask_np = (dn != NODATA_DN).any(axis=0)  # (H, W)

        # --- Standardise in DN scale -------------------------------------------------
        x = torch.from_numpy(dn)  # (C, H, W) float32 tensor
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]

        # --- Restore nodata pixels to 0 AFTER normalisation --------------------------
        if not mask_np.all():  # only do the masking when nodata exists
            mask_t = torch.from_numpy(mask_np)  # (H, W) bool tensor
            x[:, ~mask_t] = 0.0                # broadcast over channel dim


        if self.return_mask:
            return x, torch.from_numpy(mask_np)
        return x

# -----------------------------------------------------------------------------
# Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ds = S2Dataset()
