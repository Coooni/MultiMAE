"""modis_dataset.py
===================
PyTorch `Dataset` for MODIS patch tensors (DN scale).

Hard‑coded resources
--------------------
LIST_FILE  = "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/valid_list/valid_MODIS.txt"
STATS_FILE = "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/statistic/modis_stats_from_list.npz"

Assumptions
-----------
* MODIS patches are **raw Digital Numbers (DN)** with nodata pixels
  encoded as **32767**.
* The statistics file stores channel‑wise ``mean`` and ``std`` in the same DN
  scale.

Extra features
--------------
* Optional **on‑the‑fly nodata interpolation** using a local mean kernel
  (SciPy ``generic_filter``).  Enable with ``interpolate=True``.
* If interpolation is used, the **original mask is still returned** so that
  loss functions can ignore those pixels.

Example
-------
>>> ds = MODISDataset(interpolate=True, kernel_size=9)
>>> x, mask = ds[0]
>>> x.shape, mask.shape
"""
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from scipy.ndimage import generic_filter

# -----------------------------------------------------------------------------
# Hard‑coded paths
# -----------------------------------------------------------------------------
LIST_FILE: Path = Path(
    "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/valid_list/valid_MODIS.txt"
)
STATS_FILE: Path = Path(
    "/work/mech-ai-scratch/bgekim/project/imputation/imputation_prithvi/statistic/modis_stats_from_list.npz"
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NODATA_DN: int = 32767  # MODIS nodata value

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_stats(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load channel‑wise mean/std tensors from ``path`` (DN scale)."""
    stats = np.load(path)
    mean = torch.from_numpy(stats["mean"]).float()
    std  = torch.from_numpy(stats["std"]).float()
    return mean, std


def _read_list(file_path: Path) -> List[Path]:
    with open(file_path, "r") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def _fill_nodata(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple spatial interpolation: replace nodata with local mean (ignoring NaN).

    Parameters
    ----------
    arr  : (C, H, W) float32 array in DN scale.
    size : int, kernel window size (odd).
    """
    filled = arr.copy()
    for c in range(arr.shape[0]):
        band = filled[c]
        mask = band == NODATA_DN
        if not mask.any():
            continue
        band = band.astype(np.float32)
        band_masked = band.copy()
        band_masked[mask] = np.nan

        band_filled = generic_filter(
            band_masked,
            lambda x: np.nanmean(x),
            size=size,
            mode="nearest",
        )
        # any remaining NaN (all‑NaN window) → 0
        band_filled = np.nan_to_num(band_filled, nan=0.0)
        filled[c] = band_filled
    return filled

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MODISDataset(Dataset):
    """MODIS patch dataset with optional on‑the‑fly interpolation."""

    def __init__(
        self,
        list_file: Optional[Path] = None,
        stats_file: Optional[Path] = None,
        return_mask: bool = True,
        *,
        interpolate: bool = False,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.paths = _read_list(list_file or LIST_FILE)
        self.mean, self.std = _load_stats(stats_file or STATS_FILE)
        self.return_mask = return_mask
        self.interpolate = interpolate
        self.kernel_size = kernel_size

    # ----------------------------------------------------------- PyTorch API --
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with rasterio.open(path) as src:
            dn = src.read().astype(np.float32)  # (C, H, W)

                # mask_original: True where any band is valid (pre‑interpolation)
        mask_np_original = (dn != NODATA_DN).any(axis=0)  # (H, W)

        # (optional) simple spatial interpolation before normalisation
        if self.interpolate and not mask_np_original.all():
            dn = _fill_nodata(dn, size=self.kernel_size)
            # after fill, every pixel has a value → new mask is all True
            pixel_mask_np = np.ones_like(mask_np_original, dtype=bool)
        else:
            pixel_mask_np = mask_np_original 
        if self.interpolate and not pixel_mask_np.all():
            dn = _fill_nodata(dn, size=self.kernel_size)

        # --- Standardise in DN scale ----------------------------------------
        x = torch.from_numpy(dn)  # (C, H, W)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]

        # --- Restore nodata pixels to 0 across all channels ------------------
        if not pixel_mask_np.all():
            mask3d = torch.from_numpy(pixel_mask_np).unsqueeze(0).expand_as(x)
            x = x.masked_fill(~mask3d, 0.0)

        if self.return_mask:
            return x, torch.from_numpy(pixel_mask_np)
        return x

# -----------------------------------------------------------------------------
# Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ds = MODISDataset(interpolate=True, kernel_size=9)
    # x, m = ds[0]
    # print("sample:", x.shape, m.float().mean().item())
