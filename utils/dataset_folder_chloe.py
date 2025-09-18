# dataset_folder_chloe.py
# --------------------------------------------------------
# Minimal, cleaned version for MODIS / S2 txt-file loading
# --------------------------------------------------------

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torchvision.datasets.vision import VisionDataset

import pdb

# --------------------------------------------------------
# Basic utilities
# --------------------------------------------------------

IMG_EXTENSIONS: Tuple[str, ...] = (".tif", )


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


# *** 125m resolution
# def rasterio_loader(path: str) -> torch.Tensor:
#     if "MODIS" in path:
#         """MODIS 로더"""
#         with rasterio.open(path) as src:
#             img = src.read(out_dtype='float32')          # (C, H, W) 0‑10000 DN
#             # ---- nodata 처리 ----
#             img[img == 32767] = 0.0                      # 또는 np.nan
#             rgb = img[[0, 3, 2], ...]
        
#     else:
#         """S2 로더: [C, H, W] float32, reflectance 0–1 스케일"""
#         with rasterio.open(path) as src:
#             img = src.read(out_dtype='float32')          # (C, H, W) 0‑10000 DN
#             rgb = img[[3, 2, 1], ...]
    
#     rgb = rgb / 10000.0                          # reflectance 0‑1

#     return torch.from_numpy(rgb.copy()).float()

# *** 30m resolution
def rasterio_loader(path: str) -> torch.Tensor:
    if "S1" in path:
        """S1 로더"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')         
            # ---- nodata 처리 ----
            img[img == -9999] = np.nan                      # 또는 np.nan
            # ---------------------
            img = img
    elif "S2" in path:
        """S2 로더: [C, H, W] float32, reflectance 0–1 스케일"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')          # (C, H, W) 0‑10000 DN
            # ---- nodata 처리 ----
            img[img == -9999] = np.nan                      # 또는 np.nan
            # ---------------------
            img = img / 10000.0                          # reflectance 0‑1
    else:
        # CDL
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')         
            img = img
    return torch.from_numpy(img).float()             # torch.Tensor
    



# --------------------------------------------------------
# Multi-modal dataset driven by txt lists
# --------------------------------------------------------

class MultiTaskImageFolder(VisionDataset):
    """
    Multi-modal loader that reads full file paths from per-task txt files.

    Args:
        tasks (list[str]): e.g. ['modis', 's2']
        txt_paths (dict[str,str]): {'modis': '/abs/modis.txt', 's2': '/abs/s2.txt', ...}
        transform (callable): dict[str, Tensor[C,H,W]] -> dict[str, Tensor[C,H,W]]
        loader (callable): path -> Tensor[C,H,W]
        root (str or None): base dir for relative paths (optional)
        max_images (int or None): subsampling
    """
    def __init__(
        self,
        tasks: List[str],
        txt_paths: Dict[str, str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = rasterio_loader,
        root: Optional[str] = None,
        max_images: Optional[int] = None,
    ):
        super().__init__(root or "", transform=transform, target_transform=target_transform)
        self.tasks = tasks
        self.loader = loader
        self.root = root
        self.samples: Dict[str, List[str]] = {}

        def _resolve(p: str) -> str:
            return p if os.path.isabs(p) else (os.path.join(self.root, p) if self.root else p)

        # read lists
        for t in tasks:
            with open(txt_paths[t], "r") as f:
                paths = [line.strip() for line in f if is_image_file(line.strip())]
            self.samples[t] = [_resolve(p) for p in paths]

        # length check
        lens = [len(self.samples[t]) for t in tasks]
        if len(set(lens)) != 1:
            raise ValueError(f"Txt lengths must match: {dict(zip(tasks, lens))}")
        self.num_samples = lens[0]

        # subsample if requested
        if max_images is not None and max_images < self.num_samples:
            idx = random.sample(range(self.num_samples), max_images)
            for t in tasks:
                self.samples[t] = [self.samples[t][i] for i in idx]
            self.num_samples = max_images

        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index not in self._cache:
            d: Dict[str, torch.Tensor] = {}
            for t in self.tasks:
                d[t] = self.loader(self.samples[t][index])
            self._cache[index] = d
        sample = self._cache[index].copy()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
