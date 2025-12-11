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

# corn_soy_classes = [
#     1,   # Corn
#     12,  # Sweet Corn
#     13,  # Pop/Orn Corn
#     225, 226, 228, 237, 241,  # Double crop with corn
#     5, 26,   # Soybeans
#     239, 240,           # Double crop soybeans
# ]

# -----------------------------
# 3-class mapping for CDL
# Classes:
#   0 = non-vegetation
#   1 = natural vegetation
#   2 = crops
# -----------------------------
crop_classes = [
    1, 2, 3, 4, 5, 6,
    10, 11, 12, 13, 14,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35,
    36, 37, 38, 39,
    41, 42, 43, 44, 45, 46,
    47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57,
    66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77,
    204, 205,
    206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250,
    254,
]

natural_veg_classes = [
    63, 64, 141, 142, 143,         # Forest + shrub
    152,                           # Shrub
    176,                           # Grass/Pasture
    190, 195,                      # Wetlands
    58, 59, 60,                    # Wildflowers / Grass seed / Switchgrass
]

nonveg_classes = [
    0, 65, 131,                    # Barren + background
    81, 83, 111, 112,              # Cloud, Water, Ice/Snow
    82, 121, 122, 123, 124,        # Developed
    88, 92,                        # Non-ag / Aquaculture
    61,                            # Fallow / Idle cropland -> bare soil
]

# tensor로 미리 변환 (성능 향상)
crop_classes = torch.tensor(crop_classes)
natural_veg_classes = torch.tensor(natural_veg_classes)
nonveg_classes = torch.tensor(nonveg_classes)



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
            img[img == -9999] = 0                      # 또는 np.nan
        return torch.from_numpy(img).float()             # torch.Tensor

    elif "S2" in path:
        """S2 로더: [C, H, W] float32, reflectance 0–1 스케일"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')          # (C, H, W) 0‑10000 DN
            img[img == -9999] = 0                  # 또는 np.nan                        
        return torch.from_numpy(img).float()             # torch.Tensor

    elif "Soil" in path or "soil" in path:
        """Soil 로더: [10, H, W] float32, 원본 물리 단위"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')          # (10, H, W)
            img[img == -9999] = 0                        # NoData 처리
        return torch.from_numpy(img).float()
    
    elif "Elevation" in path or "elevation" in path:
        """Elevation 로더: [1, H, W] float32, 미터 단위"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')          # (1, H, W)
            img[img == -9999] = 0                        # NoData 처리
        return torch.from_numpy(img).float()
    
    elif "Weather" in path or "weather" in path:
        """Weather 로더: [1, H, W] float32, 미터 단위"""
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')          # (1, H, W)
            img[img == -9999] = 0                        # NoData 처리
        return torch.from_numpy(img).float()
    
    else:
        # CDL
        with rasterio.open(path) as src:
            img = src.read(1, out_dtype='int32')  # 첫 채널만 읽기
        img = torch.from_numpy(img).long()

        # # ** 3 classes for corn, soybean, others **
        # others = (img != 1) & (img != 5)
        # # 0: others
        # img = torch.where(others, torch.tensor(0, device=img.device), img)
        # # 1: corn (already 1)
        # img = torch.where(img == 1, torch.tensor(1, device=img.device), img)
        # # 2: soybean
        # img = torch.where(img == 5, torch.tensor(2, device=img.device), img)


        # # ** 2 classes for corn & soybean and others **
        # img = torch.where((img == 1) | (img == 5), 1, 0)

        # # 2 classes for every corn & every soybean (not only for class 1,class 5)
        # img = torch.where(torch.isin(img, torch.tensor(corn_soy_classes, device=img.device)), 1, 0)

        # default = non-veg (0)
        new = torch.zeros_like(img)

        # natural vegetation → 1
        new[torch.isin(img, natural_veg_classes)] = 1

        # crops → 2
        new[torch.isin(img, crop_classes)] = 2

        return new
        # return img


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

        # self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.num_samples

    # def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
    #     if index not in self._cache:
    #         d: Dict[str, torch.Tensor] = {}
    #         for t in self.tasks:
    #             d[t] = self.loader(self.samples[t][index])
    #         self._cache[index] = d
    #     sample = self._cache[index].copy()

    #     if self.transform is not None:
    #         sample = self.transform(sample)

    #     return sample

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        d: Dict[str, torch.Tensor] = {}
        for t in self.tasks:
            d[t] = self.loader(self.samples[t][index])

        if self.transform is not None:
            d = self.transform(d)

        return d
