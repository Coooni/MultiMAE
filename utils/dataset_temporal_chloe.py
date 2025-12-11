# temporal_dataset.py
# --------------------------------------------------------
import os
import random
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import rasterio
from .data_constants_chloe import (
    S2_DEFAULT_MEAN, S2_DEFAULT_STD,
    S1_DEFAULT_MEAN, S1_DEFAULT_STD,
    SOIL_DEFAULT_MEAN, SOIL_DEFAULT_STD,
    ELEVATION_DEFAULT_MEAN, ELEVATION_DEFAULT_STD,
    WEATHER_DEFAULT_MEAN, WEATHER_DEFAULT_STD
)



corn_soy_classes = [
    1,   # Corn
    12,  # Sweet Corn
    13,  # Pop/Orn Corn
    225, 226, 228, 237, 241,  # Double crop with corn
    5, 26,   # Soybeans
    239, 240,           # Double crop soybeans
]


# --------------------------------------------------------
# Loader
# --------------------------------------------------------
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
        # # img ∈ [0..254]
        # others = (img != 1) & (img != 5)

        # # 0: others
        # img = torch.where(others, torch.tensor(0, device=img.device), img)

        # # 1: corn (already 1)
        # img = torch.where(img == 1, torch.tensor(1, device=img.device), img)

        # # 2: soybean
        # img = torch.where(img == 5, torch.tensor(2, device=img.device), img)


        # # ** 2 classes for corn & soybean and others **
        # img = torch.where((img == 1) | (img == 5), 1, 0)


        # 2 classes for every corn & every soybean (not only for class 1,class 5)
        img = torch.where(torch.isin(img, torch.tensor(corn_soy_classes, device=img.device)), 1, 0)

        return img

# --------------------------------------------------------
# Normalize + Augmentation
# --------------------------------------------------------
class DataAugmentationForMultiMAE:
    """
    Simple normalization for downstream tasks (CDL prediction).
    No random augmentations are applied.
    """
    def __init__(self, input_size=224, hflip=False, all_domains=["s1", "s2", "cdl", "soil", "elevation", "weather"]):
        self.mean = {'s1': S1_DEFAULT_MEAN, 's2': S2_DEFAULT_MEAN, 'soil': SOIL_DEFAULT_MEAN, 'elevation': ELEVATION_DEFAULT_MEAN, 'weather': WEATHER_DEFAULT_MEAN}
        self.std = {'s1': S1_DEFAULT_STD, 's2': S2_DEFAULT_STD, "soil": SOIL_DEFAULT_STD, "elevation": ELEVATION_DEFAULT_STD, "weather": WEATHER_DEFAULT_STD}
        self.input_size = input_size
        self.hflip = hflip
        self.all_domains = all_domains

    def __call__(self, task_dict: Dict[str, torch.Tensor]):
        out = {}
        for task in self.all_domains:
            x = task_dict[task]
            if task == "cdl":
                x = x.long()
            else:
                mean = torch.tensor(self.mean[task]).view(-1, 1, 1)
                std = torch.tensor(self.std[task]).view(-1, 1, 1)
                x = (x - mean) / std
                x = torch.clamp(x, -3, 3)
            out[task] = x
        return out

    def __repr__(self):
        return f"(DataAugmentationForMultiMAE input_size={self.input_size}, hflip={self.hflip})"

# --------------------------------------------------------
# Temporal Dataset
# --------------------------------------------------------
class MultiTaskTemporalImageFolder(Dataset):
    """
    Temporal multi-modal dataset for CDL prediction.
    Each patch contains multiple timestamps (e.g., T=5 time points: June–Oct).

    Returns:
        {
            "s1": [T, 2, H, W],
            "s2": [T, 12, H, W],
            "cdl": [H, W]
        }
    """
    def __init__(
        self,
        tasks: List[str],
        txt_paths: Dict[str, str],
        transform: Optional[Callable] = None,
        loader: Callable[[str], torch.Tensor] = rasterio_loader,
        T: int = 2,
        root: Optional[str] = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.txt_paths = txt_paths
        self.loader = loader
        self.transform = transform
        self.T = T
        self.root = root

        # --- Group file paths by patch ---
        self.samples = {t: self._group_by_patch(txt_paths[t]) for t in tasks}

        # --- Align patches across domains ---
        patch_sets = [set(self.samples[t].keys()) for t in tasks]
        self.common_patches = sorted(set.intersection(*patch_sets))
        print(f"✅ Found {len(self.common_patches)} patches with full {T}-month series for all domains")

    def _group_by_patch(self, txt_path):
        patch_dict = defaultdict(list)
        with open(txt_path, "r") as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                patch_name = "_".join(os.path.basename(os.path.dirname(path)).split("_")[-2:])
                patch_dict[patch_name].append(path)

        # Sort by date (YYYY-MM-DD in filename)
        for k in patch_dict.keys():
            patch_dict[k] = sorted(patch_dict[k])
        return patch_dict

    def __len__(self):
        return len(self.common_patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        patch_id = self.common_patches[index]
        data_dict: Dict[str, torch.Tensor] = {}

        for task in self.tasks:
            paths = self.samples[task][patch_id]
            if len(paths) != self.T:
                raise ValueError(f"Patch {patch_id} in {task} has {len(paths)} != {self.T}")

            imgs = [self.loader(p) for p in paths]
            stack = torch.stack(imgs, dim=0)  # [T, C, H, W] (or [T,H,W] for CDL)
            data_dict[task] = stack

        # CDL은 시간축 제거
        if data_dict["cdl"].dim() == 3:
            data_dict["cdl"] = data_dict["cdl"][0]  # [H, W]

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

# --------------------------------------------------------
# Example usage
# --------------------------------------------------------
if __name__ == "__main__":
    args = type("Args", (), {})()
    args.all_domains = ["s1", "s2", "cdl", "soil", "elevation", "weather"]
    args.input_size = 224
    args.hflip = False

    txt_paths = {
        "s1": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_S1.txt",
        "s2": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_S2.txt",
        "cdl": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_CDL.txt",
        "soil": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_Soil.txt",
        "elevation": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_Elevation.txt",
        "weather": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_0708_Weather.txt"
        }

    dataset = MultiTaskTemporalImageFolder(
        tasks=args.all_domains,
        txt_paths=txt_paths,
        transform=DataAugmentationForMultiMAE(args.input_size, args.hflip, args.all_domains),
        T=2,
    )

    sample = dataset[0]
    print({k: v.shape for k, v in sample.items()})
