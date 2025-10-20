# temporal_dataset.py
# --------------------------------------------------------
import os
import random
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import rasterio
from torchvision import transforms

# --------------------------------------------------------
# Default Mean / Std (30m Resolution)
# --------------------------------------------------------
S2_DEFAULT_MEAN = (0.056615, 0.06887008, 0.09391985, 0.10163148, 0.14381698, 0.25419976,
                   0.30873655, 0.32210684, 0.33362151, 0.34459104, 0.27966201, 0.19123411)
S2_DEFAULT_STD = (0.06765407, 0.06848592, 0.06620852, 0.08064312, 0.0791716, 0.08433618,
                  0.1150472, 0.11199426, 0.11933007, 0.13678636, 0.09880431, 0.10527262)


S1_DEFAULT_MEAN = (0.12419162, 0.02826689)
S1_DEFAULT_STD = (0.41080412, 0.04929494)

# --------------------------------------------------------
# Loader
# --------------------------------------------------------
def rasterio_loader(path: str) -> torch.Tensor:
    if "S1" in path:
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')
            img[img == -9999] = 0
        return torch.from_numpy(img).float()

    elif "S2" in path:
        with rasterio.open(path) as src:
            img = src.read(out_dtype='float32')
            img[img == -9999] = 0
            img = img / 10000.0
        return torch.from_numpy(img).float()

    else:  # CDL
        with rasterio.open(path) as src:
            img = src.read(1, out_dtype='int32')
        img = torch.from_numpy(img).long()
        # Label remapping: Corn=1, Soybean=5 → 1/2, Others=0
        img = torch.where(img == 1, torch.tensor(1, device=img.device), img)
        img = torch.where(img == 5, torch.tensor(2, device=img.device), img)
        img = torch.where((img != 1) & (img != 2),
                          torch.tensor(0, device=img.device), img)
        return img

# --------------------------------------------------------
# Normalize + Augmentation
# --------------------------------------------------------
class DataAugmentationForMultiMAE:
    """
    Simple normalization for downstream tasks (CDL prediction).
    No random augmentations are applied.
    """
    def __init__(self, input_size=224, hflip=False, all_domains=["s1", "s2", "cdl"]):
        self.mean = {'s1': S1_DEFAULT_MEAN, 's2': S2_DEFAULT_MEAN}
        self.std = {'s1': S1_DEFAULT_STD, 's2': S2_DEFAULT_STD}
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
        T: int = 5,
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
        if data_dict["cdl"].dim() == 4:
            data_dict["cdl"] = data_dict["cdl"][0]  # [H, W]

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

# --------------------------------------------------------
# Example usage
# --------------------------------------------------------
if __name__ == "__main__":
    args = type("Args", (), {})()
    args.all_domains = ["s1", "s2", "cdl"]
    args.input_size = 224
    args.hflip = False

    txt_paths = {
        "s1": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_temporal_S1_NEW.txt",
        "s2": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_temporal_S2_NEW.txt",
        "cdl": "/work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/valid_list/nova/30m/pair_temporal_CDL_NEW.txt",
    }

    dataset = MultiTaskTemporalImageFolder(
        tasks=args.all_domains,
        txt_paths=txt_paths,
        transform=DataAugmentationForMultiMAE(args.input_size, args.hflip, args.all_domains),
        T=5,
    )

    sample = dataset[0]
    print({k: v.shape for k, v in sample.items()})
