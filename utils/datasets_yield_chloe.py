"""
yield_dataset.py — ISA Yield Prediction Dataset
- Reads .npy files
- Temporal flatten: (T, C, H, W) -> (T*C, H, W)
- Train/val/test split based on txt files
- Modalities: S1GRD, S2L2A, DEM, WEATHER, CDL (no SOIL)
- Label: yield (1, 224, 224) float32, already normalized [-1, 1]
- Normalization stats: dynamically loaded from pretrain's IA npz files
  (must match pretrain distribution — see refac_data_constants_chloe.py)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from utils.refac_data_constants_chloe import load_state_stats


# Modality folder name mapping
MODALITY_FOLDER = {
    's1':        'S1GRD',
    's2':        'S2L2A',
    'elevation': 'DEM',
    'weather':   'WEATHER',
    'cdl':       'CDL',
    'yield':     'yield_geotiffs',
}


def build_yield_stats(stats_dir: str, state: str = 'IA') -> dict:
    """
    Load normalization stats from pretrain's npz files (same stats used
    during pretraining), so the pretrained encoder sees the same input
    distribution it was trained on.

    cdl is a class label (not normalized) so mean=0/std=1 (no-op).
    """
    stats = load_state_stats(stats_dir, state)
    mean_dict = dict(stats['mean'])
    std_dict  = dict(stats['std'])

    mean_dict['cdl'] = [0]
    std_dict['cdl']  = [1]

    return {'mean': mean_dict, 'std': std_dict}


class YieldDataset(Dataset):
    """
    ISA Yield Prediction Dataset.

    Args:
        txt_path (str): path to train/val/test_corn.txt
        data_root (str): base directory containing S1GRD, S2L2A, DEM, WEATHER, CDL, yield_geotiffs
        in_domains (list): input modalities e.g. ['s1', 's2', 'elevation', 'weather', 'cdl']
        temporal_steps (int): T (number of timesteps in npy, e.g. 15)
        stats (dict): {'mean': {...}, 'std': {...}} — from build_yield_stats()
    """
    def __init__(
        self,
        txt_path: str,
        data_root: str,
        in_domains: List[str],
        stats: dict,
        temporal_steps: int = 15,
    ):
        self.data_root      = data_root
        self.in_domains     = in_domains
        self.temporal_steps = temporal_steps
        self.stats          = stats

        with open(txt_path, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]

        print(f"✅ YieldDataset loaded: {len(self.samples)} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, domain: str, filename: str) -> torch.Tensor:
        folder = MODALITY_FOLDER[domain]
        path   = os.path.join(self.data_root, folder, f"{filename}.npy")
        arr    = np.load(path).astype(np.float32)
        return torch.from_numpy(arr)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        filename = self.samples[index]
        out = {}

        for domain in self.in_domains:
            x = self._load_npy(domain, filename)  # (T, C, H, W)

            # if domain == 'cdl':
            #     # binary remap: corn(1)/soybean(5)=0, others=1
            #     x = x.long()  # (15, 1, H, W)
            #     remapped = torch.ones_like(x)
            #     remapped[(x == 1) | (x == 5)] = 0
            #     # temporal flatten: (15, 1, H, W) → (15, H, W)
            #     T, C, H, W = remapped.shape
            #     remapped = remapped.reshape(T * C, H, W).float()
            #     out[domain] = remapped  # (15, H, W)
            #     continue

            if domain == 'cdl':
                # TerraMind와 동일하게: remap 없이 원본 5-class 값(0~4) 그대로 사용
                x = x.float()  # (15, 1, H, W)
                T, C, H, W = x.shape
                x = x.reshape(T * C, H, W)
                out[domain] = x
                continue

            # Normalize (using pretrain-consistent stats)
            mean = torch.tensor(self.stats['mean'][domain]).float().view(1, -1, 1, 1)
            std  = torch.tensor(self.stats['std'][domain]).float().view(1, -1, 1, 1)
            x = (x - mean) / (std + 1e-6)
            x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
            x = torch.clamp(x, -5, 5)

            # Temporal flatten: (T, C, H, W) -> (T*C, H, W)
            T, C, H, W = x.shape
            x = x.reshape(T * C, H, W)
            out[domain] = x

        # Yield label: (1, 224, 224) float32, range [-1, 1]
        out['yield'] = self._load_npy('yield', filename)

        return out


def build_yield_datasets(args):
    """
    Build train/val/test YieldDataset.

    Required args:
        args.yield_data_root : path to processed_data_bs20_rm_11_14_2_10_9
        args.in_domains      : ['s1', 's2', 'elevation', 'weather', 'cdl']
        args.temporal_steps  : 15
        args.stats_dir        : path to pretrain's stats npz directory
        args.yield_state      : e.g. 'IA' (default 'IA' if not set)
    """
    base  = args.yield_data_root
    state = getattr(args, 'yield_state', 'IA')

    stats = build_yield_stats(stats_dir=args.stats_dir, state=state)
    print(f"✅ Yield normalization stats loaded from pretrain ({state}): {list(stats['mean'].keys())}")

    train_ds = YieldDataset(
        txt_path=os.path.join(base, 'train_corn.txt'),
        data_root=base,
        in_domains=args.in_domains,
        stats=stats,
        temporal_steps=args.temporal_steps,
    )
    val_ds = YieldDataset(
        txt_path=os.path.join(base, 'val_corn.txt'),
        data_root=base,
        in_domains=args.in_domains,
        stats=stats,
        temporal_steps=args.temporal_steps,
    )
    test_ds = YieldDataset(
        txt_path=os.path.join(base, 'test_corn.txt'),
        data_root=base,
        in_domains=args.in_domains,
        stats=stats,
        temporal_steps=args.temporal_steps,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds