# datasets_chloe.py
# --------------------------------------------------------
# - Normalize using per-state npz stats
# - Combine multiple states into ConcatDataset
# --------------------------------------------------------
import random
from typing import Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset

from .refac_data_constants_chloe import load_state_stats
from .dataset_folder_chloe import MultiTaskImageFolder


def denormalize(img: torch.Tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std  = torch.tensor(std).view(-1, 1, 1)
    return img * std + mean


class DataAugmentationForMultiMAE:
    """
    Pretraining transform: random crop/flip + normalize.
    CDL is not normalized.

    Args:
        args:  input_size, hflip, all_domains
        stats: return value of load_state_stats()
               {'mean': {'s1': tuple, ...}, 'std': {'s1': tuple, ...}}
    """
    def __init__(self, args, stats: dict):
        self.mean        = stats['mean']
        self.std         = stats['std']
        self.input_size  = args.input_size
        self.hflip       = args.hflip
        self.all_domains = args.all_domains
        self.scale       = (0.5, 1.0)
        self.ratio       = (0.75, 1.3333)

    def __call__(self, task_dict: Dict[str, torch.Tensor]):
        # Use the first non-CDL domain as reference for crop params
        first = next(d for d in self.all_domains if d != 'cdl')
        _, H, W = task_dict[first].shape
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            torch.empty(1, H, W), scale=self.scale, ratio=self.ratio
        )
        do_flip = random.random() < self.hflip

        out = {}
        for task in self.all_domains:
            x = task_dict[task]  # [C, H, W]

            # crop + resize
            x = x[:, i:i+h, j:j+w]
            x = F.interpolate(
                x.unsqueeze(0), size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

            if do_flip:
                x = torch.flip(x, dims=[2])

            # normalize (skip CDL)
            if task != 'cdl':
                mean = torch.tensor(self.mean[task]).view(-1, 1, 1)
                std  = torch.tensor(self.std[task]).view(-1, 1, 1)
                x = (x - mean) / (std + 1e-6)
                x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
                x = torch.clamp(x, -5, 5)

                if not torch.isfinite(x).all():
                    print(f"[NaN after normalize: {task}] "
                          f"min={x.min().item():.3f} max={x.max().item():.3f}")

            out[task] = x

        return out

    def __repr__(self):
        return f"DataAugmentationForMultiMAE(input_size={self.input_size}, hflip={self.hflip})"


class DownstreamAugmentationForMultiMAE:
    """
    Downstream (CDL prediction) transform.
    - No augmentation (no crop/flip)
    - Normalize only
    - CDL: binary remap → corn(1)/soybean(5)=0, others=1, long [H, W]
    """
    def __init__(self, args, stats: dict):
        self.mean        = stats['mean']
        self.std         = stats['std']
        self.input_size  = args.input_size
        self.all_domains = args.all_domains

    def __call__(self, task_dict: Dict[str, torch.Tensor]):
        out = {}
        for task in self.all_domains:
            x = task_dict[task]  # [C, H, W]

            if task == 'cdl':
                x = x.long().squeeze(0)          # [H, W] long
                # binary remap: corn(1)/soybean(5) → 0, others → 1
                remapped = torch.ones_like(x)
                remapped[(x == 1) | (x == 5)] = 0
                out[task] = remapped             # [H, W] long
            else:
                mean = torch.tensor(self.mean[task]).view(-1, 1, 1)
                std  = torch.tensor(self.std[task]).view(-1, 1, 1)
                x = (x - mean) / (std + 1e-6)
                x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
                x = torch.clamp(x, -5, 5)
                out[task] = x

        return out

    def __repr__(self):
        return f"DownstreamAugmentationForMultiMAE(input_size={self.input_size})"


def _build_single_state_dataset(args, state: str, transform_cls):
    """Build dataset for a single state with the given transform class."""
    stats     = load_state_stats(args.stats_dir, state)
    transform = transform_cls(args, stats)
    txt_paths = args.txt_paths_by_state[state]

    return MultiTaskImageFolder(
        tasks=args.all_domains,
        txt_paths=txt_paths,
        transform=transform,
        root=args.data_path,
        max_images=getattr(args, 'max_images', None),
    )


def _build_dataset(args, transform_cls):
    """
    Build datasets for all states and combine into ConcatDataset.
    Returns a single dataset if only one state is given.
    """
    states = args.states  # already a list after post_process_args

    if len(states) == 1:
        ds = _build_single_state_dataset(args, states[0], transform_cls)
        print(f"  [{states[0]}] dataset size: {len(ds)}")
        return ds

    datasets = []
    for state in states:
        ds = _build_single_state_dataset(args, state, transform_cls)
        print(f"  [{state}] dataset size: {len(ds)}")
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    print(f"  Combined dataset size: {len(combined)}")
    return combined


def build_multimae_pretraining_dataset(args):
    """Pretraining dataset with random crop/flip + normalize."""
    return _build_dataset(args, DataAugmentationForMultiMAE)


def build_multimae_downstream_dataset(args):
    """Downstream dataset with normalize only (no augmentation)."""
    return _build_dataset(args, DownstreamAugmentationForMultiMAE)