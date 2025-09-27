# datasets_chloe.py
# --------------------------------------------------------
import random
import numpy as np
from typing import Dict

import torch
import torch.nn.functional as F
from torchvision import transforms

from .data_constants_chloe import (
    S2_DEFAULT_MEAN, S2_DEFAULT_STD,
    S1_DEFAULT_MEAN, S1_DEFAULT_STD
)
from .dataset_folder_chloe import MultiTaskImageFolder

def denormalize(img: torch.Tensor, mean, std):
    # img: [C,H,W]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std  = torch.tensor(std).view(-1, 1, 1)
    return img * std + mean


class DataAugmentationForMultiMAE:
    """
    task_dict: {task_name: Tensor[C,H,W]}
    """
    def __init__(self, args):
        self.mean = {'s1': S1_DEFAULT_MEAN, 's2': S2_DEFAULT_MEAN}
        self.std  = {'s1': S1_DEFAULT_STD, 's2': S2_DEFAULT_STD}
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.all_domains = args.all_domains
        # Crop params
        self.scale = (0.5, 1.0)
        self.ratio = (0.75, 1.3333)

    def __call__(self, task_dict: Dict[str, torch.Tensor]):
        # 1) 하나의 랜덤 크롭 파라미터 생성 (첫 도메인 기준)
        first = self.all_domains[0]
        _, H, W = task_dict[first].shape
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            torch.empty(1, H, W), scale=self.scale, ratio=self.ratio
        )
        do_flip = random.random() < self.hflip

        # 2) 모든 도메인 동일하게 crop/resize/flip + normalize
        out = {}
        for task in self.all_domains:
            x = task_dict[task]  # [C,H,W]
            x = x[:, i:i+h, j:j+w]
            x = F.interpolate(
                x.unsqueeze(0), size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            if do_flip:
                x = torch.flip(x, dims=[2])  # horizontal flip (W dim)

            mean = torch.tensor(self.mean[task]).view(-1, 1, 1)
            std  = torch.tensor(self.std[task]).view(-1, 1, 1)
            x = (x - mean) / std

            if not torch.isfinite(x).all():
                print(f"[NaN after normalization in {task}] min={x.min().item()} max={x.max().item()} mean={x.mean().item()}")


            x = np.clip(x, -3, 3) # 튀는값 제거 추가 *chloe*

            out[task] = x

            # if task == 's2':
            #     print("s2 raw min/max:", task_dict['s2'].min().item(), task_dict['s2'].max().item())
            #     print("s2 norm min/max:", x.min().item(), x.max().item())
            # else:
            #     print("s1 raw min/max:", task_dict['s1'].min().item(), task_dict['s1'].max().item())
            #     print("s1 norm min/max:", x.min().item(), x.max().item())

        return out

    def __repr__(self):
        return f"(DataAugmentationForMultiMAE input_size={self.input_size}, hflip={self.hflip})"


def build_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    txt_paths = {d: getattr(args, f"{d}_txt") for d in args.all_domains}

    return MultiTaskImageFolder(
        tasks=args.all_domains,
        txt_paths=txt_paths,
        transform=transform,
        root=args.data_path,                      # None도 허용됨
        max_images=getattr(args, "max_images", None)
    )
