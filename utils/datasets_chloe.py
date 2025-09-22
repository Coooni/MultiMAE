# datasets_chloe.py
# --------------------------------------------------------
import random
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

        self.clip_z = getattr(args, 'clip_z', {'s1': 3.0, 's2': 3.0})

    # def __call__(self, task_dict: Dict[str, torch.Tensor]):
    #     # 1) 하나의 랜덤 크롭 파라미터 생성 (첫 도메인 기준)
    #     first = self.all_domains[0]
    #     _, H, W = task_dict[first].shape
    #     i, j, h, w = transforms.RandomResizedCrop.get_params(
    #         torch.empty(1, H, W), scale=self.scale, ratio=self.ratio
    #     )
    #     do_flip = random.random() < self.hflip

    #     # 2) 모든 도메인 동일하게 crop/resize/flip + normalize
    #     out = {}
    #     for task in self.all_domains:
    #         x = task_dict[task]  # [C,H,W]
    #         x = x[:, i:i+h, j:j+w]
    #         x = F.interpolate(
    #             x.unsqueeze(0), size=(self.input_size, self.input_size),
    #             mode='bilinear', align_corners=False
    #         ).squeeze(0)
    #         if do_flip:
    #             x = torch.flip(x, dims=[2])  # horizontal flip (W dim)

    #         mean = torch.tensor(self.mean[task]).view(-1, 1, 1)
    #         std  = torch.tensor(self.std[task]).view(-1, 1, 1)
    #         x = (x - mean) / std

    #         out[task] = x

    #     return out

    def __call__(self, task_dict: Dict[str, torch.Tensor]):
        first = self.all_domains[0]
        _, H, W = task_dict[first].shape
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            torch.empty(1, H, W), scale=self.scale, ratio=self.ratio
        )
        do_flip = random.random() < self.hflip

        out = {}
        for task in self.all_domains:
            x = task_dict[task].clone()  # [C,H,W]; 로더에서 NoData=NaN

            # (A) 유효 픽셀 마스크(모든 채널 finite) [1,H,W]
            valid = torch.isfinite(x).all(dim=0, keepdim=True).float()

            # (B) 같은 파라미터로 crop
            x = x[:, i:i+h, j:j+w]              # [C,h,w]
            m = valid[:, i:i+h, j:j+w]          # [1,h,w]

            # (C) ★ 보간 전에 NaN 채우기 (invalid→μ)
            mean = torch.tensor(self.mean[task], device=x.device, dtype=x.dtype).view(-1,1,1)
            std  = torch.tensor(self.std[task],  device=x.device, dtype=x.dtype).view(-1,1,1)
            x = torch.where(m > 0.5, x, mean.expand_as(x))  # 이제 x에는 NaN 없음

            # (D) resize (x=bilinear, m=nearest)
            x = F.interpolate(
                x.unsqueeze(0), size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            m = F.interpolate(
                m.unsqueeze(0), size=(self.input_size, self.input_size),
                mode='nearest'
            ).squeeze(0)

            # (E) optional flip (x, m 동일하게)
            if do_flip:
                x = torch.flip(x, dims=[2])
                m = torch.flip(m, dims=[2])

            # (F) 정규화 (invalid였던 픽셀은 이미 μ로 채워졌으니, 정규화 후 0이 됨)
            x = (x - mean) / std

            # ★ z-클리핑 (튀는 값 방어)
            k = None if self.clip_z is None else self.clip_z.get(task, None)
            if k is not None:
                x = torch.clamp(x, min=-k, max=k)

            out[task] = x                        # 모델 입력
            out[f"{task}_valid_mask"] = m        # [1,H,W], 손실 가중치용

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
