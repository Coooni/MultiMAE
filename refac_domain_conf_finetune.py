"""
Finetuning domain config — temporal stacking (T timesteps)
채널 수 = base_channels * T
base channels:
  s1: 2,  s2: 12,  elevation: 1,  soil: 10,  weather: 7
T=5 기준:
  s1: 10, s2: 60, elevation: 5, soil: 50, weather: 35
label:
  cdl   (segmentation) : [H, W] long   — temporal 없음
  yield (regression)   : [1, H, W] float — temporal 없음
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from multimae.criterion import MaskedMSELoss
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from multimae.output_adapters import DPTOutputAdapter

# base channels per modality (single time point)
BASE_CHANNELS = {
    's1':        2,
    's2':        12,
    'elevation': 1,
    'soil':      10,
    'weather':   7,
    'cdl':       1,
}


# class MaskedMSELossWrapper(nn.Module):
#     """
#     SpatialOutputAdapter output (B, 1, H, W) vs target (B, 1, H, W) 용
#     patch_size / stride 인자를 받지만 pixel-level MSE로 동작.
#     """
#     def __init__(self, patch_size: int = 16, stride: int = 1):
#         super().__init__()
#         self.loss = nn.MSELoss()

#     def forward(self, pred, target):
#         return self.loss(pred.float(), target.float())

class MaskedMSELossWrapper(nn.Module):
    """Exclude target=-1 pixels from yield MSE."""

    def __init__(self, patch_size: int = 16, stride: int = 1):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={pred.shape}, target={target.shape}"
            )

        valid = torch.isfinite(target) & (target != -1)

        if not valid.any():
            # Empty selection gives a differentiable zero.
            return pred[valid].sum()

        return F.mse_loss(pred[valid], target[valid])


class CDLCrossEntropyLoss(nn.Module):
    """
    CDL segmentation용 CrossEntropyLoss wrapper.
    pred  : [B, num_classes, H, W] float (logits)
    target: [B, H, W] long
    AMP 환경에서 NaN 없이 안정적으로 동작.
    """
    def __init__(self, patch_size: int = 16, stride: int = 1, num_classes: int = 2):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # pred: [B, num_classes, H, W], target: [B, H, W] long
        pred = pred.float()
        if target.dim() == 4:
            target = target.squeeze(1)   # [B, H, W]
        return self.loss(pred, target.long())


def build_finetune_domain_conf(
    temporal_steps: int,
    num_classes: int = 2,
    task_type: str = 'segmentation',   # 'segmentation' | 'regression'
) -> dict:
    """
    T(temporal_steps)를 받아서 동적으로 DOMAIN_CONF를 생성.
    채널 수 = base_channels * T
    task_type='segmentation' → cdl 포함
    task_type='regression'   → yield 포함
    """
    conf = {}

    # ── input modalities (공통) ──────────────────────────────────────
    for domain, base_ch in BASE_CHANNELS.items():
        ch = base_ch * temporal_steps
        conf[domain] = {
            'channels':       ch,
            'stride_level':   1,
            'input_adapter':  partial(PatchedInputAdapter, num_channels=ch),
            'output_adapter': partial(SpatialOutputAdapter, num_channels=ch),
            'loss':           MaskedMSELoss,
        }

    # ── label / output domain ────────────────────────────────────────
    if task_type == 'segmentation':
        # CDL: [B, H, W] long → CrossEntropyLoss (AMP 환경에서 안정적)
        conf['cdl'] = {
            'channels':       1,
            'stride_level':   1,
            'input_adapter':  partial(PatchedInputAdapter, num_channels=1),
            'output_adapter': partial(SpatialOutputAdapter, num_channels=num_classes),
            'loss':           partial(CDLCrossEntropyLoss, num_classes=num_classes),
        }

    # elif task_type == 'regression':
    #     # Yield: [1, H, W] float, MSELoss
    #     conf['yield'] = {
    #         'channels':       1,
    #         'stride_level':   1,
    #         'input_adapter':  partial(PatchedInputAdapter, num_channels=1),
    #         'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    #         'loss':           MaskedMSELossWrapper,
    #     }

    elif task_type == 'regression':
        conf['yield'] = {
            'channels':       1,
            'stride_level':   1,
            'input_adapter':  partial(PatchedInputAdapter, num_channels=1),
            'output_adapter': partial(
                DPTOutputAdapter,
                num_classes=1,
                hooks=[2, 5, 8, 11],
                head_type='regression',
                main_tasks=tuple(BASE_CHANNELS.keys()),  # 나중에 실제 in_domains로 덮어씀
            ),
            'loss': MaskedMSELossWrapper,
        }


    else:
        raise ValueError(f"task_type must be 'segmentation' or 'regression', got '{task_type}'")

    return conf