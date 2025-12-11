# temporal_multimae_seg.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMultiMAESeg(nn.Module):
    """
    MultiMAE 기반 downstream segmentation용 temporal wrapper.
    
    - backbone: get_model에서 만든 create_model(...) 결과 (MultiMAE 모델)
    - seg_domain: preds 딕셔너리에서 segmentation 출력에 해당하는 key (여기서는 'cdl')
    """
    def __init__(self, backbone, seg_domain: str = "cdl"):
        super().__init__()
        self.backbone = backbone
        self.seg_domain = seg_domain

    def forward(self, x):
        """
        x: Dict[str, Tensor]
           각 도메인은 [B, T, C, H, W] 형태라고 가정.

           예:
           {
               's1':       [B, T, C1, H, W],
               's2':       [B, T, C2, H, W],
               'elevation':[B, T, C3, H, W],
               'weather':  [B, T, C4, H, W],
               'soil':     [B, T, C5, H, W],
           }
        """
        x_flat = {}
        B = T = H = W = None

        # 1) 각 도메인마다 time(T)을 batch로 펼치기
        for domain, tensor in x.items():
            # tensor: [B, T, C, H, W]
            b, t, c, h, w = tensor.shape
            if B is None:
                B, T, H, W = b, t, h, w
            x_flat[domain] = tensor.view(b * t, c, h, w)  # [B*T, C, H, W]

        # 2) 기존 MultiMAE backbone 그대로 사용
        #    backbone.forward는 dict[str, Tensor]를 입력으로 받는 구조
        preds, _ = self.backbone(
            x_flat,
            mask_inputs=False,
            num_encoded_tokens=None,
        )
        # preds[self.seg_domain] : [B*T, num_classes, H_seg, W_seg]
        seg_logits_t = preds[self.seg_domain]

        # 3) [B, T, num_classes, H_seg, W_seg]로 reshape
        Bt, num_classes, H_seg, W_seg = seg_logits_t.shape
        seg_logits_t = seg_logits_t.view(B, T, num_classes, H_seg, W_seg)

        # 4) 시간 축(T) 방향으로 평균 (가장 단순한 temporal aggregation)
        seg_logits = seg_logits_t.mean(dim=1)  # [B, num_classes, H_seg, W_seg]

        # 5) 필요 시 원래 H, W로 upsample
        if (H_seg, W_seg) != (H, W):
            seg_logits = F.interpolate(
                seg_logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return seg_logits
