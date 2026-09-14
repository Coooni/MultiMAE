# """
# models/build_model.py
# pretrain / finetune 공용 model builder.
# domain_conf를 인자로 받아서 두 곳 모두에서 재사용 가능.
# """
# from functools import partial

# import torch
# import torch.nn as nn
# from multimae.output_adapters import SegmenterMaskTransformerAdapter
# from utils import create_model


# class WrappedSegmenterAdapter(SegmenterMaskTransformerAdapter):
#     """
#     MultiMAE forward가 넘겨주는 ids_keep / ids_restore 등
#     불필요한 kwargs를 무시하는 wrapper.
#     """
#     def forward(self, encoder_tokens, input_info=None, **kwargs):
#         return super().forward(encoder_tokens=encoder_tokens, input_info=input_info)


# def get_model(
#     in_domains: list,
#     out_domains: list,
#     domain_conf: dict,
#     patch_size: int = 16,
#     decoder_dim: int = 256,
#     decoder_depth: int = 2,
#     decoder_num_heads: int = 8,
#     num_global_tokens: int = 1,
#     drop_path_rate: float = 0.0,
#     num_classes: int = 2,
# ) -> nn.Module:
#     """
#     Args:
#         in_domains:        입력으로 사용할 modality 리스트
#         out_domains:       출력(복원/예측) 대상 modality 리스트
#         domain_conf:       DOMAIN_CONF dict (pretrain or finetune용)
#         patch_size:        ViT patch size
#         decoder_dim:       decoder token dimension
#         decoder_depth:     decoder self-attention 층 수
#         decoder_num_heads: decoder attention head 수
#         num_global_tokens: global token 수
#         drop_path_rate:    stochastic depth rate
#         num_classes:       CDL segmentation class 수 (finetune에서만 사용)
#     """
#     # ------------------------------------------------------------------ #
#     # Input adapters
#     # ------------------------------------------------------------------ #
#     input_adapters = {
#         d: domain_conf[d]['input_adapter'](
#             stride_level=1,
#             patch_size_full=patch_size,
#         )
#         for d in in_domains
#         if 'input_adapter' in domain_conf[d]
#     }

#     # ------------------------------------------------------------------ #
#     # Output adapters
#     # ------------------------------------------------------------------ #
#     output_adapters = {}
#     for d in out_domains:
#         if 'output_adapter' not in domain_conf[d]:
#             continue

#         adapter_factory = domain_conf[d]['output_adapter']

#         # partial인 경우 원래 클래스 꺼내기
#         base_cls = adapter_factory.func if isinstance(adapter_factory, partial) else adapter_factory

#         if issubclass(base_cls, SegmenterMaskTransformerAdapter):
#             # CDL segmentation head
#             output_adapters[d] = WrappedSegmenterAdapter(
#                 num_classes=num_classes,
#                 depth=decoder_depth,
#                 num_heads=decoder_num_heads,
#                 embed_dim=decoder_dim,
#                 patch_size=patch_size,
#             )
#         else:
#             # 일반 spatial reconstruction head
#             output_adapters[d] = adapter_factory(
#                 stride_level=1,
#                 patch_size_full=patch_size,
#                 dim_tokens=decoder_dim,
#                 depth=decoder_depth,
#                 num_heads=decoder_num_heads,
#                 use_task_queries=True,
#                 task=d,
#                 context_tasks=in_domains,
#                 use_xattn=True,
#             )

#     # ------------------------------------------------------------------ #
#     # Model
#     # ------------------------------------------------------------------ #
#     model = create_model(
#         'pretrain_multimae_base',
#         input_adapters=input_adapters,
#         output_adapters=output_adapters,
#         num_global_tokens=num_global_tokens,
#         drop_path_rate=drop_path_rate,
#     )

#     return model


# def load_pretrained_weights(
#     model: nn.Module,
#     ckpt_path: str,
#     device: torch.device,
#     load_input_adapters: bool = False,
# ) -> nn.Module:
#     """
#     Pretrained weight 로드.
#     load_input_adapters=False (기본값):
#         temporal stacking으로 채널 수가 바뀐 input_adapter proj 레이어는 건너뜀.
#     """
#     state_dict = torch.load(ckpt_path, map_location=device)

#     # checkpoint가 dict 형태로 저장된 경우 처리
#     if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
#         state_dict = state_dict['model_state_dict']

#     if not load_input_adapters:
#         filtered = {
#             k: v for k, v in state_dict.items()
#             if not ('input_adapters' in k and ('proj.weight' in k or 'proj.bias' in k))
#         }
#         missing, unexpected = model.load_state_dict(filtered, strict=False)
#     else:
#         missing, unexpected = model.load_state_dict(state_dict, strict=False)

#     print(f"[load_pretrained_weights] Missing keys : {len(missing)}")
#     print(f"[load_pretrained_weights] Unexpected keys: {len(unexpected)}")
#     return model


"""
refac_build_model.py
pretrain / finetune 공용 model builder.
domain_conf를 인자로 받아서 두 곳 모두에서 재사용 가능.

  CDL (segmentation): pretrain_multimae_base (MultiMAE, masking 지원) 그대로 사용
  Yield (regression):  multivit_base (MultiViT, return_all_layers 지원) 사용
                        → DPTOutputAdapter가 필요로 하는 multi-scale
                          중간 encoder layer 접근을 위함
"""
from functools import partial

import torch
import torch.nn as nn
from multimae.output_adapters import SegmenterMaskTransformerAdapter, DPTOutputAdapter
from utils import create_model


class WrappedSegmenterAdapter(SegmenterMaskTransformerAdapter):
    """
    MultiMAE forward가 넘겨주는 ids_keep / ids_restore 등
    불필요한 kwargs를 무시하는 wrapper.
    """
    def forward(self, encoder_tokens, input_info=None, **kwargs):
        return super().forward(encoder_tokens=encoder_tokens, input_info=input_info)


class WrappedDPTAdapter(DPTOutputAdapter):
    """
    MultiViT forward가 넘겨주는 kwargs를 무시하는 wrapper.
    DPTOutputAdapter.forward(encoder_tokens: List[Tensor], input_info: Dict)만 필요.
    """
    def forward(self, encoder_tokens, input_info=None, **kwargs):
        return super().forward(encoder_tokens=encoder_tokens, input_info=input_info)


def get_model(
    in_domains: list,
    out_domains: list,
    domain_conf: dict,
    patch_size: int = 16,
    decoder_dim: int = 256,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    num_global_tokens: int = 1,
    drop_path_rate: float = 0.0,
    num_classes: int = 2,
    model_name: str = 'pretrain_multimae_base',   # ← 새 파라미터, 기존 호출부는 기본값 그대로 써서 영향 없음
) -> nn.Module:
    """
    Args:
        in_domains:        입력으로 사용할 modality 리스트
        out_domains:       출력(복원/예측) 대상 modality 리스트
        domain_conf:       DOMAIN_CONF dict (pretrain or finetune용)
        patch_size:        ViT patch size
        decoder_dim:       decoder token dimension
        decoder_depth:     decoder self-attention 층 수
        decoder_num_heads: decoder attention head 수
        num_global_tokens: global token 수
        drop_path_rate:    stochastic depth rate
        num_classes:       CDL segmentation class 수 (finetune에서만 사용)
        model_name:        'pretrain_multimae_base'(기본, CDL/pretrain용) 또는
                            'multivit_base'(yield regression, DPT decoder용)
    """
    # ------------------------------------------------------------------ #
    # Input adapters
    # ------------------------------------------------------------------ #
    input_adapters = {
        d: domain_conf[d]['input_adapter'](
            stride_level=1,
            patch_size_full=patch_size,
        )
        for d in in_domains
        if 'input_adapter' in domain_conf[d]
    }

    # ------------------------------------------------------------------ #
    # Output adapters
    # ------------------------------------------------------------------ #
    output_adapters = {}
    for d in out_domains:
        if 'output_adapter' not in domain_conf[d]:
            continue

        adapter_factory = domain_conf[d]['output_adapter']

        # partial인 경우 원래 클래스 꺼내기
        base_cls = adapter_factory.func if isinstance(adapter_factory, partial) else adapter_factory

        if issubclass(base_cls, SegmenterMaskTransformerAdapter):
            # CDL segmentation head — 기존 그대로, 손 안 댐
            output_adapters[d] = WrappedSegmenterAdapter(
                num_classes=num_classes,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                embed_dim=decoder_dim,
                patch_size=patch_size,
            )
        elif issubclass(base_cls, DPTOutputAdapter):
            # Yield regression head — DPT-style multi-scale decoder
            output_adapters[d] = WrappedDPTAdapter(
                num_classes=1,
                hooks=[2, 5, 8, 11],
                main_tasks=tuple(in_domains),
                head_type='regression',
                patch_size=patch_size,
            )
        else:
            # 일반 spatial reconstruction head (pretrain, 기존 yield SpatialOutputAdapter 등)
            output_adapters[d] = adapter_factory(
                stride_level=1,
                patch_size_full=patch_size,
                dim_tokens=decoder_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                use_task_queries=True,
                task=d,
                context_tasks=in_domains,
                use_xattn=True,
            )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = create_model(
        model_name,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=num_global_tokens,
        drop_path_rate=drop_path_rate,
    )

    return model


def load_pretrained_weights(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    load_input_adapters: bool = False,
) -> nn.Module:
    """
    Pretrained weight 로드.
    load_input_adapters=False (기본값):
        temporal stacking으로 채널 수가 바뀐 input_adapter proj 레이어는 건너뜀.

    MultiViT는 MultiMAE를 상속만 하고 __init__을 재정의하지 않으므로,
    encoder/input_adapters/global_tokens의 state_dict key/shape이 완전히 동일함.
    따라서 model_name='multivit_base'로 만든 모델에도 이 함수를 그대로 사용 가능.
    """
    state_dict = torch.load(ckpt_path, map_location=device)

    # checkpoint가 dict 형태로 저장된 경우 처리
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if not load_input_adapters:
        filtered = {
            k: v for k, v in state_dict.items()
            if not ('input_adapters' in k and ('proj.weight' in k or 'proj.bias' in k))
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[load_pretrained_weights] Missing keys : {len(missing)}")
    print(f"[load_pretrained_weights] Unexpected keys: {len(unexpected)}")
    return model