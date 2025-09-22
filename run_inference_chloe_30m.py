# # run_inference_chloe.py
# import os, argparse, torch, numpy as np
# from pathlib import Path
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

# from multimae.input_adapters import PatchedInputAdapter
# from multimae.output_adapters import SpatialOutputAdapter
# from utils import create_model, init_distributed_mode
# from utils.datasets_chloe import build_multimae_pretraining_dataset

# DOMAIN_CONF = {
#     's1': {
#         'channels': 2, 'stride_level': 1,
#         'input_adapter': lambda patch: PatchedInputAdapter(num_channels=2, stride_level=1, patch_size_full=patch),
#         'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
#             SpatialOutputAdapter(num_channels=2, stride_level=1, patch_size_full=patch,
#                                  dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
#                                  use_task_queries=use_tq, task='s1', context_tasks=ctx, use_xattn=use_xattn)
#     },
#     's2': {
#         'channels': 12, 'stride_level': 1,
#         'input_adapter': lambda patch: PatchedInputAdapter(num_channels=12, stride_level=1, patch_size_full=patch),
#         'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
#             SpatialOutputAdapter(num_channels=12, stride_level=1, patch_size_full=patch,
#                                  dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
#                                  use_task_queries=use_tq, task='s2', context_tasks=ctx, use_xattn=use_xattn)
#     },
# }

# def build_model(args):
#     in_domains  = args.in_domains.split('-')
#     out_domains = args.out_domains.split('-')

#     input_adapters = {d: DOMAIN_CONF[d]['input_adapter'](args.patch_size) for d in in_domains}
#     output_adapters = {
#         d: DOMAIN_CONF[d]['output_adapter'](
#             args.patch_size, args.decoder_dim, args.decoder_depth, args.decoder_num_heads,
#             args.decoder_use_task_queries, in_domains, args.decoder_use_xattn
#         ) for d in out_domains
#     }

#     model = create_model(
#         args.model,
#         input_adapters=input_adapters,
#         output_adapters=output_adapters,
#         num_global_tokens=args.num_global_tokens,
#         drop_path_rate=0.0
#     )
#     ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
#     state = ckpt.get('model', ckpt)
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
#     return model

# @torch.no_grad()
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--resume', required=True)
#     p.add_argument('--s2_txt', required=True, help='있으면 GT로 MSE 평가')
#     p.add_argument('--s1_txt', required=True, help='있으면 GT 로딩 & 평가')
#     p.add_argument('--data_path', default=None)
#     p.add_argument('--save_dir', required=True)
#     p.add_argument('--batch_size', type=int, default=1)
#     p.add_argument('--num_workers', type=int, default=4)

#     p.add_argument('--in_domains',  default='s1', type=str)
#     p.add_argument('--out_domains', default='s2', type=str)

#     # must match training
#     p.add_argument('--model', default='pretrain_multimae_base')
#     p.add_argument('--input_size', type=int, default=224)
#     p.add_argument('--patch_size', type=int, default=16)
#     p.add_argument('--num_global_tokens', type=int, default=1)
#     p.add_argument('--decoder_dim', type=int, default=256)
#     p.add_argument('--decoder_depth', type=int, default=2)
#     p.add_argument('--decoder_num_heads', type=int, default=8)
#     p.add_argument('--decoder_use_task_queries', action='store_true', default=True)
#     p.add_argument('--decoder_use_xattn', action='store_true', default=True)
#     p.add_argument('--num_encoded_tokens', type=int, default=196)

#     # dist (single-gpu도 OK)
#     p.add_argument('--world_size', default=1, type=int)
#     p.add_argument('--local_rank', default=-1, type=int)
#     p.add_argument('--dist_on_itp', action='store_true')
#     p.add_argument('--dist_url', default='env://')

#     args = p.parse_args()

#     Path(args.save_dir).mkdir(parents=True, exist_ok=True)
#     init_distributed_mode(args)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cudnn.benchmark = True

#     # ===== Dataset (추론용: 랜덤 크롭/플립 끔) =====
#     class DummyArgs: pass
#     dargs = DummyArgs()
#     dargs.data_path = args.data_path
#     dargs.s2_txt = args.s2_txt
#     dargs.s1_txt = args.s1_txt
#     dargs.input_size = args.input_size
#     dargs.in_domains  = args.in_domains.split('-')
#     dargs.out_domains = args.out_domains.split('-')
#     dargs.all_domains = list(set(dargs.in_domains) | set(dargs.out_domains))
#     dargs.patch_size  = args.patch_size
#     dargs.hflip       = 0.0
#     # ★ 랜덤 크롭/플립 완전 OFF
#     dargs.scale = (1.0, 1.0)
#     dargs.ratio = (1.0, 1.0)
#     dargs.train_interpolation = 'bicubic'
#     dargs.imagenet_default_mean_and_std = False
#     dargs.sample_tasks_uniformly = False
#     dargs.alphas = 1.0

#     dataset = build_multimae_pretraining_dataset(dargs)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False,
#         drop_last=False,
#     )

#     # ===== Model =====
#     model = build_model(args).to(device).eval()
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"[Model Info] Total parameters: {total_params:,}")
#     print(f"[Model Info] Trainable parameters: {trainable_params:,}")

#     out_domains = args.out_domains.split('-')

#     # 누적자(도메인별)
#     num_sum_masked = {d: 0.0 for d in out_domains}
#     den_sum_masked = {d: 0.0 for d in out_domains}
#     num_sum_raw    = {d: 0.0 for d in out_domains}
#     den_sum_raw    = {d: 0.0 for d in out_domains}

#     idx0 = 0
#     for batch in loader:
#         # 입력 dict 구성
#         inputs = {d: batch[d].to(device, non_blocking=True) for d in args.in_domains.split('-')}

#         with torch.amp.autocast('cuda'):
#             preds, _ = model(
#                 inputs,
#                 num_encoded_tokens=args.num_encoded_tokens,
#                 alphas=1.0,
#                 sample_tasks_uniformly=False,
#                 fp32_output_adapters=[]
#             )

#         # 각 출력 도메인별 저장 + 평가
#         for d in out_domains:
#             pred = preds[d].float()                        # [B,C,H,W]
#             gt   = batch[d].to(device).float()            # [B,C,H,W]
#             px   = batch.get(f"{d}_valid_mask", None)     # [B,1,H,W] or None
#             if px is not None:
#                 px = px.to(device).float()

#             # 저장 (CPU로 이동해서 npy)
#             pred_cpu = pred.cpu()
#             gt_cpu   = gt.cpu()
#             B = pred_cpu.size(0)
#             for b in range(B):
#                 np.save(os.path.join(args.save_dir, f"pred_30m{d}_{idx0+b:06d}.npy"), pred_cpu[b].numpy())
#                 np.save(os.path.join(args.save_dir, f"gt_30m{d}_{idx0+b:06d}.npy"),   gt_cpu[b].numpy())

#             # 손실 맵
#             loss_map = ((pred - gt) ** 2).mean(dim=1, keepdim=True)  # [B,1,H,W]

#             # --- Masked ---
#             if px is not None:
#                 num_sum_masked[d] += (loss_map * px).sum().item()
#                 den_sum_masked[d] += px.sum().item()
#             else:
#                 num_sum_masked[d] += loss_map.sum().item()
#                 den_sum_masked[d] += loss_map.numel()

#             # --- Raw ---
#             num_sum_raw[d] += loss_map.sum().item()
#             den_sum_raw[d] += loss_map.numel()

#         idx0 += next(iter(preds.values())).size(0)

#     # 결과 출력
#     for d in out_domains:
#         masked_mse = (num_sum_masked[d] / max(den_sum_masked[d], 1.0)) if den_sum_masked[d] > 0 else float('nan')
#         raw_mse    = (num_sum_raw[d]    / max(den_sum_raw[d],    1.0)) if den_sum_raw[d]    > 0 else float('nan')
#         print(f"[Eval:{d}] Masked MSE={masked_mse:.6e}")
#         print(f"[Eval:{d}] Raw    MSE={raw_mse:.6e}")

# if __name__ == "__main__":
#     main()

# # run_inference_chloe.py
# import os, argparse, torch, numpy as np
# from pathlib import Path
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

# from multimae.input_adapters import PatchedInputAdapter
# from multimae.output_adapters import SpatialOutputAdapter
# from utils import create_model, init_distributed_mode
# from utils.datasets_chloe import build_multimae_pretraining_dataset

# DOMAIN_CONF = {
#     's1': {
#         'channels': 2, 'stride_level': 1,
#         'input_adapter': lambda patch: PatchedInputAdapter(num_channels=2, stride_level=1, patch_size_full=patch),
#         'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
#             SpatialOutputAdapter(num_channels=2, stride_level=1, patch_size_full=patch,
#                                  dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
#                                  use_task_queries=use_tq, task='s1', context_tasks=ctx, use_xattn=use_xattn)
#     },
#     's2': {
#         'channels': 12, 'stride_level': 1,
#         'input_adapter': lambda patch: PatchedInputAdapter(num_channels=12, stride_level=1, patch_size_full=patch),
#         'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
#             SpatialOutputAdapter(num_channels=12, stride_level=1, patch_size_full=patch,
#                                  dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
#                                  use_task_queries=use_tq, task='s2', context_tasks=ctx, use_xattn=use_xattn)
#     },
# }

# def build_model(args):
#     in_domains  = args.in_domains.split('-')
#     out_domains = args.out_domains.split('-')

#     input_adapters = {d: DOMAIN_CONF[d]['input_adapter'](args.patch_size) for d in in_domains}
#     output_adapters = {
#         d: DOMAIN_CONF[d]['output_adapter'](
#             args.patch_size, args.decoder_dim, args.decoder_depth, args.decoder_num_heads,
#             args.decoder_use_task_queries, in_domains, args.decoder_use_xattn
#         ) for d in out_domains
#     }

#     model = create_model(
#         args.model,
#         input_adapters=input_adapters,
#         output_adapters=output_adapters,
#         num_global_tokens=args.num_global_tokens,
#         drop_path_rate=0.0
#     )
#     ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
#     state = ckpt.get('model', ckpt)
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
#     return model

# @torch.no_grad()
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--resume', required=True)
#     p.add_argument('--s2_txt', required=True, help='있으면 GT로 MSE 평가')
#     p.add_argument('--s1_txt', required=True, help='있으면 GT 로딩 & 평가')
#     p.add_argument('--data_path', default=None)
#     p.add_argument('--save_dir', required=True)
#     p.add_argument('--batch_size', type=int, default=1)
#     p.add_argument('--num_workers', type=int, default=4)

#     p.add_argument('--in_domains',  default='s1', type=str)
#     p.add_argument('--out_domains', default='s2', type=str)

#     # must match training
#     p.add_argument('--model', default='pretrain_multimae_base')
#     p.add_argument('--input_size', type=int, default=224)
#     p.add_argument('--patch_size', type=int, default=16)
#     p.add_argument('--num_global_tokens', type=int, default=1)
#     p.add_argument('--decoder_dim', type=int, default=256)
#     p.add_argument('--decoder_depth', type=int, default=2)
#     p.add_argument('--decoder_num_heads', type=int, default=8)
#     p.add_argument('--decoder_use_task_queries', action='store_true', default=True)
#     p.add_argument('--decoder_use_xattn', action='store_true', default=True)
#     p.add_argument('--num_encoded_tokens', type=int, default=196)

#     # dist (single-gpu도 OK)
#     p.add_argument('--world_size', default=1, type=int)
#     p.add_argument('--local_rank', default=-1, type=int)
#     p.add_argument('--dist_on_itp', action='store_true')
#     p.add_argument('--dist_url', default='env://')

#     args = p.parse_args()

#     Path(args.save_dir).mkdir(parents=True, exist_ok=True)
#     init_distributed_mode(args)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cudnn.benchmark = True

#     # ===== Dataset (추론용: 랜덤 크롭/플립 끔) =====
#     class DummyArgs: pass
#     dargs = DummyArgs()
#     dargs.data_path = args.data_path
#     dargs.s2_txt = args.s2_txt
#     dargs.s1_txt = args.s1_txt
#     dargs.input_size = args.input_size
#     dargs.in_domains  = args.in_domains.split('-')
#     dargs.out_domains = args.out_domains.split('-')
#     dargs.all_domains = list(set(dargs.in_domains) | set(dargs.out_domains))
#     dargs.patch_size  = args.patch_size
#     dargs.hflip       = 0.0
#     dargs.scale       = (1.0, 1.0)  # no random crop
#     dargs.ratio       = (1.0, 1.0)  # no aspect jitter
#     dargs.train_interpolation = 'bicubic'
#     dargs.imagenet_default_mean_and_std = False
#     dargs.sample_tasks_uniformly = False
#     dargs.alphas = 1.0

#     dataset = build_multimae_pretraining_dataset(dargs)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False,
#         drop_last=False,
#     )

#     # ===== Model =====
#     model = build_model(args).to(device).eval()
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"[Model Info] Total parameters: {total_params:,}")
#     print(f"[Model Info] Trainable parameters: {trainable_params:,}")

#     out_domains = args.out_domains.split('-')

#     # 누적자(도메인별)
#     num_sum_masked = {d: 0.0 for d in out_domains}
#     den_sum_masked = {d: 0.0 for d in out_domains}
#     num_sum_raw    = {d: 0.0 for d in out_domains}
#     den_sum_raw    = {d: 0.0 for d in out_domains}

#     # 저장은 도메인별로 첫 샘플 1개만
#     saved_once = {d: False for d in out_domains}

#     for step, batch in enumerate(loader):
#         # 입력 dict 구성
#         inputs = {d: batch[d].to(device, non_blocking=True) for d in args.in_domains.split('-')}

#         with torch.amp.autocast('cuda'):
#             preds, _ = model(
#                 inputs,
#                 num_encoded_tokens=args.num_encoded_tokens,
#                 alphas=1.0,
#                 sample_tasks_uniformly=False,
#                 fp32_output_adapters=[]
#             )

#         # 각 출력 도메인별 저장 + 평가
#         for d in out_domains:
#             pred = preds[d].float()                 # [B,C,H,W]
#             gt   = batch[d].to(device).float()      # [B,C,H,W]
#             px   = batch.get(f"{d}_valid_mask", None)
#             if px is not None:
#                 px = px.to(device).float()          # [B,1,H,W]

#             # ----- 저장: 첫 배치의 첫 샘플만 1회 -----
#             if not saved_once[d]:
#                 np.save(os.path.join(args.save_dir, f"pred_{d}_000000.npy"), pred[0].cpu().numpy())
#                 np.save(os.path.join(args.save_dir, f"gt_{d}_000000.npy"),   gt[0].cpu().numpy())
#                 saved_once[d] = True

#             # ----- 평가 누적 -----
#             loss_map = ((pred - gt) ** 2).mean(dim=1, keepdim=True)  # [B,1,H,W]

#             # Masked MSE (유효 픽셀만)
#             if px is not None:
#                 num_sum_masked[d] += (loss_map * px).sum().item()
#                 den_sum_masked[d] += px.sum().item()
#             else:
#                 num_sum_masked[d] += loss_map.sum().item()
#                 den_sum_masked[d] += loss_map.numel()

#             # Raw MSE (전체 픽셀)
#             num_sum_raw[d] += loss_map.sum().item()
#             den_sum_raw[d] += loss_map.numel()

#         break

#     # 결과 출력
#     for d in out_domains:
#         masked_mse = (num_sum_masked[d] / max(den_sum_masked[d], 1.0)) if den_sum_masked[d] > 0 else float('nan')
#         raw_mse    = (num_sum_raw[d]    / max(den_sum_raw[d],    1.0)) if den_sum_raw[d]    > 0 else float('nan')
#         print(f"[Eval:{d}] Masked MSE={masked_mse:.6e}")
#         print(f"[Eval:{d}] Raw    MSE={raw_mse:.6e}")

# if __name__ == "__main__":
#     main()


import argparse
import os
import torch
import numpy as np
import utils
from utils import create_model
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from multimae.criterion import MaskedMSELoss
from utils.datasets_chloe import build_multimae_pretraining_dataset

DOMAIN_CONF = {
    's1': {
        'channels': 2,
        'stride_level': 1,
        'input_adapter': lambda patch: PatchedInputAdapter(num_channels=2, patch_size_full=patch, stride_level=1),
        'output_adapter': lambda patch, dim, depth, heads, taskq, ctx, xattn: SpatialOutputAdapter(
            num_channels=2, patch_size_full=patch, stride_level=1,
            dim_tokens=dim, depth=depth, num_heads=heads,
            use_task_queries=taskq, context_tasks=ctx, use_xattn=xattn, task="s1"),
        'loss': MaskedMSELoss
    },
    's2': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': lambda patch: PatchedInputAdapter(num_channels=12, patch_size_full=patch, stride_level=1),
        'output_adapter': lambda patch, dim, depth, heads, taskq, ctx, xattn: SpatialOutputAdapter(
            num_channels=12, patch_size_full=patch, stride_level=1,
            dim_tokens=dim, depth=depth, num_heads=heads,
            use_task_queries=taskq, context_tasks=ctx, use_xattn=xattn, task="s2"),
        'loss': MaskedMSELoss
    },
}

def get_args():
    parser = argparse.ArgumentParser("MultiMAE inference script")

    # 필수
    parser.add_argument('--resume', required=True, help='Path to checkpoint')
    parser.add_argument('--s1_txt', type=str, required=True)
    parser.add_argument('--s2_txt', type=str, required=True)
    parser.add_argument('--in_domains', default='s1', type=str)
    parser.add_argument('--out_domains', default='s2', type=str)

    # Dataset / Augmentation
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--hflip', type=float, default=0)  # inference에서는 augment OFF
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    # Model params
    parser.add_argument('--model', default='pretrain_multimae_base', type=str)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--num_global_tokens', default=1, type=int)
    parser.add_argument('--decoder_dim', default=256, type=int)
    parser.add_argument('--decoder_depth', default=2, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true')
    parser.add_argument('--num_encoded_tokens', default=1024, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0, 
                    help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')
    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)

    # Loader
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=False)

    # Output / device
    parser.add_argument('--output_dir', default='./inference_result')
    parser.add_argument('--device', default='cuda')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser.parse_args()

def build_model(args):
    in_domains  = args.in_domains
    out_domains = args.out_domains

    input_adapters = {d: DOMAIN_CONF[d]['input_adapter'](args.patch_size) for d in in_domains}
    output_adapters = {
        d: DOMAIN_CONF[d]['output_adapter'](
            args.patch_size, args.decoder_dim, args.decoder_depth, args.decoder_num_heads,
            args.decoder_use_task_queries, in_domains, args.decoder_use_xattn
        ) for d in out_domains
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=0.0
    )

    # checkpoint load
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] checkpoint: {args.resume}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")

    return model

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # 도메인 설정: 훈련 때와 동일하게 s1, s2 둘 다 포함
    args.in_domains = ['s1', 's2']
    args.out_domains = ['s1', 's2']
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = build_model(args).to(device)

    # dataset
    dataset = build_multimae_pretraining_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # model.eval()
    # with torch.no_grad():
    #     for i, batch in enumerate(dataloader):
    #         # ✅ 입력은 s1만 사용
    #         inputs = {'s1': batch['s1'].to(device)}
    #         preds, _ = model(inputs)

    #         # ✅ 출력 중 s2만 저장/평가
    #         if 's2' in preds:
    #             pred = preds['s2']
    #             np_pred = pred.cpu().numpy()
    #             np.save(os.path.join(args.output_dir, f"pred_30m_s2_batch{i}.npy"), np_pred)

    #             # GT 저장
    #             if 's2' in batch:
    #                 gt = batch['s2'].to(device)
    #                 np_gt = gt.cpu().numpy()
    #                 np.save(os.path.join(args.output_dir, f"gt_30m_s2_batch{i}.npy"), np_gt)

    #                 # Loss 계산 (MSE)
    #                 loss_fn = DOMAIN_CONF['s2']['loss'](patch_size=args.patch_size, stride=1)
    #                 loss_val = loss_fn(pred.float(), gt.float()).item()
    #                 print(f"[inference] batch {i} s2 loss (MSE): {loss_val:.6f}")

    #         if i % 10 == 0:
    #             print(f"[inference] processed {i} batches")
            
    #         break

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # ✅ 입력은 s1만 사용
            inputs = {'s1': batch['s1'].to(device)}

            # preds와 masks 같이 반환
            preds, masks = model(
                inputs,
                num_encoded_tokens=args.num_encoded_tokens,
                alphas=args.alphas,
                sample_tasks_uniformly=args.sample_tasks_uniformly
            )

            # ✅ 출력 중 s2만 저장/평가
            if 's2' in preds:
                pred = preds['s2']

                # 저장 (numpy 변환)
                np_pred = pred.cpu().numpy()
                np.save(os.path.join(args.output_dir, f"pred_30m_s2_batch{i}.npy"), np_pred)

                # GT + 마스크 가져오기
                if 's2' in batch:
                    gt = batch['s2'].to(device)
                    np_gt = gt.cpu().numpy()
                    np.save(os.path.join(args.output_dir, f"gt_30m_s2_batch{i}.npy"), np_gt)

                    pxmask = batch.get('s2_valid_mask', None)   # 픽셀 마스크
                    pmask  = masks.get('s2', None)              # 패치 마스크

                    # ✅ Loss 계산 (훈련 방식과 동일)
                    loss_fn = DOMAIN_CONF['s2']['loss'](patch_size=args.patch_size, stride=1)

                    if args.loss_on_unmasked:
                        # unmasked 전체 (픽셀 마스크만 적용)
                        loss_val = loss_fn(pred.float(), gt, pixel_mask=pxmask).item()
                    else:
                        # masked 부분만 (패치 마스크 + 픽셀 마스크 교집합)
                        loss_val = loss_fn(pred.float(), gt, pixel_mask=pxmask, patch_mask=pmask).item()

                    print(f"[inference] batch {i} s2 loss (MSE): {loss_val:.6f}")

            if i % 10 == 0:
                print(f"[inference] processed {i} batches")
            
            break


if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
