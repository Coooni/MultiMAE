# run_inference_crossmodal.py
import os, torch, numpy as np
from pathlib import Path
from typing import Dict, Iterable
import torch.backends.cudnn as cudnn
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from utils import create_model
from utils import init_distributed_mode  # no-op if single gpu
from utils.datasets_chloe import build_multimae_pretraining_dataset

import argparse

DOMAIN_CONF = {
    'modis': {
        'channels': 7, 'stride_level': 1,
        'input_adapter': lambda patch: PatchedInputAdapter(num_channels=7, stride_level=1, patch_size_full=patch),
        'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
            SpatialOutputAdapter(num_channels=7, stride_level=1, patch_size_full=patch,
                                 dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
                                 use_task_queries=use_tq, task='modis', context_tasks=ctx, use_xattn=use_xattn)
    },
    's1': {
        'channels': 2, 'stride_level': 1,
        'input_adapter': lambda patch: PatchedInputAdapter(num_channels=2, stride_level=1, patch_size_full=patch),
        'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
            SpatialOutputAdapter(num_channels=2, stride_level=1, patch_size_full=patch,
                                 dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
                                 use_task_queries=use_tq, task='s1', context_tasks=ctx, use_xattn=use_xattn)
    },
    's2': {
        'channels': 12, 'stride_level': 1,
        'input_adapter': lambda patch: PatchedInputAdapter(num_channels=12, stride_level=1, patch_size_full=patch),
        'output_adapter': lambda patch, dec_dim, dec_depth, dec_heads, use_tq, ctx, use_xattn:
            SpatialOutputAdapter(num_channels=12, stride_level=1, patch_size_full=patch,
                                 dim_tokens=dec_dim, depth=dec_depth, num_heads=dec_heads,
                                 use_task_queries=use_tq, task='s2', context_tasks=ctx, use_xattn=use_xattn)
    },
}

def build_model(args):
    in_domains  = args.in_domains.split('-')
    out_domains = args.out_domains.split('-')
        

    input_adapters = {d: DOMAIN_CONF[d]['input_adapter'](args.patch_size) for d in in_domains}
    output_adapters = {d: DOMAIN_CONF[d]['output_adapter'](
        args.patch_size, args.decoder_dim, args.decoder_depth, args.decoder_num_heads,
        args.decoder_use_task_queries, in_domains, args.decoder_use_xattn
    ) for d in out_domains}

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=0.0
    )
    # load checkpoint (trained with modis-s2 ↔ modis-s2)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    return model

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', required=True)
    p.add_argument('--modis_txt', required=True)
    p.add_argument('--s2_txt', required=True, help='있으면 GT로 MSE 평가')
    p.add_argument('--s1_txt', required=True, help='있으면 GT 로딩 & 평가')
    p.add_argument('--data_path', default=None)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--in_domains',  default='modis', type=str, help='입력 도메인들(하이픈 구분)')
    p.add_argument('--out_domains', default='s2',   type=str, help='출력 도메인들(하이픈 구분)')


    # must match training
    p.add_argument('--model', default='pretrain_multimae_base')
    p.add_argument('--input_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--num_global_tokens', type=int, default=1)
    p.add_argument('--decoder_dim', type=int, default=256)
    p.add_argument('--decoder_depth', type=int, default=2)
    p.add_argument('--decoder_num_heads', type=int, default=8)
    p.add_argument('--decoder_use_task_queries', action='store_true', default=True)
    p.add_argument('--decoder_use_xattn', action='store_true', default=True)
    p.add_argument('--num_encoded_tokens', type=int, default=196)  # 14x14=196 : encode all MODIS tokens

    # Distributed training parameters
    p.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url', default='env://', help='url used to set up distributed training')



    args = p.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)  # single-GPU ok
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # dataset (reuses training preprocessing; returns dict with 'modis' (and 's2' if provided))
    class DummyArgs: pass
    dargs = DummyArgs()
    dargs.data_path = args.data_path
    dargs.modis_txt = args.modis_txt
    dargs.s2_txt = args.s2_txt if args.s2_txt else args.modis_txt  # placeholder to satisfy builder
    dargs.s1_txt = args.s1_txt if args.s1_txt else args.modis_txt
    dargs.input_size = args.input_size
    dargs.in_domains  = args.in_domains.split('-')     # 예) ['modis','s1']
    dargs.out_domains = args.out_domains.split('-')    # 예) ['s2']
    dargs.all_domains = list(set(dargs.in_domains) | set(dargs.out_domains))



    dargs.input_size                  = args.input_size   # 224
    dargs.patch_size                  = args.patch_size   # 16
    dargs.hflip                       = 0.0               # ★추가★ Inference는 뒤집기 off
    dargs.train_interpolation         = 'bicubic'
    dargs.imagenet_default_mean_and_std = False
    dargs.sample_tasks_uniformly = False
    dargs.alphas = 1.0



    dataset = build_multimae_pretraining_dataset(dargs)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,        # 한 번에 한 샘플
        shuffle=False,
        num_workers=args.num_workers,       # worker 0 → 첫 배치가 바로 나옴
        pin_memory=False,
        drop_last=False,
        )

    model = build_model(args).to(device).eval()

    # 파라미터 수 출력 추가
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Info] Total parameters: {total_params:,}")
    print(f"[Model Info] Trainable parameters: {trainable_params:,}")

    import torch.nn.functional as F
    mse_sum, n_pix = 0.0, 0
    idx0 = 0

    for batch in loader:
        inputs = {
            's2': batch['s2'].to(device, non_blocking=True),
            's1':    batch['s1'].to(device,    non_blocking=True)
            }

        with torch.cuda.amp.autocast():
            preds, _ = model(inputs,
                            num_encoded_tokens=args.num_encoded_tokens,
                            alphas=1.0,
                            sample_tasks_uniformly=False,
                            fp32_output_adapters=[])
        pred_s2 = preds['modis'].float().cpu()
        gt_s2   = batch['modis'].float().cpu()  

        # save .npy per item (간단)
        for b in range(pred_s2.size(0)):
            out = pred_s2[b].numpy()
            np.save(os.path.join(args.save_dir, f"pred_modis_new_{idx0+b:06d}.npy"), out)
            out = gt_s2[b].numpy()
            np.save(os.path.join(args.save_dir, f"gt_modis_new_{idx0+b:06d}.npy"), out)


        # optional: GT가 있으면 평가
        if 'modis' in batch:
            gt = batch['modis'].float()
            mse = F.mse_loss(pred_s2, gt, reduction='sum').item()
            mse_sum += mse
            n_pix += np.prod(pred_s2.shape)

        # idx0 += pred_s2.size(0)
        break

    if n_pix > 0:
        print(f"[Eval] MSE={mse_sum/n_pix:.6e}")

if __name__ == "__main__":
    main()
