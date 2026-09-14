"""
refac_pretrain.py — MultiMAE Pretraining Entry Point (with Scaler Resume)

실행 예시 (단일 GPU):
    torchrun --nproc_per_node=1 refac_pretrain.py --config config_pretrain.yaml

resume 예시 (명시적):
    torchrun --nproc_per_node=1 refac_pretrain.py --config config_pretrain.yaml --resume /path/to/pretrain_latest.pth
"""
import os
import glob
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import wandb

from refac_pretrain_args import get_pretrain_args, post_process_args
from refac_domain_conf_pretrain import DOMAIN_CONF
from refac_build_model import get_model
from refac_pretrain_engine import train_one_epoch, test_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils.refac_datasets_chloe import build_multimae_pretraining_dataset


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    args.rank        = int(os.environ.get('RANK', 0))
    args.local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size  = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    torch.cuda.set_device(args.local_rank)

    if args.distributed:
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()
    else:
        print('Single GPU mode — DDP disabled.')


def is_main_process(args) -> bool:
    return args.rank == 0


def build_dataloaders(args, dataset):
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        test_sampler  = DistributedSampler(test_ds,  shuffle=False)
        train_shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )
    return train_loader, val_loader, test_loader, train_sampler


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(args):
    init_distributed(args)
    set_seed(args.seed + args.rank)
    cudnn.benchmark = True

    device = torch.device(f'cuda:{args.local_rank}')

    # ---------------------------------------------------------------- #
    # Dataset
    # ---------------------------------------------------------------- #
    dataset = build_multimae_pretraining_dataset(args)
    train_loader, val_loader, test_loader, train_sampler = build_dataloaders(args, dataset)

    if is_main_process(args):
        print(f'Dataset size: {len(dataset)} | '
              f'Train: {len(train_loader.dataset)} | '
              f'Val: {len(val_loader.dataset)} | '
              f'Test: {len(test_loader.dataset)}')

    # ---------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------- #
    model = get_model(
        in_domains=args.in_domains,
        out_domains=args.out_domains,
        domain_conf=DOMAIN_CONF,
        patch_size=args.patch_size,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
    ).to(device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_params,
        )

    raw_model = model.module if args.distributed else model

    # ---------------------------------------------------------------- #
    # Optimizer & Loss
    # ---------------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.blr,
        eps=args.opt_eps,
        betas=tuple(args.opt_betas),
        weight_decay=args.weight_decay,
    )
    loss_scaler = NativeScaler()

    tasks_loss_fn = {
        d: DOMAIN_CONF[d]['loss'](patch_size=args.patch_size, stride=1)
        for d in args.out_domains
    }

    # ---------------------------------------------------------------- #
    # Resume (이어서 학습)
    # ---------------------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch   = args.start_epoch

    # --resume 인자 또는 output_dir의 latest checkpoint 자동 탐지
    resume_path = getattr(args, 'resume', None)
    if not resume_path: # None이거나 빈 문자열인 경우 탐지
        latest = os.path.join(args.output_dir, 'pretrain_latest.pth')
        if os.path.isfile(latest):
            resume_path = latest

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # [핵심 추가] Scaler 상태 복구 - Grad Norm 튀는 현상 방지
        if 'scaler_state_dict' in ckpt and loss_scaler is not None:
            loss_scaler.load_state_dict(ckpt['scaler_state_dict'])
            
        start_epoch   = ckpt['epoch']          # 저장된 epoch 다음부터 시작
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if is_main_process(args):
            print(f'▶ Resumed from {resume_path} (epoch {start_epoch}, best_val_loss={best_val_loss:.6f})')
    else:
        if is_main_process(args):
            print(f'▶ Training from scratch. (Path checked: {resume_path})')

    # ---------------------------------------------------------------- #
    # W&B (main process only)
    # ---------------------------------------------------------------- #
    if is_main_process(args):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            resume='allow',                    # resume 시 같은 run에 이어서 log
            id=args.wandb_run_name,            # 같은 run에 이어서 log하기 위해 고정 id 사용
        )

    # ---------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model, train_loader, tasks_loss_fn, optimizer,
            device, epoch, loss_scaler,
            args.in_domains, args.out_domains, args,
            split='train',
        )

        torch.cuda.empty_cache()

        val_stats = train_one_epoch(
            model, val_loader, tasks_loss_fn, optimizer,
            device, epoch, loss_scaler,
            args.in_domains, args.out_domains, args,
            split='valid',
        )

        val_loss = val_stats.get('loss', float('inf'))

        if is_main_process(args):
            # ── 매 epoch마다 latest checkpoint 저장 (resume용) ──
            latest_path = os.path.join(args.output_dir, 'pretrain_latest.pth')
            torch.save({
                'epoch':              epoch + 1,   # 다음 시작 epoch
                'model_state_dict':   raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict':  loss_scaler.state_dict(), # [핵심 추가] Scaler 저장
                'val_loss':           val_loss,
                'best_val_loss':      best_val_loss,
                'args':               vars(args),
            }, latest_path)

            # ── val_loss 개선 시 best checkpoint도 저장 ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output_dir, f'pretrain_best_epoch{epoch+1}.pth')
                torch.save({
                    'epoch':              epoch + 1,
                    'model_state_dict':   raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict':  loss_scaler.state_dict(), # [핵심 추가] Scaler 저장
                    'val_loss':           best_val_loss,
                    'best_val_loss':      best_val_loss,
                    'args':               vars(args),
                }, best_path)
                print(f'💾 Best model saved → {best_path} (val_loss={best_val_loss:.6f})')
                # wandb.save(best_path) # 필요 시 활성화

        if args.distributed:
            dist.barrier()

    # ---------------------------------------------------------------- #
    # Test (best model 로드 후 평가)
    # ---------------------------------------------------------------- #
    if is_main_process(args):
        print(f'\n🔍 Running test with best model ...')

    ckpt_files = sorted(glob.glob(os.path.join(args.output_dir, 'pretrain_best_epoch*.pth')))
    if ckpt_files:
        ckpt = torch.load(ckpt_files[-1], map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        if is_main_process(args):
            print(f'   Loaded best checkpoint: {ckpt_files[-1]}')

    torch.cuda.empty_cache()

    test_stats = test_one_epoch(
        model, test_loader, tasks_loss_fn,
        device, args.epochs,
        args.in_domains, args.out_domains, args,
    )

    if is_main_process(args):
        print('✅ Pretraining test finished.')
        print(f'   avg R²={test_stats.get("avg_r2", 0):.4f} | '
              f'avg RMSE={test_stats.get("avg_rmse", 0):.4f} | '
              f'avg MSE={test_stats.get("avg_mse", 0):.4f}')
        wandb.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_pretrain_args()
    args   = parser.parse_args()
    args   = post_process_args(args)
    main(args)