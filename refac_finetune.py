# """
# refac_finetune.py — MultiMAE Downstream Finetuning Entry Point
#   - segmentation : CDL prediction  (--task_type segmentation)
#   - regression   : Yield prediction (--task_type regression)

# 실행 예시 (CDL, single GPU):
#     torchrun --nproc_per_node=1 refac_finetune.py --config config_cdl.yaml

# resume 예시:
#     torchrun --nproc_per_node=1 refac_finetune.py --config config_cdl.yaml --resume output/finetune/cdl/finetune_latest.pth
# """
# import os
# import copy
# import random
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler
# import wandb

# from refac_finetune_args import get_finetune_args, post_process_args
# from refac_domain_conf_finetune import build_finetune_domain_conf
# from refac_build_model import get_model, load_pretrained_weights
# from refac_finetune_engine import train_one_epoch, test_one_epoch
# from utils import NativeScalerWithGradNormCount as NativeScaler
# from utils.refac_datasets_chloe import build_multimae_downstream_dataset


# # ------------------------------------------------------------------ #
# # Helpers
# # ------------------------------------------------------------------ #

# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def init_distributed(args):
#     args.rank        = int(os.environ.get('RANK', 0))
#     args.local_rank  = int(os.environ.get('LOCAL_RANK', 0))
#     args.world_size  = int(os.environ.get('WORLD_SIZE', 1))
#     args.distributed = args.world_size > 1

#     torch.cuda.set_device(args.local_rank)

#     if args.distributed:
#         dist.init_process_group(
#             backend='nccl',
#             init_method=args.dist_url,
#             world_size=args.world_size,
#             rank=args.rank,
#         )
#         dist.barrier()
#     else:
#         print('Single GPU mode — DDP disabled.')


# def is_main_process(args) -> bool:
#     return args.rank == 0


# def build_dataloaders(args, dataset):
#     n_total = len(dataset)
#     n_train = int(n_total * 0.70)
#     n_val   = int(n_total * 0.15)
#     n_test  = n_total - n_train - n_val

#     train_ds, val_ds, test_ds = random_split(
#         dataset, [n_train, n_val, n_test],
#         generator=torch.Generator().manual_seed(args.seed),
#     )

#     if args.distributed:
#         train_sampler = DistributedSampler(train_ds, shuffle=True)
#         val_sampler   = DistributedSampler(val_ds,   shuffle=False)
#         test_sampler  = DistributedSampler(test_ds,  shuffle=False)
#         train_shuffle = False
#     else:
#         train_sampler = val_sampler = test_sampler = None
#         train_shuffle = True

#     train_loader = DataLoader(
#         train_ds, batch_size=args.batch_size, sampler=train_sampler,
#         shuffle=train_shuffle,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=args.batch_size, sampler=val_sampler,
#         shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=args.batch_size, sampler=test_sampler,
#         shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
#     )
#     return train_loader, val_loader, test_loader, train_sampler


# # ------------------------------------------------------------------ #
# # Main
# # ------------------------------------------------------------------ #

# def main(args):
#     init_distributed(args)
#     set_seed(args.seed + args.rank)
#     cudnn.benchmark = True

#     device = torch.device(f'cuda:{args.local_rank}')

#     is_regression = (args.task_type == 'regression')

#     if is_main_process(args):
#         print(f'▶ Task type : {args.task_type}')
#         print(f'  in_domains : {args.in_domains}')
#         print(f'  out_domains: {args.out_domains}')

#     # ---------------------------------------------------------------- #
#     # Domain config
#     # ---------------------------------------------------------------- #
#     domain_conf = build_finetune_domain_conf(
#         temporal_steps=args.temporal_steps,
#         num_classes=args.num_classes,
#         task_type=args.task_type,
#     )

#     # ---------------------------------------------------------------- #
#     # Dataset — pretrain 방식 (state별 ConcatDataset)
#     # ---------------------------------------------------------------- #
#     dataset = build_multimae_downstream_dataset(args)
#     train_loader, val_loader, test_loader, train_sampler = build_dataloaders(args, dataset)

#     if is_main_process(args):
#         print(f'Dataset size: {len(dataset)} | '
#               f'Train: {len(train_loader.dataset)} | '
#               f'Val: {len(val_loader.dataset)} | '
#               f'Test: {len(test_loader.dataset)}')

#     # ---------------------------------------------------------------- #
#     # Model
#     # ---------------------------------------------------------------- #
#     model = get_model(
#         in_domains=args.in_domains,
#         out_domains=args.out_domains,
#         domain_conf=domain_conf,
#         patch_size=args.patch_size,
#         decoder_dim=args.decoder_dim,
#         decoder_depth=args.decoder_depth,
#         decoder_num_heads=args.decoder_num_heads,
#         num_global_tokens=args.num_global_tokens,
#         drop_path_rate=args.drop_path,
#         num_classes=args.num_classes if not is_regression else 1,
#     ).to(device)

#     if args.pretrained_weights:
#         model = load_pretrained_weights(
#             model, args.pretrained_weights, device,
#             load_input_adapters=args.load_input_adapters,
#         )
#         if is_main_process(args):
#             print(f'✅ Loaded pretrained weights from {args.pretrained_weights}')

#     if args.distributed:
#         model = DDP(
#             model,
#             device_ids=[args.local_rank],
#             find_unused_parameters=args.find_unused_params,
#         )

#     raw_model = model.module if args.distributed else model

#     # ---------------------------------------------------------------- #
#     # Optimizer & Loss
#     # ---------------------------------------------------------------- #
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=args.blr,
#         eps=args.opt_eps,
#         betas=tuple(args.opt_betas),
#         weight_decay=args.weight_decay,
#     )
#     loss_scaler = NativeScaler()

#     tasks_loss_fn = {
#         d: domain_conf[d]['loss'](patch_size=args.patch_size, stride=1)
#         for d in args.out_domains
#     }

#     # ---------------------------------------------------------------- #
#     # Resume (이어서 학습)
#     # ---------------------------------------------------------------- #
#     os.makedirs(args.output_dir, exist_ok=True)
#     best_val_loss  = float('inf')
#     best_model_wts = None
#     best_epoch     = -1
#     start_epoch    = args.start_epoch

#     # --resume 인자 또는 output_dir의 latest checkpoint 자동 탐지
#     resume_path = args.resume if args.resume else None
#     if resume_path is None and getattr(args, 'auto_resume', True):
#         latest = os.path.join(args.output_dir, 'finetune_latest.pth')
#         if os.path.isfile(latest):
#             resume_path = latest

#     if resume_path and os.path.isfile(resume_path):
#         ckpt = torch.load(resume_path, map_location=device)
#         raw_model.load_state_dict(ckpt['model_state_dict'])
#         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         start_epoch    = ckpt['epoch']
#         best_val_loss  = ckpt.get('best_val_loss', float('inf'))
#         best_epoch     = ckpt.get('best_epoch', -1)
#         if is_main_process(args):
#             print(f'▶ Resumed from {resume_path} (epoch {start_epoch}, best_val_loss={best_val_loss:.6f})')
#     else:
#         if is_main_process(args):
#             print('▶ Training from scratch.')

#     # ---------------------------------------------------------------- #
#     # W&B (main process only)
#     # ---------------------------------------------------------------- #
#     if is_main_process(args):
#         wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity,
#             name=args.wandb_run_name,
#             config=vars(args),
#             resume='allow',
#             id='telgx9t8',
#         )

#     # ---------------------------------------------------------------- #
#     # Training loop
#     # ---------------------------------------------------------------- #
#     for epoch in range(start_epoch, args.epochs):
#         if args.distributed:
#             train_sampler.set_epoch(epoch)

#         train_one_epoch(
#             model, train_loader, tasks_loss_fn, optimizer,
#             device, epoch, loss_scaler,
#             args.in_domains, args.out_domains, args,
#             split='train',
#         )

#         torch.cuda.empty_cache()

#         val_stats = train_one_epoch(
#             model, val_loader, tasks_loss_fn, optimizer,
#             device, epoch, loss_scaler,
#             args.in_domains, args.out_domains, args,
#             split='val',
#         )

#         val_loss = val_stats.get('loss', float('inf'))

#         if is_main_process(args):
#             # ── 매 epoch마다 latest checkpoint 저장 (resume용) ──
#             latest_path = os.path.join(args.output_dir, 'finetune_latest.pth')
#             torch.save({
#                 'epoch':                epoch + 1,
#                 'model_state_dict':     raw_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss':             val_loss,
#                 'best_val_loss':        best_val_loss,
#                 'best_epoch':           best_epoch,
#                 'args':                 vars(args),
#             }, latest_path)

#             # ── val_loss 개선 시 best checkpoint도 저장 ──
#             if val_loss < best_val_loss:
#                 best_val_loss  = val_loss
#                 best_epoch     = epoch
#                 best_model_wts = copy.deepcopy(raw_model.state_dict())
#                 save_path = os.path.join(
#                     args.output_dir,
#                     f'finetune_best_epoch{epoch+1}_valloss{best_val_loss:.4f}.pth',
#                 )
#                 torch.save(best_model_wts, save_path)
#                 print(f'✨ Best model → {save_path} (val_loss={best_val_loss:.4f})')
#                 wandb.save(save_path)

#         torch.cuda.empty_cache()
#         if args.distributed:
#             dist.barrier()

#     # ---------------------------------------------------------------- #
#     # Test (best model 로드 후 평가)
#     # ---------------------------------------------------------------- #
#     if is_main_process(args):
#         print(f'\n🔍 Running test with best model (epoch {best_epoch+1}) ...')

#     if best_model_wts is not None:
#         raw_model.load_state_dict(best_model_wts)

#     torch.cuda.empty_cache()

#     test_stats = test_one_epoch(
#         model, test_loader, tasks_loss_fn,
#         device, args.epochs,
#         args.in_domains, args.out_domains, args,
#         num_classes=args.num_classes if not is_regression else 1,
#     )

#     if is_main_process(args):
#         print('✅ Test performance:', test_stats)
#         wandb.finish()

#     if args.distributed:
#         dist.destroy_process_group()


# if __name__ == '__main__':
#     parser = get_finetune_args()
#     args   = parser.parse_args()
#     args   = post_process_args(args)
#     main(args)
"""
refac_finetune.py — MultiMAE Downstream Finetuning Entry Point
  - segmentation : CDL prediction  (--task_type segmentation)
  - regression   : Yield prediction (--task_type regression)

Example (CDL, single GPU):
    torchrun --nproc_per_node=1 refac_finetune.py --config refac_config_cdl.yaml

Example (Yield, single GPU):
    torchrun --nproc_per_node=1 refac_finetune.py --config refac_config_yield.yaml

Resume:
    torchrun --nproc_per_node=1 refac_finetune.py --config refac_config_cdl.yaml --resume output/finetune/cdl/finetune_latest.pth

Note:
    LR schedule (warmup+cosine) is OPT-IN via --use_lr_schedule / config `use_lr_schedule: true`.
    Default (False) preserves legacy fixed-lr behavior — used by CDL to keep existing results reproducible.
"""
import os
import copy
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import wandb

from refac_finetune_args import get_finetune_args, post_process_args
from refac_domain_conf_finetune import build_finetune_domain_conf
from refac_build_model import get_model, load_pretrained_weights
from refac_finetune_engine import train_one_epoch, test_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils.native_scaler import cosine_scheduler
from utils.refac_datasets_chloe import build_multimae_downstream_dataset
from utils.datasets_yield_chloe import build_yield_datasets


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
    """For CDL segmentation — split combined dataset into train/val/test."""
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


def build_yield_dataloaders(args, train_ds, val_ds, test_ds):
    """For Yield regression — use pre-split train/val/test datasets."""
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

    is_regression = (args.task_type == 'regression')

    if is_main_process(args):
        print(f'Task type  : {args.task_type}')
        print(f'in_domains : {args.in_domains}')
        print(f'out_domains: {args.out_domains}')

    # ---------------------------------------------------------------- #
    # Domain config
    # ---------------------------------------------------------------- #
    domain_conf = build_finetune_domain_conf(
        temporal_steps=args.temporal_steps,
        num_classes=args.num_classes,
        task_type=args.task_type,
    )

    # ---------------------------------------------------------------- #
    # Dataset
    # ---------------------------------------------------------------- #
    if args.task_type == 'segmentation':
        # CDL: combined IA+IL dataset, split into train/val/test
        dataset = build_multimae_downstream_dataset(args)
        train_loader, val_loader, test_loader, train_sampler = build_dataloaders(args, dataset)

        if is_main_process(args):
            print(f'Dataset size: {len(dataset)} | '
                  f'Train: {len(train_loader.dataset)} | '
                  f'Val: {len(val_loader.dataset)} | '
                  f'Test: {len(test_loader.dataset)}')

    elif args.task_type == 'regression':
        # Yield: pre-split train/val/test from txt files
        train_ds, val_ds, test_ds = build_yield_datasets(args)
        train_loader, val_loader, test_loader, train_sampler = build_yield_dataloaders(
            args, train_ds, val_ds, test_ds
        )

        if is_main_process(args):
            print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    # ---------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------- #
    # model = get_model(
    #     in_domains=args.in_domains,
    #     out_domains=args.out_domains,
    #     domain_conf=domain_conf,
    #     patch_size=args.patch_size,
    #     decoder_dim=args.decoder_dim,
    #     decoder_depth=args.decoder_depth,
    #     decoder_num_heads=args.decoder_num_heads,
    #     num_global_tokens=args.num_global_tokens,
    #     drop_path_rate=args.drop_path,
    #     num_classes=args.num_classes if not is_regression else 1,
    # ).to(device)

    model = get_model(
        in_domains=args.in_domains,
        out_domains=args.out_domains,
        domain_conf=domain_conf,
        patch_size=args.patch_size,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        num_classes=args.num_classes if not is_regression else 1,
        model_name='multivit_base' if is_regression else 'pretrain_multimae_base',  # ← 추가
    ).to(device)


    if args.pretrained_weights:
        model = load_pretrained_weights(
            model, args.pretrained_weights, device,
            load_input_adapters=args.load_input_adapters,
        )
        if is_main_process(args):
            print(f'Loaded pretrained weights from {args.pretrained_weights}')

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

    # ---------------------------------------------------------------- #
    # LR schedule: warmup + cosine decay (opt-in via args.use_lr_schedule)
    # Default False → legacy fixed-lr behavior (CDL keeps existing results reproducible)
    # ---------------------------------------------------------------- #
    use_lr_schedule = getattr(args, 'use_lr_schedule', False)
    niter_per_ep = len(train_loader)
    lr_schedule_values = None
    if use_lr_schedule:
        lr_schedule_values = cosine_scheduler(
            base_value=args.blr,
            final_value=args.min_lr,
            epochs=args.epochs,
            niter_per_ep=niter_per_ep,
            warmup_epochs=args.warmup_epochs,
            start_warmup_value=args.warmup_lr,
        )
        if is_main_process(args):
            print(f'[LR schedule] ENABLED — warmup {args.warmup_epochs} epochs '
                  f'({args.warmup_lr:.2e} -> {args.blr:.2e}), cosine decay to {args.min_lr:.2e}')
    else:
        if is_main_process(args):
            print(f'[LR schedule] DISABLED (legacy) — fixed lr={args.blr:.2e}')

    tasks_loss_fn = {
        d: domain_conf[d]['loss'](patch_size=args.patch_size, stride=1)
        for d in args.out_domains
    }

    # ---------------------------------------------------------------- #
    # Resume
    # ---------------------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss  = float('inf')
    best_model_wts = None
    best_epoch     = -1
    start_epoch    = args.start_epoch

    resume_path = args.resume if args.resume else None
    if resume_path is None and getattr(args, 'auto_resume', True):
        latest = os.path.join(args.output_dir, 'finetune_latest.pth')
        if os.path.isfile(latest):
            resume_path = latest

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_epoch    = ckpt.get('best_epoch', -1)
        if is_main_process(args):
            print(f'Resumed from {resume_path} (epoch {start_epoch}, best_val_loss={best_val_loss:.6f})')
    else:
        if is_main_process(args):
            print('Training from scratch.')

    # ---------------------------------------------------------------- #
    # W&B
    # ---------------------------------------------------------------- #
    if is_main_process(args):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ---------------------------------------------------------------- #
    # Training loop (with optional LR schedule + early stopping)
    # ---------------------------------------------------------------- #
    patience   = getattr(args, 'patience', 75)
    min_delta  = getattr(args, 'min_delta', 0.0001)
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ---- LR schedule 적용 (opt-in, 기본은 기존처럼 고정 lr) ---- #
        if use_lr_schedule:
            current_lr = float(lr_schedule_values[epoch * niter_per_ep])
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = args.blr   # legacy: 고정 lr, 아무것도 안 바뀜

        if is_main_process(args):
            print(f'[LR] epoch {epoch+1} | lr={current_lr:.2e}')
            wandb.log({'epoch': epoch, 'train/lr': current_lr})

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
            split='val',
        )

        val_loss = val_stats.get('loss', float('inf'))
        improved = False

        if is_main_process(args):
            improved = val_loss < (best_val_loss - min_delta)

            # Save latest checkpoint every epoch (for resume)
            latest_path = os.path.join(args.output_dir, 'finetune_latest.pth')
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'best_val_loss':        best_val_loss,
                'best_epoch':           best_epoch,
                'args':                 vars(args),
            }, latest_path)

            if improved:
                best_val_loss  = val_loss
                best_epoch     = epoch
                best_model_wts = copy.deepcopy(raw_model.state_dict())
                save_path = os.path.join(
                    args.output_dir,
                    f'finetune_best_epoch{epoch+1}_valloss{best_val_loss:.4f}.pth',
                )
                torch.save(best_model_wts, save_path)
                print(f'Best model saved: {save_path} (val_loss={best_val_loss:.4f})')
                wandb.save(save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f'[EarlyStopping] epoch {epoch+1} | val_loss={val_loss:.6f} | '
                  f'best={best_val_loss:.6f} (epoch {best_epoch+1}) | '
                  f'no_improve={epochs_no_improve}/{patience}')

        torch.cuda.empty_cache()
        if args.distributed:
            dist.barrier()

        # ---- early stopping decision, synced across processes ---- #
        if args.distributed:
            stop_flag = torch.tensor(
                [1 if (is_main_process(args) and epochs_no_improve >= patience) else 0],
                device=device,
            )
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(stop_flag.item())
        else:
            should_stop = epochs_no_improve >= patience

        if should_stop:
            if is_main_process(args):
                print(f'\nEarly stopping at epoch {epoch+1} '
                      f'(no val_loss improvement for {patience} epochs)')
            break

    # ---------------------------------------------------------------- #
    # Test (load best model)
    # ---------------------------------------------------------------- #
    if is_main_process(args):
        print(f'\nRunning test with best model (epoch {best_epoch+1}) ...')

    if best_model_wts is not None:
        raw_model.load_state_dict(best_model_wts)

    torch.cuda.empty_cache()

    test_stats = test_one_epoch(
        model, test_loader, tasks_loss_fn,
        device, args.epochs,
        args.in_domains, args.out_domains, args,
        num_classes=args.num_classes if not is_regression else 1,
    )

    if is_main_process(args):
        print('Test performance:', test_stats)
        wandb.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_finetune_args()
    args   = parser.parse_args()
    args   = post_process_args(args)
    main(args)