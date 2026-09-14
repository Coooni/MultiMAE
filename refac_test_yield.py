"""
test_yield.py — Run test only with best checkpoint
Usage:
    python test_yield.py --config refac_config_yield.yaml --checkpoint output/finetune/yield/finetune_best_epochXX_vallossX.pth
"""
import os
import torch
import torch.backends.cudnn as cudnn
import wandb

from refac_finetune_args import get_finetune_args, post_process_args
from refac_domain_conf_finetune import build_finetune_domain_conf
from refac_build_model import get_model, load_pretrained_weights
from refac_finetune_engine import test_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets_yield_chloe import build_yield_datasets
from torch.utils.data import DataLoader


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    is_regression = (args.task_type == 'regression')

    # Domain config
    domain_conf = build_finetune_domain_conf(
        temporal_steps=args.temporal_steps,
        num_classes=args.num_classes,
        task_type=args.task_type,
    )

    # Dataset
    _, test_ds, _ = build_yield_datasets(args)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
    )
    print(f'Test samples: {len(test_ds)}')

    # Model
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
        num_classes=1,
    ).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        # try best checkpoint in output_dir
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, 'finetune_best_epoch*.pth')))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f'Auto-selected checkpoint: {ckpt_path}')
        else:
            raise FileNotFoundError(f'No checkpoint found in {args.output_dir}')

    state_dict = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {ckpt_path}')

    # Loss
    tasks_loss_fn = {
        d: domain_conf[d]['loss'](patch_size=args.patch_size, stride=1)
        for d in args.out_domains
    }

    # W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name + '-test-only',
        config=vars(args),
    )

    # Test
    test_stats = test_one_epoch(
        model, test_loader, tasks_loss_fn,
        device, 0,
        args.in_domains, args.out_domains, args,
        num_classes=1,
    )

    print('Final Test Results:')
    for k, v in test_stats.items():
        if 'field' in k or 'r2' in k or 'rmse' in k or 'mae' in k:
            print(f'  {k}: {v:.4f}')

    wandb.finish()


if __name__ == '__main__':
    parser = get_finetune_args()
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint to test')
    args = parser.parse_args()
    args = post_process_args(args)
    main(args)