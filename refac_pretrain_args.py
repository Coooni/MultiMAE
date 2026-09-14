import argparse
import yaml


def get_pretrain_args():
    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to yaml config file')


    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int,
                        help='Checkpoint saving frequency in epochs')

    # ------------------------------------------------------------------ #
    # Task / Domain
    # ------------------------------------------------------------------ #
    parser.add_argument('--in_domains', default='s1-s2-elevation-weather-soil-cdl', type=str,
                        help='Input domain names, separated by hyphen')
    parser.add_argument('--out_domains', default='s1-s2-elevation-weather-soil-cdl', type=str,
                        help='Output domain names, separated by hyphen')
    parser.add_argument('--all_domains', default='s1-s2-elevation-weather-soil-cdl', type=str,
                        help='All domain names, separated by hyphen')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    parser.add_argument('--model', default='pretrain_multimae_base', type=str)
    parser.add_argument('--num_encoded_tokens', default=784, type=int)
    parser.add_argument('--num_global_tokens', default=1, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--decoder_dim', default=256, type=int)
    parser.add_argument('--decoder_depth', default=2, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true')
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--alphas', type=float, default=1.0)
    parser.add_argument('--sample_tasks_uniformly', default=True, action='store_true')
    parser.add_argument('--loss_on_unmasked', default=False, action='store_true')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)
    parser.add_argument('--fp32_output_adapters', type=str, default='')

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--skip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--decoder_decay', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-5,
                        help='Base LR: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--warmup_lr', type=float, default=5e-6)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--task_balancer', type=str, default='none',
                        help='Task balancing scheme: [uncertainty, none]')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0)

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #
    parser.add_argument('--hflip', type=float, default=0.5)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    # ------------------------------------------------------------------ #
    # Dataset paths
    # ------------------------------------------------------------------ #
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--stats_dir', type=str, default=None,
                        help='stats npz 파일들이 있는 디렉토리')
    parser.add_argument('--states', type=str, default='IA',
                        help='학습에 사용할 주 목록, 하이픈으로 구분 (예: IA-IL)')

    # IA txt paths
    parser.add_argument('--s1_txt_IA',        type=str, default=None)
    parser.add_argument('--s2_txt_IA',        type=str, default=None)
    parser.add_argument('--cdl_txt_IA',       type=str, default=None)
    parser.add_argument('--elevation_txt_IA', type=str, default=None)
    parser.add_argument('--soil_txt_IA',      type=str, default=None)
    parser.add_argument('--weather_txt_IA',   type=str, default=None)

    # IL txt paths
    parser.add_argument('--s1_txt_IL',        type=str, default=None)
    parser.add_argument('--s2_txt_IL',        type=str, default=None)
    parser.add_argument('--cdl_txt_IL',       type=str, default=None)
    parser.add_argument('--elevation_txt_IL', type=str, default=None)
    parser.add_argument('--soil_txt_IL',      type=str, default=None)
    parser.add_argument('--weather_txt_IL',   type=str, default=None)

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='Path to save checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint path')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=False)
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    parser.add_argument('--wandb_project', default='MultiMAE-pretrain', type=str)
    parser.add_argument('--wandb_entity', default='goeulkim', type=str)
    parser.add_argument('--wandb_run_name', default='pretrain-run', type=str)

    # ------------------------------------------------------------------ #
    # Distributed
    # ------------------------------------------------------------------ #
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def post_process_args(args):
    """String → list 변환 등 args 후처리"""
    # yaml config 파일이 있으면 로드해서 args에 덮어쓰기
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    args.in_domains  = args.in_domains.split('-') if isinstance(args.in_domains, str) else args.in_domains
    args.out_domains = args.out_domains.split('-') if isinstance(args.out_domains, str) else args.out_domains
    args.all_domains = args.all_domains.split('-') if isinstance(args.all_domains, str) else args.all_domains
    args.states      = args.states.split('-') if isinstance(args.states, str) else args.states

    # txt_paths_by_state 딕셔너리 자동 구성
    args.txt_paths_by_state = {}
    for state in args.states:
        args.txt_paths_by_state[state] = {
            d: getattr(args, f"{d}_txt_{state}")
            for d in args.all_domains
        }
    return args