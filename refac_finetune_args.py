# import argparse
# import yaml


# def get_finetune_args():
#     parser = argparse.ArgumentParser('MultiMAE finetuning script', add_help=False)
#     parser.add_argument('--config', type=str, default=None,
#                         help='Path to yaml config file')

#     # ------------------------------------------------------------------ #
#     # Training
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--batch_size', default=16, type=int)
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--save_ckpt_freq', default=1, type=int)

#     # ------------------------------------------------------------------ #
#     # Task / Domain
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--task_type', default='segmentation',
#                         choices=['segmentation', 'regression'],
#                         help='segmentation=CDL, regression=Yield')
#     parser.add_argument('--in_domains', default='s1-s2-elevation-soil-weather', type=str)
#     parser.add_argument('--out_domains', default='cdl', type=str)
#     parser.add_argument('--all_domains', default='s1-s2-elevation-weather-soil-cdl', type=str)
#     parser.add_argument('--num_classes', default=2, type=int,
#                         help='segmentation에서만 사용 (regression은 무시)')

#     # ------------------------------------------------------------------ #
#     # Temporal
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--temporal_steps', default=1, type=int)

#     # ------------------------------------------------------------------ #
#     # Model
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--model', default='pretrain_multimae_base', type=str)
#     parser.add_argument('--num_encoded_tokens', default=784, type=int)
#     parser.add_argument('--num_global_tokens', default=1, type=int)
#     parser.add_argument('--patch_size', default=16, type=int)
#     parser.add_argument('--input_size', default=224, type=int)
#     parser.add_argument('--decoder_dim', default=256, type=int)
#     parser.add_argument('--decoder_depth', default=2, type=int)
#     parser.add_argument('--decoder_num_heads', default=8, type=int)
#     parser.add_argument('--drop_path', type=float, default=0.0)

#     # ------------------------------------------------------------------ #
#     # Pretrained weights
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--pretrained_weights', default='', type=str)
#     parser.add_argument('--load_input_adapters', default=False, action='store_true')

#     # ------------------------------------------------------------------ #
#     # Optimizer
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--opt', default='adamw', type=str)
#     parser.add_argument('--opt_eps', default=1e-8, type=float)
#     parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+')
#     parser.add_argument('--clip_grad', type=float, default=1.0)
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--blr', type=float, default=5e-5)
#     parser.add_argument('--warmup_lr', type=float, default=5e-6)
#     parser.add_argument('--min_lr', type=float, default=0.0)
#     parser.add_argument('--warmup_epochs', type=int, default=5)

#     # ------------------------------------------------------------------ #
#     # Augmentation
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--hflip', type=float, default=0.0)
#     parser.add_argument('--train_interpolation', type=str, default='bicubic')

#     # ------------------------------------------------------------------ #
#     # Dataset — pretrain 방식 (state별 txt 경로)
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--stats_dir', type=str, default=None)
#     parser.add_argument('--states', default='IA-IL', type=str)
#     parser.add_argument('--data_path', type=str, default=None)

#     # IA txt paths
#     parser.add_argument('--s1_txt_IA',        type=str, default=None)
#     parser.add_argument('--s2_txt_IA',        type=str, default=None)
#     parser.add_argument('--cdl_txt_IA',       type=str, default=None)
#     parser.add_argument('--elevation_txt_IA', type=str, default=None)
#     parser.add_argument('--soil_txt_IA',      type=str, default=None)
#     parser.add_argument('--weather_txt_IA',   type=str, default=None)
#     parser.add_argument('--yield_txt_IA',     type=str, default=None)

#     # IL txt paths
#     parser.add_argument('--s1_txt_IL',        type=str, default=None)
#     parser.add_argument('--s2_txt_IL',        type=str, default=None)
#     parser.add_argument('--cdl_txt_IL',       type=str, default=None)
#     parser.add_argument('--elevation_txt_IL', type=str, default=None)
#     parser.add_argument('--soil_txt_IL',      type=str, default=None)
#     parser.add_argument('--weather_txt_IL',   type=str, default=None)
#     parser.add_argument('--yield_txt_IL',     type=str, default=None)

#     # ------------------------------------------------------------------ #
#     # Misc
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--output_dir', default='./output/finetune')
#     parser.add_argument('--device', default='cuda')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--resume', default='')
#     parser.add_argument('--start_epoch', default=0, type=int)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--pin_mem', action='store_true')
#     parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
#     parser.set_defaults(pin_mem=True)
#     parser.add_argument('--find_unused_params', action='store_true')
#     parser.set_defaults(find_unused_params=False)

#     # ------------------------------------------------------------------ #
#     # W&B
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--wandb_project', default='MultiMAE-finetune', type=str)
#     parser.add_argument('--wandb_entity', default='goeulkim', type=str)
#     parser.add_argument('--wandb_run_name', default='finetune-cdl', type=str)

#     # ------------------------------------------------------------------ #
#     # Distributed
#     # ------------------------------------------------------------------ #
#     parser.add_argument('--world_size', default=1, type=int)
#     parser.add_argument('--local_rank', default=-1, type=int)
#     parser.add_argument('--dist_on_itp', action='store_true')
#     parser.add_argument('--dist_url', default='env://')

#     return parser


# def post_process_args(args):
#     # yaml config 파일이 있으면 로드해서 args에 덮어쓰기
#     if args.config is not None:
#         with open(args.config) as f:
#             cfg = yaml.safe_load(f)
#         for k, v in cfg.items():
#             setattr(args, k, v)

#     args.in_domains  = args.in_domains.split('-') if isinstance(args.in_domains, str) else args.in_domains
#     args.out_domains = args.out_domains.split('-') if isinstance(args.out_domains, str) else args.out_domains
#     args.all_domains = args.all_domains.split('-') if isinstance(args.all_domains, str) else args.all_domains
#     args.states      = args.states.split('-') if isinstance(args.states, str) else args.states

#     # regression일 때 out_domains 자동 설정
#     if args.task_type == 'regression' and 'yield' not in args.out_domains:
#         args.out_domains = ['yield']
#     if args.task_type == 'regression' and 'yield' not in args.all_domains:
#         args.all_domains = [d for d in args.all_domains if d != 'cdl'] + ['yield']

#     # state별 txt_paths 딕셔너리 구성
#     is_regression = (args.task_type == 'regression')
#     label_key = 'yield' if is_regression else 'cdl'

#     args.txt_paths_by_state = {}
#     for state in args.states:
#         label_txt = getattr(args, f'{label_key}_txt_{state}', None)
#         args.txt_paths_by_state[state] = {
#             's1':        getattr(args, f's1_txt_{state}',        None),
#             's2':        getattr(args, f's2_txt_{state}',        None),
#             'elevation': getattr(args, f'elevation_txt_{state}', None),
#             'soil':      getattr(args, f'soil_txt_{state}',      None),
#             'weather':   getattr(args, f'weather_txt_{state}',   None),
#             label_key:   label_txt,
#         }

#     return args

import argparse
import yaml


def get_finetune_args():
    parser = argparse.ArgumentParser('MultiMAE finetuning script', add_help=False)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to yaml config file')

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation steps (TerraMind comparison: 8)')
    parser.add_argument('--patience', default=75, type=int,
                        help='Early stopping patience (epochs without val_loss improvement)')
    parser.add_argument('--min_delta', default=0.0001, type=float,
                        help='Minimum val_loss improvement to reset early stopping counter')
    parser.add_argument('--use_lr_schedule', default=False, action='store_true',
                        help='Enable warmup+cosine LR schedule. If False, lr stays fixed at --blr (legacy/CDL behavior).')

    # ------------------------------------------------------------------ #
    # Task / Domain
    # ------------------------------------------------------------------ #
    parser.add_argument('--task_type', default='segmentation',
                        choices=['segmentation', 'regression'],
                        help='segmentation=CDL, regression=Yield')
    parser.add_argument('--in_domains', default='s1-s2-elevation-soil-weather', type=str)
    parser.add_argument('--out_domains', default='cdl', type=str)
    parser.add_argument('--all_domains', default='s1-s2-elevation-weather-soil-cdl', type=str)
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of classes for segmentation (ignored for regression)')

    # ------------------------------------------------------------------ #
    # Temporal
    # ------------------------------------------------------------------ #
    parser.add_argument('--temporal_steps', default=1, type=int,
                        help='Number of temporal steps. For yield regression, set to match dataset (e.g. 15)')

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
    parser.add_argument('--drop_path', type=float, default=0.0)

    # ------------------------------------------------------------------ #
    # Pretrained weights
    # ------------------------------------------------------------------ #
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--load_input_adapters', default=False, action='store_true')

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--blr', type=float, default=5e-5)
    parser.add_argument('--warmup_lr', type=float, default=5e-6)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #
    parser.add_argument('--hflip', type=float, default=0.0)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    # ------------------------------------------------------------------ #
    # Dataset — CDL (state-based txt paths)
    # ------------------------------------------------------------------ #
    parser.add_argument('--stats_dir', type=str, default=None)
    parser.add_argument('--states', default='IA-IL', type=str)
    parser.add_argument('--data_path', type=str, default=None)

    # IA txt paths
    parser.add_argument('--s1_txt_IA',        type=str, default=None)
    parser.add_argument('--s2_txt_IA',        type=str, default=None)
    parser.add_argument('--cdl_txt_IA',       type=str, default=None)
    parser.add_argument('--elevation_txt_IA', type=str, default=None)
    parser.add_argument('--soil_txt_IA',      type=str, default=None)
    parser.add_argument('--weather_txt_IA',   type=str, default=None)
    parser.add_argument('--yield_txt_IA',     type=str, default=None)

    # IL txt paths
    parser.add_argument('--s1_txt_IL',        type=str, default=None)
    parser.add_argument('--s2_txt_IL',        type=str, default=None)
    parser.add_argument('--cdl_txt_IL',       type=str, default=None)
    parser.add_argument('--elevation_txt_IL', type=str, default=None)
    parser.add_argument('--soil_txt_IL',      type=str, default=None)
    parser.add_argument('--weather_txt_IL',   type=str, default=None)
    parser.add_argument('--yield_txt_IL',     type=str, default=None)

    # ------------------------------------------------------------------ #
    # Dataset — Yield regression
    # ------------------------------------------------------------------ #
    parser.add_argument('--yield_data_root', type=str, default=None,
                        help='Path to yield dataset root directory (contains train/val/test_corn.txt)')

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    parser.add_argument('--output_dir', default='./output/finetune')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', default=True, type=bool)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.set_defaults(find_unused_params=False)

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    parser.add_argument('--wandb_project', default='MultiMAE-finetune', type=str)
    parser.add_argument('--wandb_entity', default='goeulkim', type=str)
    parser.add_argument('--wandb_run_name', default='finetune-cdl', type=str)

    # ------------------------------------------------------------------ #
    # Distributed
    # ------------------------------------------------------------------ #
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def post_process_args(args):
    # Load yaml config and override args
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # Parse domain strings
    args.in_domains  = args.in_domains.split('-') if isinstance(args.in_domains, str) else args.in_domains
    args.out_domains = args.out_domains.split('-') if isinstance(args.out_domains, str) else args.out_domains
    args.all_domains = args.all_domains.split('-') if isinstance(args.all_domains, str) else args.all_domains
    args.states      = args.states.split('-') if isinstance(args.states, str) else args.states

    # Auto-set out_domains and all_domains for regression
    if args.task_type == 'regression' and 'yield' not in args.out_domains:
        args.out_domains = ['yield']
    if args.task_type == 'regression' and 'yield' not in args.all_domains:
        args.all_domains = [d for d in args.all_domains if d != 'cdl'] + ['yield']

    # Build txt_paths_by_state for CDL segmentation
    is_regression = (args.task_type == 'regression')
    label_key = 'yield' if is_regression else 'cdl'

    args.txt_paths_by_state = {}
    if not is_regression:  # Only build for segmentation tasks  
        for state in args.states:
            label_txt = getattr(args, f'{label_key}_txt_{state}', None)
            args.txt_paths_by_state[state] = {
                's1':        getattr(args, f's1_txt_{state}',        None),
                's2':        getattr(args, f's2_txt_{state}',        None),
                'elevation': getattr(args, f'elevation_txt_{state}', None),
                'soil':      getattr(args, f'soil_txt_{state}',      None),
                'weather':   getattr(args, f'weather_txt_{state}',   None),
                label_key:   label_txt,
            }

    return args