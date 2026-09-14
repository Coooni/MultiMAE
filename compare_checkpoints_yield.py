"""
compare_checkpoints_yield.py — 저장된 모든 finetune checkpoint를 val set으로 평가해서
                                 field R² 기준 best checkpoint를 찾고,
                                 그 checkpoint를 TerraMind(ISA Yield project)와
                                 완전히 동일한 방식으로 test 평가하는 스크립트.

  TerraMind와 동일한 evaluation 로직 (regression_tasks.py의 on_test_epoch_end 그대로 따라감):
    1. 모든 픽셀을 (YieldGT, Prediction, Filename) 테이블로 쌓기
    2. YieldGT > 0 인 행만 사용 (배경/-1 clip 픽셀 제외)
    3. Filename(필드)별로 groupby → 평균
    4. corn min-max로 denormalize (data_min=50, data_max=370) → bu/acre
    5. sklearn의 r2_score, mean_absolute_error로 최종 지표 계산

Usage:
    python compare_checkpoints_yield.py --config refac_config_yield.yaml
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from refac_finetune_args import get_finetune_args, post_process_args
from refac_domain_conf_finetune import build_finetune_domain_conf
from refac_build_model import get_model
from utils.datasets_yield_chloe import build_yield_datasets
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ #
# Corn denormalization constants (TerraMind regression_tasks.py와 동일)
# ------------------------------------------------------------------ #
CORN_DATA_MIN = 50.0
CORN_DATA_MAX = 370.0


def compute_r2(pred, target):
    pred, target = pred.detach().cpu().float().reshape(-1), target.detach().cpu().float().reshape(-1)
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


@torch.no_grad()
def evaluate_checkpoint(model, loader, in_domains, device, num_encoded_tokens):
    """
    Val set checkpoint 순위 매기기용 (patch 전체 평균, 정규화 스케일).
    R²는 스케일 무관이라 순위 비교엔 문제없음. 빠르고 단순하게 유지.
    """
    model.eval()
    field_preds, field_targets = [], []

    for batch in loader:
        tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
        input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}
        with torch.cuda.amp.autocast():
            preds, _ = model(input_dict, mask_inputs=False, num_encoded_tokens=num_encoded_tokens)

        p, t = preds['yield'], tasks_dict['yield']
        field_preds.append(p.mean(dim=[1, 2, 3]).cpu())
        field_targets.append(t.mean(dim=[1, 2, 3]).cpu())

    all_preds   = torch.cat(field_preds)
    all_targets = torch.cat(field_targets)

    return {'field_r2': compute_r2(all_preds, all_targets)}


@torch.no_grad()
def evaluate_test_terramind_way(model, test_loader, in_domains, device, num_encoded_tokens):
    """
    TerraMind(regression_tasks.py, on_test_epoch_end)와 동일한 방식으로 field-level 평가.
    가능한 한 원본 코드 구조를 그대로 따라감 (pandas DataFrame 기반).
    """
    model.eval()

    all_gt, all_pred, all_fname = [], [], []

    for batch_idx, batch in enumerate(test_loader):
        tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
        input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}
        with torch.cuda.amp.autocast():
            preds, _ = model(input_dict, mask_inputs=False, num_encoded_tokens=num_encoded_tokens)

        p = preds['yield']       # (B, 1, H, W), 정규화된 스케일
        t = tasks_dict['yield']  # (B, 1, H, W), 정규화된 스케일
        B = p.shape[0]

        # TerraMind처럼: 배치 안 각 샘플(=필드 하나)을 픽셀 단위로 펼쳐서 쌓기
        for i in range(B):
            gt_flat   = t[i].flatten().cpu().numpy()
            pred_flat = p[i].flatten().cpu().float().numpy()
            fname     = f'test_{batch_idx}_{i}'   # 필드 식별용 임시 id (실제 파일명 없어도 groupby엔 문제없음)

            all_gt.append(gt_flat)
            all_pred.append(pred_flat)
            all_fname.extend([fname] * len(gt_flat))

    # ── 여기서부터는 TerraMind on_test_epoch_end 코드와 동일한 순서 ── #
    df = pd.DataFrame({
        'YieldGT': np.concatenate(all_gt).astype(float),
        'Prediction': np.concatenate(all_pred).astype(float),
        'Filename': all_fname,
    })

    df = df[df["YieldGT"] > 0]   # 배경/-1 clip 픽셀 제외

    df_agg = df.groupby('Filename').agg({
        'YieldGT': 'mean',
        'Prediction': 'mean',
    }).reset_index()

    # denormalize (corn 기준, TerraMind와 동일한 공식)
    df_agg['YieldGT']     = df_agg['YieldGT']     * (CORN_DATA_MAX - CORN_DATA_MIN) + CORN_DATA_MIN
    df_agg['Prediction']  = df_agg['Prediction']  * (CORN_DATA_MAX - CORN_DATA_MIN) + CORN_DATA_MIN

    y_true = df_agg['YieldGT'].values
    y_pred = df_agg['Prediction'].values

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    return {
        'field_r2':   r2,
        'field_mae':  mae,    # bu/acre
        'field_rmse': rmse,   # bu/acre
        'n_fields':   len(df_agg),
        'df_agg':     df_agg,
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_conf = build_finetune_domain_conf(
        temporal_steps=args.temporal_steps,
        num_classes=args.num_classes,
        task_type=args.task_type,
    )

    train_ds, val_ds, test_ds = build_yield_datasets(args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.pin_mem)

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

    # ---------------------------------------------------------------- #
    # 1단계: val set으로 모든 checkpoint 스캔 → best 찾기 (빠르고 단순한 방식)
    # ---------------------------------------------------------------- #
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.output_dir, 'finetune_best_epoch*_valloss*.pth')),
        key=lambda p: int(re.search(r'epoch(\d+)_', p).group(1)),
    )
    print(f'Found {len(ckpt_paths)} checkpoints in {args.output_dir}\n')

    results = []
    for ckpt_path in ckpt_paths:
        epoch = int(re.search(r'epoch(\d+)_', ckpt_path).group(1))
        val_loss_tag = re.search(r'valloss([\d.]+)\.pth', ckpt_path).group(1)

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        metrics = evaluate_checkpoint(model, val_loader, args.in_domains, device, args.num_encoded_tokens)
        metrics['epoch'] = epoch
        metrics['ckpt_val_loss'] = float(val_loss_tag)
        metrics['ckpt_path'] = ckpt_path
        results.append(metrics)

        print(f"epoch {epoch:4d} | val_loss(saved)={val_loss_tag:>8s} | VAL field_R2={metrics['field_r2']:.4f}")

    results_sorted = sorted(results, key=lambda r: r['field_r2'], reverse=True)

    print('\n' + '=' * 70)
    print('TOP 5 checkpoints by VAL field R²:')
    print('=' * 70)
    for r in results_sorted[:5]:
        print(f"epoch {r['epoch']:4d} | VAL field_R2={r['field_r2']:.4f} | ckpt_val_loss={r['ckpt_val_loss']:.4f}")

    best = results_sorted[0]
    print(f"\n>>> Best by VAL field R²: epoch {best['epoch']} ({best['ckpt_path']})")

    # ---------------------------------------------------------------- #
    # 2단계: best checkpoint 하나만 TerraMind 방식으로 test 평가
    # ---------------------------------------------------------------- #
    state_dict = torch.load(best['ckpt_path'], map_location=device)
    model.load_state_dict(state_dict)

    print("\n>>> Running TEST evaluation (TerraMind 방식: filter + groupby + denormalize)...\n")
    test_result = evaluate_test_terramind_way(
        model, test_loader, args.in_domains, device, args.num_encoded_tokens
    )

    print('=' * 70)
    print(f"FINAL TEST RESULTS (checkpoint = epoch {best['epoch']}):")
    print(f"  (유효 field 수: {test_result['n_fields']})")
    print(f"  Field R²   : {test_result['field_r2']:.4f}")
    print(f"  Field RMSE : {test_result['field_rmse']:.2f} bu/acre")
    print(f"  Field MAE  : {test_result['field_mae']:.2f} bu/acre")
    print('=' * 70)
    print("  참고 — TerraMind M5 (fine-tuned): Field R²=0.680, Field MAE=12.22 bu/acre")
    print('=' * 70)

    # 확인용: 처음 5개 필드 실제 값 미리보기
    print('\n처음 5개 필드 미리보기:')
    print(test_result['df_agg'].head())


if __name__ == '__main__':
    parser = get_finetune_args()
    args   = parser.parse_args()
    args   = post_process_args(args)
    main(args)