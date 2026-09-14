"""
engine/pretrain_engine.py
Pretraining 전용 train / validation / test loop.
  - MaskedMSELoss reconstruction
  - NaN 디버깅 포함
  - test: R², RMSE, MAE, MSE (modality별 + 전체 평균)
  - args를 명시적 파라미터로 받음 (글로벌 참조 없음)
  - wandb.log()는 main process (rank 0)에서만 호출
  - test loop: 배치별 누적 통계로 OOM 방지 (결과값 동일)
  - valid/test wandb.log()는 100 step마다 호출 (속도 향상)
"""
import numpy as np
import torch
import torch.distributed as dist
import wandb

import utils


def is_main() -> bool:
    """rank 0인지 확인 (단일 GPU도 True 반환)"""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def train_one_epoch(
    model,
    loader,
    tasks_loss_fn: dict,
    optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    in_domains: list,
    out_domains: list,
    args,
    split: str = 'train',
) -> dict:
    """
    Args:
        split: 'train' or 'valid'
    Returns:
        dict of averaged metrics
    """
    if split == 'train':
        model.train()
    else:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'[{split.upper()}] Epoch [{epoch}]'

    for step, batch in enumerate(metric_logger.log_every(loader, 10, header)):
        tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
        input_dict  = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}

        # ---------------------------------------------------------------- #
        # Forward
        # ---------------------------------------------------------------- #
        ctx = torch.no_grad() if split == 'valid' else torch.enable_grad()
        with ctx:
            with torch.cuda.amp.autocast():
                preds, masks = model(input_dict, num_encoded_tokens=args.num_encoded_tokens)

                task_losses = {}
                for task in out_domains:
                    target = tasks_dict[task]
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)

                # NaN per-modality 디버깅
                for tname, tval in task_losses.items():
                    if not torch.isfinite(tval).all():
                        pred = preds[tname]
                        tgt  = tasks_dict[tname]
                        print(f'🚨 NaN in {tname}_loss | epoch {epoch} step {step}')
                        print(f'   pred  : min={pred.min():.3f} max={pred.max():.3f} mean={pred.mean():.3f}')
                        print(f'   target: min={tgt.min():.3f}  max={tgt.max():.3f}  mean={tgt.mean():.3f}')

                loss = sum(task_losses.values())

        # ---------------------------------------------------------------- #
        # Backward (train only)
        # ---------------------------------------------------------------- #
        if split == 'train':
            optimizer.zero_grad()

            if not torch.isfinite(loss):
                print(f'⚠️  NaN loss BEFORE backward | epoch {epoch} step {step}')
                for k, v in task_losses.items():
                    val_str = f'{v.detach().mean().item():.4f}' if torch.isfinite(v).all() else 'NaN'
                    print(f'   {k}_loss: {val_str}')
                continue  # skip batch

            scaler = loss_scaler._scaler
            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f'🚨 RuntimeError during backward | epoch {epoch} step {step}: {e}')
                for n, p in model.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f'   NaN grad : {n}')
                    if torch.isnan(p).any():
                        print(f'   NaN weight: {n}')
                continue

            # Unscale → Clip grad
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            # Grad norm 모니터링
            grad_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach(), 2)
                    for p in model.parameters()
                    if p.grad is not None
                ]), 2
            ).item()

            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()

            if step % 50 == 0:
                print(f'[Epoch {epoch} | Step {step}] Grad norm: {grad_norm:.2f} | '
                      f'AMP scale: {scaler.get_scale():.1f}')
        else:
            grad_norm = 0.0

        # ---------------------------------------------------------------- #
        # Logging
        # - train: 매 step마다
        # - valid: 100 step마다 (wandb 오버헤드 감소)
        # ---------------------------------------------------------------- #
        metric_logger.update(loss=loss.item(), grad_norm=grad_norm)
        for task, l in task_losses.items():
            metric_logger.update(**{f'{split}_{task}_loss': l.item()})

        if is_main():
            wandb.log({
                'epoch': epoch,
                'step': step,
                f'{split}/loss': loss.item(),
                f'{split}/grad_norm': grad_norm,
                **{f'{split}/{task}_loss': l.item() for task, l in task_losses.items()},
            })

    metric_logger.synchronize_between_processes()
    print(f'[{split.upper()}] Epoch {epoch} averaged stats:', metric_logger)

    # Epoch-level summary (main process only)
    if is_main():
        epoch_log = {
            'epoch': epoch,
            f'{split}/loss_avg': metric_logger.meters['loss'].global_avg,
        }
        for task in out_domains:
            key = f'{split}_{task}_loss'
            if key in metric_logger.meters:
                epoch_log[f'{split}/{task}_loss_avg'] = metric_logger.meters[key].global_avg
        wandb.log(epoch_log)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ------------------------------------------------------------------ #
# Test loop
# ------------------------------------------------------------------ #
def test_one_epoch(
    model,
    test_loader,
    tasks_loss_fn: dict,
    device: torch.device,
    epoch: int,
    in_domains: list,
    out_domains: list,
    args,
) -> dict:
    """
    Pretraining test loop.
    GPU tensor로 누적 통계 계산 → CPU 변환 최소화로 속도 향상.
    step별 wandb log 없음 — 최종 metrics만 저장.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'[TEST] Epoch [{epoch}]'
 
    # GPU tensor로 누적 (CPU numpy 변환 제거)
    stats = {
        t: {
            'n':       torch.tensor(0,   dtype=torch.float64, device=device),
            'sum_y':   torch.tensor(0.0, dtype=torch.float64, device=device),
            'sum_y2':  torch.tensor(0.0, dtype=torch.float64, device=device),
            'sum_res': torch.tensor(0.0, dtype=torch.float64, device=device),
            'sum_mae': torch.tensor(0.0, dtype=torch.float64, device=device),
        }
        for t in out_domains
    }
 
    with torch.no_grad():
        for step, batch in enumerate(metric_logger.log_every(test_loader, 10, header)):
            tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
            input_dict  = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}
 
            with torch.cuda.amp.autocast():
                preds, masks = model(input_dict, num_encoded_tokens=args.num_encoded_tokens)
 
                task_losses = {}
                for task in out_domains:
                    target = tasks_dict[task]
                    pred   = preds[task].float()
                    task_losses[task] = tasks_loss_fn[task](pred, target)
 
                    # GPU에서 바로 누적 (CPU 변환 없음)
                    p = pred.detach().reshape(-1).double()
                    y = target.detach().reshape(-1).double()
                    stats[task]['n']       += y.numel()
                    stats[task]['sum_y']   += y.sum()
                    stats[task]['sum_y2']  += (y ** 2).sum()
                    stats[task]['sum_res'] += ((y - p) ** 2).sum()
                    stats[task]['sum_mae'] += (y - p).abs().sum()
 
                loss = sum(task_losses.values())
 
            metric_logger.update(loss=loss.item(), grad_norm=0.0)
            for task, l in task_losses.items():
                metric_logger.update(**{f'test_{task}_loss': l.item()})
 
    # ---------------------------------------------------------------- #
    # Per-modality metrics
    # ---------------------------------------------------------------- #
    results = {}
    all_mse, all_r2 = [], []
 
    print(f'\n{"─"*60}')
    print(f'  Test Epoch {epoch} — Reconstruction Metrics')
    print(f'{"─"*60}')
 
    for task in out_domains:
        n       = stats[task]['n'].item()
        sum_y   = stats[task]['sum_y'].item()
        sum_y2  = stats[task]['sum_y2'].item()
        ss_res  = stats[task]['sum_res'].item()
        sum_mae = stats[task]['sum_mae'].item()
 
        mean_y = sum_y / n
        ss_tot = sum_y2 - n * mean_y ** 2
 
        mse  = float(ss_res / n)
        rmse = float(np.sqrt(mse))
        mae  = float(sum_mae / n)
        r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
 
        results[task] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        all_mse.append(mse)
        all_r2.append(r2)
 
        print(f'  [{task:>10s}]  MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}')
 
    # ---------------------------------------------------------------- #
    # Macro-average across modalities
    # ---------------------------------------------------------------- #
    avg_mse  = float(np.mean(all_mse))
    avg_rmse = float(np.sqrt(avg_mse))
    avg_r2   = float(np.mean(all_r2))
 
    print(f'{"─"*60}')
    print(f'  [   avg    ]  MSE={avg_mse:.4f}  RMSE={avg_rmse:.4f}  R²={avg_r2:.4f}')
    print(f'{"─"*60}\n')
 
    metric_logger.synchronize_between_processes()
    print('Test loss stats:', metric_logger)
 
    # 최종 metrics만 wandb에 한 번 저장
    if is_main():
        test_log = {
            'epoch': epoch,
            'test/avg_mse':  avg_mse,
            'test/avg_rmse': avg_rmse,
            'test/avg_r2':   avg_r2,
            'test/loss_avg': metric_logger.meters['loss'].global_avg,
        }
        for task, m in results.items():
            test_log[f'test/{task}_mse']  = m['mse']
            test_log[f'test/{task}_rmse'] = m['rmse']
            test_log[f'test/{task}_mae']  = m['mae']
            test_log[f'test/{task}_r2']   = m['r2']
        wandb.log(test_log)
 
    return {
        'avg_mse':  avg_mse,
        'avg_rmse': avg_rmse,
        'avg_r2':   avg_r2,
        **{f'{task}_{k}': v for task, metrics in results.items() for k, v in metrics.items()},
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
    }
 