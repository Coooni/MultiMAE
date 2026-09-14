# """
# refac_finetune_engine.py
# Downstream finetuning 전용 train / validation / test loop.

#   [segmentation]  CDL prediction
#     - BinaryFocalLoss
#     - metrics: pixel accuracy, mIoU, per-class accuracy, confusion matrix

#   [regression]    Yield prediction
#     - MSELoss
#     - metrics: R², RMSE, MAE

#   공통:
#     - mask_inputs=False (마스킹 없이 전체 입력 사용)
#     - wandb.log()는 main process (rank 0)에서만 호출
# """
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import wandb
# from sklearn.metrics import confusion_matrix

# import utils


# def is_main() -> bool:
#     return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


# # ------------------------------------------------------------------ #
# # Segmentation metric helpers
# # ------------------------------------------------------------------ #

# def compute_confusion_matrix(logits, target, num_classes):
#     pred   = logits.argmax(dim=1).cpu().numpy().ravel()
#     target = target.cpu().numpy().ravel()
#     return confusion_matrix(target, pred, labels=list(range(num_classes)))


# def pixel_accuracy(logits, target):
#     pred    = logits.argmax(dim=1)
#     correct = (pred == target).sum().item()
#     return correct / max(1, target.numel())


# def mean_iou(logits, target, num_classes):
#     pred   = logits.argmax(dim=1).cpu().numpy()
#     target = target.cpu().numpy()
#     ious = []
#     for c in range(num_classes):
#         inter = np.logical_and(pred == c, target == c).sum()
#         union = np.logical_or(pred == c, target == c).sum()
#         if union > 0:
#             ious.append(inter / union)
#     return float(np.mean(ious)) if ious else 0.0


# def per_class_accuracy(logits, target, num_classes):
#     pred = logits.argmax(dim=1)
#     accs = []
#     for c in range(num_classes):
#         mask = (target == c)
#         n = mask.sum().item()
#         accs.append(((pred == c) & mask).sum().item() / n if n > 0 else float('nan'))
#     return accs


# # ------------------------------------------------------------------ #
# # Regression metric helpers
# # ------------------------------------------------------------------ #

# def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     ss_res = ((target - pred) ** 2).sum().item()
#     ss_tot = ((target - target.mean()) ** 2).sum().item()
#     return 1.0 - ss_res / max(ss_tot, 1e-8)


# def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     return float(torch.sqrt(((pred - target) ** 2).mean()).item())


# def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     return float((pred - target).abs().mean().item())


# # ------------------------------------------------------------------ #
# # Train / Validation loop  (segmentation + regression 공통)
# # ------------------------------------------------------------------ #

# def train_one_epoch(
#     model,
#     loader,
#     tasks_loss_fn: dict,
#     optimizer,
#     device: torch.device,
#     epoch: int,
#     loss_scaler,
#     in_domains: list,
#     out_domains: list,
#     args,
#     split: str = 'train',
# ) -> dict:
#     is_regression = (args.task_type == 'regression')

#     model.train() if split == 'train' else model.eval()

#     metric_logger = utils.MetricLogger(delimiter='  ')
#     header = f'[{split.upper()}] Epoch [{epoch}]'

#     r2_sum, rmse_sum, mae_sum, n_reg = 0.0, 0.0, 0.0, 0

#     ctx = torch.no_grad() if split != 'train' else torch.enable_grad()
#     with ctx:
#         for step, batch in enumerate(metric_logger.log_every(loader, 10, header)):
#             tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
#             input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}

#             with torch.cuda.amp.autocast():
#                 preds, masks = model(
#                     input_dict,
#                     mask_inputs=False,
#                     num_encoded_tokens=args.num_encoded_tokens,
#                 )

#                 task_losses = {}
#                 for task in out_domains:
#                     target = tasks_dict[task]
#                     task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)

#                 loss = sum(task_losses.values())

#             if split == 'train':
#                 optimizer.zero_grad()
#                 grad_norm = loss_scaler(
#                     loss, optimizer,
#                     parameters=model.parameters(),
#                     clip_grad=args.clip_grad,
#                 )
#                 torch.cuda.synchronize()
#             else:
#                 grad_norm = 0.0

#             metric_logger.update(loss=loss.item(), grad_norm=grad_norm)
#             for task, l in task_losses.items():
#                 metric_logger.update(**{f'{split}_{task}_loss': l.item()})

#             # regression 추가 metrics
#             step_log = {}
#             if is_regression and 'yield' in preds:
#                 p = preds['yield']
#                 t = tasks_dict['yield']
#                 r2   = compute_r2(p, t)
#                 rmse = compute_rmse(p, t)
#                 mae  = compute_mae(p, t)
#                 r2_sum += r2; rmse_sum += rmse; mae_sum += mae; n_reg += 1
#                 step_log = {f'{split}/r2_step': r2,
#                             f'{split}/rmse_step': rmse,
#                             f'{split}/mae_step': mae}

#             if is_main():
#                 wandb.log({
#                     'epoch': epoch, 'step': step,
#                     f'{split}/loss': loss.item(),
#                     f'{split}/grad_norm': grad_norm,
#                     **{f'{split}/{task}_loss': l.item() for task, l in task_losses.items()},
#                     **step_log,
#                 })

#     metric_logger.synchronize_between_processes()
#     print(f'[{split.upper()}] Epoch {epoch} averaged stats:', metric_logger)

#     if is_main():
#         epoch_log = {
#             'epoch': epoch,
#             f'{split}/loss_avg': metric_logger.meters['loss'].global_avg,
#         }
#         for task in out_domains:
#             key = f'{split}_{task}_loss'
#             if key in metric_logger.meters:
#                 epoch_log[f'{split}/{task}_loss_avg'] = metric_logger.meters[key].global_avg

#         if is_regression and n_reg > 0:
#             epoch_log.update({
#                 f'{split}/r2_avg':   r2_sum   / n_reg,
#                 f'{split}/rmse_avg': rmse_sum / n_reg,
#                 f'{split}/mae_avg':  mae_sum  / n_reg,
#             })
#         wandb.log(epoch_log)

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# # ------------------------------------------------------------------ #
# # Test loop  (segmentation + regression 공통)
# # ------------------------------------------------------------------ #

# def test_one_epoch(
#     model,
#     test_loader,
#     tasks_loss_fn: dict,
#     device: torch.device,
#     epoch: int,
#     in_domains: list,
#     out_domains: list,
#     args,
#     num_classes: int = 2,
# ) -> dict:
#     is_regression = (args.task_type == 'regression')
#     model.eval()

#     metric_logger = utils.MetricLogger(delimiter='  ')
#     header = f'[TEST] Epoch [{epoch}]'

#     # segmentation accumulators
#     total_acc, total_miou, n_seg = 0.0, 0.0, 0
#     per_class_acc_sum  = np.zeros(num_classes, dtype=np.float64)
#     per_class_counts   = np.zeros(num_classes, dtype=np.int64)
#     cm_total           = np.zeros((num_classes, num_classes), dtype=np.int64)

#     # regression accumulators
#     r2_sum, rmse_sum, mae_sum, n_reg = 0.0, 0.0, 0.0, 0

#     with torch.no_grad():
#         for step, batch in enumerate(metric_logger.log_every(test_loader, 10, header)):
#             tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
#             input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}

#             with torch.cuda.amp.autocast():
#                 preds, masks = model(
#                     input_dict,
#                     mask_inputs=False,
#                     num_encoded_tokens=args.num_encoded_tokens,
#                 )
#                 task_losses = {}
#                 for task in out_domains:
#                     target = tasks_dict[task]
#                     task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
#                 loss = sum(task_losses.values())

#             metric_logger.update(loss=loss.item(), grad_norm=0.0)
#             for task, l in task_losses.items():
#                 metric_logger.update(**{f'test_{task}_loss': l.item()})

#             step_log = {}

#             # ── segmentation metrics ──────────────────────────────────
#             if not is_regression and 'cdl' in preds:
#                 target    = tasks_dict['cdl']
#                 step_acc  = pixel_accuracy(preds['cdl'], target)
#                 step_miou = mean_iou(preds['cdl'], target, num_classes)
#                 accs      = per_class_accuracy(preds['cdl'], target, num_classes)
#                 cm        = compute_confusion_matrix(preds['cdl'], target, num_classes)

#                 total_acc  += step_acc; total_miou += step_miou
#                 cm_total   += cm;       n_seg      += 1
#                 for c in range(num_classes):
#                     if not np.isnan(accs[c]):
#                         per_class_acc_sum[c] += accs[c]
#                         per_class_counts[c]  += 1

#                 step_log = {'test/acc_step': step_acc, 'test/mIoU_step': step_miou}

#             # ── regression metrics ────────────────────────────────────
#             elif is_regression and 'yield' in preds:
#                 p = preds['yield']
#                 t = tasks_dict['yield']
#                 r2   = compute_r2(p, t)
#                 rmse = compute_rmse(p, t)
#                 mae  = compute_mae(p, t)
#                 r2_sum += r2; rmse_sum += rmse; mae_sum += mae; n_reg += 1
#                 step_log = {'test/r2_step': r2,
#                             'test/rmse_step': rmse,
#                             'test/mae_step': mae}

#             if is_main():
#                 wandb.log({
#                     'epoch': epoch, 'step': step,
#                     'test/loss': loss.item(),
#                     **{f'test/{task}_loss': l.item() for task, l in task_losses.items()},
#                     **step_log,
#                 })

#     metric_logger.synchronize_between_processes()
#     print('Test stats:', metric_logger)

#     result = {
#         **{k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     }

#     if is_main():
#         epoch_log = {
#             'epoch': epoch,
#             'test/loss_avg': metric_logger.meters['loss'].global_avg,
#         }

#         # ── segmentation summary ──────────────────────────────────────
#         if not is_regression and n_seg > 0:
#             avg_acc  = total_acc  / n_seg
#             avg_miou = total_miou / n_seg
#             per_class_avg = per_class_acc_sum / np.maximum(1, per_class_counts)

#             print(f'✅ Test | Acc: {avg_acc:.4f} | mIoU: {avg_miou:.4f}')
#             for c, acc_c in enumerate(per_class_avg):
#                 print(f'   Class {c} Acc: {acc_c:.4f}')
#             print('Confusion Matrix:\n', cm_total)

#             epoch_log.update({'test/acc_avg': avg_acc, 'test/mIoU_avg': avg_miou})
#             for c, acc_c in enumerate(per_class_avg):
#                 epoch_log[f'test/acc_class{c}'] = acc_c
#             wandb.log(epoch_log)
#             wandb.log({
#                 'test/confusion_matrix': wandb.Table(
#                     data=cm_total.tolist(),
#                     columns=[f'Pred_{i}' for i in range(num_classes)],
#                 )
#             })

#             result.update({'test_acc': avg_acc, 'test_mIoU': avg_miou,
#                            'confusion_matrix': cm_total,
#                            **{f'test_acc_class{c}': per_class_avg[c]
#                               for c in range(num_classes)}})

#         # ── regression summary ────────────────────────────────────────
#         elif is_regression and n_reg > 0:
#             avg_r2   = r2_sum   / n_reg
#             avg_rmse = rmse_sum / n_reg
#             avg_mae  = mae_sum  / n_reg

#             print(f'✅ Test | R²: {avg_r2:.4f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f}')

#             epoch_log.update({'test/r2_avg': avg_r2,
#                               'test/rmse_avg': avg_rmse,
#                               'test/mae_avg': avg_mae})
#             wandb.log(epoch_log)

#             result.update({'test_r2': avg_r2,
#                            'test_rmse': avg_rmse,
#                            'test_mae': avg_mae})

#     return result
"""
refac_finetune_engine.py
Downstream finetuning train / validation / test loop.
  [segmentation]  CDL prediction
    - CrossEntropyLoss
    - metrics: pixel accuracy, mIoU, per-class accuracy, confusion matrix
  [regression]    Yield prediction
    - MSELoss
    - metrics: pixel-level R², RMSE, MAE + field-level R², RMSE, MAE
  Common:
    - mask_inputs=False (no masking, use full input)
    - wandb.log() only called from main process (rank 0)
    - gradient accumulation via args.accum_iter (default 1 = no accumulation)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from sklearn.metrics import confusion_matrix
import utils


def is_main() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


# ------------------------------------------------------------------ #
# Segmentation metric helpers
# ------------------------------------------------------------------ #
def compute_confusion_matrix(logits, target, num_classes):
    pred   = logits.argmax(dim=1).cpu().numpy().ravel()
    target = target.cpu().numpy().ravel()
    return confusion_matrix(target, pred, labels=list(range(num_classes)))


def pixel_accuracy(logits, target):
    pred    = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / max(1, target.numel())


def mean_iou(logits, target, num_classes):
    pred   = logits.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def per_class_accuracy(logits, target, num_classes):
    pred = logits.argmax(dim=1)
    accs = []
    for c in range(num_classes):
        mask = (target == c)
        n = mask.sum().item()
        accs.append(((pred == c) & mask).sum().item() / n if n > 0 else float('nan'))
    return accs


# # ------------------------------------------------------------------ #
# # Regression metric helpers
# # ------------------------------------------------------------------ #
# def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     ss_res = ((target - pred) ** 2).sum().item()
#     ss_tot = ((target - target.mean()) ** 2).sum().item()
#     return 1.0 - ss_res / max(ss_tot, 1e-8)


# def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     return float(torch.sqrt(((pred - target) ** 2).mean()).item())


# def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
#     pred   = pred.detach().cpu().float().reshape(-1)
#     target = target.detach().cpu().float().reshape(-1)
#     return float((pred - target).abs().mean().item())

def _valid_regression_pixels(pred, target):
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {pred.shape} vs {target.shape}"
        )

    pred = pred.detach().reshape(-1)
    target = target.detach().reshape(-1)

    valid = torch.isfinite(target) & (target != -1)
    pred = pred[valid].cpu().double()
    target = target[valid].cpu().double()

    if not torch.isfinite(pred).all():
        raise ValueError("Non-finite predictions at valid pixels.")

    return pred, target


def compute_r2(pred, target):
    pred, target = _valid_regression_pixels(pred, target)
    if target.numel() < 2:
        return float("nan")

    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_rmse(pred, target):
    pred, target = _valid_regression_pixels(pred, target)
    if target.numel() == 0:
        return float("nan")
    return (pred - target).square().mean().sqrt().item()


def compute_mae(pred, target):
    pred, target = _valid_regression_pixels(pred, target)
    if target.numel() == 0:
        return float("nan")
    return (pred - target).abs().mean().item()


# ------------------------------------------------------------------ #
# Train / Validation loop (segmentation + regression)
# ------------------------------------------------------------------ #
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
    is_regression = (args.task_type == 'regression')
    model.train() if split == 'train' else model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'[{split.upper()}] Epoch [{epoch}]'
    r2_sum, rmse_sum, mae_sum, n_reg = 0.0, 0.0, 0.0, 0

    accum_iter = getattr(args, 'accum_iter', 1)   # gradient accumulation steps (TerraMind: 8)
    num_steps_in_epoch = len(loader)

    ctx = torch.no_grad() if split != 'train' else torch.enable_grad()
    with ctx:
        if split == 'train':
            optimizer.zero_grad()   # accumulation 시작 전 한 번만

        for step, batch in enumerate(metric_logger.log_every(loader, 10, header)):
            tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
            input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}
            with torch.cuda.amp.autocast():
                preds, masks = model(
                    input_dict,
                    mask_inputs=False,
                    num_encoded_tokens=args.num_encoded_tokens,
                )
                task_losses = {}
                for task in out_domains:
                    target = tasks_dict[task]
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                loss = sum(task_losses.values())

            loss_value = loss.item()   # 로깅용 (accumulation으로 나누기 전 원래 loss)

            if split == 'train':
                loss = loss / accum_iter
                update_grad = ((step + 1) % accum_iter == 0) or ((step + 1) == num_steps_in_epoch)

                grad_norm = loss_scaler(
                    loss, optimizer,
                    parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=update_grad,
                )
                if update_grad:
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                grad_norm = grad_norm if grad_norm is not None else 0.0
            else:
                grad_norm = 0.0

            metric_logger.update(loss=loss_value, grad_norm=grad_norm)
            for task, l in task_losses.items():
                metric_logger.update(**{f'{split}_{task}_loss': l.item()})

            step_log = {}
            if is_regression and 'yield' in preds:
                p = preds['yield']
                t = tasks_dict['yield']
                r2   = compute_r2(p, t)
                rmse = compute_rmse(p, t)
                mae  = compute_mae(p, t)
                r2_sum += r2; rmse_sum += rmse; mae_sum += mae; n_reg += 1
                step_log = {f'{split}/r2_step': r2,
                            f'{split}/rmse_step': rmse,
                            f'{split}/mae_step': mae}

            if is_main():
                wandb.log({
                    'epoch': epoch, 'step': step,
                    f'{split}/loss': loss_value,
                    f'{split}/grad_norm': grad_norm,
                    **{f'{split}/{task}_loss': l.item() for task, l in task_losses.items()},
                    **step_log,
                })

    metric_logger.synchronize_between_processes()
    print(f'[{split.upper()}] Epoch {epoch} averaged stats:', metric_logger)
    if is_main():
        epoch_log = {
            'epoch': epoch,
            f'{split}/loss_avg': metric_logger.meters['loss'].global_avg,
        }
        for task in out_domains:
            key = f'{split}_{task}_loss'
            if key in metric_logger.meters:
                epoch_log[f'{split}/{task}_loss_avg'] = metric_logger.meters[key].global_avg
        if is_regression and n_reg > 0:
            epoch_log.update({
                f'{split}/r2_avg':   r2_sum   / n_reg,
                f'{split}/rmse_avg': rmse_sum / n_reg,
                f'{split}/mae_avg':  mae_sum  / n_reg,
            })
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
    num_classes: int = 2,
) -> dict:
    # Corn regression uses the TerraMind filename-mean evaluation protocol.
    # CDL follows the original test implementation below.
    # if args.task_type == 'regression':
    #     return _test_yield_terramind(
    #         model, test_loader, tasks_loss_fn, device, epoch,
    #         in_domains, out_domains, args, num_classes=num_classes,
    #     )
    
    # train_one_epoch, test 관련 forward 호출 부분에서
    if args.task_type == 'regression':
        preds = model(input_dict, return_all_layers=True)   # MultiViT: dict만 반환
    else:
        preds, masks = model(
            input_dict, mask_inputs=False,
            num_encoded_tokens=args.num_encoded_tokens,
        )   # MultiMAE: 기존 그대로

    is_regression = (args.task_type == 'regression')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'[TEST] Epoch [{epoch}]'

    # segmentation accumulators
    total_acc, total_miou, n_seg = 0.0, 0.0, 0
    per_class_acc_sum  = np.zeros(num_classes, dtype=np.float64)
    per_class_counts   = np.zeros(num_classes, dtype=np.int64)
    cm_total           = np.zeros((num_classes, num_classes), dtype=np.int64)

    # regression accumulators
    r2_sum, rmse_sum, mae_sum, n_reg = 0.0, 0.0, 0.0, 0
    field_preds_list   = []   # field-level predictions (patch mean)
    field_targets_list = []   # field-level targets (patch mean)

    with torch.no_grad():
        for step, batch in enumerate(metric_logger.log_every(test_loader, 10, header)):
            tasks_dict = {t: ten.to(device, non_blocking=True) for t, ten in batch.items()}
            input_dict = {t: tasks_dict[t] for t in in_domains if t in tasks_dict}
            with torch.cuda.amp.autocast():
                preds, masks = model(
                    input_dict,
                    mask_inputs=False,
                    num_encoded_tokens=args.num_encoded_tokens,
                )
                task_losses = {}
                for task in out_domains:
                    target = tasks_dict[task]
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                loss = sum(task_losses.values())
            metric_logger.update(loss=loss.item(), grad_norm=0.0)
            for task, l in task_losses.items():
                metric_logger.update(**{f'test_{task}_loss': l.item()})
            step_log = {}

            # ── segmentation metrics ──────────────────────────────────
            if not is_regression and 'cdl' in preds:
                target    = tasks_dict['cdl']
                step_acc  = pixel_accuracy(preds['cdl'], target)
                step_miou = mean_iou(preds['cdl'], target, num_classes)
                accs      = per_class_accuracy(preds['cdl'], target, num_classes)
                cm        = compute_confusion_matrix(preds['cdl'], target, num_classes)
                total_acc  += step_acc; total_miou += step_miou
                cm_total   += cm;       n_seg      += 1
                for c in range(num_classes):
                    if not np.isnan(accs[c]):
                        per_class_acc_sum[c] += accs[c]
                        per_class_counts[c]  += 1
                step_log = {'test/acc_step': step_acc, 'test/mIoU_step': step_miou}

            # ── regression metrics ────────────────────────────────────
            elif is_regression and 'yield' in preds:
                p = preds['yield']
                t = tasks_dict['yield']
                r2   = compute_r2(p, t)
                rmse = compute_rmse(p, t)
                mae  = compute_mae(p, t)
                r2_sum += r2; rmse_sum += rmse; mae_sum += mae; n_reg += 1
                # field-level: mean over all pixels per patch
                field_preds_list.append(p.mean(dim=[1, 2, 3]).cpu())
                field_targets_list.append(t.mean(dim=[1, 2, 3]).cpu())
                step_log = {'test/r2_step': r2,
                            'test/rmse_step': rmse,
                            'test/mae_step': mae}

            if is_main():
                wandb.log({
                    'epoch': epoch, 'step': step,
                    'test/loss': loss.item(),
                    **{f'test/{task}_loss': l.item() for task, l in task_losses.items()},
                    **step_log,
                })

    metric_logger.synchronize_between_processes()
    print('Test stats:', metric_logger)
    result = {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()}
    }

    if is_main():
        epoch_log = {
            'epoch': epoch,
            'test/loss_avg': metric_logger.meters['loss'].global_avg,
        }

        # ── segmentation summary ──────────────────────────────────────
        if not is_regression and n_seg > 0:
            avg_acc  = total_acc  / n_seg
            avg_miou = total_miou / n_seg
            per_class_avg = per_class_acc_sum / np.maximum(1, per_class_counts)
            print(f'✅ Test | Acc: {avg_acc:.4f} | mIoU: {avg_miou:.4f}')
            for c, acc_c in enumerate(per_class_avg):
                print(f'   Class {c} Acc: {acc_c:.4f}')
            print('Confusion Matrix:\n', cm_total)
            epoch_log.update({'test/acc_avg': avg_acc, 'test/mIoU_avg': avg_miou})
            for c, acc_c in enumerate(per_class_avg):
                epoch_log[f'test/acc_class{c}'] = acc_c
            wandb.log(epoch_log)
            wandb.log({
                'test/confusion_matrix': wandb.Table(
                    data=cm_total.tolist(),
                    columns=[f'Pred_{i}' for i in range(num_classes)],
                )
            })
            result.update({'test_acc': avg_acc, 'test_mIoU': avg_miou,
                           'confusion_matrix': cm_total,
                           **{f'test_acc_class{c}': per_class_avg[c]
                              for c in range(num_classes)}})

        # ── regression summary ────────────────────────────────────────
        elif is_regression and n_reg > 0:
            # pixel-level metrics
            avg_r2   = r2_sum   / n_reg
            avg_rmse = rmse_sum / n_reg
            avg_mae  = mae_sum  / n_reg

            # field-level metrics (patch mean aggregation)
            all_preds   = torch.cat(field_preds_list)    # (N_fields,)
            all_targets = torch.cat(field_targets_list)  # (N_fields,)
            field_r2   = compute_r2(all_preds, all_targets)
            field_rmse = compute_rmse(all_preds, all_targets)
            field_mae  = compute_mae(all_preds, all_targets)

            print(f'✅ Test | Pixel R²: {avg_r2:.4f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f}')
            print(f'✅ Test | Field R²: {field_r2:.4f} | Field RMSE: {field_rmse:.4f} | Field MAE: {field_mae:.4f}')

            epoch_log.update({
                'test/r2_avg':      avg_r2,
                'test/rmse_avg':    avg_rmse,
                'test/mae_avg':     avg_mae,
                'test/field_r2':    field_r2,
                'test/field_rmse':  field_rmse,
                'test/field_mae':   field_mae,
            })
            wandb.log(epoch_log)
            result.update({
                'test_r2':       avg_r2,
                'test_rmse':     avg_rmse,
                'test_mae':      avg_mae,
                'test_field_r2':   field_r2,
                'test_field_rmse': field_rmse,
                'test_field_mae':  field_mae,
            })

    return result

# TerraMind-style corn test evaluator (single-process sequential loader).
# target > 0; basename means; inverse transform x*320+50.
# Field means filename groups; polygon identity is not inferred.
# Legacy train/validation metrics above are intentionally unchanged.
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _yield_reference_metrics(n, sum_t, sum_t2, squared_error, absolute_error):
    if n == 0:
        return {'r2': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
    sst = max(0.0, sum_t2 - sum_t * sum_t / n)
    # Match sklearn's finite constant-target convention; n<2 is undefined.
    r2 = (float('nan') if n < 2 else
          (1.0 - squared_error / sst if sst > 0 else
           (1.0 if squared_error == 0 else 0.0)))
    return {'r2': r2, 'rmse': math.sqrt(squared_error / n),
            'mae': absolute_error / n}


def _yield_reference_finish(groups, pixel_stats):
    rows = []
    for name, (count, target_sum, prediction_sum) in sorted(groups.items()):
        tn, pn = target_sum / count, prediction_sum / count
        rows.append({'Filename': name, 'YieldGT': tn * 320 + 50,
                     'Prediction': pn * 320 + 50, 'ValidPixels': count,
                     'YieldGT_normalized': tn, 'Prediction_normalized': pn})
    t = [row['YieldGT_normalized'] for row in rows]
    p = [row['Prediction_normalized'] for row in rows]
    field = _yield_reference_metrics(len(t), math.fsum(t), math.fsum(x*x for x in t),
                     math.fsum((a-b)**2 for a, b in zip(t, p)),
                     math.fsum(abs(a-b) for a, b in zip(t, p)))
    pixel = _yield_reference_metrics(*pixel_stats)
    result = {}
    for prefix, metrics in [('test', pixel), ('test_field', field)]:
        for key, value in metrics.items():
            result[f'{prefix}_{key}'] = value
        result[f'{prefix}_rmse_bu_acre'] = metrics['rmse'] * 320
        result[f'{prefix}_mae_bu_acre'] = metrics['mae'] * 320
    result['test_n_valid_pixels'] = pixel_stats[0]
    result['test_n_filename_groups'] = len(rows)
    return rows, result


def _test_yield_terramind(model, test_loader, tasks_loss_fn, device, epoch,
                   in_domains, out_domains, args, num_classes=1):
    import numpy as np
    import torch
    from contextlib import nullcontext
    from torch.utils.data import SequentialSampler, BatchSampler

    if args.task_type != 'regression' or list(out_domains) != ['yield']:
        raise ValueError('This evaluator supports yield-only regression.')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise ValueError('Use the supplied single-process test_yield.py, not DDP.')
    if (type(test_loader.sampler) is not SequentialSampler or test_loader.drop_last
            or type(test_loader.batch_sampler) is not BatchSampler
            or test_loader.batch_sampler.sampler is not test_loader.sampler
            or not getattr(test_loader, 'in_order', True)):
        raise ValueError('Filename matching requires an ordinary ordered DataLoader: shuffle=False, drop_last=False.')
    samples = getattr(test_loader.dataset, 'samples', None)
    if samples is None or len(samples) != len(test_loader.dataset):
        raise ValueError('Expected the supplied YieldDataset with .samples.')
    names = [Path(str(s)).stem for s in samples]
    if any('corn' not in name.lower() or 'soybean' in name.lower() for name in names):
        raise ValueError('Corn-only evaluator: filenames must contain corn, as in the reference protocol.')
    if not names:
        raise ValueError('Empty test dataset.')

    model.eval()
    groups = defaultdict(lambda: [0, 0.0, 0.0])
    pixel_stats = [0, 0.0, 0.0, 0.0, 0.0]
    offset, skipped = 0, 0
    loss_sum, loss_samples = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            tensors = {k: v.to(device, non_blocking=True)
                       for k, v in batch.items() if torch.is_tensor(v)}
            # Require all configured inputs; do not silently drop a modality.
            inputs = {k: tensors[k] for k in in_domains}
            amp = torch.cuda.amp.autocast() if torch.device(device).type == 'cuda' else nullcontext()
            with amp:
                preds, _ = model(inputs, mask_inputs=False,
                                 num_encoded_tokens=args.num_encoded_tokens)
                pred, target = preds['yield'].float(), tensors['yield']
                if pred.ndim == 3:
                    pred = pred.unsqueeze(1)
                if target.ndim == 3:
                    target = target.unsqueeze(1)
                if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 1:
                    raise ValueError(f'Expected matching [B,1,H,W], got {pred.shape}, {target.shape}.')
                loss = tasks_loss_fn['yield'](pred, target)
            size = target.shape[0]
            loss_sum += float(loss.item()) * size
            loss_samples += size
            if offset + size > len(names):
                raise ValueError('Batch order/size does not match dataset.samples.')
            p = pred.detach().cpu().numpy().astype(np.float64)
            t = target.detach().cpu().numpy().astype(np.float64)
            for i, name in enumerate(names[offset:offset + size]):
                # EXACT reference selection, with explicit error on invalid retained values.
                valid = t[i] > 0
                tv, pv = t[i][valid], p[i][valid]
                if not np.isfinite(tv).all() or not np.isfinite(pv).all():
                    raise ValueError(f'Non-finite retained target/prediction: {name}')
                n = tv.size
                if n == 0:
                    skipped += 1
                    continue
                st, sp = float(tv.sum()), float(pv.sum())
                group = groups[name]
                group[0] += n
                group[1] += st
                group[2] += sp
                err = pv - tv
                values = [n, st, float(np.square(tv).sum()),
                          float(np.square(err).sum()), float(np.abs(err).sum())]
                for j, value in enumerate(values):
                    pixel_stats[j] += value
            offset += size
            if step % 10 == 0:
                print(f'[TEST] {offset}/{len(names)} samples; {pixel_stats[0]} retained pixels', flush=True)
    if offset != len(names) or not groups:
        raise ValueError('Incomplete test coverage or no target > 0 pixels.')
    rows, result = _yield_reference_finish(groups, pixel_stats)
    result['test_n_samples'] = offset
    result['test_n_empty_samples'] = skipped
    # Original loss is retained as a diagnostic, not relabeled as filtered MSE.
    result['test_yield_loss_original'] = loss_sum / loss_samples
    destination = Path(args.output_dir) / 'terramind_style_eval'
    destination.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    csv_path = destination / f'Corn_field_predictions_{tag}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / f'Corn_metrics_{tag}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerows(result.items())
    print('Field aggregation: target > 0; filename means; corn x*320+50.')
    print(f"Field R2: {result['test_field_r2']:.6f}; "
          f"RMSE: {result['test_field_rmse_bu_acre']:.4f} bu/acre; "
          f"MAE: {result['test_field_mae_bu_acre']:.4f} bu/acre")
    print(f'Predictions: {csv_path}')
    try:
        import wandb
    except ImportError:
        wandb = None
    if wandb is not None and wandb.run is not None:
        wandb.log({f'test_terramind_style/{k}': v for k, v in result.items()})
    return result
