from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
try:
    from hausdorff import hausdorff_distance
except ImportError:
    hausdorff_distance = None
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.generate_prompts import get_click_prompt
from utils.task_registry import get_primary_task, get_task_spec, task_uses_video_path


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _segmentation_stats(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())

    dice = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_divide(tp, tp + fp + fn)
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    stats = {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
    }
    if hausdorff_distance is not None:
        if pred.sum() == 0 and gt.sum() == 0:
            stats["hd"] = 0.0
        else:
            try:
                stats["hd"] = float(hausdorff_distance(pred, gt, distance="manhattan"))
            except Exception:
                stats["hd"] = float(max(pred.shape))
    return stats


def _classification_stats(probs: List[float], labels: List[int]) -> Dict[str, float]:
    if not probs:
        return {
            "accuracy": 0.0,
            "auc": 0.0,
            "f1": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
    probs_np = np.asarray(probs, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)
    preds_np = (probs_np >= 0.5).astype(np.int32)

    tp = float(np.logical_and(preds_np == 1, labels_np == 1).sum())
    fp = float(np.logical_and(preds_np == 1, labels_np == 0).sum())
    tn = float(np.logical_and(preds_np == 0, labels_np == 0).sum())
    fn = float(np.logical_and(preds_np == 0, labels_np == 1).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    if len(np.unique(labels_np)) == 2:
        auc = float(roc_auc_score(labels_np, probs_np))
        auprc = float(average_precision_score(labels_np, probs_np))
    else:
        auc = auprc = None
    return {
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
        "auc": auc,
        "auprc": auprc,
        "brier": float(np.mean((probs_np - labels_np) ** 2)),
        "high_confidence_errors": int(np.sum(((probs_np <= .1) | (probs_np >= .9)) & (preds_np != labels_np))),
        "num_positive": int(labels_np.sum()),
        "num_predictions": len(labels_np),
        "f1": f1,
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
    }


def _forward_batch(datapack, model, criterion, opt):
    imgs = datapack["image"].to(dtype=torch.float32, device=opt.device)
    masks = datapack["mask"].to(dtype=torch.float32, device=opt.device)
    low_masks = datapack["low_mask"].to(dtype=torch.float32, device=opt.device)
    labels = datapack["label"].to(dtype=torch.float32, device=opt.device)
    prompts = {
        "modality_prompt": datapack["modality_prompt"].to(device=opt.device),
        "mode_prompt": datapack["mode_prompt"].to(device=opt.device),
        "type_prompt": datapack["type_prompt"].to(device=opt.device),
        "position_prompt": datapack["position_prompt"].to(device=opt.device),
        "task_prompt": datapack["task_prompt"].to(device=opt.device),
    }
    task_name = get_primary_task(datapack["task"])
    spec = get_task_spec(task_name)
    dataset_names = datapack.get("dataset", [])
    use_roi = torch.tensor([name in ["BUS-BRA", "TN3K"] for name in dataset_names], device=opt.device, dtype=torch.bool)

    model_kwargs = {
        "pt": get_click_prompt(datapack, opt) if "pt" in datapack else None,
        "prompts": prompts,
        "use_roi": use_roi,
    }
    if spec.is_video:
        time_steps = imgs.shape[1] if imgs.ndim == 5 else imgs.shape[0]
        mask_valid = torch.ones((imgs.shape[0], time_steps), dtype=torch.bool, device=opt.device)
        model_kwargs["mask_valid"] = mask_valid
        if "frame_valid" in datapack:
            model_kwargs["frame_valid"] = datapack["frame_valid"].to(device=opt.device)
            model_kwargs["frame_time"] = datapack["frame_time"].to(device=opt.device)
            # Only explicit clicks can reserve a conditioned memory slot.
            model_kwargs["mask_valid"] = torch.zeros_like(model_kwargs["frame_valid"], dtype=torch.bool)
            model_kwargs["use_roi"] = torch.ones(imgs.shape[0], device=opt.device, dtype=torch.bool)

    with torch.no_grad():
        pred = model(imgs, **model_kwargs)
        pred["task"] = [task_name] * imgs.shape[0]
        pred["cls_target"] = labels
        pred["enable_prototype_loss"] = True
        pred["enable_auxiliary_losses"] = False
        target_tensor = low_masks if spec.target_key == "low_mask" else masks
        loss = criterion(pred, target_tensor)
    return pred, loss.item(), labels, masks


def get_eval(valloader, model, criterion, opt, args, predictions=None):
    model.eval()
    task_name = None
    dataset_names = set()
    losses: List[float] = []
    seg_stats: List[Dict[str, float]] = []
    cls_probs: List[float] = []
    cls_labels: List[int] = []

    for datapack in valloader:
        task_name = get_primary_task(datapack["task"])
        spec = get_task_spec(task_name)
        if "dataset" in datapack:
            if isinstance(datapack["dataset"], list):
                dataset_names.update(datapack["dataset"])
            else:
                dataset_names.add(str(datapack["dataset"]))

        pred, loss_value, labels, masks = _forward_batch(datapack, model, criterion, opt)
        losses.append(loss_value)

        if spec.is_segmentation:
            probs = torch.sigmoid(pred["masks"]).detach().cpu().numpy()
            gt = masks.detach().cpu().numpy()
            if probs.ndim == 4:
                probs = probs[:, None, ...]
                gt = gt[:, None, ...]
            if probs.ndim == 5 and probs.shape[2] == 1:
                probs = probs[:, :, 0]
            if gt.ndim == 5 and gt.shape[2] == 1:
                gt = gt[:, :, 0]
            for batch_index in range(probs.shape[0]):
                for frame_index in range(probs.shape[1]):
                    seg_stats.append(_segmentation_stats(probs[batch_index, frame_index] > 0.5, gt[batch_index, frame_index]))

        if spec.is_classification and "cls_logits" in pred:
            logits = pred["cls_logits"]
            if logits.dim() == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            cls_probs.extend(float(x) for x in probs)
            batch_labels = labels.detach().cpu().numpy().reshape(-1)
            cls_labels.extend(int(x) for x in batch_labels)
            if predictions is not None:
                for i, (probability, label) in enumerate(zip(probs, batch_labels)):
                    predictions.append({
                        "task": task_name,
                        "dataset": datapack["dataset"][i],
                        "case_id": datapack["case_name"][i],
                        "patient_id": datapack.get("patient_id", [None] * len(probs))[i],
                        "label": int(label),
                        "probability": float(probability),
                    })

    summary = {
        "task": task_name,
        "datasets": sorted(dataset_names),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "num_batches": len(losses),
    }
    if task_name is None:
        return summary

    spec = get_task_spec(task_name)
    if spec.is_segmentation:
        keys = ("dice", "iou", "sensitivity", "specificity", "accuracy")
        if hausdorff_distance is not None:
            keys += ("hd",)
        for key in keys:
            values = [stat[key] for stat in seg_stats] if seg_stats else [0.0]
            summary[key] = float(np.mean(values))
    if spec.is_classification:
        summary.update(_classification_stats(cls_probs, cls_labels))
    return summary
