"""
Train a frame-level phase classifier with leave-one-video-out cross-validation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class PhaseFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None, temporal_window: int = 1):
        self.df = df.sort_values(["video_number", "timestamp"]).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.temporal_window = temporal_window
        self.half_window = temporal_window // 2
        self.window_indices = self._build_window_indices() if temporal_window > 1 else None

    def _build_window_indices(self):
        window_indices = [None] * len(self.df)
        for _, group in self.df.groupby("video_number", sort=False):
            idxs = group.index.to_list()
            last = len(idxs) - 1
            for pos, idx in enumerate(idxs):
                win = []
                for offset in range(-self.half_window, self.half_window + 1):
                    pos2 = min(max(pos + offset, 0), last)
                    win.append(idxs[pos2])
                window_indices[idx] = win
        return window_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.temporal_window <= 1:
            img_path = self.root_dir / row["image_path"]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            images = []
            for win_idx in self.window_indices[idx]:
                win_row = self.df.iloc[win_idx]
                img_path = self.root_dir / win_row["image_path"]
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            image = torch.stack(images, dim=0)  # (T, C, H, W)

        sample = {
            "image": image,
            "label": int(row["label_id"]),
            "video_number": int(row["video_number"]),
            "timestamp": float(row["timestamp"]),
            "image_path": str(row["image_path"]),
        }
        return sample


class TemporalPhaseClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        temporal_window: int = 1,
        temporal_fusion: str = "avg",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.temporal_fusion = temporal_fusion

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone_type = "resnet"
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone_type = "resnet"
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone_type = "resnet"
        elif model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone_type = "efficientnet"
        elif model_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b1(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone_type = "efficientnet"
        elif model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b2(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone_type = "efficientnet"
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if backbone_type == "resnet":
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.pool = nn.Identity()
        else:
            self.backbone = backbone.features
            self.pool = backbone.avgpool

        if temporal_fusion not in {"avg", "conv1d"}:
            raise ValueError("temporal_fusion must be 'avg' or 'conv1d'")

        self.temporal_conv = None
        if temporal_fusion == "conv1d":
            self.temporal_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feats = self.backbone(x)
        feats = self.pool(feats)
        feats = feats.view(feats.size(0), -1)
        feats = feats.view(b, t, -1)

        if self.temporal_fusion == "avg":
            fused = feats.mean(dim=1)
        else:
            feats_t = feats.transpose(1, 2)  # (B, D, T)
            feats_t = F.relu(self.temporal_conv(feats_t))
            fused = feats_t.mean(dim=2)

        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def prob_col(label_name: str) -> str:
    safe = re.sub(r"\s+", "_", label_name.strip())
    safe = re.sub(r"[^A-Za-z0-9_]", "", safe)
    return f"prob_{safe}"


def smooth_predictions(df_preds: pd.DataFrame, window: int, pred_col: str = "pred") -> pd.Series:
    if window <= 1:
        return df_preds[pred_col].copy()

    half = window // 2
    smoothed = []

    for vid, group in df_preds.sort_values("timestamp").groupby("video_number"):
        preds = group[pred_col].tolist()
        for i in range(len(preds)):
            start = max(0, i - half)
            end = min(len(preds), i + half + 1)
            window_preds = preds[start:end]
            smoothed.append(Counter(window_preds).most_common(1)[0][0])

    return pd.Series(smoothed, index=df_preds.sort_values(["video_number", "timestamp"]).index)


def evaluate_predictions(y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1_macro, report, cm


def drop_boundary_frames(df: pd.DataFrame, boundary_sec: float) -> tuple[pd.DataFrame, int]:
    if boundary_sec <= 0:
        return df, 0

    kept = []
    removed = 0
    for _, group in df.groupby("video_number"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        times = g["timestamp"].values
        labels = g["label_id"].values
        change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
        if len(change_idx) == 0:
            kept.append(g)
            continue

        change_times = (times[change_idx - 1] + times[change_idx]) / 2.0
        mask = np.ones(len(g), dtype=bool)
        for ct in change_times:
            mask &= np.abs(times - ct) > boundary_sec

        removed += int((~mask).sum())
        kept.append(g[mask])

    filtered = pd.concat(kept, ignore_index=True)
    return filtered, removed


def compute_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["label_id"].value_counts().to_dict()
    weights = df["label_id"].map(lambda x: 1.0 / counts.get(int(x), 1)).values
    return torch.tensor(weights, dtype=torch.float32)


def infer_phase_order(train_df: pd.DataFrame, num_classes: int) -> list[int]:
    med = train_df.groupby("label_id")["timestamp"].median()
    order = med.sort_values().index.tolist()
    missing = [i for i in range(num_classes) if i not in order]
    return order + missing


def viterbi_decode(
    log_probs: np.ndarray,
    order: list[int],
    stay_prob: float = 0.8,
    advance_prob: float = 0.2,
    allow_skip: bool = False,
    skip_prob: float = 0.05,
) -> list[int]:
    if log_probs.size == 0:
        return []

    order = list(order)
    k = len(order)
    logp = log_probs[:, order]

    total = stay_prob + advance_prob + (skip_prob if allow_skip else 0.0)
    if total <= 0:
        stay_prob, advance_prob, skip_prob = 0.5, 0.5, 0.0
        total = 1.0
    stay_prob /= total
    advance_prob /= total
    if allow_skip:
        skip_prob /= total

    trans = np.full((k, k), -1e9, dtype=np.float32)
    trans[np.arange(k), np.arange(k)] = math.log(max(stay_prob, 1e-8))
    trans[np.arange(k - 1), np.arange(1, k)] = math.log(max(advance_prob, 1e-8))
    if allow_skip and k > 2:
        trans[np.arange(k - 2), np.arange(2, k)] = math.log(max(skip_prob, 1e-8))

    t_len = logp.shape[0]
    dp = np.full((t_len, k), -1e9, dtype=np.float32)
    back = np.zeros((t_len, k), dtype=np.int64)
    dp[0] = logp[0]

    for t in range(1, t_len):
        prev = dp[t - 1][:, None] + trans
        back[t] = prev.argmax(axis=0)
        dp[t] = prev.max(axis=0) + logp[t]

    best = int(dp[-1].argmax())
    path = [best]
    for t in range(t_len - 1, 0, -1):
        best = int(back[t, best])
        path.append(best)
    path = path[::-1]
    return [order[i] for i in path]


def plot_confusion_matrix(cm, labels, output_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list,
    dataset_root: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: int,
    device: str,
    smooth_window: int,
    early_patience: int,
    lr_patience: int,
    lr_factor: float,
    temporal_window: int,
    temporal_fusion: str,
    balance_sampler: bool,
    focal_loss: bool,
    focal_gamma: float,
    use_class_weights: bool,
    dropout: float,
    drop_boundary_sec: float,
    order_decode: bool,
    order_allow_skip: bool,
    order_stay_prob: float,
    order_advance_prob: float,
    order_skip_prob: float,
):
    train_tf, eval_tf = get_transforms(image_size=image_size)

    if drop_boundary_sec > 0:
        train_df, dropped_train = drop_boundary_frames(train_df, drop_boundary_sec)
        val_df, dropped_val = drop_boundary_frames(val_df, drop_boundary_sec)
        if dropped_train + dropped_val > 0:
            print(f"Dropped {dropped_train} train + {dropped_val} val frames near boundaries")

    train_ds = PhaseFrameDataset(train_df, dataset_root, transform=train_tf, temporal_window=temporal_window)
    val_ds = PhaseFrameDataset(val_df, dataset_root, transform=eval_tf, temporal_window=temporal_window)
    test_ds = PhaseFrameDataset(test_df, dataset_root, transform=eval_tf, temporal_window=temporal_window)

    train_sampler = None
    if balance_sampler:
        sample_weights = compute_sample_weights(train_df)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(label_names)
    model = TemporalPhaseClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        temporal_window=temporal_window,
        temporal_fusion=temporal_fusion,
        dropout=dropout,
    ).to(device)

    class_weights = compute_class_weights(train_df["label_id"].values, num_classes).to(device)
    if focal_loss:
        alpha = class_weights if use_class_weights else None
        if alpha is not None:
            alpha = alpha / alpha.mean()
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights if use_class_weights else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_kwargs = {
        "optimizer": optimizer,
        "mode": "max",
        "factor": lr_factor,
        "patience": lr_patience,
        "min_lr": 1e-6,
    }
    if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs["verbose"] = False
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**scheduler_kwargs)

    best_val = -1.0
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step(val_f1)

        if val_f1 > best_val + 1e-4:
            best_val = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_macro_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if no_improve >= early_patience:
            break

    # Load best for test
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    prob_columns = [prob_col(name) for name in label_names]
    test_rows = []
    video_buffers = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            log_probs = torch.log_softmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(preds)):
                vid = int(batch["video_number"][i])
                ts = float(batch["timestamp"][i])
                true_label = int(labels[i].cpu().item())
                video_buffers.setdefault(vid, []).append((ts, true_label, log_probs[i]))
                row = {
                    "image_path": batch["image_path"][i],
                    "video_number": vid,
                    "timestamp": ts,
                    "true": true_label,
                    "pred": int(preds[i].cpu().item()),
                }
                for j, col in enumerate(prob_columns):
                    row[col] = float(probs[i, j])
                test_rows.append(row)

    pred_df = pd.DataFrame(test_rows)

    # Raw metrics
    acc, f1_macro, report, cm = evaluate_predictions(
        pred_df["true"], pred_df["pred"], label_names
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "classification_report": report,
        "probability_columns": {
            name: col for name, col in zip(label_names, prob_columns)
        },
        "temporal": {
            "window": temporal_window,
            "fusion": temporal_fusion,
            "dropout": dropout,
        },
        "loss": {
            "type": "focal" if focal_loss else "cross_entropy",
            "focal_gamma": focal_gamma if focal_loss else None,
            "use_class_weights": use_class_weights,
        },
        "sampling": {
            "balance_sampler": balance_sampler,
            "drop_boundary_sec": drop_boundary_sec,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "epochs_ran": len(history),
        "early_stopping_patience": early_patience,
        "lr_scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": lr_factor,
            "patience": lr_patience,
        },
    }

    plot_confusion_matrix(cm, label_names, output_dir / "confusion_matrix.png", "Confusion Matrix (Raw)")

    # Smoothed metrics (optional)
    if smooth_window and smooth_window > 1:
        sorted_df = pred_df.sort_values(["video_number", "timestamp"]).copy()
        sorted_df["pred_smooth"] = smooth_predictions(sorted_df, smooth_window, pred_col="pred")
        acc_s, f1_s, report_s, cm_s = evaluate_predictions(
            sorted_df["true"], sorted_df["pred_smooth"], label_names
        )
        metrics["smoothed"] = {
            "window": smooth_window,
            "accuracy": acc_s,
            "macro_f1": f1_s,
            "classification_report": report_s,
        }
        plot_confusion_matrix(cm_s, label_names, output_dir / "confusion_matrix_smoothed.png",
                              f"Confusion Matrix (Smooth window={smooth_window})")

    # Order-constrained decoding (optional)
    if order_decode:
        order = infer_phase_order(train_df, num_classes=num_classes)
        sorted_df = pred_df.sort_values(["video_number", "timestamp"]).copy()
        order_preds = []
        for vid, _ in sorted_df.groupby("video_number"):
            items = sorted(video_buffers.get(vid, []), key=lambda x: x[0])
            logp = np.stack([it[2] for it in items], axis=0)
            vpreds = viterbi_decode(
                logp,
                order=order,
                stay_prob=order_stay_prob,
                advance_prob=order_advance_prob,
                allow_skip=order_allow_skip,
                skip_prob=order_skip_prob,
            )
            order_preds.extend(vpreds)

        sorted_df["pred_order"] = order_preds
        acc_o, f1_o, report_o, cm_o = evaluate_predictions(
            sorted_df["true"], sorted_df["pred_order"], label_names
        )
        metrics["order_decode"] = {
            "order": order,
            "accuracy": acc_o,
            "macro_f1": f1_o,
            "classification_report": report_o,
            "stay_prob": order_stay_prob,
            "advance_prob": order_advance_prob,
            "allow_skip": order_allow_skip,
            "skip_prob": order_skip_prob if order_allow_skip else 0.0,
        }
        plot_confusion_matrix(cm_o, label_names, output_dir / "confusion_matrix_order.png",
                              "Confusion Matrix (Order-constrained)")

        if smooth_window and smooth_window > 1:
            sorted_df["pred_order_smooth"] = smooth_predictions(
                sorted_df, smooth_window, pred_col="pred_order"
            )
            acc_os, f1_os, report_os, cm_os = evaluate_predictions(
                sorted_df["true"], sorted_df["pred_order_smooth"], label_names
            )
            metrics["order_decode_smoothed"] = {
                "window": smooth_window,
                "accuracy": acc_os,
                "macro_f1": f1_os,
                "classification_report": report_os,
            }
            plot_confusion_matrix(
                cm_os,
                label_names,
                output_dir / "confusion_matrix_order_smoothed.png",
                f"Confusion Matrix (Order+Smooth window={smooth_window})",
            )

        merge_cols = ["image_path", "pred_order"]
        if smooth_window and smooth_window > 1:
            merge_cols.append("pred_order_smooth")
        pred_df = pred_df.merge(
            sorted_df[merge_cols],
            on="image_path",
            how="left",
        )

    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train phase classifier (leave-one-video-out CV)")
    parser.add_argument("--dataset", type=str, default="data/phase_dataset_v3",
                        help="Dataset root with labels.csv and images/")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50",
                                 "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--lr-patience", type=int, default=2, help="LR scheduler patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR scheduler decay factor")
    parser.add_argument("--temporal-window", type=int, default=5, help="Odd number of frames per sample")
    parser.add_argument("--temporal-fusion", type=str, default="conv1d", choices=["avg", "conv1d"])
    parser.add_argument("--balance-sampler", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--no-balance-sampler", dest="balance_sampler", action="store_false",
                        help="Disable class-balanced sampler")
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss")
    parser.add_argument("--no-focal-loss", dest="focal_loss", action="store_false",
                        help="Disable focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted loss")
    parser.add_argument("--dropout", type=float, default=0.0, help="Classifier dropout")
    parser.add_argument("--drop-boundary-sec", type=float, default=1.0,
                        help="Drop frames within this many seconds of phase transitions (train/val only)")
    parser.add_argument("--order-decode", action="store_true", help="Apply order-constrained decoding")
    parser.add_argument("--no-order-decode", dest="order_decode", action="store_false",
                        help="Disable order-constrained decoding")
    parser.add_argument("--order-allow-skip", action="store_true", help="Allow skipping a phase in decoding")
    parser.add_argument("--order-stay-prob", type=float, default=0.8)
    parser.add_argument("--order-advance-prob", type=float, default=0.2)
    parser.add_argument("--order-skip-prob", type=float, default=0.05)
    parser.set_defaults(balance_sampler=True, focal_loss=True, order_decode=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    labels_path = dataset_root / "labels.csv"
    classes_path = dataset_root / "classes.json"

    df = pd.read_csv(labels_path)
    with open(classes_path, "r") as f:
        label_map = json.load(f)

    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    video_ids = sorted(df["video_number"].unique())

    if args.temporal_window % 2 == 0:
        raise ValueError("--temporal-window must be an odd number")

    out_root = Path("outputs") / f"phase_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for i, test_vid in enumerate(video_ids):
        remaining = [v for v in video_ids if v != test_vid]
        if len(remaining) < 2:
            raise ValueError("Need at least 2 training videos to hold out a validation video.")

        val_vid = remaining[i % len(remaining)]
        train_vids = [v for v in remaining if v != val_vid]

        train_df = df[df["video_number"].isin(train_vids)]
        val_df = df[df["video_number"] == val_vid]
        test_df = df[df["video_number"] == test_vid]

        fold_dir = out_root / f"fold_test_{test_vid}_val_{val_vid}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFold: test={test_vid}, val={val_vid}, train={train_vids}")
        metrics = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_names=label_names,
            dataset_root=dataset_root,
            output_dir=fold_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            device=args.device,
            smooth_window=args.smooth_window,
            early_patience=args.patience,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            temporal_window=args.temporal_window,
            temporal_fusion=args.temporal_fusion,
            balance_sampler=args.balance_sampler,
            focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            use_class_weights=not args.no_class_weights,
            dropout=args.dropout,
            drop_boundary_sec=args.drop_boundary_sec,
            order_decode=args.order_decode,
            order_allow_skip=args.order_allow_skip,
            order_stay_prob=args.order_stay_prob,
            order_advance_prob=args.order_advance_prob,
            order_skip_prob=args.order_skip_prob,
        )
        all_metrics[str(test_vid)] = metrics

    with open(out_root / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll results saved to: {out_root}")


if __name__ == "__main__":
    main()
