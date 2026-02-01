"""
Create a single combined timeline plot for all folds (GT vs Pred per video),
and compute ROC-AUC / PR-AUC metrics from saved probabilities.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def load_label_names(classes_path: Path) -> List[str]:
    with open(classes_path, "r") as f:
        label_map = json.load(f)
    return [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]


def prob_col(label_name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", label_name).strip("_")
    return f"prob_{safe}"


def majority_smooth(labels: List[int], window: int) -> List[int]:
    if window <= 1:
        return labels
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        vals = labels[start:end]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        smoothed.append(max(counts.items(), key=lambda x: x[1])[0])
    return smoothed


def to_segments(times: List[float], labels: List[int]) -> List[Tuple[float, float, int]]:
    if not times:
        return []
    if len(times) == 1:
        return [(times[0], times[0] + 1.0, labels[0])]

    diffs = np.diff(times)
    step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
    segments = []
    start = times[0]
    prev = labels[0]
    for t, lbl in zip(times[1:], labels[1:]):
        if lbl != prev:
            segments.append((start, t, prev))
            start = t
            prev = lbl
    segments.append((start, times[-1] + step, prev))
    return segments


def plot_segments(ax, segments, label_names):
    cmap = plt.get_cmap("tab10")
    for start, end, lbl in segments:
        color = cmap(lbl % 10)
        ax.broken_barh([(start, end - start)], (0, 9), facecolors=color)
    ax.set_ylim(0, 10)
    ax.set_yticks([])
    ax.set_xlim(min(s[0] for s in segments), max(s[1] for s in segments))
    ax.set_xlabel("Time (s)")


def pick_video_from_predictions(df: pd.DataFrame) -> int:
    counts = df["video_number"].value_counts()
    return int(counts.idxmax())


def compute_auc_metrics(df: pd.DataFrame, label_names: List[str]) -> dict:
    prob_cols = [prob_col(name) for name in label_names]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        return {
            "roc_auc_macro": None,
            "pr_auc_macro": None,
            "roc_auc_per_class": {},
            "pr_auc_per_class": {},
            "missing_prob_columns": missing,
        }

    y_true = df["true"].astype(int).to_numpy()
    y_score = df[prob_cols].to_numpy()

    roc_per_class = {}
    pr_per_class = {}
    for i, name in enumerate(label_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            roc_per_class[name] = None
            pr_per_class[name] = None
            continue
        roc_per_class[name] = float(roc_auc_score(y_bin, y_score[:, i]))
        pr_per_class[name] = float(average_precision_score(y_bin, y_score[:, i]))

    roc_vals = [v for v in roc_per_class.values() if v is not None]
    pr_vals = [v for v in pr_per_class.values() if v is not None]

    return {
        "roc_auc_macro": float(np.mean(roc_vals)) if roc_vals else None,
        "pr_auc_macro": float(np.mean(pr_vals)) if pr_vals else None,
        "roc_auc_per_class": roc_per_class,
        "pr_auc_per_class": pr_per_class,
        "missing_prob_columns": [],
    }


def plot_auc_curves(df: pd.DataFrame, label_names: List[str], output_path: Path, smooth_window: int) -> None:
    prob_cols = [prob_col(name) for name in label_names]
    if any(c not in df.columns for c in prob_cols):
        print("Skipping AUC curves (missing probability columns).")
        return

    y_true = df["true"].astype(int).to_numpy()
    y_score = df[prob_cols].to_numpy()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for i, name in enumerate(label_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        precision, recall, _ = precision_recall_curve(y_bin, y_score[:, i])
        ax_roc.plot(fpr, tpr, label=f"{i}: {name}")
        ax_pr.plot(recall, precision, label=f"{i}: {name}")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_title("ROC Curves (OvR)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

    ax_pr.set_title("PR Curves (OvR)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")

    handles, labels = ax_roc.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle(f"ROC/PR Curves (smooth_window={smooth_window})", fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combined timeline plot for all folds")
    parser.add_argument("--folds-root", type=str, required=True,
                        help="Path to outputs/phase_classifier_*/ directory")
    parser.add_argument("--dataset", type=str, default="data/phase_dataset_v3",
                        help="Dataset root for classes.json")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    folds_root = Path(args.folds_root)
    pred_files = sorted(folds_root.glob("**/predictions.csv"))
    if not pred_files:
        print("No predictions.csv found under:", folds_root)
        return

    label_names = load_label_names(Path(args.dataset) / "classes.json")

    fold_auc_rows = []
    fold_auc_detail = []
    video_blocks = []
    for p in pred_files:
        df_all = pd.read_csv(p)
        if df_all.empty:
            continue
        fold_id = str(p.parent.relative_to(folds_root))

        auc_metrics = compute_auc_metrics(df_all, label_names)
        if not auc_metrics["missing_prob_columns"]:
            fold_auc_rows.append({
                "fold": fold_id,
                "roc_auc_macro": auc_metrics["roc_auc_macro"],
                "pr_auc_macro": auc_metrics["pr_auc_macro"],
            })
            fold_auc_detail.append({
                "fold": fold_id,
                "roc_auc_macro": auc_metrics["roc_auc_macro"],
                "pr_auc_macro": auc_metrics["pr_auc_macro"],
                "roc_auc_per_class": auc_metrics["roc_auc_per_class"],
                "pr_auc_per_class": auc_metrics["pr_auc_per_class"],
            })

            plot_auc_curves(df_all, label_names, p.parent / "roc_pr_curves.png", args.smooth_window)
        else:
            print(f"Skipping AUC metrics for {fold_id}; missing columns: {auc_metrics['missing_prob_columns']}")

        vid = pick_video_from_predictions(df_all)
        df = df_all[df_all["video_number"] == vid].sort_values("timestamp")
        times = df["timestamp"].tolist()
        gt = df["true"].astype(int).tolist()
        pred = df["pred"].astype(int).tolist()
        pred_sm = majority_smooth(pred, args.smooth_window)

        gt_segments = to_segments(times, gt)
        pred_segments = to_segments(times, pred_sm)
        y_true = np.array(gt, dtype=int)
        y_pred = np.array(pred, dtype=int)
        y_pred_sm = np.array(pred_sm, dtype=int)

        metrics = {
            "acc": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
            "acc_s": accuracy_score(y_true, y_pred_sm),
            "f1_s": f1_score(y_true, y_pred_sm, average="macro"),
            "roc_auc_macro": auc_metrics["roc_auc_macro"],
            "pr_auc_macro": auc_metrics["pr_auc_macro"],
        }
        video_blocks.append((vid, gt_segments, pred_segments, metrics))

    video_blocks.sort(key=lambda x: x[0])

    n = len(video_blocks)
    fig, axes = plt.subplots(nrows=2 * n, ncols=1, figsize=(14, max(3, 2.2 * n)))
    if n == 1:
        axes = [axes[0], axes[1]]

    for i, (vid, gt_seg, pred_seg, metrics) in enumerate(video_blocks):
        ax_gt = axes[2 * i]
        ax_pr = axes[2 * i + 1]

        plot_segments(ax_gt, gt_seg, label_names)
        ax_gt.set_title(f"Video {vid} - Ground Truth")

        plot_segments(ax_pr, pred_seg, label_names)
        ax_pr.set_title(f"Video {vid} - Predicted (smooth={args.smooth_window})")

        roc_auc_text = "n/a" if metrics["roc_auc_macro"] is None else f"{metrics['roc_auc_macro']:.3f}"
        pr_auc_text = "n/a" if metrics["pr_auc_macro"] is None else f"{metrics['pr_auc_macro']:.3f}"
        metric_text = (
            f"Acc: {metrics['acc']:.3f}  F1: {metrics['f1']:.3f}\n"
            f"Acc_s: {metrics['acc_s']:.3f}  F1_s: {metrics['f1_s']:.3f}\n"
            f"ROC-AUC: {roc_auc_text}  PR-AUC: {pr_auc_text}"
        )
        ax_pr.text(
            0.99,
            0.5,
            metric_text,
            transform=ax_pr.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    handles = []
    cmap = plt.get_cmap("tab10")
    for idx, name in enumerate(label_names):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(idx % 10), label=f"{idx}: {name}"))

    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = folds_root / "combined_phase_timelines.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    if fold_auc_rows:
        pd.DataFrame(fold_auc_rows).to_csv(folds_root / "fold_auc_metrics.csv", index=False)
        with open(folds_root / "fold_auc_metrics.json", "w") as f:
            json.dump(fold_auc_detail, f, indent=2)
        print(f"Saved: {folds_root / 'fold_auc_metrics.csv'}")
        print(f"Saved: {folds_root / 'fold_auc_metrics.json'}")


if __name__ == "__main__":
    main()
