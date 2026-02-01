"""
Visualize ground-truth vs predicted phase timelines for a video.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_label_names(classes_path: Path) -> List[str]:
    with open(classes_path, "r") as f:
        label_map = json.load(f)
    return [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]


def majority_smooth(labels: List[int], window: int) -> List[int]:
    if window <= 1:
        return labels
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        vals = labels[start:end]
        # Majority vote
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        smoothed.append(max(counts.items(), key=lambda x: x[1])[0])
    return smoothed


def to_segments(times: List[float], labels: List[int]) -> List[Tuple[float, float, int]]:
    if not times:
        return []
    times = list(times)
    labels = list(labels)
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


def plot_timeline(segments, label_names, ax, title):
    cmap = plt.get_cmap("tab10")
    for start, end, lbl in segments:
        color = cmap(lbl % 10)
        ax.broken_barh([(start, end - start)], (0, 9), facecolors=color)
    ax.set_ylim(0, 10)
    ax.set_xlim(min(s[0] for s in segments), max(s[1] for s in segments))
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted phase timelines")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.csv")
    parser.add_argument("--dataset", type=str, default="data/phase_dataset_v3",
                        help="Dataset root (for classes.json)")
    parser.add_argument("--video-number", type=int, default=None, help="Filter to a single video")
    parser.add_argument("--smooth-window", type=int, default=5, help="Majority smoothing window")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output) if args.output else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names(Path(args.dataset) / "classes.json")

    df = pd.read_csv(pred_path)
    if args.video_number is not None:
        df = df[df["video_number"] == args.video_number]

    if df.empty:
        print("No rows to visualize.")
        return

    for vid, group in df.groupby("video_number"):
        group = group.sort_values("timestamp")
        times = group["timestamp"].tolist()
        gt = group["true"].astype(int).tolist()
        pred = group["pred"].astype(int).tolist()
        pred_sm = majority_smooth(pred, args.smooth_window)

        gt_segments = to_segments(times, gt)
        pred_segments = to_segments(times, pred_sm)

        fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        plot_timeline(gt_segments, label_names, axes[0], f"Video {vid} - Ground Truth")
        plot_timeline(pred_segments, label_names, axes[1], f"Video {vid} - Predicted (smooth={args.smooth_window})")

        # Legend
        handles = []
        cmap = plt.get_cmap("tab10")
        for idx, name in enumerate(label_names):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(idx % 10), label=f"{idx}: {name}"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)

        fig.tight_layout(rect=[0, 0.1, 1, 1])
        out_path = out_dir / f"phase_timeline_video_{vid}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
