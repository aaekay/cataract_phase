"""
Plot cross-validation summary metrics from summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot CV summary metrics")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.output) if args.output else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "r") as f:
        data = json.load(f)

    fold_ids = []
    macro_f1 = []
    macro_f1_s = []

    for fold, metrics in data.items():
        fold_ids.append(fold)
        macro_f1.append(metrics.get("macro_f1", 0.0))
        if "smoothed" in metrics:
            macro_f1_s.append(metrics["smoothed"].get("macro_f1", 0.0))
        else:
            macro_f1_s.append(None)

    x = np.arange(len(fold_ids))

    # Raw macro-F1
    plt.figure(figsize=(10, 4))
    plt.bar(x, macro_f1, color="steelblue")
    plt.xticks(x, fold_ids, rotation=45, ha="right")
    plt.ylabel("Macro F1")
    plt.title("CV Macro F1 per Fold (Raw)")
    plt.tight_layout()
    out_raw = out_dir / "cv_macro_f1.png"
    plt.savefig(out_raw, dpi=150)
    plt.close()

    # Smoothed macro-F1 if available
    if any(v is not None for v in macro_f1_s):
        vals = [v if v is not None else 0.0 for v in macro_f1_s]
        plt.figure(figsize=(10, 4))
        plt.bar(x, vals, color="seagreen")
        plt.xticks(x, fold_ids, rotation=45, ha="right")
        plt.ylabel("Macro F1")
        plt.title("CV Macro F1 per Fold (Smoothed)")
        plt.tight_layout()
        out_s = out_dir / "cv_macro_f1_smoothed.png"
        plt.savefig(out_s, dpi=150)
        plt.close()

    print(f"Saved: {out_raw}")
    if any(v is not None for v in macro_f1_s):
        print(f"Saved: {out_s}")


if __name__ == "__main__":
    main()
