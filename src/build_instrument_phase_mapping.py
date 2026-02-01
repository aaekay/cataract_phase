"""
Build instrument↔phase mapping from annotated CSV using instrument_standardized.
Outputs JSON matrix and a heatmap plot of P(instrument | phase).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def split_instruments(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[;,]", text) if p and p.strip()]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build instrument-phase mapping from CSV")
    parser.add_argument("--csv", type=str, default="data/cataract_coach_clean_v2.csv")
    parser.add_argument("--phase-col", type=str, default="phase_main")
    parser.add_argument("--instrument-col", type=str, default="instrument_standardized")
    parser.add_argument("--output", type=str, default="outputs/instrument_phase_mapping.json")
    parser.add_argument("--plot", type=str, default="outputs/instrument_phase_mapping.png")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing alpha")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[[args.phase_col, args.instrument_col]].dropna()

    counts = defaultdict(Counter)
    phase_totals = Counter()
    instrument_totals = Counter()

    for _, row in df.iterrows():
        phase = str(row[args.phase_col]).strip()
        instruments = split_instruments(str(row[args.instrument_col]))
        if not phase or not instruments:
            continue
        for inst in instruments:
            counts[phase][inst] += 1
            phase_totals[phase] += 1
            instrument_totals[inst] += 1

    phases = sorted(counts.keys())
    instruments = sorted(instrument_totals.keys())

    phase_index = {p: i for i, p in enumerate(phases)}
    inst_index = {i: j for j, i in enumerate(instruments)}

    count_matrix = np.zeros((len(phases), len(instruments)), dtype=np.float32)
    for phase, inst_counts in counts.items():
        for inst, c in inst_counts.items():
            count_matrix[phase_index[phase], inst_index[inst]] = float(c)

    alpha = float(args.smoothing)
    # P(inst | phase)
    p_inst_given_phase = (count_matrix + alpha)
    p_inst_given_phase = p_inst_given_phase / p_inst_given_phase.sum(axis=1, keepdims=True)

    # P(phase | inst)
    p_phase_given_inst = (count_matrix + alpha)
    p_phase_given_inst = p_phase_given_inst / p_phase_given_inst.sum(axis=0, keepdims=True)

    output = {
        "phase_list": phases,
        "instrument_list": instruments,
        "counts": count_matrix.tolist(),
        "p_instrument_given_phase": p_inst_given_phase.tolist(),
        "p_phase_given_instrument": p_phase_given_inst.tolist(),
        "phase_totals": {k: int(v) for k, v in phase_totals.items()},
        "instrument_totals": {k: int(v) for k, v in instrument_totals.items()},
        "phase_column": args.phase_col,
        "instrument_column": args.instrument_col,
        "smoothing": alpha,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot heatmap
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(instruments) * 0.6), max(4, len(phases) * 0.5)))
    sns.heatmap(p_inst_given_phase, annot=False, cmap="Blues",
                xticklabels=instruments, yticklabels=phases)
    plt.title("P(instrument | phase)")
    plt.xlabel("Instrument")
    plt.ylabel("Phase")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved mapping to {output_path}")
    print(f"Saved heatmap to {plot_path}")


if __name__ == "__main__":
    main()
