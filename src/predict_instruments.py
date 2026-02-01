"""
Run YOLO on phase_dataset_v3 images to produce instrument detections per frame.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def build_name_mapper(instrument_list: list[str]):
    norm_to_inst = {normalize(i): i for i in instrument_list}

    alias = {
        "ia probe": "Irrigation and aspiration probe",
        "i a probe": "Irrigation and aspiration probe",
        "irrigation aspiration probe": "Irrigation and aspiration probe",
        "phaco probe": "Phacoemulsification probe",
    }

    def mapper(name: str) -> str | None:
        n = normalize(name)
        if n in norm_to_inst:
            return norm_to_inst[n]
        if n in alias:
            return alias[n]
        for inst_norm, inst in norm_to_inst.items():
            if inst_norm in n or n in inst_norm:
                return inst
        return None

    return mapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict instruments on phase dataset images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--labels", type=str, default="data/phase_dataset_v3/labels.csv")
    parser.add_argument("--dataset-root", type=str, default="data/phase_dataset_v3")
    parser.add_argument("--mapping", type=str, default="outputs/instrument_phase_mapping.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    dataset_root = Path(args.dataset_root)

    df = pd.read_csv(labels_path)

    instrument_list = []
    mapping_path = Path(args.mapping)
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            instrument_list = mapping.get("instrument_list", [])

    name_mapper = build_name_mapper(instrument_list) if instrument_list else (lambda x: x)

    out_dir = Path(args.output) if args.output else Path("outputs") / f"instrument_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model_names = model.names

    results_rows = []
    score_rows = []

    for _, row in df.iterrows():
        img_path = dataset_root / row["image_path"]
        if not img_path.exists():
            continue

        res = model.predict(source=str(img_path), conf=args.conf, device=args.device, verbose=False)[0]
        detections = []
        score_map = defaultdict(float)

        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            name = model_names.get(cls_id, str(cls_id))
            mapped = name_mapper(name)
            if mapped is None:
                continue
            score_map[mapped] += conf
            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "mapped_name": mapped,
                "conf": conf,
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            })

        results_rows.append({
            "image_path": row["image_path"],
            "video_number": int(row["video_number"]),
            "timestamp": float(row["timestamp"]),
            "true": int(row["label_id"]),
            "detections": detections,
            "instrument_scores": dict(score_map),
        })

        score_row = {
            "image_path": row["image_path"],
            "video_number": int(row["video_number"]),
            "timestamp": float(row["timestamp"]),
            "true": int(row["label_id"]),
        }
        for inst, score in score_map.items():
            score_row[inst] = score
        score_rows.append(score_row)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results_rows, f, indent=2)

    pd.DataFrame(score_rows).to_csv(out_dir / "instrument_scores.csv", index=False)

    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
