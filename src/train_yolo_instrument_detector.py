"""
Train YOLO instrument detector on the prepared YOLO dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO instrument detector")
    parser.add_argument("--data", type=str, default="data/yolo_instruments/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.name or f"yolo_instruments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=run_name,
    )

    project_path = Path(args.project)
    candidates = []
    if project_path.is_absolute():
        candidates.append(project_path / run_name)
    else:
        candidates.append(Path("runs") / "detect" / project_path / run_name)
        candidates.append(project_path / run_name)

    run_dir = None
    for cand in candidates:
        if (cand / "weights" / "best.pt").exists():
            run_dir = cand
            break

    if run_dir is None:
        matches = list(Path("runs").glob(f"**/{run_name}/weights/best.pt"))
        if matches:
            run_dir = matches[0].parent.parent

    if run_dir:
        print(f"Training complete. Results in {run_dir}")
        print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    else:
        print(f"Training complete. Results in {project / run_name}")


if __name__ == "__main__":
    main()
