# Cataract Surgery Phase Classification (ResNet18 / EfficientNet-B0)

This repo contains the scripts and experiment recipes used for frame-level cataract phase prediction and instrument detection/fusion. The recommended order is to train the phase classifier (ResNet baseline) before the YOLO instrument detector.

## What's included
- `src/` scripts for dataset preparation, training, inference, and visualization
- `requirements.txt` and `setup_env.sh` for environment setup
- This README for experiment notes and commands

## What's excluded (not committed)
- Raw videos, derived datasets, training outputs, and model weights

These are ignored via `.gitignore`. Place local copies under `data/`, `outputs/`, and `runs/` as shown in commands below.

## Data requirements
You will need:
- `data/cataract_coach_clean_v2.csv` (phase annotations; uses `phase_main`)
- `data/cataract_coach_videos.zip` or extracted `data/cataract_coach_videos/`
- Local-only videos (not committed). Place these exact files under `data/phase_dataset_v3/videos/cataract_coach_videos/`:
  - `CataractCoach 1087_ how to perfect your case - cataract surgery.mp4`
  - `CataractCoach 1148_ surgical skills after 2000 cataract surgeries.mp4`
  - `CataractCoach 1206_ the soft and gummy cataract.mp4`
- COCO instrument annotation zips for YOLO (place under `data/`)
Note: `data/phase_dataset_v3/labels.csv` is created by step 1 and is used by the YOLO dataset prep step.

Optional helper (downloads missing videos referenced by the CSV):
```bash
uv run python src/download_missing_videos.py \
  --csv data/cataract_coach_clean_v2.csv \
  --video-zip data/cataract_coach_videos.zip \
  --video-dir data/cataract_coach_videos
```
Requires `yt-dlp` (install with `uv pip install yt-dlp` inside the venv).

Note: video sources may be restricted; ensure you have permission to access and use the data.

## Environment setup
```bash
./setup_env.sh
source venv/bin/activate
```

## Reproduce experiments (core steps)
### 1) Prepare phase dataset
```bash
uv run python src/prepare_phase_dataset.py \
  --csv data/cataract_coach_clean_v2.csv \
  --video-zip data/cataract_coach_videos.zip \
  --video-dir data/cataract_coach_videos \
  --output data/phase_dataset_v3 \
  --fps 1.0 \
  --label-column phase_main
```

### 2) Train phase classifier
```bash
uv run python src/train_phase_classifier.py \
  --dataset data/phase_dataset_v3 \
  --model resnet18 \
  --epochs 10 \
  --batch 32 \
  --device cuda

uv run python src/train_phase_classifier.py \
  --dataset data/phase_dataset_v3 \
  --model efficientnet_b0 \
  --epochs 50 \
  --batch 32 \
  --device cuda
```

Other supported backbones: `resnet34`, `resnet50`, `efficientnet_b1`, `efficientnet_b2`.

### 3) Accuracy boosters (optional)
```bash
# Temporal context + class balancing + focal loss + order decoding
uv run python src/train_phase_classifier.py \
  --dataset data/phase_dataset_v3 \
  --model resnet18 \
  --temporal-window 5 \
  --temporal-fusion conv1d \
  --balance-sampler \
  --focal-loss \
  --drop-boundary-sec 1.0 \
  --order-decode \
  --epochs 20 \
  --batch 32 \
  --device cuda
```

Key flags:
- `--temporal-window`: odd number of frames per sample (e.g., 5 or 9)
- `--temporal-fusion`: `avg` or `conv1d`
- `--balance-sampler`: class-balanced sampling
- `--focal-loss`: focal loss for rare classes
- `--drop-boundary-sec`: drop frames near phase transitions (train/val only)
- `--order-decode`: Viterbi decoding to enforce phase order

### 4) Visualize cross-validation folds
Use the output directory created by the phase classifier (the `outputs/phase_classifier_*` path printed during training) as `--folds-root`.
```bash
uv run python src/visualize_phase_folds.py \
  --folds-root outputs/phase_classifier_20260127_235759_resnet18 \
  --dataset data/phase_dataset_v3 \
  --smooth-window 5

uv run python src/visualize_phase_folds.py \
  --folds-root outputs/phase_classifier_20260128_001025_efficientnet_b0 \
  --dataset data/phase_dataset_v3 \
  --smooth-window 5
```

### 5) Build instrument to phase mapping
```bash
uv run python src/build_instrument_phase_mapping.py \
  --csv data/cataract_coach_clean_v2.csv \
  --phase-col phase_main \
  --instrument-col instrument_standardized \
  --output outputs/instrument_phase_mapping.json \
  --plot outputs/instrument_phase_mapping.png
```

### 6) Prepare YOLO instrument dataset (from COCO zips)
Place COCO zip files directly under `data/` (or point `--coco-dir` elsewhere). Each zip should include one or more `*.coco.json` files plus the referenced images. The `file_name` entries must contain the video number (3–4 digits) so they can be matched to `phase_dataset_v3/labels.csv`. Example inputs:
- `data/K_CC_1087_perfect your case.v1i.coco-segmentation.zip`
- `data/K_CC_1148_2000 cataract.v1i.coco-segmentation.zip`
```bash
uv run python src/prepare_yolo_instrument_dataset.py \
  --coco-dir data \
  --phase-labels data/phase_dataset_v3/labels.csv \
  --output data/yolo_instruments \
  --train-ratio 0.7 --val-ratio 0.2
```

### 7) Train YOLO instrument detector
```bash
uv run python src/train_yolo_instrument_detector.py \
  --data data/yolo_instruments/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### 8) Run YOLO on phase frames (instrument predictions)
```bash
uv run python src/predict_instruments.py \
  --model runs/detect/outputs/yolo_instruments_*/weights/best.pt \
  --labels data/phase_dataset_v3/labels.csv \
  --dataset-root data/phase_dataset_v3 \
  --mapping outputs/instrument_phase_mapping.json \
  --conf 0.25 \
  --device 0
```

## Notes
- Torchvision will download pretrained weights on first run.
- Ultralytics can download base YOLO weights; if it does not, place `yolov8n.pt` in the repo root.
- No system-specific `venv/` is included.
