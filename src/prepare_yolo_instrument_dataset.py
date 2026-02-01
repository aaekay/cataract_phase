"""
Prepare a YOLO detection dataset from COCO zips, split by video.
Only uses COCO zips that contain videos present in phase_dataset_v3.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path


def extract_video_number(text: str) -> int | None:
    match = re.search(r"(\d{3,4})", text)
    return int(match.group(1)) if match else None


def sanitize_prefix(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def find_coco_zips(coco_dir: Path) -> list[Path]:
    return sorted(coco_dir.glob("*.coco-segmentation.zip"))


def zip_contains_video(zf: zipfile.ZipFile, allowed_videos: set[int]) -> bool:
    json_paths = [n for n in zf.namelist() if n.endswith(".coco.json")]
    for jp in json_paths:
        data = json.loads(zf.read(jp))
        for img in data.get("images", []):
            num = extract_video_number(str(img.get("file_name", "")))
            if num in allowed_videos:
                return True
    return False


def split_videos(video_ids: list[int], train_ratio: float, val_ratio: float, seed: int) -> dict[int, str]:
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1")
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    vids = video_ids[:]
    random.Random(seed).shuffle(vids)
    n = len(vids)
    n_train = max(1, int(round(n * train_ratio))) if train_ratio > 0 else 0
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    n_test = n - n_train - n_val
    if test_ratio == 0:
        n_test = 0
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)

    split_map = {}
    for i, vid in enumerate(vids):
        if i < n_train:
            split_map[vid] = "train"
        elif i < n_train + n_val:
            split_map[vid] = "val"
        else:
            split_map[vid] = "test"
    return split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from COCO zips")
    parser.add_argument("--coco-dir", type=str, default="data", help="Directory with COCO zip files")
    parser.add_argument("--phase-labels", type=str, default="data/phase_dataset_v3/labels.csv")
    parser.add_argument("--output", type=str, default="data/yolo_instruments")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    phase_labels = Path(args.phase_labels)
    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.output)

    allowed_videos = set()
    if phase_labels.exists():
        import pandas as pd
        df = pd.read_csv(phase_labels)
        allowed_videos = set(df["video_number"].astype(int).unique().tolist())

    coco_zips = find_coco_zips(coco_dir)
    if not coco_zips:
        raise FileNotFoundError(f"No COCO zips found in {coco_dir}")

    selected_zips: list[Path] = []
    for zp in coco_zips:
        # quick filename check
        nums = {int(n) for n in re.findall(r"\d{3,4}", zp.name)}
        if nums & allowed_videos:
            selected_zips.append(zp)
            continue
        with zipfile.ZipFile(zp) as zf:
            if zip_contains_video(zf, allowed_videos):
                selected_zips.append(zp)

    if not selected_zips:
        raise RuntimeError("No COCO zips matched videos in phase dataset")

    video_split = split_videos(sorted(allowed_videos), args.train_ratio, args.val_ratio, args.seed)

    # Build class mapping across zips
    class_name_to_id: dict[str, int] = {}
    images_written = 0
    labels_written = 0
    split_counts = defaultdict(int)

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    missing_images = 0
    for zp in selected_zips:
        prefix = sanitize_prefix(zp.stem)
        with zipfile.ZipFile(zp) as zf:
            name_set = set(zf.namelist())
            json_paths = [n for n in zf.namelist() if n.endswith(".coco.json")]
            for jp in json_paths:
                base_dir = str(Path(jp).parent)
                data = json.loads(zf.read(jp))
                categories = {c["id"]: c["name"] for c in data.get("categories", [])}
                for name in categories.values():
                    if name not in class_name_to_id:
                        class_name_to_id[name] = len(class_name_to_id)

                images = {img["id"]: img for img in data.get("images", [])}
                anns_by_image = defaultdict(list)
                for ann in data.get("annotations", []):
                    anns_by_image[ann["image_id"]].append(ann)

                for image_id, img in images.items():
                    file_name = str(img.get("file_name", ""))
                    video_num = extract_video_number(file_name)
                    if video_num is None or video_num not in allowed_videos:
                        continue
                    split = video_split.get(video_num)
                    if split is None:
                        continue

                    out_name = f"{prefix}_{Path(file_name).name}"
                    out_img = images_dir / split / out_name
                    out_lbl = labels_dir / split / (Path(out_name).stem + ".txt")

                    # resolve file path inside zip
                    zip_name = file_name
                    if zip_name not in name_set:
                        candidate = f"{base_dir}/{file_name}" if base_dir not in (".", "") else file_name
                        if candidate in name_set:
                            zip_name = candidate
                        else:
                            matches = [n for n in name_set if n.endswith(f"/{file_name}")]
                            if len(matches) == 1:
                                zip_name = matches[0]
                            else:
                                missing_images += 1
                                continue

                    # write image
                    if not out_img.exists():
                        with open(out_img, "wb") as f:
                            f.write(zf.read(zip_name))
                        images_written += 1

                    width = float(img.get("width", 1))
                    height = float(img.get("height", 1))

                    lines = []
                    for ann in anns_by_image.get(image_id, []):
                        cat_name = categories.get(ann["category_id"], None)
                        if cat_name is None:
                            continue
                        class_id = class_name_to_id[cat_name]
                        bbox = ann.get("bbox", None)
                        if not bbox or len(bbox) != 4:
                            continue
                        x, y, w, h = bbox
                        x_c = (x + w / 2.0) / width
                        y_c = (y + h / 2.0) / height
                        w_n = w / width
                        h_n = h / height
                        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                    with open(out_lbl, "w") as f:
                        f.write("\n".join(lines))
                    labels_written += 1
                    split_counts[split] += 1

    # Write dataset yaml
    names = [None] * len(class_name_to_id)
    for name, idx in class_name_to_id.items():
        names[idx] = name

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join([
            f"path: {out_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            f"names: {names}",
        ]) + "\n"
    )

    summary = {
        "selected_zips": [str(p) for p in selected_zips],
        "videos": sorted(allowed_videos),
        "video_split": video_split,
        "class_names": names,
        "images_written": images_written,
        "labels_written": labels_written,
        "split_counts": dict(split_counts),
        "missing_images": missing_images,
        "dataset_yaml": str(dataset_yaml),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared dataset at {out_dir}")
    print(f"Classes: {len(names)} | Images: {images_written}")


if __name__ == "__main__":
    main()
