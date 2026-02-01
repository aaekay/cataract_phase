"""
Prepare supervised phase dataset from cataract_coach_clean_v2.csv.
Creates frame-level labels by sampling videos at a fixed fps.
"""

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


def extract_video_number(text: str) -> Optional[int]:
    match = re.search(r"(\d{3,4})", text)
    return int(match.group(1)) if match else None


def extract_videos(zip_path: Path, output_dir: Path, needed_numbers: set[int]) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, Path] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        video_files = [
            f for f in zf.namelist()
            if f.lower().endswith((".mp4", ".avi", ".mov")) and not f.startswith("__MACOSX")
        ]

        for vf in video_files:
            num = extract_video_number(vf)
            if num is None or num not in needed_numbers:
                continue

            extracted_path = output_dir / vf
            if not extracted_path.exists():
                zf.extract(vf, output_dir)

            # Ensure mapping points to the actual extracted file
            mapping[num] = extracted_path

    return mapping


def find_videos_in_dir(video_dir: Path, needed_numbers: set[int]) -> dict[int, Path]:
    if not video_dir.exists():
        return {}

    mapping: dict[int, Path] = {}
    for path in video_dir.rglob("*"):
        if path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            continue
        num = extract_video_number(path.name)
        if num is None or num not in needed_numbers:
            continue
        mapping[num] = path
    return mapping


def get_video_duration(video_path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return duration, fps, total_frames


def clean_segments(df: pd.DataFrame, video_duration: float) -> tuple[pd.DataFrame, dict]:
    report = {
        "rows_in": int(len(df)),
        "rows_swapped": 0,
        "rows_clamped": 0,
        "rows_dropped_invalid": 0,
        "overlaps_clipped": 0,
    }

    segs = df.copy()
    segs = segs.dropna(subset=["start_sec", "end_sec"]).copy()
    segs["start_sec"] = pd.to_numeric(segs["start_sec"], errors="coerce")
    segs["end_sec"] = pd.to_numeric(segs["end_sec"], errors="coerce")
    segs = segs.dropna(subset=["start_sec", "end_sec"]).copy()

    swap_mask = segs["end_sec"] < segs["start_sec"]
    if swap_mask.any():
        segs.loc[swap_mask, ["start_sec", "end_sec"]] = segs.loc[swap_mask, ["end_sec", "start_sec"]].values
        report["rows_swapped"] = int(swap_mask.sum())

    # Clamp to video duration
    before = segs[["start_sec", "end_sec"]].copy()
    segs["start_sec"] = segs["start_sec"].clip(lower=0, upper=video_duration)
    segs["end_sec"] = segs["end_sec"].clip(lower=0, upper=video_duration)
    report["rows_clamped"] = int((before != segs[["start_sec", "end_sec"]]).any(axis=1).sum())

    # Drop invalid
    segs["duration_sec"] = segs["end_sec"] - segs["start_sec"]
    segs = segs[segs["duration_sec"] > 0].copy()

    # Resolve overlaps by clipping each segment to the next segment's start
    segs = segs.sort_values("start_sec").reset_index(drop=True)
    cleaned_rows = []
    for i, row in segs.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if i < len(segs) - 1:
            next_start = float(segs.loc[i + 1, "start_sec"])
            if end > next_start:
                end = next_start
                report["overlaps_clipped"] += 1

        if end > start:
            row = row.copy()
            row["end_sec"] = end
            row["duration_sec"] = end - start
            cleaned_rows.append(row)
        else:
            report["rows_dropped_invalid"] += 1

    cleaned = pd.DataFrame(cleaned_rows)
    return cleaned, report


def build_label_map(phases: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted(set(phases)))}


def main():
    parser = argparse.ArgumentParser(description="Prepare phase dataset from annotated CSV")
    parser.add_argument("--csv", type=str, default="data/cataract_coach_clean_v2.csv")
    parser.add_argument("--video-zip", type=str, default="data/cataract_coach_videos.zip")
    parser.add_argument("--video-dir", type=str, default="data/cataract_coach_videos")
    parser.add_argument("--output", type=str, default="data/phase_dataset")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling fps for frame extraction")
    parser.add_argument("--label-column", type=str, default="phase_main",
                        choices=["phase_main", "phase_name", "phase"],
                        help="Which column to use as label")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    zip_path = Path(args.video_zip)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Keep only rows with numeric video_number
    df = df[df["video_number"].notna()].copy()
    df["video_number"] = df["video_number"].astype(int)

    needed_numbers = set(df["video_number"].unique())
    video_map = {}
    # Prefer videos from a local directory if available
    video_dir_map = find_videos_in_dir(Path(args.video_dir), needed_numbers)
    video_map.update(video_dir_map)

    # Fallback to extracting from zip for any remaining videos
    remaining = needed_numbers - set(video_map.keys())
    if remaining:
        extracted_map = extract_videos(zip_path, output_dir / "videos", remaining)
        video_map.update(extracted_map)

    skipped_videos = [int(v) for v in sorted(needed_numbers - set(video_map.keys()))]

    all_labels = []
    cleaning_reports = {}
    phase_names = []
    per_phase_counts = {}

    for vid, video_path in video_map.items():
        print(f"\nProcessing video {vid}: {video_path.name}")
        duration, video_fps, total_frames = get_video_duration(video_path)
        print(f"  duration={duration:.2f}s fps={video_fps:.2f} frames={total_frames}")

        segs = df[df["video_number"] == vid].copy()
        cleaned, report = clean_segments(segs, duration)
        cleaning_reports[str(vid)] = report

        if cleaned.empty:
            print("  ⚠️  No valid segments after cleaning, skipping.")
            continue

        # Save cleaned segments per video
        cleaned.to_csv(output_dir / f"segments_{vid}.csv", index=False)

        # Build list of segments for labeling
        segments = cleaned.sort_values("start_sec").to_dict("records")

        # Frame extraction
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  ⚠️  Could not open video, skipping.")
            continue

        # Ensure video is readable (some partial downloads open but contain no frames)
        ret, _ = cap.read()
        if not ret:
            print("  ⚠️  Could not read frames (partial/corrupt), skipping.")
            cap.release()
            continue
        # Reset to first frame for extraction
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_interval = max(1, int(round(video_fps / args.fps))) if args.fps else 1
        frame_idx = 0
        seg_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps if video_fps > 0 else 0.0

                while seg_idx < len(segments) and timestamp >= segments[seg_idx]["end_sec"]:
                    seg_idx += 1

                if seg_idx < len(segments):
                    seg = segments[seg_idx]
                    if seg["start_sec"] <= timestamp < seg["end_sec"]:
                        phase_label = str(seg[args.label_column])
                        phase_names.append(phase_label)

                        per_phase_counts[phase_label] = per_phase_counts.get(phase_label, 0) + 1

                        img_name = f"{vid}_{frame_idx:06d}.jpg"
                        img_rel = Path("images") / img_name
                        img_path = output_dir / img_rel
                        cv2.imwrite(str(img_path), frame)

                        all_labels.append({
                            "image_path": str(img_rel),
                            "video_number": vid,
                            "timestamp": float(timestamp),
                            "phase": str(seg.get("phase", "")),
                            "phase_main": str(seg.get("phase_main", "")),
                            "phase_name": str(seg.get("phase_name", "")),
                            "label": phase_label,
                        })
                        saved += 1

            frame_idx += 1

        cap.release()
        print(f"  saved frames: {saved}")

    if not all_labels:
        print("No labeled frames created. Check CSV/video availability.")
        return

    label_map = build_label_map(phase_names)
    for row in all_labels:
        row["label_id"] = label_map[row["label"]]

    labels_df = pd.DataFrame(all_labels)
    labels_df.to_csv(output_dir / "labels.csv", index=False)

    with open(output_dir / "classes.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(output_dir / "cleaning_report.json", "w") as f:
        json.dump({
            "skipped_videos": skipped_videos,
            "reports": cleaning_reports,
            "label_counts": per_phase_counts,
            "total_labeled_frames": len(all_labels),
        }, f, indent=2)

    print("\nDone.")
    print(f"Labels: {output_dir / 'labels.csv'}")
    print(f"Classes: {output_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
