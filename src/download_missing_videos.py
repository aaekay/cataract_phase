"""
Download missing cataract videos from YouTube based on cataract_coach_clean_v2.csv.
"""

import argparse
import subprocess
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_video_number(text: str) -> Optional[int]:
    match = re.search(r"(\d{3,4})", text)
    return int(match.group(1)) if match else None


def find_videos_in_zip(zip_path: Path) -> set[int]:
    if not zip_path.exists():
        return set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        video_files = [
            f for f in zf.namelist()
            if f.lower().endswith((".mp4", ".avi", ".mov")) and not f.startswith("__MACOSX")
        ]
    nums = set()
    for vf in video_files:
        num = extract_video_number(vf)
        if num is not None:
            nums.add(num)
    return nums


def find_videos_in_dir(video_dir: Path) -> set[int]:
    if not video_dir.exists():
        return set()
    nums = set()
    for path in video_dir.rglob("*"):
        if path.suffix.lower() in {".mp4", ".avi", ".mov"}:
            num = extract_video_number(path.name)
            if num is not None:
                nums.add(num)
    return nums


def get_video_urls(df: pd.DataFrame) -> dict[int, str]:
    # Pick the most common URL per video_number
    urls = {}
    for vid, group in df.groupby("video_number"):
        choices = group["video_url"].dropna().astype(str)
        if choices.empty:
            continue
        url = choices.value_counts().idxmax()
        urls[int(vid)] = url
    return urls


def ensure_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


def download_video(url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "best[ext=mp4]/best",
        "-o",
        str(output_path),
    ]

    print(f"Downloading: {url}")

    # Attempt 1: default
    result = subprocess.run(base_cmd + [url])
    if result.returncode == 0:
        return True

    # Attempt 2: use android client to avoid SABR issues
    result = subprocess.run(base_cmd + ["--extractor-args", "youtube:player_client=android", url])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download missing videos from YouTube")
    parser.add_argument("--csv", type=str, default="data/cataract_coach_clean_v2.csv")
    parser.add_argument("--video-zip", type=str, default="data/cataract_coach_videos.zip")
    parser.add_argument("--video-dir", type=str, default="data/cataract_coach_videos")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of downloads")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    video_zip = Path(args.video_zip)
    video_dir = Path(args.video_dir)

    df = pd.read_csv(csv_path)
    df = df[df["video_number"].notna()].copy()
    df["video_number"] = df["video_number"].astype(int)

    needed = set(df["video_number"].unique())
    in_zip = find_videos_in_zip(video_zip)
    in_dir = find_videos_in_dir(video_dir)

    missing = sorted(int(x) for x in (needed - in_zip - in_dir))

    if not missing:
        print("No missing videos to download.")
        return

    if not ensure_ytdlp():
        print("yt-dlp not found. Install it first (pip install yt-dlp).")
        return

    url_map = get_video_urls(df)

    to_download = missing[: args.limit] if args.limit else missing
    print(f"Missing videos: {missing}")
    print(f"Downloading: {to_download}")

    for vid in to_download:
        url = url_map.get(vid)
        if not url:
            print(f"⚠️  No URL found for video {vid}, skipping.")
            continue

        output_path = video_dir / f"CataractCoach_{vid}.%(ext)s"
        ok = download_video(url, output_path)
        if not ok:
            print(f"❌ Failed to download video {vid}")
        else:
            print(f"✓ Downloaded video {vid}")


if __name__ == "__main__":
    main()
