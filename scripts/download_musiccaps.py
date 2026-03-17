import argparse
import csv
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from datasets import load_dataset


logger = logging.getLogger("download_musiccaps")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_ffmpeg(input_url: str, output_path: Path, start: float, duration: float) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", input_url,
        "-t", str(duration),
        "-ac", "2",
        "-ar", "32000",
        "-vn",
        "-f", "wav",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def get_audio_url(ytid: str) -> Optional[str]:
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        logger.error("yt-dlp is not installed. Run `pip install yt-dlp`.")
        return None

    url = f"https://www.youtube.com/watch?v={ytid}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception:
            return None
    return info.get("url")


def download_items(
    items: list,
    output_dir: Path,
    split_name: str,
    max_items: Optional[int],
    segment_duration: float,
    metadata_csv: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ["ytid", "path", "split", "start", "end", "caption"]
    write_header = not metadata_csv.exists()

    processed = 0
    with metadata_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for item in items:
            if max_items is not None and processed >= max_items:
                break

            ytid = item["ytid"]
            start = float(item["start_s"])
            caption = item["caption"]

            filename = f"{ytid}_{int(start)}.wav"
            output_path = output_dir / filename

            if not output_path.exists():
                audio_url = get_audio_url(ytid)
                if audio_url is None:
                    continue
                ok = run_ffmpeg(
                    input_url=audio_url,
                    output_path=output_path,
                    start=start,
                    duration=segment_duration,
                )
                if not ok:
                    continue

            writer.writerow({
                "ytid": ytid,
                "path": str(output_path.resolve()),
                "split": split_name,
                "start": start,
                "end": start + segment_duration,
                "caption": caption,
            })
            processed += 1

            if processed % 50 == 0:
                logger.info("Downloaded %d items for split=%s", processed, split_name)

    logger.info("Finished split=%s, total: %d", split_name, processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--valid_limit",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=10.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    train_dir = root / "audio" / "train"
    valid_dir = root / "audio" / "valid"
    metadata_csv = root / "musiccaps_metadata.csv"
    logger.info("loading dataset...")
    ds = load_dataset("google/MusicCaps", split="train")

    train_items = [item for item in ds if not item["is_audioset_eval"]]
    valid_items = [item for item in ds if item["is_audioset_eval"]]
    logger.info("counts: train=%d valid=%d", len(train_items), len(valid_items))

    train_limit = args.train_limit if args.train_limit > 0 else None
    valid_limit = args.valid_limit if args.valid_limit > 0 else None

    download_items(
        items=train_items,
        output_dir=train_dir,
        split_name="train",
        max_items=train_limit,
        segment_duration=args.segment_duration,
        metadata_csv=metadata_csv,
    )
    download_items(
        items=valid_items,
        output_dir=valid_dir,
        split_name="valid",
        max_items=valid_limit,
        segment_duration=args.segment_duration,
        metadata_csv=metadata_csv,
    )


if __name__ == "__main__":
    main()
