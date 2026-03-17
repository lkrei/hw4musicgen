import argparse
import logging
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger("build_manifests")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_audio_dataset(audiocraft_root: Path, input_dir: Path, output_manifest: Path) -> None:
    if not input_dir.exists():
        logger.warning("missing input: %s", input_dir)
        return

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "audiocraft.data.audio_dataset",
        str(input_dir),
        str(output_manifest),
    ]

    logger.info("build: %s <- %s", output_manifest, input_dir)
    subprocess.run(command, check=True, cwd=audiocraft_root)
    logger.info("done: %s", output_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--egs_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--audiocraft_root",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_root = Path(args.audio_root)
    egs_root = Path(args.egs_root)
    audiocraft_root = Path(args.audiocraft_root).resolve()
    run_audio_dataset(
        audiocraft_root=audiocraft_root,
        input_dir=audio_root / "train",
        output_manifest=egs_root / "train" / "data.jsonl.gz",
    )
    run_audio_dataset(
        audiocraft_root=audiocraft_root,
        input_dir=audio_root / "valid",
        output_manifest=egs_root / "valid" / "data.jsonl.gz",
    )


if __name__ == "__main__":
    main()
