import argparse
import logging
import os
import subprocess
from pathlib import Path


logger = logging.getLogger("train_musicgen_finetune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audiocraft_root", type=str, required=True
    )
    parser.add_argument(
        "--model_scale", type=str, default="small", choices=["small", "medium"]
    )
    parser.add_argument(
        "--batch_size", type=int, default=4
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--epochs", type=int, default=10
    )
    parser.add_argument(
        "--updates_per_epoch", type=int, default=200
    )
    parser.add_argument(
        "--merge_text_p", type=float, default=0.8
    )
    parser.add_argument(
        "--drop_desc_p", type=float, default=0.1
    )
    parser.add_argument(
        "--drop_other_p", type=float, default=0.85
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audiocraft_root = Path(args.audiocraft_root).resolve()

    if not audiocraft_root.exists():
        raise FileNotFoundError(f"audiocraft_root not found: {audiocraft_root}")
    env = {**os.environ, "PYTHONPATH": str(audiocraft_root)}

    pretrained_name = f"facebook/musicgen-{args.model_scale}"

    command = [
        "dora", "run",
        "solver=musicgen/musicgen_base_32khz",
        f"model/lm/model_scale={args.model_scale}",
        f"continue_from=//pretrained/{pretrained_name}",
        "conditioner=text2music",
        "dset=audio/musiccaps_hw4",
        f"dataset.batch_size={args.batch_size}",
        f"dataset.merge_text_p={args.merge_text_p}",
        f"dataset.drop_desc_p={args.drop_desc_p}",
        f"dataset.drop_other_p={args.drop_other_p}",
        f"optim.lr={args.lr}",
        f"optim.epochs={args.epochs}",
        f"optim.updates_per_epoch={args.updates_per_epoch}",
    ]

    logger.info("cwd: %s", audiocraft_root)
    logger.info("cmd: %s", " ".join(command))
    logger.info(
        "params: scale=%s batch=%d lr=%g epochs=%d u/ep=%d merge=%.2f drop_desc=%.2f drop_other=%.2f",
        args.model_scale,
        args.batch_size,
        args.lr,
        args.epochs,
        args.updates_per_epoch,
        args.merge_text_p,
        args.drop_desc_p,
        args.drop_other_p,
    )

    subprocess.run(command, cwd=audiocraft_root, check=True, env=env)


if __name__ == "__main__":
    main()
