from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .ui.image_labeler import start_image_labeling


def _path_from_env(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return Path(raw).expanduser()
    return None


def _resolve_tasks_dir(raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value).expanduser()

    env_value = os.environ.get("LABEL_ANYTHING_TASKS_DIR")
    if env_value:
        return Path(env_value).expanduser()

    while True:
        entered = input("Tasks Directory: ").strip()
        if entered:
            return Path(entered).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label images with SAM 2.")
    parser.add_argument(
        "tasks_dir",
        nargs="?",
        help="Directory containing the images to label.",
    )
    parser.add_argument(
        "--rgb-to-bgr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Swap RGB channel order before labeling.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the window in fullscreen mode.",
    )
    parser.add_argument(
        "--sam-model",
        choices=["tiny", "small", "base_plus", "large"],
        default=None,
        help="SAM 2 model size.",
    )
    parser.add_argument(
        "--sam2-dir",
        type=Path,
        default=None,
        help="Path to a local SAM 2 checkout.",
    )
    parser.add_argument(
        "--custom-weights",
        type=Path,
        default=None,
        help="Optional fine-tuned weights to load into the image model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tasks_dir = _resolve_tasks_dir(args.tasks_dir)
    rgb_to_bgr = args.rgb_to_bgr
    fullscreen = args.fullscreen
    model_type = args.sam_model or os.environ.get("LABEL_ANYTHING_MODEL_TYPE", "base_plus")
    sam2_dir = (args.sam2_dir.expanduser() if args.sam2_dir else None) or _path_from_env(
        "LABEL_ANYTHING_SAM2_DIR",
        "SAM_2_DIR",
        "SAM2_DIR",
    )
    custom_weights = (args.custom_weights.expanduser() if args.custom_weights else None) or _path_from_env(
        "LABEL_ANYTHING_CUSTOM_WEIGHTS"
    )

    start_image_labeling(
        tasks_dir=tasks_dir,
        rgb_to_bgr=rgb_to_bgr,
        model_type=model_type,
        sam2_dir=sam2_dir,
        custom_weights=custom_weights,
        fullscreen=fullscreen,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
