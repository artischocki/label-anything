from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    checkpoint: str
    config_name: str


MODEL_SPECS = {
    "large": ModelSpec("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "base_plus": ModelSpec("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "small": ModelSpec("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "tiny": ModelSpec("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
}


def _resolve_sam2_dir(explicit_path: Path | None = None) -> Path:
    candidate = explicit_path
    if candidate is None:
        for env_name in ("LABEL_ANYTHING_SAM2_DIR", "SAM_2_DIR", "SAM2_DIR"):
            raw_value = os.environ.get(env_name)
            if raw_value:
                candidate = Path(raw_value).expanduser()
                break

    if candidate is None:
        raise EnvironmentError("Set LABEL_ANYTHING_SAM2_DIR, SAM_2_DIR, or SAM2_DIR to a local SAM 2 checkout.")

    sam2_dir = candidate.expanduser()
    if not sam2_dir.exists():
        raise EnvironmentError(f"SAM 2 directory does not exist: {sam2_dir}")

    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))
    return sam2_dir


def _load_sam2_modules(sam2_dir: Path):
    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return build_sam2, SAM2ImagePredictor


def _resolve_model_spec(model_type: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_type]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported SAM 2 model type '{model_type}'. Expected one of: {valid}.") from exc


def _resolve_config_name(spec: ModelSpec, sam2_dir: Path) -> str:
    config_path = Path(spec.config_name)
    config_roots = [sam2_dir, sam2_dir / "sam2"]

    for root in config_roots:
        if (root / config_path).exists():
            return spec.config_name

    basename = config_path.name
    for root in config_roots:
        matches = sorted(root.rglob(basename))
        if not matches:
            continue
        match = matches[0]
        try:
            return match.relative_to(root).as_posix()
        except ValueError:
            continue

    raise FileNotFoundError(
        "SAM 2 config not found for "
        f"{spec.config_name!r} under {sam2_dir} or {sam2_dir / 'sam2'}."
    )


def _build_sam_model(model_type: str, sam2_dir: Path):
    spec = _resolve_model_spec(model_type)
    checkpoint = sam2_dir / "checkpoints" / spec.checkpoint
    config_name = _resolve_config_name(spec, sam2_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")

    build_sam2, _ = _load_sam2_modules(sam2_dir)
    return build_sam2(config_name, str(checkpoint), device=SAM_DEVICE)


def load_image_predictor(
    model_type: str = "base_plus",
    sam2_dir: Path | None = None,
    custom_weights: Path | None = None,
):
    resolved_sam2_dir = _resolve_sam2_dir(sam2_dir)
    _, sam2_image_predictor = _load_sam2_modules(resolved_sam2_dir)
    predictor = sam2_image_predictor(_build_sam_model(model_type, resolved_sam2_dir))

    if custom_weights is not None:
        weights_path = custom_weights.expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"Custom weights file not found: {weights_path}")
        map_location = "cpu" if SAM_DEVICE == "cpu" else None
        state_dict = torch.load(weights_path, map_location=map_location)
        predictor.model.load_state_dict(state_dict)

    return predictor


@contextmanager
def sam_inference_mode() -> Iterator[None]:
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield
