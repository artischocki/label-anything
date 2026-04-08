from __future__ import annotations

from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = ("bmp", "png", "jpg", "jpeg")


def strip_exact_suffix(value: str, suffix: str) -> str:
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def list_image_files(directory: Path) -> list[Path]:
    image_files: list[Path] = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        image_files.extend(directory.glob(f"*.{extension}"))
        image_files.extend(directory.glob(f"*.{extension.upper()}"))
    return sorted(
        [path for path in image_files if not path.name.endswith("_label.png")],
        key=lambda path: path.name.lower(),
    )


def label_output_name(image_path: Path) -> str:
    label_stem = strip_exact_suffix(image_path.stem, "_raw")
    return f"{label_stem}_label.png"


def find_existing_mask(image_path: Path) -> Path | None:
    canonical_mask = image_path.with_name(label_output_name(image_path))
    if canonical_mask.exists():
        return canonical_mask

    legacy_stems = {image_path.stem, strip_exact_suffix(image_path.stem, "_raw")}
    for candidate in sorted(image_path.parent.glob("*_label.png")):
        candidate_stem = strip_exact_suffix(candidate.stem, "_label")
        if candidate_stem in legacy_stems:
            return candidate
    return None


def results_dir(base_dir: Path) -> Path:
    target = base_dir / "results"
    target.mkdir(exist_ok=True)
    return target


def final_image_output_path(image_path: Path) -> Path:
    return results_dir(image_path.parent) / image_path.name


def final_mask_output_path(image_path: Path) -> Path:
    return results_dir(image_path.parent) / label_output_name(image_path)
