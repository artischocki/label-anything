from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

BoolMask = npt.NDArray[np.bool_]


def empty_mask(size: tuple[int, int]) -> BoolMask:
    width, height = size
    return np.zeros((height, width), dtype=bool)


def load_rgb_image(image_path: str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]", rgb_to_bgr: bool = False) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    if rgb_to_bgr:
        blue, green, red = image.split()
        image = Image.merge("RGB", (red, green, blue))

    return image, image.size


def load_binary_mask(mask_path: str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]", size: tuple[int, int]) -> BoolMask:
    with Image.open(mask_path) as mask_image:
        resized_mask = mask_image.convert("L").resize(size)
    return np.asarray(resized_mask) > 128


def mask_to_overlay(mask: BoolMask, color: tuple[int, int, int], alpha: int) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.fromarray(rgba, "RGBA")


def draw_mask_output(mask: BoolMask, output_size: tuple[int, int]) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (255, 255, 255)
    image = Image.fromarray(rgb, "RGB")
    if image.size != output_size:
        image = image.resize(output_size)
    return image


def apply_morphology(mask: BoolMask, operation: int, kernel_size: int = 5) -> BoolMask:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    morphed = cv2.morphologyEx(mask_uint8, operation, kernel)
    return morphed > 0


def strokes_to_mask(
    strokes: Iterable[Image.Image],
    size: tuple[int, int],
    channel_index: int,
) -> BoolMask:
    composite = Image.new("RGBA", size)
    for stroke in strokes:
        composite.paste(stroke, (0, 0), stroke)
    return np.asarray(composite)[:, :, channel_index] > 0


def remove_from_mask(mask: BoolMask, eraser_mask: BoolMask) -> BoolMask:
    return np.logical_and(mask, np.logical_not(eraser_mask))
