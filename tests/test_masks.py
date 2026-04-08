from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_anything.masks import draw_mask_output, empty_mask, remove_from_mask, strokes_to_mask


class MaskHelpersTest(unittest.TestCase):
    def test_empty_mask_uses_width_height_order(self) -> None:
        mask = empty_mask((7, 3))
        self.assertEqual(mask.shape, (3, 7))
        self.assertFalse(mask.any())

    def test_draw_mask_output_resizes_back_to_original_size(self) -> None:
        mask = empty_mask((2, 2))
        mask[0, 0] = True
        output = draw_mask_output(mask, (4, 4))
        self.assertEqual(output.size, (4, 4))

    def test_strokes_to_mask_reads_requested_channel(self) -> None:
        stroke = Image.new("RGBA", (10, 10))
        draw = ImageDraw.Draw(stroke)
        draw.rectangle((2, 2, 5, 5), fill=(0, 255, 0, 128))

        mask = strokes_to_mask([stroke], (10, 10), channel_index=1)
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[0, 0])

    def test_remove_from_mask_only_clears_erased_pixels(self) -> None:
        mask = np.array([[True, True], [False, True]])
        eraser = np.array([[False, True], [True, False]])
        result = remove_from_mask(mask, eraser)
        self.assertTrue(np.array_equal(result, np.array([[True, False], [False, True]])))


if __name__ == "__main__":
    unittest.main()
