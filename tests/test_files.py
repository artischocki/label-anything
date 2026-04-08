from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_anything.files import find_existing_mask, label_output_name, strip_exact_suffix


class FileHelpersTest(unittest.TestCase):
    def test_strip_exact_suffix_only_removes_real_suffix(self) -> None:
        self.assertEqual(strip_exact_suffix("image_raw", "_raw"), "image")
        self.assertEqual(strip_exact_suffix("drawer", "_raw"), "drawer")

    def test_label_output_name_uses_exact_raw_suffix(self) -> None:
        self.assertEqual(label_output_name(Path("sample_raw.png")), "sample_label.png")
        self.assertEqual(label_output_name(Path("radar.png")), "radar_label.png")

    def test_find_existing_mask_prefers_canonical_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            image_path = temp_path / "example_raw.png"
            image_path.touch()
            canonical_mask = temp_path / "example_label.png"
            canonical_mask.touch()

            self.assertEqual(find_existing_mask(image_path), canonical_mask)

    def test_find_existing_mask_falls_back_to_legacy_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            image_path = temp_path / "example.png"
            image_path.touch()
            legacy_mask = temp_path / "example_label.png"
            legacy_mask.touch()

            self.assertEqual(find_existing_mask(image_path), legacy_mask)


if __name__ == "__main__":
    unittest.main()
