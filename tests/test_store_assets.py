import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "extension" / "assets" / "icons" / "icon-128.png"
STORE_ASSETS = ROOT / "store-assets"


class StoreAssetTests(unittest.TestCase):
    def test_icon_uses_solid_lower_right_signal_dot(self):
        with Image.open(ICON) as image:
            self.assertEqual(image.size, (128, 128))
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((76, 82)), (49, 95, 140, 255))
            self.assertNotEqual(image.getpixel((101, 96)), (49, 95, 140, 255))

    def test_store_assets_are_store_ready(self):
        for name, size in (
            ("screenshot-1280x800.png", (1280, 800)),
            ("small-promo-440x280.png", (440, 280)),
            ("marquee-promo-1400x560.png", (1400, 560)),
        ):
            with self.subTest(name=name):
                with Image.open(STORE_ASSETS / name) as image:
                    self.assertEqual(image.size, size)
                    self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
