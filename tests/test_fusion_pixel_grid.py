import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fusion import TTDataset


CAMERA_TEXT = """extrinsic
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
100 0 2
0 100 2
0 0 1

1 0.01 192 2.92
"""


class FusionPixelGridTest(unittest.TestCase):
    def test_mismatched_image_and_depth_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            for name in (
                "images_test",
                "cams_test",
                "depth_est",
                "confidence",
            ):
                (scene / name).mkdir()

            (scene / "pair.txt").write_text(
                "1\n0\n1 1 1.0\n",
                encoding="utf-8",
            )
            Image.fromarray(
                np.zeros((4, 5, 3), dtype=np.uint8)
            ).save(scene / "images_test" / "00000000.jpg")
            (scene / "cams_test" / "00000000_cam.txt").write_text(
                CAMERA_TEXT,
                encoding="utf-8",
            )
            np.save(
                scene / "depth_est" / "00000000.npy",
                np.ones((2, 3), dtype=np.float32),
            )
            np.save(
                scene / "confidence" / "00000000.npy",
                np.ones((2, 3), dtype=np.float32),
            )

            dataset = TTDataset(
                str(scene),
                "depth_est",
                "confidence",
                n_src_views=1,
            )
            with self.assertRaisesRegex(
                ValueError,
                "must share one pixel grid",
            ):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
