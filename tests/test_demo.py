"""The demo keeps cropping/output responsibilities while delegating inference."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import enhance_sequence as demo
from lpsr import LPSREnhancer
from lpsr.model import MFSR


class DemoTests(unittest.TestCase):
    def test_crop_margins_and_border_clipping_are_unchanged(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertEqual(demo.crop_plate(image, (50, 20, 90, 40)).shape, (40, 60, 3))
        self.assertEqual(demo.crop_plate(image, (0, 0, 40, 20)).shape, (30, 50, 3))

    def test_demo_reads_scale_from_weights_and_writes_the_api_results(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "output"
            source.mkdir()
            config = dict(scale=4, c=16, n_enc=1, n_groups=1, n_blocks=1)
            weights = root / "arbitrary_filename.pth"
            torch.save({"model_cfg": config, "mfsr": MFSR(**config).state_dict()}, weights)
            crops = [np.full((8, 16, 3), color, dtype=np.uint8) for color in (30, 180)]
            for index, crop in enumerate(crops):
                cv2.imwrite(str(source / f"{index:02d}.png"), crop)
            args = ["enhance_sequence.py", "--input", str(source), "--output", str(output),
                    "--weights", str(weights), "--device", "cpu", "--no_sheet"]
            with patch("sys.argv", args), redirect_stdout(io.StringIO()):
                demo.main()
            expected = LPSREnhancer(weights, device="cpu").enhance(crops)
            for index, image in enumerate(expected):
                np.testing.assert_array_equal(cv2.imread(str(output / f"plaque_{index:02d}_amelioree_x4.png")), image)
                self.assertTrue((output / f"plaque_{index:02d}_originale.png").is_file())
                self.assertTrue((output / f"plaque_{index:02d}_bicubique_x4.png").is_file())
            self.assertFalse((output / "planche.png").exists())
            with patch("sys.argv", args + ["--scale", "2"]), self.assertRaisesRegex(SystemExit, "modèle x4"):
                demo.main()


if __name__ == "__main__":
    unittest.main()
