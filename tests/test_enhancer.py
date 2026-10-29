"""API regression checks using small synthetic images and checkpoints only."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from lpsr import LPSREnhancer, __version__
from lpsr.inference import enhance, to_tensor
from lpsr.model import MFSR


class EnhancerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.weights = {}
        cls.models = {}
        torch.manual_seed(42)
        for scale in (2, 4):
            config = dict(scale=scale, c=16, n_enc=1, n_groups=1, n_blocks=1)
            model = MFSR(**config).eval()
            path = Path(cls.temp.name) / f"checkpoint_{scale}.pth"
            torch.save({"model_cfg": config, "mfsr": model.state_dict()}, path)
            cls.weights[scale], cls.models[scale] = path, model

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def crops(self):
        random = np.random.default_rng(42)
        return [random.integers(0, 256, (h, w, 3), dtype=np.uint8)
                for h, w in ((8, 16), (10, 18), (7, 15))]

    def test_load_once_safely_and_read_scale_from_checkpoint(self):
        with patch("lpsr.enhancer.torch.load", wraps=torch.load) as load:
            engine = LPSREnhancer(self.weights[4], device="cpu")
            engine.enhance(self.crops())
            engine.enhance(self.crops())
        load.assert_called_once_with(self.weights[4], map_location="cpu", weights_only=True)
        self.assertEqual(engine.scale, 4)
        self.assertFalse(engine._model.training)
        self.assertEqual(__version__, "0.3.0")

    def test_default_device_uses_cpu_when_cuda_is_unavailable(self):
        with patch("lpsr.enhancer.torch.cuda.is_available", return_value=False):
            engine = LPSREnhancer(self.weights[2])
        self.assertEqual(engine.device, torch.device("cpu"))

    def test_checkpoint_without_scale_keeps_the_network_x2_default(self):
        state = torch.load(self.weights[2], map_location="cpu", weights_only=True)
        del state["model_cfg"]["scale"]
        with patch("lpsr.enhancer.torch.load", return_value=state):
            engine = LPSREnhancer(self.weights[2], device="cpu")
        self.assertEqual(engine.scale, 2)
        self.assertEqual(engine.enhance(self.crops())[0].shape, (16, 32, 3))

    def test_auto_device_selects_cuda_but_explicit_cpu_is_respected(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        with patch("lpsr.enhancer.MFSR", return_value=model), patch("lpsr.enhancer.torch.cuda.is_available", return_value=True):
            automatic = LPSREnhancer(self.weights[2])
            explicit = LPSREnhancer(self.weights[2], device="cpu")
        self.assertEqual(automatic.device, torch.device("cuda"))
        self.assertEqual(explicit.device, torch.device("cpu"))
        self.assertEqual([entry.args[0] for entry in model.to.call_args_list], [torch.device("cuda"), torch.device("cpu")])

    def test_variable_size_outputs_match_reference_order_and_do_not_modify_inputs(self):
        crops = self.crops()
        originals = [crop.copy() for crop in crops]
        for scale in (2, 4):
            engine = LPSREnhancer(self.weights[scale], device="cpu")
            actual = engine.enhance(crops)
            self.assertEqual(len(actual), len(crops))
            for ref, result in enumerate(actual):
                h, w = crops[ref].shape[:2]
                reference = [crops[ref]] + [cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
                                           for index, crop in enumerate(crops) if index != ref]
                tensor = torch.from_numpy(np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                                    for crop in reference]).transpose(0, 3, 1, 2).astype(np.float32) / 255)[None]
                with torch.no_grad():
                    expected = self.models[scale](tensor)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
                expected = cv2.cvtColor((expected * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.shape, (h * scale, w * scale, 3))
                self.assertEqual(result.dtype, np.uint8)
        for before, after in zip(originals, crops):
            np.testing.assert_array_equal(before, after)

    def test_conversion_preserves_bgr_colors_and_disables_gradients(self):
        class IdentityReference(torch.nn.Module):
            def forward(self, sequence):
                self.grad_enabled = torch.is_grad_enabled()
                return sequence[:, 0]

        blue = np.zeros((4, 8, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        model = IdentityReference()
        np.testing.assert_array_equal(enhance(model, [blue], "cpu")[0], blue)
        self.assertFalse(model.grad_enabled)
        self.assertEqual(to_tensor([blue], "cpu").shape, (1, 1, 3, 4, 8))

    def test_invalid_inputs_are_rejected_before_inference(self):
        engine = LPSREnhancer(self.weights[2], device="cpu")
        for value in (None, np.zeros((8, 16, 3), dtype=np.uint8), [None], [np.zeros((8, 16, 3), dtype=np.float32)]):
            with self.subTest(value=type(value)), self.assertRaises(TypeError):
                engine.enhance(value)
        for value in ([], [np.zeros((8, 16), dtype=np.uint8)], [np.zeros((8, 16, 4), dtype=np.uint8)],
                      [np.zeros((0, 16, 3), dtype=np.uint8)]):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                engine.enhance(value)

    def test_missing_malformed_and_incompatible_weights_fail_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            LPSREnhancer(Path(self.temp.name) / "missing.pth")
        for state in ({}, {"model_cfg": {"scale": 3}, "mfsr": {}}, {"model_cfg": {"scale": 2}, "mfsr": []}):
            with patch("lpsr.enhancer.torch.load", return_value=state), self.assertRaises(ValueError):
                LPSREnhancer(self.weights[2], device="cpu")
        with patch("lpsr.enhancer.torch.load", return_value={"model_cfg": {"scale": 2}, "mfsr": {}}), self.assertRaises(RuntimeError):
            LPSREnhancer(self.weights[2], device="cpu")

    def test_unsafe_load_is_never_retried(self):
        with patch("lpsr.enhancer.torch.load", side_effect=RuntimeError("unsupported checkpoint")) as load:
            with self.assertRaisesRegex(RuntimeError, "unsupported checkpoint"):
                LPSREnhancer(self.weights[2], device="cpu")
        self.assertEqual(load.call_count, 1)
        self.assertTrue(load.call_args.kwargs["weights_only"])


if __name__ == "__main__":
    unittest.main()
