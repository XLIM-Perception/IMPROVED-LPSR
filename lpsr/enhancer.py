"""Reusable model instance for a sequence of BGR uint8 plate crops."""
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .inference import enhance
from .model import MFSR


class LPSREnhancer:
    """Load weights once, then enhance sequences without file or display I/O.

    ``device=None`` selects CUDA when available, otherwise CPU. The scale is
    read from the checkpoint's ``model_cfg`` rather than supplied by the caller.
    Cropping, including the training margins, is the caller's responsibility.
    """

    def __init__(self, weights_path, device=None):
        self.weights_path = Path(weights_path).expanduser()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not isinstance(state.get("model_cfg"), Mapping) or not isinstance(state.get("mfsr"), Mapping):
            raise ValueError("Expected a checkpoint containing 'model_cfg' and 'mfsr' mappings")
        config = dict(state["model_cfg"])
        # The released x2 checkpoint omits scale, using MFSR's x2 default.
        scale = config.get("scale", 2)
        if type(scale) is not int or scale not in (2, 4):
            raise ValueError("Checkpoint model_cfg.scale must be 2 or 4")
        model = MFSR(**config)
        model.load_state_dict(state["mfsr"], strict=True)
        self._model = model.to(self.device).eval()
        self._scale = scale

    @property
    def scale(self):
        """Output enlargement factor (2 or 4), read from the weights."""
        return self._scale

    def enhance(self, crops):
        """Return one BGR uint8 image per crop, in the same order.

        ``crops`` is a nonempty list or tuple of H×W×3 NumPy uint8 arrays of the
        same plate. Sizes may differ. Inputs are not modified. Every crop is
        used as a reference in turn, assisted by the rest of the sequence.
        """
        if not isinstance(crops, (list, tuple)):
            raise TypeError("crops must be a list or tuple of BGR uint8 arrays")
        if not crops:
            raise ValueError("crops must contain at least one image")
        for index, crop in enumerate(crops):
            if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8:
                raise TypeError(f"crops[{index}] must be a NumPy uint8 array")
            if crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
                raise ValueError(f"crops[{index}] must be a nonempty HxWx3 BGR image")
        return enhance(self._model, crops, self.device)
