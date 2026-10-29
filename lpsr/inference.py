"""Array-only inference, independent of file handling and demo presentation."""
import cv2
import numpy as np
import torch


def to_tensor(frames, device):
    x = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).transpose(0, 3, 1, 2).astype(np.float32) / 255
    return torch.from_numpy(x)[None].to(device)


@torch.no_grad()
def enhance(model, crops, device):
    """Each frame in turn is the reference; the other frames are resized onto it (same margins)."""
    out = []
    for ref in range(len(crops)):
        h, w = crops[ref].shape[:2]
        seq = [crops[ref]] + [cv2.resize(crops[i], (w, h), interpolation=cv2.INTER_CUBIC)
                              for i in range(len(crops)) if i != ref]
        y = model(to_tensor(seq, device))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        out.append(cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
    return out
