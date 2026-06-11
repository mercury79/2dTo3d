"""
LaMa inpainting (TorchScript big-lama) for dis-occlusion holes.

Loads models/big-lama.pt directly with torch.jit — no extra packages.
Used by splat_renderer as a higher-quality alternative to lateral
background fill when generating stereo views.
"""
import os
import numpy as np

_model = None
_device = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "big-lama.pt")


def is_available() -> bool:
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load():
    global _model, _device
    if _model is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = torch.jit.load(MODEL_PATH, map_location=_device)
        _model.eval()
    return _model, _device


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image: (H,W,3) uint8 RGB.  mask: (H,W) bool/uint8, True = hole to fill.
    Returns (H,W,3) uint8 with holes filled.
    """
    import torch

    model, device = _load()
    H, W = mask.shape

    # pad to multiple of 8 (LaMa requirement)
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    img_t = torch.from_numpy(image.astype(np.float32) / 255.0) \
        .permute(2, 0, 1)[None].to(device)
    mask_t = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(device)

    if pad_h or pad_w:
        img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
        mask_t = torch.nn.functional.pad(mask_t, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        out = model(img_t, mask_t)

    result = out[0].permute(1, 2, 0)[:H, :W].clamp(0, 255)
    return result.byte().cpu().numpy()
