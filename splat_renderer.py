"""
GPU point-splatting stereo renderer (torch/CUDA).

Instead of column-shifting (which smears object edges — "ghosting"),
each pixel is un-projected with its depth, forward-splatted into each
eye's view with a z-buffer (near occludes far), and dis-occlusion holes
are filled from the background side. Same interface as
stereo_builder.build_sbs so callers can switch freely.
"""
import numpy as np
from PIL import Image

_torch = None
_device = None


def _init():
    global _torch, _device
    if _torch is None:
        import torch
        _torch = torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _torch, _device


def is_available() -> bool:
    try:
        torch, device = _init()
        return device.type == "cuda"
    except Exception:
        return False


def _splat_eye(color, depth, shift, torch, device):
    """
    Forward-splat pixels shifted horizontally by `shift`, z-buffered.
    color: (H,W,3) float; depth: (H,W) [0..1] 1=near; shift: (H,W) px.
    Returns (out (H,W,3), out_depth (H,W), valid (H,W) bool).
    """
    H, W, _ = color.shape
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)

    xt = torch.round(xs + shift).long().clamp(0, W - 1)
    tgt = (ys * W + xt).reshape(-1)
    d = depth.reshape(-1)

    # Pass 1: z-buffer — keep the nearest depth landing on each target pixel
    zbuf = torch.full((H * W,), -1.0, device=device)
    zbuf.scatter_reduce_(0, tgt, d, reduce="amax", include_self=True)

    # Pass 2: write color only for pixels that won the z-test
    win = d >= zbuf[tgt] - 1e-4
    out = torch.zeros(H * W, 3, device=device)
    out_d = torch.full((H * W,), -1.0, device=device)
    idx = tgt[win]
    out[idx] = color.reshape(-1, 3)[win]
    out_d[idx] = d[win]

    valid = out_d >= 0
    return out.view(H, W, 3), out_d.view(H, W), valid.view(H, W)


def _fill_holes(out, out_d, valid, torch, device):
    """
    Fill dis-occlusion holes from the BACKGROUND side: for each hole, look
    at the nearest valid pixel left and right and copy the one that is
    farther away (background), which is what would be revealed behind an
    object — never smear the foreground.
    """
    H, W, _ = out.shape
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)
    rows = torch.arange(H, device=device).view(H, 1).expand(H, W)

    neg = torch.full_like(xs, -1)
    li = torch.cummax(torch.where(valid, xs, neg), dim=1).values
    ri_rev = torch.cummax(torch.where(valid.flip(1), xs, neg), dim=1).values
    ri = (W - 1) - ri_rev.flip(1)          # -1 on a side with no valid pixel → W

    li_ok = li >= 0
    ri_ok = ri <= W - 1
    li_c = li.clamp(0, W - 1)
    ri_c = ri.clamp(0, W - 1)

    dl = out_d[rows, li_c]
    dr = out_d[rows, ri_c]

    use_left = dl <= dr                     # smaller depth = farther = background
    use_left = torch.where(~ri_ok, torch.ones_like(use_left), use_left)
    use_left = torch.where(~li_ok, torch.zeros_like(use_left), use_left)

    src_x = torch.where(use_left, li_c, ri_c)
    filled = out[rows, src_x]
    return torch.where(valid.unsqueeze(-1), out, filled)


def build_sbs_splat(
    img: Image.Image,
    depth: np.ndarray,
    max_disparity: int = 30,
    swap_eyes: bool = False,
) -> Image.Image:
    """Drop-in replacement for stereo_builder.build_sbs using GPU splatting."""
    torch, device = _init()

    arr = torch.from_numpy(
        np.array(img.convert("RGB"), dtype=np.float32)
    ).to(device)
    H, W = arr.shape[0], arr.shape[1]

    d = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))
    if d.shape != (H, W):
        d = torch.nn.functional.interpolate(
            d[None, None].to(device), size=(H, W),
            mode="bilinear", align_corners=False,
        )[0, 0]
    else:
        d = d.to(device)

    half = (d - 0.5) * float(max_disparity)

    eyes = []
    for sign in (-1.0, 1.0):                # left eye, right eye
        out, out_d, valid = _splat_eye(arr, d, sign * half, torch, device)
        out = _fill_holes(out, out_d, valid, torch, device)
        eyes.append(out.clamp(0, 255).byte().cpu().numpy())

    left, right = eyes
    if swap_eyes:
        left, right = right, left
    return Image.fromarray(np.concatenate([left, right], axis=1))
