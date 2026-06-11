"""
Build a Side-By-Side (SBS) stereo pair from an image + depth map.
"""
import numpy as np
from PIL import Image


def warp_image(arr: np.ndarray, shift_map: np.ndarray) -> np.ndarray:
    """
    Shift each column of `arr` horizontally by `shift_map` (per-pixel float).
    Positive shift → pixel moves right.
    """
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.int32)

    src_x = (xs[np.newaxis, :] - shift_map).clip(0, w - 1)

    x0 = np.floor(src_x).astype(np.int32).clip(0, w - 1)
    x1 = (x0 + 1).clip(0, w - 1)
    t = (src_x - x0)[..., np.newaxis] if channels > 1 else (src_x - x0)

    row_idx = ys[:, np.newaxis]

    if channels > 1:
        a = arr[row_idx, x0]   # (h, w, c)
        b = arr[row_idx, x1]
    else:
        a = arr[row_idx, x0]
        b = arr[row_idx, x1]

    warped = ((1 - t) * a + t * b).clip(0, 255).astype(np.uint8)
    return warped


def build_sbs(
    img: Image.Image,
    depth: np.ndarray,
    max_disparity: int = 30,
    swap_eyes: bool = False,
) -> Image.Image:
    """
    Returns a Side-By-Side stereo image (left|right).

    max_disparity: maximum horizontal pixel shift (controls 3D depth intensity).
    swap_eyes: set True if your display needs right-left order instead of left-right.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # depth map resized to match image
    depth_resized = np.array(
        Image.fromarray(depth).resize((w, h), Image.BILINEAR)
    )

    # Convergence plane at mid-depth: nearer-than-median pops OUT of the
    # screen, farther sinks INTO it. Without this everything shifts one
    # way and the 3D feels flat/uncomfortable.
    half_disp = ((depth_resized - 0.5) * max_disparity).astype(np.float32)

    # left eye  → shift right  (+half)
    # right eye → shift left   (-half)
    left_arr = warp_image(arr.astype(np.uint8), -half_disp)
    right_arr = warp_image(arr.astype(np.uint8), half_disp)

    left_img = Image.fromarray(left_arr)
    right_img = Image.fromarray(right_arr)

    # --- assemble SBS ---
    sbs = Image.new("RGB", (w * 2, h))
    if swap_eyes:
        sbs.paste(right_img, (0, 0))
        sbs.paste(left_img, (w, 0))
    else:
        sbs.paste(left_img, (0, 0))
        sbs.paste(right_img, (w, 0))

    return sbs
