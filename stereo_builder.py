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


def apply_depth_curve(depth: np.ndarray, gamma: float) -> np.ndarray:
    """
    Non-linear depth remap. gamma > 1 expands separation between the
    foreground subject and the mid-ground while compressing the far
    background — closer to how human stereo perception works.
    """
    if abs(gamma - 1.0) < 1e-3:
        return depth
    return np.power(depth.clip(0.0, 1.0), gamma).astype(np.float32)


def build_sbs(
    img: Image.Image,
    depth: np.ndarray,
    max_disparity: int = 30,
    swap_eyes: bool = False,
    convergence: float = 0.5,
    gamma: float = 1.0,
) -> Image.Image:
    """
    Returns a Side-By-Side stereo image (left|right).

    max_disparity: maximum horizontal pixel shift (3D intensity).
    convergence:   depth value [0..1] that sits exactly AT the screen plane.
                   Lower → more of the scene pops out; higher → sinks behind.
    gamma:         non-linear depth curve (see apply_depth_curve).
    swap_eyes:     True if your display needs right-left order.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # depth map resized to match image
    depth_resized = np.array(
        Image.fromarray(depth).resize((w, h), Image.BILINEAR)
    )
    depth_resized = apply_depth_curve(depth_resized, gamma)

    # Convergence plane: depth == convergence renders at the screen plane;
    # nearer pops OUT of the screen, farther sinks INTO it.
    half_disp = ((depth_resized - convergence) * max_disparity).astype(np.float32)

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
