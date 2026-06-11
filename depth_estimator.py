"""
Depth estimation from a single 2D image.
Uses gradient-magnitude + frequency analysis heuristics.
Optionally uses Depth Anything v2 if transformers+torch are available.
"""
import numpy as np
from PIL import Image, ImageFilter


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _to_pil_l(arr: np.ndarray) -> Image.Image:
    """Convert a float array [0..1] to a PIL L-mode image."""
    return Image.fromarray((_normalize(arr) * 255).astype(np.uint8), mode="L")


def _blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur on a float [0..1] array, returns float [0..1]."""
    r = max(1, radius)
    blurred = _to_pil_l(arr).filter(ImageFilter.GaussianBlur(radius=r))
    return np.array(blurred, dtype=np.float32) / 255.0


def estimate_depth_heuristic(img: Image.Image) -> np.ndarray:
    """
    Fast heuristic depth map.
    Returns float32 array [0..1] where 1 = closer / in focus.
    """
    gray = img.convert("L")
    w, h = gray.size

    arr = np.array(gray, dtype=np.float32) / 255.0

    # --- gradient magnitude (sharp edges → closer) ---
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_smooth = _normalize(_blur(grad, max(1, min(w, h) // 40)))

    # --- high-frequency local contrast ---
    blurred_arr = _blur(arr, max(1, min(w, h) // 20))
    hf = np.abs(arr - blurred_arr)
    hf_smooth = _normalize(_blur(hf, max(1, min(w, h) // 30)))

    # --- vertical bias: lower in frame tends to be closer ---
    v_bias = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    v_bias = np.broadcast_to(v_bias, (h, w)).copy()

    depth = 0.45 * grad_smooth + 0.30 * hf_smooth + 0.25 * v_bias

    # final smooth
    depth = _normalize(_blur(depth, max(2, min(w, h) // 60)))
    return depth


_ml_pipe = None  # cached pipeline — model loads once, reused per image


def estimate_depth_ml(img: Image.Image):
    """
    Use Depth Anything V2 if transformers + torch are available.
    Returns float32 [0..1] with 1 = closer, or None on failure.
    """
    global _ml_pipe
    try:
        import torch
        from transformers import pipeline

        if _ml_pipe is None:
            has_gpu = torch.cuda.is_available()
            # Large on GPU (best quality); Small on CPU (speed)
            model = (
                "depth-anything/Depth-Anything-V2-Large-hf"
                if has_gpu
                else "depth-anything/Depth-Anything-V2-Small-hf"
            )
            dtype = torch.float16 if has_gpu else torch.float32
            try:
                _ml_pipe = pipeline(
                    task="depth-estimation", model=model,
                    device=0 if has_gpu else -1, dtype=dtype,
                )
            except TypeError:
                # older transformers use torch_dtype
                _ml_pipe = pipeline(
                    task="depth-estimation", model=model,
                    device=0 if has_gpu else -1, torch_dtype=dtype,
                )

        result = _ml_pipe(img)
        depth_arr = np.array(result["depth"], dtype=np.float32)

        # Depth Anything outputs higher = closer (inverse depth) in the
        # HF pipeline's rendered map; verify orientation by convention:
        # the pipeline returns disparity-like maps where bright = near.
        depth = _normalize(depth_arr)

        # Gentle blur to avoid hard depth edges → reduces warp artifacts
        w, h = img.size
        depth = _blur(depth, max(1, min(w, h) // 200))
        return _normalize(depth)
    except Exception as exc:
        global last_ml_error
        last_ml_error = str(exc)
        return None


last_ml_error: str | None = None


def refine_depth_edges(img: Image.Image, depth: np.ndarray) -> np.ndarray:
    """
    Joint bilateral filter: smooth the depth map using the IMAGE as guide,
    so depth edges snap to real color edges (hair, leaves, silhouettes).
    Runs on GPU if torch+CUDA available; otherwise returns depth unchanged.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return depth
        device = torch.device("cuda")

        H, W = depth.shape
        # work at <=1536 px wide to bound memory; upsample result after
        work_w = min(1536, W)
        work_h = round(H * work_w / W)

        guide = torch.from_numpy(
            np.array(img.convert("RGB").resize((work_w, work_h), Image.BILINEAR),
                     dtype=np.float32) / 255.0
        ).to(device)                                            # (h,w,3)
        d = torch.nn.functional.interpolate(
            torch.from_numpy(depth)[None, None].to(device),
            size=(work_h, work_w), mode="bilinear", align_corners=False,
        )[0, 0]                                                  # (h,w)

        K = 9                       # kernel size
        R = K // 2
        sigma_s = 3.0               # spatial sigma (px)
        sigma_c = 0.08              # color sigma (0..1 scale)

        guide_p = torch.nn.functional.pad(
            guide.permute(2, 0, 1)[None], (R, R, R, R), mode="reflect")[0]
        d_p = torch.nn.functional.pad(
            d[None, None], (R, R, R, R), mode="reflect")[0, 0]

        num = torch.zeros_like(d)
        den = torch.zeros_like(d)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                w_s = float(np.exp(-(dx * dx + dy * dy) / (2 * sigma_s ** 2)))
                g_n = guide_p[:, R + dy:R + dy + work_h, R + dx:R + dx + work_w]
                d_n = d_p[R + dy:R + dy + work_h, R + dx:R + dx + work_w]
                w_c = torch.exp(
                    -((g_n - guide.permute(2, 0, 1)) ** 2).sum(0)
                    / (2 * sigma_c ** 2)
                )
                w = w_s * w_c
                num += w * d_n
                den += w

        out = num / den.clamp_min(1e-6)
        out = torch.nn.functional.interpolate(
            out[None, None], size=(H, W), mode="bilinear", align_corners=False,
        )[0, 0]
        return out.clamp(0, 1).cpu().numpy().astype(np.float32)
    except Exception:
        return depth


def estimate_depth(img: Image.Image, use_ml: bool = False) -> np.ndarray:
    if use_ml:
        ml = estimate_depth_ml(img)
        if ml is not None:
            return refine_depth_edges(img, ml)
    return refine_depth_edges(img, estimate_depth_heuristic(img))
