"""
SBS viewer for Samsung Odyssey 3D — hybrid exclusive mode + live disparity.

Usage:
    python viewer.py <image.png> <depth.npy> <monitor_idx> <disparity> <swap:0|1>

Phase 1: exclusive fullscreen (3840x2160) → triggers Odyssey 3D Hub detection.
Phase 2: if surface is lost during the Hub's mode switch, re-open exclusive.

Live controls:
    Rueda del mouse / ↑ ↓   ajustar disparidad (intensidad 3D)
    S                       invertir ojos
    ESC                     cerrar
"""
import sys
import os
import time
import ctypes

import numpy as np

TARGET_W, TARGET_H = 3840, 2160
OVERLAY_SECONDS = 2.0


def set_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def get_monitor_offset(monitor_idx: int):
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
        if monitor_idx < len(monitors):
            return monitors[monitor_idx].x, monitors[monitor_idx].y
    except Exception:
        pass
    return 0, 0


def main():
    if len(sys.argv) < 3:
        print("Usage: viewer.py <image> <depth.npy> [monitor] [disparity] [swap]")
        sys.exit(1)

    img_path = sys.argv[1]
    depth_path = sys.argv[2]
    monitor_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    disparity = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    swap_eyes = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    set_dpi_aware()
    ox, oy = get_monitor_offset(monitor_idx)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{ox},{oy}"
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    import pygame
    from PIL import Image
    from stereo_builder import build_sbs

    # ── prepare letterboxed source (16:9 canvas, aspect preserved) ──────────
    src = Image.open(img_path).convert("RGB")
    depth_full = np.load(depth_path)  # float32 [0..1], same size as src

    scale = min(TARGET_W / src.width, TARGET_H / src.height)
    nw, nh = round(src.width * scale), round(src.height * scale)
    fitted = src.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
    pad_x, pad_y = (TARGET_W - nw) // 2, (TARGET_H - nh) // 2
    canvas.paste(fitted, (pad_x, pad_y))

    # depth letterboxed the same way (background = far = 0)
    depth_img = Image.fromarray((depth_full * 255).astype(np.uint8), mode="L")
    depth_fitted = depth_img.resize((nw, nh), Image.BILINEAR)
    depth_canvas = Image.new("L", (TARGET_W, TARGET_H), 0)
    depth_canvas.paste(depth_fitted, (pad_x, pad_y))
    depth_lb = np.array(depth_canvas, dtype=np.float32) / 255.0

    # Half-res working copies for fast preview while adjusting
    canvas_half = canvas.resize((TARGET_W // 2, TARGET_H // 2), Image.BILINEAR)
    depth_half = np.array(
        Image.fromarray((depth_lb * 255).astype(np.uint8), mode="L")
        .resize((TARGET_W // 2, TARGET_H // 2), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    def render_frame(disp: int, swap: bool, fast: bool = False) -> bytes:
        """Build half-SBS frame at TARGET resolution, return raw RGB bytes."""
        if fast:
            # quarter the pixels → ~4x faster; upscaled for display
            sbs = build_sbs(canvas_half, depth_half,
                            max_disparity=disp // 2, swap_eyes=swap)
        else:
            sbs = build_sbs(canvas, depth_lb, max_disparity=disp, swap_eyes=swap)
        # full-SBS → squeeze to half-SBS at display size
        frame = sbs.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        return frame.tobytes()

    # ── pygame setup ─────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Segoe UI", 48, bold=True)

    def open_window():
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode(
            (TARGET_W, TARGET_H),
            pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
            display=monitor_idx,
        )
        pygame.display.set_caption("SBS 3D Viewer")
        return screen

    screen = open_window()
    raw = render_frame(disparity, swap_eyes)
    surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")

    overlay_until = 0.0
    overlay_text = ""

    def show_overlay(text: str):
        nonlocal overlay_until, overlay_text
        overlay_text = text
        overlay_until = time.time() + OVERLAY_SECONDS

    rebuild_pending = False
    refine_at = None
    last_input = 0.0

    clock = pygame.time.Clock()
    while True:
        try:
            changed = False
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_PLUS):
                        disparity = min(150, disparity + 4)
                        changed = True
                    elif event.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                        disparity = max(0, disparity - 4)
                        changed = True
                    elif event.key == pygame.K_s:
                        swap_eyes = not swap_eyes
                        changed = True
                        show_overlay("Ojos invertidos" if swap_eyes else "Ojos normales")
                elif event.type == pygame.MOUSEWHEEL:
                    disparity = max(0, min(150, disparity + event.y * 4))
                    changed = True

            if changed:
                last_input = time.time()
                rebuild_pending = True
                if not overlay_text.startswith("Ojos"):
                    show_overlay(f"Disparidad: {disparity}")
                else:
                    overlay_until = time.time() + OVERLAY_SECONDS

            # Two-stage rebuild: fast half-res preview right away,
            # full-res refinement once input settles for 0.8s.
            now = time.time()
            if rebuild_pending and now - last_input > 0.1:
                raw = render_frame(disparity, swap_eyes, fast=True)
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
                rebuild_pending = False
                refine_at = now + 0.8
            if refine_at and now >= refine_at and not rebuild_pending:
                raw = render_frame(disparity, swap_eyes, fast=False)
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
                refine_at = None

            screen.blit(surf, (0, 0))

            # Floating overlay — drawn on BOTH halves so it reads in 3D
            if time.time() < overlay_until and overlay_text:
                label = font.render(overlay_text, True, (255, 255, 255))
                bg = pygame.Surface((label.get_width() + 40, label.get_height() + 20))
                bg.set_alpha(160)
                bg.fill((0, 0, 0))
                half = TARGET_W // 2
                for x_base in (0, half):
                    cx = x_base + (half - bg.get_width()) // 2
                    cy = TARGET_H - 200
                    screen.blit(bg, (cx, cy))
                    screen.blit(label, (cx + 20, cy + 10))

            pygame.display.flip()
            clock.tick(60)

        except pygame.error:
            # Surface lost during Hub's 3D mode switch — re-open exclusive
            time.sleep(1.5)
            try:
                screen = open_window()
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
            except Exception:
                time.sleep(1.0)


if __name__ == "__main__":
    main()
