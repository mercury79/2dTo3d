"""
SBS viewer for Samsung Odyssey 3D — exclusive fullscreen, live controls.

Usage:
    python viewer.py <image.png> <depth.npy> [monitor] [disparity] [swap]
                     [splat] [convergence*100] [gamma*100]

Controls:
    Rueda del mouse / ↑ ↓     disparidad (intensidad 3D)
    Slider flotante 1         disparidad
    Slider flotante 2         pop-out (plano de convergencia)
    ← →                       imagen anterior / siguiente (galería)
    S                         invertir ojos
    ESC                       cerrar (sincroniza ajustes con la app)

Gallery protocol (viewer ↔ app, via temp files):
    viewer writes odyssey_nav_request.json {"seq": n, "dir": ±1}
    app re-computes depth for the next image, overwrites the tmp image/depth
    files and writes odyssey_ready.json {"seq": n, "name": "..."}
    viewer sees the matching seq and reloads.
"""
import sys
import os
import json
import time
import ctypes
import tempfile

import numpy as np

TARGET_W, TARGET_H = 3840, 2160
OVERLAY_SECONDS = 2.0

NAV_REQUEST = os.path.join(tempfile.gettempdir(), "odyssey_nav_request.json")
NAV_READY = os.path.join(tempfile.gettempdir(), "odyssey_ready.json")
RESULT_FILE = os.path.join(tempfile.gettempdir(), "odyssey_result_tmp.json")


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
        print("Usage: viewer.py <image> <depth.npy> [monitor] [disparity] "
              "[swap] [splat] [conv*100] [gamma*100]")
        sys.exit(1)

    img_path = sys.argv[1]
    depth_path = sys.argv[2]
    monitor_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    disparity = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    swap_eyes = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    use_splat = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
    convergence = (int(sys.argv[7]) / 100.0) if len(sys.argv) > 7 else 0.5
    gamma = (int(sys.argv[8]) / 100.0) if len(sys.argv) > 8 else 1.0
    use_inpaint = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False

    set_dpi_aware()
    ox, oy = get_monitor_offset(monitor_idx)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{ox},{oy}"
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    # clean stale protocol files
    for p in (NAV_REQUEST, NAV_READY, RESULT_FILE):
        try:
            os.remove(p)
        except OSError:
            pass

    import pygame
    from PIL import Image
    from stereo_builder import build_sbs

    if use_splat:
        try:
            from splat_renderer import build_sbs_splat, is_available
            if not is_available():
                use_splat = False
        except Exception:
            use_splat = False

    # ── source loading (letterbox to 16:9, aspect preserved) ────────────────
    state = {}

    def load_source():
        src = Image.open(img_path).convert("RGB")
        depth_full = np.load(depth_path)

        scale = min(TARGET_W / src.width, TARGET_H / src.height)
        nw, nh = round(src.width * scale), round(src.height * scale)
        fitted = src.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        pad_x, pad_y = (TARGET_W - nw) // 2, (TARGET_H - nh) // 2
        canvas.paste(fitted, (pad_x, pad_y))

        depth_img = Image.fromarray((depth_full * 255).astype(np.uint8), mode="L")
        depth_fitted = depth_img.resize((nw, nh), Image.BILINEAR)
        depth_canvas = Image.new("L", (TARGET_W, TARGET_H), 0)
        depth_canvas.paste(depth_fitted, (pad_x, pad_y))
        depth_lb = np.array(depth_canvas, dtype=np.float32) / 255.0

        canvas_half = canvas.resize((TARGET_W // 2, TARGET_H // 2), Image.BILINEAR)
        depth_half = np.array(
            Image.fromarray((depth_lb * 255).astype(np.uint8), mode="L")
            .resize((TARGET_W // 2, TARGET_H // 2), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        state["canvas"] = canvas
        state["depth"] = depth_lb
        state["canvas_half"] = canvas_half
        state["depth_half"] = depth_half

    load_source()

    def render_frame(disp: int, swap: bool, conv: float, fast: bool = False) -> bytes:
        if use_splat:
            # LaMa inpainting only on the final (slow) pass, not while dragging
            sbs = build_sbs_splat(state["canvas"], state["depth"],
                                  max_disparity=disp, swap_eyes=swap,
                                  convergence=conv, gamma=gamma,
                                  inpaint=use_inpaint and not fast)
        elif fast:
            sbs = build_sbs(state["canvas_half"], state["depth_half"],
                            max_disparity=disp // 2, swap_eyes=swap,
                            convergence=conv, gamma=gamma)
        else:
            sbs = build_sbs(state["canvas"], state["depth"],
                            max_disparity=disp, swap_eyes=swap,
                            convergence=conv, gamma=gamma)
        frame = sbs.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        return frame.tobytes()

    # ── pygame setup ─────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Segoe UI", 48, bold=True)
    font_sm = pygame.font.SysFont("Segoe UI", 34, bold=True)

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
    raw = render_frame(disparity, swap_eyes, convergence)
    surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")

    overlay_until = 0.0
    overlay_text = ""

    def show_overlay(text: str):
        nonlocal overlay_until, overlay_text
        overlay_text = text
        overlay_until = time.time() + OVERLAY_SECONDS

    # ── floating sliders (per half → read as ONE control in 3D) ──────────────
    HALF = TARGET_W // 2
    SL_W, SL_H = 1100, 14
    SL_X = (HALF - SL_W) // 2
    DISP_Y = TARGET_H - 130          # slider 1: disparity
    CONV_Y = TARGET_H - 240          # slider 2: pop-out
    SL_MAX = 150
    slider_until = 0.0
    drag_target = None               # None | "disp" | "conv"

    def value_from_x(mx: int, vmax: int) -> int:
        t = ((mx % HALF) - SL_X) / SL_W
        return max(0, min(vmax, round(t * vmax)))

    def slider_hit(mx: int, my: int):
        x_in_half = mx % HALF
        if not (SL_X - 40 <= x_in_half <= SL_X + SL_W + 40):
            return None
        if DISP_Y - 50 <= my <= DISP_Y + 50:
            return "disp"
        if CONV_Y - 50 <= my <= CONV_Y + 50:
            return "conv"
        return None

    # pop-out as 0..100 (100 = everything pops out → convergence 0)
    popout = round((1.0 - convergence) * 100)

    # ── gallery state ─────────────────────────────────────────────────────────
    nav_seq = 0
    waiting_nav = False
    nav_started = 0.0

    def request_nav(direction: int):
        nonlocal nav_seq, waiting_nav, nav_started
        nav_seq += 1
        try:
            with open(NAV_REQUEST, "w") as f:
                json.dump({"seq": nav_seq, "dir": direction}, f)
            waiting_nav = True
            nav_started = time.time()
            show_overlay("Cargando…")
        except Exception:
            pass

    def check_nav_ready():
        nonlocal waiting_nav
        if not waiting_nav:
            return False
        try:
            with open(NAV_READY) as f:
                ready = json.load(f)
            if ready.get("seq") == nav_seq:
                load_source()
                waiting_nav = False
                show_overlay(ready.get("name", ""))
                return True
        except Exception:
            pass
        if time.time() - nav_started > 30:
            waiting_nav = False
            show_overlay("No se pudo cargar")
        return False

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
                        try:
                            with open(RESULT_FILE, "w") as f:
                                json.dump({
                                    "disparity": disparity,
                                    "swap": swap_eyes,
                                    "convergence": 1.0 - popout / 100.0,
                                    "image": img_path,
                                }, f)
                        except Exception:
                            pass
                        pygame.quit()
                        return
                    elif event.key in (pygame.K_UP, pygame.K_PLUS):
                        disparity = min(SL_MAX, disparity + 4)
                        changed = True
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                        disparity = max(0, disparity - 4)
                        changed = True
                    elif event.key == pygame.K_RIGHT:
                        request_nav(+1)
                    elif event.key == pygame.K_LEFT:
                        request_nav(-1)
                    elif event.key == pygame.K_s:
                        swap_eyes = not swap_eyes
                        changed = True
                        show_overlay("Ojos invertidos" if swap_eyes else "Ojos normales")
                elif event.type == pygame.MOUSEWHEEL:
                    disparity = max(0, min(SL_MAX, disparity + event.y * 4))
                    changed = True
                elif event.type == pygame.MOUSEMOTION:
                    slider_until = time.time() + 3.0
                    if drag_target == "disp":
                        disparity = value_from_x(event.pos[0], SL_MAX)
                        changed = True
                    elif drag_target == "conv":
                        popout = value_from_x(event.pos[0], 100)
                        changed = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    drag_target = slider_hit(*event.pos)
                    if drag_target == "disp":
                        disparity = value_from_x(event.pos[0], SL_MAX)
                        changed = True
                    elif drag_target == "conv":
                        popout = value_from_x(event.pos[0], 100)
                        changed = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drag_target = None

            if check_nav_ready():
                changed = True

            if changed:
                last_input = time.time()
                slider_until = time.time() + 3.0
                rebuild_pending = True

            now = time.time()
            if rebuild_pending and now - last_input > 0.1:
                raw = render_frame(disparity, swap_eyes,
                                   1.0 - popout / 100.0, fast=True)
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
                rebuild_pending = False
                refine_at = now + 0.8
            if refine_at and now >= refine_at and not rebuild_pending:
                raw = render_frame(disparity, swap_eyes,
                                   1.0 - popout / 100.0, fast=False)
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
                refine_at = None

            screen.blit(surf, (0, 0))

            now2 = time.time()

            # ── floating sliders ───────────────────────────────────────────────
            if now2 < slider_until or drag_target:
                panel = pygame.Surface((SL_W + 120, 230))
                panel.set_alpha(170)
                panel.fill((10, 10, 16))
                sliders = [
                    ("3D", disparity, SL_MAX, DISP_Y, (137, 180, 250), font),
                    ("Pop-out", popout, 100, CONV_Y, (166, 227, 161), font_sm),
                ]
                for x_base in (0, HALF):
                    screen.blit(panel, (x_base + SL_X - 60, CONV_Y - 50))
                    for name, val, vmax, sy, color, fnt in sliders:
                        hx = SL_X + round(val / vmax * SL_W)
                        pygame.draw.rect(screen, (90, 95, 120),
                                         (x_base + SL_X, sy, SL_W, SL_H),
                                         border_radius=7)
                        pygame.draw.rect(screen, color,
                                         (x_base + SL_X, sy, hx - SL_X, SL_H),
                                         border_radius=7)
                        pygame.draw.circle(screen, (240, 240, 255),
                                           (x_base + hx, sy + SL_H // 2), 24)
                        lbl = fnt.render(f"{name}  {val}", True, (255, 255, 255))
                        screen.blit(lbl, (x_base + SL_X, sy - 44))

            elif now2 < overlay_until and overlay_text:
                lbl = font.render(overlay_text, True, (255, 255, 255))
                bg = pygame.Surface((lbl.get_width() + 40, lbl.get_height() + 20))
                bg.set_alpha(160)
                bg.fill((0, 0, 0))
                for x_base in (0, HALF):
                    cx = x_base + (HALF - bg.get_width()) // 2
                    cy = TARGET_H - 200
                    screen.blit(bg, (cx, cy))
                    screen.blit(lbl, (cx + 20, cy + 10))

            pygame.display.flip()
            clock.tick(60)

        except pygame.error:
            time.sleep(1.5)
            try:
                screen = open_window()
                surf = pygame.image.frombytes(raw, (TARGET_W, TARGET_H), "RGB")
            except Exception:
                time.sleep(1.0)


if __name__ == "__main__":
    main()
