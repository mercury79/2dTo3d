"""
2D → 3D SBS Converter
Converts 2D photos to Side-By-Side 3D format and shows them fullscreen
for Samsung (or any passive/active) 3D monitors.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Register HEIC/HEIF support if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except ImportError:
    _HEIC_OK = False

from depth_estimator import estimate_depth
from stereo_builder import build_sbs

try:
    from splat_renderer import build_sbs_splat, is_available as _splat_available
    _SPLAT_OK = _splat_available()
except Exception:
    _SPLAT_OK = False


try:
    from lama_inpaint import is_available as _lama_available
    _LAMA_OK = _lama_available()
except Exception:
    _LAMA_OK = False


def build_stereo(img, depth, max_disparity, swap_eyes, use_splat,
                 convergence=0.5, gamma=1.0, inpaint=False):
    """Route to GPU splatting or CPU column warp."""
    if use_splat and _SPLAT_OK:
        try:
            return build_sbs_splat(img, depth, max_disparity=max_disparity,
                                   swap_eyes=swap_eyes,
                                   convergence=convergence, gamma=gamma,
                                   inpaint=inpaint and _LAMA_OK)
        except Exception:
            pass
    return build_sbs(img, depth, max_disparity=max_disparity,
                     swap_eyes=swap_eyes,
                     convergence=convergence, gamma=gamma)

SUPPORTED = (
    ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.heic"),
    ("Todos los archivos", "*.*"),
)

# ── globals ────────────────────────────────────────────────────────────────────
current_sbs: Image.Image | None = None
current_img: Image.Image | None = None     # original photo
current_depth = None                       # np.ndarray float32 [0..1]
fullscreen_win: tk.Toplevel | None = None


# ── helpers ────────────────────────────────────────────────────────────────────

def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def _process(path: str, use_ml: bool, cb_done, cb_err):
    """Heavy stage: load image + estimate depth (once per image/ML toggle)."""
    try:
        img = Image.open(path).convert("RGB")
        depth = estimate_depth(img, use_ml=use_ml)
        cb_done(img, depth)
        if use_ml:
            import depth_estimator
            if depth_estimator.last_ml_error:
                cb_err(f"ML falló, usando heurística: {depth_estimator.last_ml_error}")
                depth_estimator.last_ml_error = None
    except Exception as exc:
        cb_err(str(exc))


# ── main window ────────────────────────────────────────────────────────────────

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _BaseTk = TkinterDnD.Tk
    _DND_OK = True
except ImportError:
    _BaseTk = tk.Tk
    _DND_OK = False


class App(_BaseTk):
    def __init__(self):
        super().__init__()
        self.title("2D → 3D SBS Converter")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")
        self._build_ui()
        if _DND_OK:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        # event.data: paths, brace-wrapped when they contain spaces
        raw = event.data.strip()
        path = raw[1:].split("}")[0] if raw.startswith("{") else raw.split()[0]
        if os.path.isfile(path):
            self._img_path = path
            self._rebuild()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 10
        BG = "#1e1e2e"
        FG = "#cdd6f4"
        ACC = "#89b4fa"
        BTN = {"bg": "#313244", "fg": FG, "activebackground": "#45475a",
               "activeforeground": FG, "relief": tk.FLAT, "padx": 10, "pady": 6,
               "cursor": "hand2"}

        F_LBL = ("Segoe UI", 13, "bold")
        F_CHK = ("Segoe UI", 12)
        F_BTN = ("Segoe UI", 12, "bold")
        BTN = {**BTN, "font": F_BTN, "padx": 16, "pady": 8}

        # ── row 1: buttons ────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=PAD, pady=(PAD, 0))

        tk.Button(top, text="📂  Abrir imagen", command=self._open, **BTN).pack(side=tk.LEFT)
        tk.Button(top, text="💾  Guardar SBS", command=self._save, **BTN).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="⛶  Pantalla completa 3D", command=self._fullscreen,
                  bg="#a6e3a1", fg="#1e1e2e", activebackground="#94e2d5",
                  activeforeground="#1e1e2e", relief=tk.FLAT, padx=16, pady=8,
                  font=F_BTN, cursor="hand2").pack(side=tk.LEFT)

        # ── row 2: sliders (grid so they stretch with the window) ────────────
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=PAD, pady=(PAD // 2, 0))
        ctrl.columnconfigure(1, weight=3)
        ctrl.columnconfigure(3, weight=2)
        ctrl.columnconfigure(5, weight=2)

        def make_slider(col, text, color, var, vfrom, vto):
            tk.Label(ctrl, text=text, bg=BG, fg=color, font=F_LBL) \
                .grid(row=0, column=col, sticky="sw", padx=(0, 6), pady=(0, 8))
            s = tk.Scale(ctrl, from_=vfrom, to=vto, orient=tk.HORIZONTAL,
                         variable=var, width=22, bg=BG, fg=color,
                         troughcolor="#45475a", highlightthickness=0,
                         font=("Segoe UI", 12, "bold"),
                         command=self._on_slider)
            s.grid(row=0, column=col + 1, sticky="ew", padx=(0, 18))
            return s

        self.disp_var = tk.IntVar(value=28)
        self.disp_slider = make_slider(0, "Disparidad 3D:", "#f9e2af",
                                       self.disp_var, 0, 100)
        self.conv_var = tk.IntVar(value=50)
        make_slider(2, "Pop-out:", "#a6e3a1", self.conv_var, 0, 100)
        self.gamma_var = tk.IntVar(value=140)   # gamma ×100: 1.40 default
        make_slider(4, "Curva:", "#cba6f7", self.gamma_var, 50, 250)

        # ── row 3: processing options ─────────────────────────────────────────
        opts = tk.Frame(self, bg=BG)
        opts.pack(fill=tk.X, padx=PAD, pady=(4, 0))

        def make_check(text, var, command=None, enabled=True):
            c = tk.Checkbutton(opts, text=text, variable=var, font=F_CHK,
                               bg=BG, fg=FG, selectcolor="#313244",
                               activebackground=BG, activeforeground=FG,
                               command=command)
            c.pack(side=tk.LEFT, padx=(0, 18))
            if not enabled:
                c.configure(state=tk.DISABLED)
            return c

        self.swap_var = tk.BooleanVar(value=False)
        make_check("Invertir ojos (R/L)", self.swap_var, self._on_slider)
        self.ml_var = tk.BooleanVar(value=False)
        make_check("Depth Anything (IA)", self.ml_var, self._rebuild)
        self.splat_var = tk.BooleanVar(value=_SPLAT_OK)
        make_check("Splatting GPU", self.splat_var, self._on_slider, _SPLAT_OK)
        self.lama_var = tk.BooleanVar(value=_LAMA_OK)
        make_check("Relleno IA (LaMa)", self.lama_var, None, _LAMA_OK)

        # ── monitor selector ──────────────────────────────────────────────────
        ctrl2 = tk.Frame(self, bg=BG)
        ctrl2.pack(fill=tk.X, padx=PAD, pady=(2, 0))

        tk.Label(ctrl2, text="Monitor Odyssey 3D:", bg=BG, fg=FG,
                 font=F_CHK).pack(side=tk.LEFT)
        self.monitor_var = tk.StringVar()
        self.monitor_combo = ttk.Combobox(ctrl2, textvariable=self.monitor_var,
                                          state="readonly", width=40,
                                          font=F_CHK)
        self.monitor_combo.pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(ctrl2, text="↺", command=self._refresh_monitors,
                  bg="#313244", fg=FG, relief=tk.FLAT, padx=10, pady=4,
                  font=F_CHK, cursor="hand2").pack(side=tk.LEFT, padx=6)
        self._refresh_monitors()

        # ── preview area ──────────────────────────────────────────────────────
        self.canvas = tk.Label(self, bg="#11111b", cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=PAD)
        self.canvas.bind("<Button-1>", lambda e: self._fullscreen())

        # ── status bar ────────────────────────────────────────────────────────
        heic_note = "" if _HEIC_OK else "  ⚠ HEIC no disponible"
        self.status_var = tk.StringVar(value=f"Abre una imagen para comenzar.{heic_note}")
        tk.Label(self, textvariable=self.status_var, bg="#11111b", fg="#a6adc8",
                 anchor=tk.W, padx=6).pack(fill=tk.X, side=tk.BOTTOM)

        # ── progress bar ──────────────────────────────────────────────────────
        self.progress = ttk.Progressbar(self, mode="indeterminate")

        self.geometry("1280x760")
        self.minsize(900, 600)

        # internal state
        self._img_path: str | None = None
        self._photo_ref = None     # keep-alive for preview PhotoImage
        self._slider_pending = None
        self._preview_img = None   # downscaled image for real-time preview
        self._preview_depth = None
        self._depth_cache = {}     # (path, ml) → (img, depth); gallery prefetch
        self._depth_lock = threading.Lock()  # serialize model inference

    # ── actions ───────────────────────────────────────────────────────────────

    def _open(self):
        path = filedialog.askopenfilename(filetypes=SUPPORTED)
        if path:
            self._img_path = path
            self._rebuild()

    def _save(self):
        if current_img is None or current_depth is None:
            messagebox.showinfo("Aviso", "Primero abre una imagen.")
            return
        base = os.path.splitext(os.path.basename(self._img_path))[0] \
            if self._img_path else "imagen"
        out = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"{base}_SBS.jpg",
            initialdir=os.path.dirname(self._img_path) if self._img_path else None,
        )
        if out:
            # Build full-resolution SBS with the current settings
            sbs = build_stereo(current_img, current_depth,
                               max_disparity=self.disp_var.get(),
                               swap_eyes=self.swap_var.get(),
                               use_splat=self.splat_var.get(),
                               convergence=1.0 - self.conv_var.get() / 100.0,
                               gamma=self.gamma_var.get() / 100.0,
                               inpaint=self.lama_var.get())
            sbs.save(out, quality=95)
            self.status_var.set(f"Guardado: {out}")

    def _refresh_monitors(self):
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
        except Exception:
            monitors = []

        entries = []
        default_idx = 0
        for i, m in enumerate(monitors):
            tag = " ★ UHD" if (m.width == ODYSSEY_3D_W and m.height == ODYSSEY_3D_H) else ""
            entries.append(f"Monitor {i+1}: {m.width}×{m.height}  @({m.x},{m.y}){tag}")
            if m.width == ODYSSEY_3D_W and m.height == ODYSSEY_3D_H:
                default_idx = i

        if not entries:
            entries = [f"Principal: {self.winfo_screenwidth()}×{self.winfo_screenheight()}  @(0,0)"]

        self.monitor_combo["values"] = entries
        self.monitor_combo.current(default_idx)
        self._monitors_raw = monitors

    def _fullscreen(self):
        if current_img is None or current_depth is None:
            messagebox.showinfo("Aviso", "Primero genera una imagen SBS.")
            return
        idx = self.monitor_combo.current()
        # Disparity is in display pixels at 3840-wide — scale the slider value.
        # Cap the starting value at 30 so the 3D opens comfortable, not aggressive.
        disp_scaled = round(self.disp_var.get() * ODYSSEY_3D_W / (2 * current_img.width))
        proc = _launch_exclusive_viewer(
            current_img, current_depth,
            monitor_idx=idx,
            disparity=min(30, max(4, disp_scaled)),
            swap=self.swap_var.get(),
            splat=self.splat_var.get(),
            convergence=1.0 - self.conv_var.get() / 100.0,
            gamma=self.gamma_var.get() / 100.0,
            inpaint=self.lama_var.get(),
        )
        # When the viewer closes (ESC), pull back its final settings
        threading.Thread(target=self._wait_viewer_sync,
                         args=(proc,), daemon=True).start()

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic", ".heif")

    def _gallery_files(self):
        """Sorted list of images in the current image's folder."""
        folder = os.path.dirname(self._img_path)
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(self._IMG_EXTS)
        )
        return files

    def _depth_for(self, path: str):
        """Image + depth with caching (gallery prefetch hits this)."""
        key = (path, self.ml_var.get())
        if key in self._depth_cache:
            return self._depth_cache[key]
        img = Image.open(path).convert("RGB")
        with self._depth_lock:
            if key in self._depth_cache:        # raced with prefetch
                return self._depth_cache[key]
            depth = estimate_depth(img, use_ml=self.ml_var.get())
        # keep the cache small
        if len(self._depth_cache) > 6:
            self._depth_cache.pop(next(iter(self._depth_cache)))
        self._depth_cache[key] = (img, depth)
        return img, depth

    def _prefetch_neighbors(self, files, i):
        """Warm the cache for prev/next so arrow keys feel instant."""
        def work():
            for j in (i + 1, i - 1):
                try:
                    self._depth_for(files[j % len(files)])
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _handle_nav(self, direction: int, seq: int):
        """Viewer asked for prev/next image: compute depth, hand it over."""
        import json
        import numpy as np
        global current_img, current_depth

        files = self._gallery_files()
        cur = os.path.normcase(os.path.normpath(os.path.abspath(self._img_path)))
        i = next((k for k, f in enumerate(files)
                  if os.path.normcase(os.path.normpath(os.path.abspath(f))) == cur), 0)
        new_i = (i + direction) % len(files)
        new_path = files[new_i]

        img, depth = self._depth_for(new_path)
        self._prefetch_neighbors(files, new_i)

        tmp_img = os.path.join(tempfile.gettempdir(), "odyssey_src_tmp.png")
        tmp_depth = os.path.join(tempfile.gettempdir(), "odyssey_depth_tmp.npy")
        img.save(tmp_img, format="PNG")
        np.save(tmp_depth, depth)
        with open(os.path.join(tempfile.gettempdir(), "odyssey_ready.json"), "w") as f:
            json.dump({"seq": seq, "name": os.path.basename(new_path)}, f)

        # keep the app in sync with what the viewer shows
        self._img_path = new_path
        current_img = img
        current_depth = depth
        self.after(0, self._prepare_preview)

    def _wait_viewer_sync(self, proc):
        import json
        nav_path = os.path.join(tempfile.gettempdir(), "odyssey_nav_request.json")
        last_seq = 0
        try:
            # serve gallery requests while the viewer is open
            while proc.poll() is None:
                try:
                    with open(nav_path) as f:
                        req = json.load(f)
                    if req.get("seq", 0) > last_seq:
                        last_seq = req["seq"]
                        self._handle_nav(int(req.get("dir", 1)), last_seq)
                except (OSError, ValueError):
                    pass
                time.sleep(0.3)

            result_path = os.path.join(tempfile.gettempdir(),
                                       "odyssey_result_tmp.json")
            if not os.path.exists(result_path):
                return
            with open(result_path) as f:
                result = json.load(f)
            os.remove(result_path)
            # viewer disparity is in 3840-display pixels → back to image pixels
            disp_app = round(result["disparity"] * 2 * current_img.width
                             / ODYSSEY_3D_W)
            disp_app = max(0, min(100, disp_app))
            conv = float(result.get("convergence", 0.5))

            def apply():
                self.disp_var.set(disp_app)
                self.swap_var.set(bool(result.get("swap", False)))
                self.conv_var.set(round((1.0 - conv) * 100))
                self._refresh_sbs_preview()
                self.status_var.set(
                    f"Ajustes del visor 3D aplicados: disparidad {disp_app}")
            self.after(0, apply)
        except Exception:
            pass

    def _on_slider(self, *_):
        # Real-time: only re-warp the cached preview (no depth recompute).
        if self._preview_img is None:
            return
        if self._slider_pending:
            self.after_cancel(self._slider_pending)
        self._slider_pending = self.after(15, self._refresh_sbs_preview)

    def _rebuild(self):
        """Heavy path: re-estimate depth (new image or ML toggle)."""
        if not self._img_path:
            return
        self._set_busy(True)
        threading.Thread(
            target=_process,
            args=(self._img_path, self.ml_var.get(),
                  self._on_done, self._on_err),
            daemon=True,
        ).start()

    def _on_done(self, img: Image.Image, depth):
        global current_img, current_depth
        current_img = img
        current_depth = depth
        self.after(0, self._prepare_preview)

    def _prepare_preview(self):
        """Build downscaled copies once; slider then re-warps them instantly."""
        import numpy as np
        img, depth = current_img, current_depth

        pw = 1000
        scale = min(1.0, pw / img.width)
        nw, nh = round(img.width * scale), round(img.height * scale)
        self._preview_img = img.resize((nw, nh), Image.LANCZOS)
        self._preview_depth = np.array(
            Image.fromarray((depth * 255).astype(np.uint8), mode="L")
            .resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0

        self._set_busy(False)
        self._refresh_sbs_preview()

        # warm the gallery cache for prev/next
        try:
            files = self._gallery_files()
            cur = os.path.normcase(os.path.normpath(os.path.abspath(self._img_path)))
            i = next((k for k, f in enumerate(files)
                      if os.path.normcase(os.path.normpath(os.path.abspath(f))) == cur), 0)
            self._prefetch_neighbors(files, i)
        except Exception:
            pass

    def _refresh_sbs_preview(self):
        """Fast re-warp of the preview pair — runs on every slider tick."""
        if self._preview_img is None:
            return
        # Slider is calibrated for full-res; scale down for the preview copy
        disp = round(self.disp_var.get() * self._preview_img.width
                     / max(1, current_img.width))
        sbs = build_stereo(self._preview_img, self._preview_depth,
                           max_disparity=max(1, disp),
                           swap_eyes=self.swap_var.get(),
                           use_splat=self.splat_var.get(),
                           convergence=1.0 - self.conv_var.get() / 100.0,
                           gamma=self.gamma_var.get() / 100.0)

        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 400
        preview = sbs
        if sbs.width > cw or sbs.height > ch:
            preview = sbs.copy()
            preview.thumbnail((cw, ch), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(preview)
        self.canvas.configure(image=self._photo_ref)
        name = os.path.basename(self._img_path)
        self.status_var.set(
            f"{name}   [disparidad {self.disp_var.get()}]   "
            f"Clic en la imagen o ⛶ para verla en 3D"
        )

    def _on_err(self, msg: str):
        self.after(0, lambda: (self._set_busy(False),
                                self.status_var.set(f"Error: {msg}")))

    def _set_busy(self, busy: bool):
        if busy:
            self.progress.pack(fill=tk.X, before=self.canvas)
            self.progress.start(12)
            self.status_var.set("Procesando…")
        else:
            self.progress.stop()
            self.progress.pack_forget()


# ── exclusive fullscreen launcher ─────────────────────────────────────────────

ODYSSEY_3D_W = 3840
ODYSSEY_3D_H = 2160

_VIEWER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.py")
_tmp_sbs_path: str | None = None


def _launch_exclusive_viewer(img: Image.Image, depth, monitor_idx: int = 0,
                             disparity: int = 40, swap: bool = False,
                             splat: bool = False, convergence: float = 0.5,
                             gamma: float = 1.0, inpaint: bool = False):
    """
    Save the original image + depth map to temp files and launch viewer.py.
    The viewer builds the SBS itself, so disparity can be adjusted live
    (mouse wheel / arrow keys) while watching in 3D.
    """
    import numpy as np

    tmp_img = os.path.join(tempfile.gettempdir(), "odyssey_src_tmp.png")
    tmp_depth = os.path.join(tempfile.gettempdir(), "odyssey_depth_tmp.npy")
    img.save(tmp_img, format="PNG")
    np.save(tmp_depth, depth)

    return subprocess.Popen(
        [sys.executable, _VIEWER_SCRIPT, tmp_img, tmp_depth,
         str(monitor_idx), str(disparity), str(int(swap)), str(int(splat)),
         str(round(convergence * 100)), str(round(gamma * 100)),
         str(int(inpaint))],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
