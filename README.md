# 2D → 3D SBS Converter para Samsung Odyssey 3D

Convierte fotografías 2D a estéreo Side-By-Side (SBS) y las proyecta en
pantalla completa exclusiva para activar el modo 3D del monitor Samsung
Odyssey 3D (27", sin gafas, con eye-tracking).

## Características

**Profundidad**
- **Depth Anything V2 Large** en GPU (CUDA, float16) o heurística rápida sin GPU
- Filtro bilateral guiado por color (GPU): alinea los bordes de profundidad
  con las siluetas reales — menos halo en pelo, hojas, contornos finos
- Curva gamma ajustable: expande la separación del sujeto, comprime el fondo

**Render estéreo**
- **Splatting GPU**: reproyección 3D con z-buffer — oclusión correcta,
  sin ghosting en los bordes (30 ms en HD, 150 ms en 4K en una RTX 4090)
- **Inpainting LaMa**: las des-oclusiones se rellenan con IA generativa en
  vez de estirar el fondo (solo en el render final, no al arrastrar sliders)
- Pop-out ajustable (plano de convergencia): decide cuánto sale la imagen
  de la pantalla y cuánto se hunde detrás
- Fallback CPU (warp por columnas) si no hay GPU

**Visor 3D fullscreen**
- Fullscreen exclusivo (pygame/SDL2/DirectX) — requisito para que el
  Odyssey 3D Hub detecte el contenido y active el 3D
- Sliders flotantes de disparidad y pop-out, ajuste en vivo viendo el 3D
- **Modo galería**: flechas ← → navegan la carpeta sin salir del 3D,
  con pre-cálculo de profundidad de las fotos vecinas
- Al salir (ESC) los ajustes se sincronizan con la app principal

**App**
- Preview SBS en tiempo real (la profundidad se calcula una vez por imagen)
- Drag & drop, soporte HEIC/HEIF, selector de monitor multi-pantalla
- Guardado como `<nombre-original>_SBS.jpg` en la carpeta de la foto

## Requisitos

- Windows 11, Python 3.13+
- Samsung Odyssey 3D + Odyssey 3D Hub (cable DisplayPort/HDMI **y** USB)
- Resolución del monitor en UHD 3840×2160 (requisito del modo 3D del Hub)
- GPU NVIDIA recomendada (probado en RTX 4090)

## Instalación

```bash
pip install numpy pillow pillow-heif screeninfo pygame tkinterdnd2
# IA de profundidad (recomendado, requiere GPU NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate
```

### Modelo de inpainting LaMa (opcional, ~196 MB)

Descargar [big-lama.pt](https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt)
y colocarlo en `models/big-lama.pt`:

```bash
curl -L -o models/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

> No usar `pip install simple-lama-inpainting`: exige numpy<2 y no es
> compatible con Python 3.13. `lama_inpaint.py` carga el TorchScript directo.

### Modelos Depth Anything (descarga automática o manual)

La primera vez que actives "Depth Anything (IA)" se descargan de HuggingFace
(~1.3 GB el Large). Para instalación offline, copiar las carpetas
`models--depth-anything--Depth-Anything-V2-Large-hf` (y/o `-Small-hf`)
a `%USERPROFILE%\.cache\huggingface\hub\`.

## Uso

1. Doble clic en `Abrir 3D Converter.bat`
2. Abre o arrastra una imagen; ajusta **Disparidad**, **Pop-out** y **Curva**
   viendo el preview en tiempo real
3. Pulsa **Pantalla completa 3D** — el Odyssey 3D Hub detecta el SBS
   (elige "SBS" en su popup si no está en automático)
4. Ajusta en vivo y navega tu carpeta sin salir del 3D

### Controles en el visor 3D

| Control | Acción |
|---|---|
| Rueda del mouse / ↑ ↓ | Disparidad (intensidad 3D) |
| Sliders flotantes (aparecen al mover el mouse) | Disparidad y Pop-out |
| ← → | Foto anterior / siguiente (galería) |
| `S` | Invertir ojos |
| `ESC` | Cerrar y sincronizar ajustes con la app |

## Arquitectura

| Archivo | Rol |
|---|---|
| `app.py` | GUI (tkinter): preview en vivo, galería, lanzador del visor |
| `viewer.py` | Visor fullscreen exclusivo (pygame, proceso aparte) |
| `depth_estimator.py` | Depth Anything V2 / heurística + refinado bilateral |
| `stereo_builder.py` | Warp estéreo CPU (fallback) |
| `splat_renderer.py` | Splatting GPU con z-buffer y relleno de des-oclusiones |
| `lama_inpaint.py` | Inpainting LaMa (TorchScript) para des-oclusiones |

El visor corre como subproceso porque pygame necesita fullscreen **exclusivo**
(DirectX) para que el Odyssey 3D Hub detecte el contenido; una ventana
borderless normal no dispara la detección. App y visor se comunican por
archivos temporales (navegación de galería y sincronización de ajustes).
