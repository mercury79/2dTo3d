# 2D → 3D SBS Converter para Samsung Odyssey 3D

Convierte fotografías 2D a formato estéreo Side-By-Side (SBS) y las proyecta en
pantalla completa exclusiva para activar el modo 3D del monitor Samsung Odyssey 3D
(27", sin gafas, con eye-tracking).

## Características

- Estimación de profundidad con IA (**Depth Anything V2 Large** en GPU) o heurística rápida sin GPU
- Plano de convergencia centrado: lo cercano sale de la pantalla, lo lejano se hunde
- Visor en fullscreen exclusivo (pygame/SDL2) que dispara la detección del Odyssey 3D Hub
- Ajuste de disparidad **en vivo** mientras ves el 3D (rueda del mouse / flechas)
- Soporte HEIC/HEIF, JPG, PNG, WebP, TIFF
- Letterbox automático que respeta el aspect ratio original
- Selector de monitor para configuraciones multi-pantalla

## Requisitos

- Windows 11, Python 3.13+
- Monitor Samsung Odyssey 3D + Odyssey 3D Hub instalado (cable DisplayPort/HDMI **y** USB)
- Resolución del monitor en UHD 3840×2160 (requisito del modo 3D)

```bash
pip install numpy pillow pillow-heif screeninfo pygame
# Para profundidad con IA (recomendado, requiere GPU NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate
```

## Uso

1. Doble clic en `Abrir 3D Converter.bat` (o `pythonw app.py`)
2. Abre una imagen y ajusta la disparidad con el slider
3. Marca **Usar Depth Anything (ML)** para profundidad con IA
4. Pulsa **Pantalla completa** — el Odyssey 3D Hub detecta el SBS y activa el 3D

### Controles en el visor 3D

| Tecla | Acción |
|---|---|
| Rueda del mouse / ↑ ↓ | Ajustar disparidad en vivo |
| `S` | Invertir ojos |
| `ESC` | Cerrar |

## Arquitectura

- `app.py` — interfaz gráfica (tkinter)
- `depth_estimator.py` — mapa de profundidad (Depth Anything V2 o heurística)
- `stereo_builder.py` — warping estéreo y ensamblado SBS
- `viewer.py` — visor fullscreen exclusivo con ajuste en vivo (proceso aparte, pygame)

El visor corre como subproceso porque pygame requiere fullscreen **exclusivo**
(DirectX) para que el Odyssey 3D Hub detecte el contenido — una ventana borderless
normal no dispara la detección.
