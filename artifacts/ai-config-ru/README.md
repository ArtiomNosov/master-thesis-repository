# AI configuration screenshot localization

This folder keeps the source screenshot, the Russian localized raster output, and the script used to reproduce it.

Files:

- `source-original.png` - original English screenshot.
- `ai-configuration-ru-raster.png` - Russian localized raster image, `752x831`.
- `render_from_original.py` - deterministic overlay script that masks the original text areas and renders Russian text.

Re-render:

```powershell
python artifacts/ai-config-ru/render_from_original.py
```

The script requires Pillow.
