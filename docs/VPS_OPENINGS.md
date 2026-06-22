# VPS — Door & Window Mask Endpoints

CubiCasa5K outputs 44 segmentation classes including doors and windows.
The current `app.py` only extracts the wall mask. This guide shows how to
add `/predict/door-mask` and `/predict/window-mask`.

---

## CubiCasa5K Output Format

The model returns a dict with two outputs:

```python
output = model(input_tensor)
# output['room_type']    → [1, 12, H, W]  room classes
# output['heatmaps']     → [1, 21, H, W]  icon/element heatmaps
# output['room_type']    → [1, 12, H, W]
```

Relevant heatmap channels (from CubiCasa5K paper):
- Channel 0: Background
- Channel 1: Walls
- Channel 2: Doors (single)
- Channel 3: Doors (double)
- Channel 4: Windows
- ...

> **Note:** Exact channel indices may vary depending on training config.
> Inspect `output['heatmaps'].shape` and compare with ground truth to verify.

---

## Add to `app.py` on VPS

Add these endpoints after the existing `/predict/wall-mask`:

```python
import torch
import numpy as np
import cv2
from fastapi import File, UploadFile, Query
from fastapi.responses import Response


@app.post("/predict/door-mask")
async def predict_door_mask(
    file: UploadFile = File(...),
    max_dim: int = Query(1024),
    threshold: float = Query(0.5),
):
    """Return binary door mask (white=door, black=background)."""
    img = _load_image(await file.read(), max_dim)
    input_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(input_tensor)

    # Heatmap channel for doors — verify channel index for your model
    # Single door: channel 2, Double door: channel 3
    heatmaps = output["heatmaps"][0]  # [21, H, W]
    door_prob = torch.sigmoid(heatmaps[2] + heatmaps[3])  # combine single+double
    door_mask = (door_prob.cpu().numpy() > threshold).astype(np.uint8) * 255

    # Resize back to original image dimensions
    door_mask = cv2.resize(door_mask, (img.shape[1], img.shape[0]))

    _, png_bytes = cv2.imencode(".png", door_mask)
    return Response(content=png_bytes.tobytes(), media_type="image/png")


@app.post("/predict/window-mask")
async def predict_window_mask(
    file: UploadFile = File(...),
    max_dim: int = Query(1024),
    threshold: float = Query(0.5),
):
    """Return binary window mask (white=window, black=background)."""
    img = _load_image(await file.read(), max_dim)
    input_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(input_tensor)

    # Window channel — verify index for your model
    heatmaps = output["heatmaps"][0]  # [21, H, W]
    window_prob = torch.sigmoid(heatmaps[4])
    window_mask = (window_prob.cpu().numpy() > threshold).astype(np.uint8) * 255

    window_mask = cv2.resize(window_mask, (img.shape[1], img.shape[0]))

    _, png_bytes = cv2.imencode(".png", window_mask)
    return Response(content=png_bytes.tobytes(), media_type="image/png")


@app.post("/predict/all-masks")
async def predict_all_masks(
    file: UploadFile = File(...),
    max_dim: int = Query(1024),
):
    """Return all masks as JSON: wall, door, window as base64 PNG."""
    import base64

    img = _load_image(await file.read(), max_dim)
    input_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(input_tensor)

    h, w = img.shape[:2]
    results = {}

    # Wall mask
    room_pred = torch.argmax(output["room_type"], dim=1)[0].cpu().numpy()
    wall_mask = (room_pred == 2).astype(np.uint8) * 255
    wall_mask = cv2.resize(wall_mask, (w, h))
    _, buf = cv2.imencode(".png", wall_mask)
    results["wall_mask"] = base64.b64encode(buf).decode()

    # Door + window masks from heatmaps
    heatmaps = output["heatmaps"][0]
    for name, channels, idx in [
        ("door_mask",   [2, 3], None),  # single + double door
        ("window_mask", [4],    None),
    ]:
        if len(channels) == 1:
            prob = torch.sigmoid(heatmaps[channels[0]])
        else:
            prob = torch.sigmoid(sum(heatmaps[c] for c in channels))
        mask = (prob.cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))
        _, buf = cv2.imencode(".png", mask)
        results[name] = base64.b64encode(buf).decode()

    return results
```

---

## Usage from local engine

### Fetch door mask separately

```bash
# Fetch door mask
curl -X POST http://YOUR_VPS_IP:8000/predict/door-mask \
  -F "file=@sample/raw/image-4.jpeg" \
  -o output/door_mask_image-4.png

# Fetch window mask
curl -X POST http://YOUR_VPS_IP:8000/predict/window-mask \
  -F "file=@sample/raw/image-4.jpeg" \
  -o output/window_mask_image-4.png
```

### Run engine with dedicated masks

```bash
python3 running.py sample/raw/image-4.jpeg \
  --api --contour \
  --close 8 --wall-thickness 6 \
  --door-mask output/door_mask_image-4.png \
  --window-mask output/window_mask_image-4.png \
  -o out.json
```

### Or geometric detection (no extra API call needed)

```bash
python3 running.py sample/raw/image-4.jpeg \
  --api --contour \
  --close 8 --wall-thickness 6 \
  --detect-openings \
  -o out.json
```

---

## Output JSON Format

```json
{
  "type": "shape",
  "image_size": [1024, 768],
  "shapes": [
    { "outer": [[x, y], ...], "holes": [[[x, y], ...]] }
  ],
  "doors": [
    {
      "id": "D0000",
      "x": 245.3,
      "y": 182.0,
      "width": 48.0,
      "depth": 14.0,
      "wall_axis": "h",
      "door_type": "swing"
    }
  ],
  "windows": [
    {
      "id": "Win0000",
      "x": 410.5,
      "y": 98.0,
      "width": 32.0,
      "depth": 10.0,
      "wall_axis": "v"
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | float | Center position (pixels, image coordinates) |
| `width` | float | Opening width along the wall (px) |
| `depth` | float | Depth through the wall (px) |
| `wall_axis` | `"h"` / `"v"` | Axis the wall runs along |
| `door_type` | string | `"swing"` \| `"sliding"` \| `"double_swing"` |

---

## Verify Channel Indices on VPS

```python
# Run this on VPS to inspect heatmap channels
import torch
ckpt = torch.load("model_best_val_loss_var.pkl", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load test image and run inference
# ...
output = model(input_tensor)
print("heatmaps shape:", output["heatmaps"].shape)  # [1, N, H, W]
print("room_type shape:", output["room_type"].shape)

# Compare argmax prediction with ground truth SVG annotation
# to map channel index → class name
```
