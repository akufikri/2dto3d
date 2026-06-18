# wallgraph — Wall-First CAD Reconstruction Engine

Ekstraksi dinding dari gambar floorplan 2D → wall graph topologi → validasi geometri.

## Filosofi

**Walls are source of truth.** Rooms derived from walls. Never derive walls from rooms.

Pipeline deterministik (OpenCV, tanpa ML). Lihat [`methodlogy.md`](methodlogy.md) dan [`Plan3D_Wall_First_CAD_Blueprint.md`](Plan3D_Wall_First_CAD_Blueprint.md) untuk detail arsitektur.

## Pipeline

```
Gambar Floorplan
↓
WallDetector (OpenCV: threshold/morfologi/Hough/contour)
↓
Raw Wall Segments
↓
WallGraphBuilder (snap endpoint → resolve junction → filter → dedup)
↓
WallGraph (topologi clean)
↓
WallGraphValidator (cek floating/duplicate/disconnected)
```

## Install

```bash
pip install -e .
```

Dependencies: opencv-python-headless, numpy, networkx, pydantic, shapely.

Python ≥3.11.

## CLI

```bash
# Deteksi default (binary threshold + contour)
wallgraph sample/raw/image-3.jpg

# Output ke file JSON
wallgraph sample/raw/image-3.jpg -o hasil.json

# Dengan visualisasi
wallgraph sample/raw/image-3.jpg -v debug.png

# Method lain
wallgraph sample/raw/image-3.jpg --method canny
wallgraph sample/raw/image-3.jpg --method hough

# Parameter
wallgraph sample/raw/image-3.jpg --snap 5 --min-length 40
```

## Python API

```python
from wallgraph import WallDetector, WallGraphBuilder, WallGraphValidator

# 1. Deteksi dinding
detector = WallDetector(params={"min_wall_length": 40})
walls = detector.detect("sample/raw/image-3.jpg")

# 2. Bangun wall graph
builder = WallGraphBuilder(snap_distance=5.0, min_wall_length=40)
graph = builder.build(walls)

# 3. Validasi
validator = WallGraphValidator(snap_distance=5.0)
report = validator.validate(graph)
print(report.passed)  # True/False

# 4. Visualisasi
detector.visualize("sample/raw/image-3.jpg", graph.walls, "output.png")

# 5. Export JSON
print(graph.model_dump_json(indent=2))
```

## Detection Methods

| Method | Kapan pakai |
|--------|-------------|
| `binary` (default) | Adaptive threshold + morfologi + contour + Hough. Cocok untuk gambar arsitektur bersih. |
| `hough` | Skeletonisasi + Hough. Segmen lebih sedikit, cenderung axis-aligned. |
| `canny` | Canny edges + Hough. Untuk gambar dengan garis tepi tajam. |

## Arsitektur Package

```
wallgraph/
├── __init__.py    # Public API exports
├── models.py      # Point, Wall, Door, Window, WallGraph (Pydantic)
├── detector.py    # WallDetector — ekstraksi dinding dari gambar
├── builder.py     # WallGraphBuilder — raw segments → clean graph
├── validator.py   # WallGraphValidator — topology & geometry check
├── cli.py         # Command-line interface
├── constants.py   # Semua threshold & parameter
└── utils.py       # Geometri helpers
```

## Constraints

Semua parameter threshold ada di `constants.py`. Override via `WallDetector(params={...})`:

```python
detector = WallDetector(params={
    "min_wall_length": 30,
    "hough_threshold": 40,
    "parallel_merge_dist": 10,
})
```

## Tests

```bash
pytest tests/ -v
```

Sample gambar di `sample/raw/`. Script debug di `_run_debug_178.py` dan `test_img3_new.py`.

## Output Format (JSON)

```json
{
  "walls": [
    {
      "id": "W0000",
      "start": {"x": 100.0, "y": 200.0},
      "end": {"x": 500.0, "y": 200.0},
      "thickness": 0,
      "exterior": false,
      "center": null,
      "radius": 0,
      "start_angle": 0,
      "end_angle": 0
    }
  ],
  "doors": [],
  "windows": []
}
```

Arc walls punya `center`, `radius`, `start_angle`, `end_angle` terisi.
