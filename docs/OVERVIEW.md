# 3D FloorPlan — Project Overview

> **Pertama baca? Mulai dari [README.md](README.md) — panduan komprehensif satu file.**

Sistem konversi denah lantai 2D → 3D. Pipeline dua bagian:
1. **ML server (VPS)** — CubiCasa5K model segmentasi denah → binary wall mask
2. **Local pipeline** — wall mask → WallGraph → JSON → Three.js 3D viewer

---

## Arsitektur Pipeline

```
[Gambar denah lantai]
        │
        ▼ (HTTP POST)
┌─────────────────────┐
│  VPS ML Server      │  YOUR_VPS_IP:8000
│  CubiCasa5K model   │  FastAPI + uvicorn
│  → wall_mask.png    │  (Debian 12, CPU)
└─────────────────────┘
        │
        ▼ (binary PNG, white=wall)
┌─────────────────────┐
│  Local Pipeline     │
│  WallDetector       │  skeleton → Hough → Wall list
│  WallGraphBuilder   │  snap → resolve → filter → WallGraph
│  WallGraph JSON     │  serialized walls/doors/windows
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3D Viewer          │
│  Three.js (browser) │  BoxGeometry per wall
│  viewer/index.html  │  OrbitControls
└─────────────────────┘
```

---

## Struktur Folder

```
3dfloorplan/
├── docs/                    ← dokumentasi (folder ini)
│   ├── OVERVIEW.md
│   ├── INSTALLATION.md
│   ├── CUBICASA.md
│   └── USAGE.md
│
├── wallgraph/               ← Python package utama
│   ├── __init__.py
│   ├── models.py            ← Point, Wall, Door, Window, WallGraph
│   ├── detector.py          ← WallDetector (OpenCV Hough)
│   ├── builder.py           ← WallGraphBuilder (topology cleanup)
│   ├── validator.py         ← WallGraphValidator
│   ├── constants.py         ← semua parameter/threshold
│   ├── utils.py             ← geometry helpers
│   └── cli.py               ← CLI entry point (python -m wallgraph)
│
├── viewer/
│   └── index.html           ← Three.js 3D viewer (standalone HTML)
│
├── sample/raw/              ← sample denah lantai untuk testing
├── output/                  ← wall mask hasil ML API
│
├── running.py               ← ENTRY POINT UTAMA (satu command)
├── test_api.py              ← CLI tester untuk ML API
│
├── pyproject.toml           ← Python dependencies
└── uv.lock
```

---

## Komponen Utama

### `wallgraph/models.py`
Data models Pydantic:
- `Point(x, y)` — koordinat 2D
- `Wall(id, start, end, thickness, exterior, center, radius, ...)` — segment dinding. `center/radius` diisi = arc wall (dinding melengkung)
- `Door`, `Window` — bukaan
- `WallGraph(walls, doors, windows)` — output akhir

### `wallgraph/detector.py` — `WallDetector`
Mendeteksi dinding dari image/mask menggunakan OpenCV.

| Method | Input | Keterangan |
|--------|-------|-----------|
| `detect_with_doors(path)` | raw image | Hough langsung di image asli |
| `detect_from_mask(path)` | binary mask PNG | **Recommended.** Skeletonize → Hough |
| `detect_hough(path)` | raw image | Alternatif: skeleton dari image asli |

#### `detect_from_mask` pipeline:
```
mask PNG → binarize → remove noise → skeletonize → prune spurs
→ Hough (t=15, ml=15, gap=25) → snap H/V → merge collinear
→ door gap detection → filter short walls
```

### `wallgraph/builder.py` — `WallGraphBuilder`
Membersihkan wall list menjadi topologi yang konsisten:

```
build(walls):
  1. _snap_endpoints         ← cluster endpoint berdekatan → shared node
  2. _resolve_junctions      ← split wall di T/X intersection
  3. _filter_short_walls     ← buang wall < min_wall_length
  4. _deduplicate            ← hapus duplikat
  5. _snap_endpoints         ← pass kedua
  6. _filter_noise           ← buang isolated degree-1 stubs
  7. _close_dangling_endpoints ← Layer C: extend endpoint ke wall terdekat
  8. _filter_diagonal_walls  ← buang diagonal (bukan H/V)
  9. _filter_small_components ← buang komponen terisolasi kecil
```

### `wallgraph/constants.py`
Semua magic number terpusat di sini. Lihat file untuk deskripsi lengkap tiap konstanta.

### `viewer/index.html`
Three.js viewer standalone:
- Load WallGraph JSON via file picker / drag-drop / `?json=URL`
- Straight wall → `BoxGeometry` (panjang=wall.length, tinggi=wallHeight, tebal=wall.thickness)
- Arc wall → `TubeGeometry` via `CatmullRomCurve3`
- Bridge walls (gap-fill hasil Layer C) → warna hijau
- Sliders: wall height, scale (px→m), default thickness
