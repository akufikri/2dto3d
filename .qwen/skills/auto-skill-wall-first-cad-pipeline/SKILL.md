---
name: wall-first-cad-pipeline
description: Deterministic wall-first CAD floorplan reconstruction — wall detection, graph building, topology validation before room inference
source: auto-skill
extracted_at: '2026-06-18T11:40:49.511Z'
---

# Wall-First CAD Reconstruction Pipeline

## When to Use

Use this approach when:
- Reconstructing 3D floorplans from 2D architectural drawings
- Converting raster floorplan images to structured geometry
- Building CAD tools where geometry accuracy matters more than semantic labeling
- Working with clean architectural drawings (not photos of sketches)

## Core Principle

**Walls are the source of truth. Rooms are derived from walls. Never derive walls from rooms.**

### Why Not LLM-First?

Traditional pipeline:
```
Image → LLM → Rooms → Walls → 3D
```

**Problem:** LLMs do *interpretation* ("this looks like a bedroom"), not *measurement* ("wall at coordinate 1250,4800"). Geometry errors propagate to 3D. Accuracy depends on model reasoning, not deterministic CV.

Wall-first pipeline:
```
Image → Wall Detection → Wall Graph → Topology Validation → Room Discovery → Room Labeling (LLM) → 3D
```

**Why it works:** Deterministic CV extracts exact geometry. LLM only assigns semantic labels at the end. Expected +20-40% accuracy improvement without model changes or GPU.

## Pipeline Stages

### 1. Wall Mask Extraction
- Grayscale → **adaptive threshold** (better than Otsu for varying contrast) → morphological close → open
- Output: binary wall mask (preserve thickness, don't skeletonize yet)
- Tools: OpenCV `cv2.adaptiveThreshold`, `cv2.morphologyEx`
- **Block size**: `min(51, max(11, width // 10 | 1))` — adapts to image width

### 2. Wall Detection (Binary method — recommended)

**Critical: Run HoughLinesP on the cleaned MASK directly, NOT on skeleton.**

Why: Skeletonization + pruning destroys thin 3-5px perimeter walls. The skeleton of thin walls has too few pixels for Hough to detect lines (e.g., left wall: 29px skeleton → 1px after pruning → no Hough detection). Running Hough on the raw mask preserves all wall pixels, and duplicate detections from thick walls collapse via merge logic.

Pipeline inside `_detect_binary`:
```
cleaned_mask → HoughLinesP → snap_to_axis → create Walls
→ merge_nearby_walls → filter_outside_boundary → merge_nearby_walls
→ merge_parallel_pairs → merge_nearby_walls → merge_parallel_pairs
→ arc_detection → remove_arc_overlaps → merge_nearby_walls → filter_small_walls
```

#### Merge Pipeline (iterative, 3 passes needed)
1. `_merge_nearby_walls` (collinear merge) — groups walls at same axis position within `collinear_tol=8px`, merges overlapping/adjacent segments with gap tolerance (`COLLINEAR_MERGE_GAP=60px` for door openings)
2. `_filter_outside_boundary` — removes isolated noise walls (text/dimension lines) using: (a) mask percentile boundary, (b) structural isolation check (no perpendicular intersection AND no endpoint near any structural wall)
3. `_merge_parallel_pairs` — merges closely-spaced parallel walls into centerlines (`PARALLEL_MERGE_DIST=20px`)
4. Repeat `_merge_nearby_walls` + `_merge_parallel_pairs` to consolidate centerlines from previous step

**Key: `collinear_tol=8px` in `_merge_axis` prevents chain-merging.** Without it, iterative averaging would merge x=68 → x=70 → x=114 → x=118, collapsing the left wall into an internal wall. The `ref_pos` (original wall position) is the anchor; only walls within 8px of `ref_pos` can merge collinearly.

#### Arc Detection
- Uses thickness-filtered mask (5×5 open, text-free) for contour tracing
- Dual strategy: (1) fillet detection at `approxPolyDP` vertices (angle 30°-150°), (2) curvature-based curved segment detection
- **Critical: `_remove_arc_overlaps` only removes SHORT walls (< 2× arc diameter).** Long structural walls (>4× arc radius) passing through an arc must be kept — the arc is a small rounded corner at one end, not a replacement for the entire wall.

#### Noise Filtering (`_filter_outside_boundary`)
Two heuristics:
1. **Mask percentile boundary**: walls outside 5th-95th percentile + 30px margin of mask content area are removed
2. **Structural isolation**: walls with no perpendicular intersection AND no endpoint near any *structural* wall (one that has intersections) are noise. **Only count proximity to structural walls, not to other isolated walls** — two text labels near each other are still noise.

### 3. Wall Graph Building
Pipeline:
1. **Snap endpoints** — cluster nearby points (within `snap_distance`) → centroid
2. **Resolve junctions** — split walls at T/X intersections (iterate 3 passes, skip if >300 walls)
3. **Filter short** — discard walls < `min_wall_length`
4. **Deduplicate** — remove exact duplicates (both directions)
5. **Snap again** — second pass after dedup
6. **Filter noise** — remove degree-1 stubs shorter than threshold

Output: connected graph with nodes (endpoints/junctions) and edges (wall segments).

### 4. Topology Validation
Check:
- Zero-length walls (error)
- Graph connectivity (warning if disconnected components)
- Floating walls — degree-1 nodes (warning)
- Duplicate segments (warning)

Auto-repair before room discovery if validation fails.

### 5. Room Discovery (not yet implemented)
- Use Shapely `polygonize()` on wall graph
- Output: closed room polygons
- **Never** ask LLM for rooms — use geometry

### 6. Room Labeling (LLM layer)
Input: polygon, area, adjacency graph
Output: semantic labels (Kitchen, Bedroom, Bathroom)

LLM is semantic layer only, not geometry layer.

### 7. Dimension Solver (not yet implemented)
Dimension lines are hard constraints. If pixel coordinates disagree with labeled dimensions, dimensions win.

### 8. Door/Window Insertion (not yet implemented)
- Doors: openings only, never walls. Arc graphics = metadata
- Windows: openings only, never modify topology
- Insert into existing wall graph after topology complete

### 9. 3D Extrusion (not yet implemented)
Only after topology finalized:
```
Wall Graph → Extrusion → Three.js
```
Never extrude from segmentation masks or room inference directly.

## Tech Stack

- **Python 3.11+** with `uv` package manager
- **OpenCV** — thresholding, morphology, contours, Hough, skeletonization
- **NumPy** — array operations, least-squares circle fitting
- **Shapely** — polygon operations, `polygonize()` for room discovery
- **NetworkX** — graph topology, connectivity checks, loop detection
- **Pydantic** — data models (Wall, Point, WallGraph)

No GPU required. No ML training. Runs on 4 vCPU / 8GB RAM.

## Key Parameters (from constants.py)

- `MIN_WALL_LENGTH = 20` px
- `MIN_COMPONENT_AREA = 50` px²
- `SNAP_DISTANCE = 5` px (endpoint clustering)
- `ANGLE_TOLERANCE = 0.3` (H/V classification: `|dy| < |dx| * 0.3`)
- `PARALLEL_MERGE_DIST = 20` px (centerline merging — must handle thick walls with ~20px faces)
- `COLLINEAR_MERGE_GAP = 60` px (merge through door openings up to 60px wide)
- `ARC_MIN_RADIUS = 25`, `ARC_MAX_RADIUS = 3000` px (structural arcs, not junction noise)
- `ARC_MIN_ANGLE_SPAN = 45°`, `ARC_MAX_ANGLE_SPAN = 270°` (filter near-full-circle noise)
- `ARC_CURVATURE_THRESHOLD = 0.05` (curvature detection on raw contours)
- `collinear_tol = 8` px (in `_merge_axis` — prevents chain-merging across image)
- `HOUGH_THRESHOLD = 30`, `HOUGH_MIN_LINE_LENGTH = 30`, `HOUGH_MAX_LINE_GAP = 10`

## Common Pitfalls

1. **Don't skeletonize+prune before Hough for thin walls** — skeleton of 3-5px walls has too few pixels. Pruning removes most of what's left. Use Hough on cleaned mask directly.
2. **Don't trust LLM for geometry** — use deterministic CV, LLM only for labels.
3. **Don't derive walls from rooms** — rooms are derived from walls.
4. **Arc detection: use thickness-filtered mask** — morphological opening with 5×5 kernel preserves wall body shape including fillets. Raw mask includes text that creates false arcs at junctions.
5. **Junction resolution is O(n²)** — skip if wall count > 300 to avoid latency.
6. **Diagonal walls often dropped** — typical floorplans are orthogonal. Add explicit support if needed.
7. **`_remove_arc_overlaps` must preserve long structural walls** — only remove walls shorter than 2× arc diameter. A 463px top wall should NOT be removed because a 28px-radius arc overlaps its midpoint.
8. **Chain-merging in `_merge_axis`** — without `collinear_tol` anchor and `ref_pos`, iterative averaging merges x=68→x=70→x=114→x=118, collapsing left wall into internal wall. Fix: only merge walls within 8px of the reference wall's original position.
9. **Noise wall isolation** — two text/dimension lines near each other shouldn't count as "structural proximity" for the isolation filter. Only count proximity to walls that have perpendicular intersections (structural walls).
10. **Merge pipeline needs multiple passes** — `_merge_nearby_walls` → `_merge_parallel_pairs` → `_merge_nearby_walls` → `_merge_parallel_pairs`. Centerlines from the first parallel merge need collinear consolidation before the second parallel merge can pair them correctly.

## Debug Visualization

Save intermediate outputs as PNG:
- `debug_img3_new.png` — final wall overlay on original image
- `debug_img3_wall_mask.png` — thick mask (5×5 open, wall body only)
- `debug_img3_text_mask.png` — thin mask subtracted from thick mask (text/dimension remnants)
- `debug_img3_cleaned.png` — cleaned binary mask (before Hough)

Use `WallDetector.visualize(image, walls, output_path)` to draw walls (green = straight, cyan = arcs, blue/red dots = start/end).

## Related Projects

- **Plan3D** — this pipeline is the Python optimization layer for a larger LLM-based floorplan-to-3D system
- **DeepFloorplan**, **Floor-SP**, **CubiCasa5K** — ML-based alternatives (require GPU, training data)
