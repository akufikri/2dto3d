# Plan3D Refactor Blueprint: Wall-First CAD Reconstruction Engine

## Executive Summary

Tujuan utama refactor Plan3D adalah mengubah pendekatan dari:

```text
Image
↓
LLM Room Classification
↓
Room Polygons
↓
Infer Walls
↓
3D Model
```

menjadi:

```text
Image
↓
Wall Detection
↓
Wall Graph Reconstruction
↓
Topology Validation
↓
Dimension Solver
↓
Room Discovery
↓
Room Labeling
↓
CAD / 3D Export
```

Prinsip utama:

> Walls are the source of truth.
>
> Rooms are derived from walls.
>
> Never derive walls from rooms.

---

# Current Plan3D Flow

## Existing Pipeline

```text
Upload
↓
PASS 1 Structure (LLM)
↓
Rooms + Walls + Doors + Windows
↓
Wall Retrace (LLM)
↓
deriveWallsFromRooms()
↓
CV Fallback
↓
Geometry Cleanup
↓
3D Extrusion
```

### Problems

1. Rooms become authoritative.
2. Walls are inferred from room understanding.
3. Geometry accuracy depends on LLM reasoning.
4. Topology errors propagate to 3D.
5. Dimensions are not true constraints.
6. CV is only a fallback.

---

# Root Cause Analysis

## Why Larger Models Don't Solve It

Even a large VLM:

- Qwen VL
- GPT-4o
- Gemini 2.5
- Claude Vision

still performs:

```text
Interpretation
```

rather than:

```text
Measurement
```

The model estimates geometry.

CAD reconstruction requires exact geometry.

### LLM Behavior

```text
"This looks like a bedroom."
```

### CAD Behavior

```text
Wall exists at coordinate (1250, 4800)
```

These are fundamentally different tasks.

---

# New Architecture

## Wall-First Architecture

```text
Image
↓
Wall Detection
↓
Wall Segments
↓
Wall Graph Builder
↓
Topology Validation
↓
Dimension Solver
↓
Room Discovery
↓
Room Labeling (LLM)
↓
Door / Window Placement
↓
3D Extrusion
```

---

# Core Principle

## Walls Are Authoritative

Correct:

```text
Walls
↓
Rooms
↓
Labels
```

Wrong:

```text
Labels
↓
Rooms
↓
Walls
```

---

# Recommended Data Model

```ts
interface Wall {
  id: string;
  start: { x: number; y: number };
  end: { x: number; y: number };
  thickness: number;
  exterior: boolean;
}

interface Door {
  wallId: string;
  offset: number;
  width: number;
}

interface Window {
  wallId: string;
  offset: number;
  width: number;
}

interface WallGraph {
  walls: Wall[];
  doors: Door[];
  windows: Window[];
}
```

Source of truth:

```text
WallGraph
```

Not:

```text
Room[]
```

---

# Detailed Pipeline Design

## PASS A — Wall Detection

Goal:

```text
Detect wall geometry only.
```

Ignore:

- Room labels
- Furniture
- Toilet icons
- Door arcs
- Dimensions
- Windows

Output:

```json
{
  "walls": [
    {
      "x1": 100,
      "y1": 200,
      "x2": 600,
      "y2": 200
    }
  ]
}
```

---

## PASS B — Wall Graph Builder

Input:

```text
Raw Wall Segments
```

Operations:

- Endpoint snapping
- Collinear merge
- T-junction detection
- Cross-junction detection
- Connectivity graph creation

Output:

```text
Validated Wall Graph
```

---

## PASS C — Topology Validation

Checks:

- Floating walls
- Disconnected islands
- Broken perimeter
- Duplicate segments
- Invalid intersections

Validation Rules:

```text
Every room must be closed.
Every wall must connect.
Perimeter must be continuous.
```

---

## PASS D — Dimension Solver

Extract dimensions:

```text
2550
1300
1700
3900
3000
```

Constraint:

```text
2550 + 1300 + 1700 + 3900 + 3000
= 12750
```

Dimension constraints override raster pixel locations.

---

## PASS E — Room Discovery

Never guess rooms.

Instead:

```text
Wall Graph
↓
Polygon Detection
↓
Room Candidates
```

Use:

```python
polygonize()
```

from Shapely.

---

## PASS F — Room Labeling

This is where LLM belongs.

Example:

```text
Polygon #1 → Kitchen
Polygon #2 → Bedroom
Polygon #3 → Bathroom
```

LLM is semantic layer only.

Not geometry layer.

---

## PASS G — Door & Window Placement

Rules:

### Doors

Door arcs are NOT walls.

Door swing graphics:

```text
Metadata only
```

### Windows

Windows are openings.

They do not modify topology.

---

## PASS H — 3D Extrusion

Only after topology is finalized.

```text
Wall Graph
↓
Extrusion
↓
Three.js
```

---

# Open Source Stack Recommendation

## OpenCV

Purpose:

- Thresholding
- Morphology
- Contours
- Hough lines
- Wall masks

Role:

```text
Primary wall detector
```

not fallback.

---

## Shapely

Purpose:

- Polygon operations
- Intersections
- Buffers
- Unions
- Geometry cleanup

Critical for:

```text
WallGraphBuilder
```

---

## NetworkX

Purpose:

```text
Wall topology graph
```

Nodes:

```text
Wall endpoints
```

Edges:

```text
Wall segments
```

Capabilities:

- Loop detection
- Connectivity checks
- Topology validation

---

## Potrace

Purpose:

```text
Raster → Vector
```

Pipeline:

```text
Wall Mask
↓
Potrace
↓
Vector Geometry
```

---

## Detectron2

Purpose:

- Wall segmentation
- Door detection
- Window detection

Optional future enhancement.

---

## MMDetection

Alternative to Detectron2.

Useful for:

- Symbol detection
- Floorplan object detection

---

## DeepFloorplan

Provides:

- Wall segmentation
- Room segmentation
- Icon detection

Useful as baseline model.

---

## Floor-SP

Interesting because it focuses on:

```text
Topology
Graph Reconstruction
```

which is close to desired architecture.

---

## CubiCasa5K Dataset

Recommended training dataset.

Useful for:

- Wall detection
- Door detection
- Window detection

---

# Pure Python Stack

A nearly complete implementation can be built in Python.

```text
FastAPI
│
├── OpenCV
├── NumPy
├── Shapely
├── NetworkX
├── Pydantic
├── Potrace
├── Trimesh
└── Optional LLM
```

---

# Hardware Requirements

## Deterministic Pipeline

```text
OpenCV
Shapely
NetworkX
Potrace
```

Runs on:

```text
4 vCPU
8 GB RAM
```

No GPU required.

---

## Heavy Vision Models

Require GPU only if:

- DeepFloorplan
- Detectron2
- Custom segmentation models

are run locally.

---

# Recommended MVP Roadmap

## Phase 1

Build:

```text
WallGraph Model
```

---

## Phase 2

Build:

```text
WallGraphBuilder
```

---

## Phase 3

Build:

```text
Topology Validator
```

---

## Phase 4

Build:

```text
Room Discovery Engine
```

---

## Phase 5

Build:

```text
Dimension Solver
```

---

## Phase 6

Integrate with Plan3D

---

## Phase 7

Add LLM labeling layer

---

# Final Target Architecture

```text
Upload Image
        │
        ▼
OpenCV
        │
        ▼
Wall Segmentation
        │
        ▼
Potrace
        │
        ▼
Wall Graph Builder
(Shapely + NetworkX)
        │
        ▼
Topology Validator
        │
        ▼
Dimension Solver
        │
        ▼
Room Discovery
        │
        ▼
LLM Room Labeling
        │
        ▼
Door / Window Placement
        │
        ▼
Three.js / CAD Export
```

## Final Principle

The biggest accuracy gain will not come from switching:

```text
Gemini → Qwen → GPT
```

The biggest gain comes from changing:

```text
LLM-generated geometry
```

into:

```text
Deterministic geometry
+
LLM semantic labeling
```
