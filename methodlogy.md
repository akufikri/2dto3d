# PLAN3D PYTHON OPTIMIZATION BLUEPRINT (NO TRAINING, NO MODEL CHANGES)

## OBJECTIVE

Improve floorplan-to-3D conversion accuracy by fixing geometry reconstruction in Python before introducing model training or changing LLM providers.

Current bottleneck is NOT Gemini, Qwen, GPT, or prompt quality.

Current bottleneck is:

* Wall extraction
* Wall topology reconstruction
* Geometry validation
* Missing wall graph layer

The goal is to transform Plan3D from:

Image
→ LLM
→ Rooms
→ Walls
→ 3D

into:

Image
→ Wall Detection
→ Wall Graph
→ Topology Validation
→ Room Discovery
→ 3D

---

# CORE PRINCIPLE

Walls are the source of truth.

Rooms are derived from walls.

Never derive walls from rooms.

Wrong:

Image
→ Room Classification
→ Generate Walls

Correct:

Image
→ Detect Walls
→ Build Wall Graph
→ Discover Rooms

---

# TARGET PYTHON STACK

FastAPI
OpenCV
NumPy
Scikit-Image
Shapely
NetworkX
Pydantic

Optional:

Potrace
Trimesh

No GPU required.

No model training required.

Can run on:

4 vCPU
8 GB RAM VPS

---

# STAGE 1 — WALL MASK EXTRACTION

Current issue:

Walls are being treated as lines.

Instead:

Treat walls as solid objects.

Process:

Image
↓
Grayscale
↓
Adaptive Threshold
↓
Morphological Close
↓
Morphological Open
↓
Wall Binary Mask

Output:

wall_mask.png

Goal:

Preserve full wall thickness.

Do not convert to centerlines yet.

---

# STAGE 2 — WALL THICKNESS PRESERVATION

Store:

{
centerline,
thickness
}

Do not lose thickness information.

Reason:

Structural walls
Household shelters
Load-bearing walls

must remain thicker than normal walls.

Current extraction loses this information.

---

# STAGE 3 — SKELETONIZATION

After wall mask is stable:

wall mask
↓
skeletonize()

Output:

single-pixel wall centerline

Important:

Keep thickness metadata separately.

Do not replace wall geometry with skeleton only.

Store:

Wall {
centerline,
thickness
}

---

# STAGE 4 — JUNCTION DETECTION

Detect:

L Junction

└

T Junction

├

Cross Junction

┼

These become graph nodes.

Output:

nodes[]
edges[]

Use:

NetworkX

---

# STAGE 5 — WALL GRAPH BUILDER

Create:

WallGraph

Nodes:

wall endpoints
junctions

Edges:

wall segments

Process:

snap endpoints
merge collinear walls
remove duplicates
repair small gaps

Output:

Connected WallGraph

---

# STAGE 6 — TOPOLOGY VALIDATION

Validate:

1. No floating walls
2. No disconnected islands
3. No overlapping walls
4. Closed perimeter
5. Closed rooms

If validation fails:

auto repair

before room discovery.

---

# STAGE 7 — ARC CLASSIFICATION

Current issue:

Door arcs and curved walls are confused.

Classifier:

Door Arc:

* thin
* single line
* attached to doorway
* approximately 90° arc

Curved Wall:

* thick
* double boundary
* continuous wall object

Rule:

Door arcs must never become walls.

---

# STAGE 8 — ROOM DISCOVERY

Do NOT ask LLM for rooms.

Use geometry.

Process:

Wall Graph
↓
Polygon Detection
↓
Room Polygons

Use:

Shapely polygonize()

Output:

RoomPolygon[]

Only after this step:

assign labels.

---

# STAGE 9 — DIMENSION SOLVER

Dimension lines are hard constraints.

Example:

2550
1300
1700
3900
3000

must satisfy:

2550 + 1300 + 1700 + 3900 + 3000 = 12750

If wall coordinates disagree:

dimensions win.

Do not trust raster positions more than dimensions.

---

# STAGE 10 — ROOM LABELING

Only here use LLM.

Input:

Polygon
Area
Adjacency

Output:

Kitchen
Bedroom
Bathroom
Living Room

LLM should classify rooms only.

Never generate geometry.

---

# STAGE 11 — DOOR / WINDOW INSERTION

After topology is complete.

Rules:

Doors:

* opening only
* never walls

Windows:

* opening only
* never topology

Insert into existing wall graph.

Never generate wall structure from doors/windows.

---

# STAGE 12 — 3D EXTRUSION

Final step only.

WallGraph
↓
Extrusion
↓
Three.js

Never extrude directly from segmentation masks.

Never extrude directly from room inference.

Always extrude from validated wall graph.

---

# HIGH PRIORITY REFACTOR TASKS

1. Remove dependency on deriveWallsFromRooms()

2. Create WallGraphBuilder

3. Add TopologyValidator

4. Add Junction Detection

5. Add Thickness Preservation

6. Add Arc Classification

7. Add Dimension Solver

8. Use LLM only for room labels

---

# EXPECTED IMPACT

Current system:

Accuracy heavily dependent on LLM reasoning.

After refactor:

Accuracy primarily dependent on geometry.

Expected improvement:

+20% to +40% accuracy

without changing model

without training

without GPU

without fine-tuning

---

# FINAL TARGET PIPELINE

Upload Image
↓
OpenCV Threshold
↓
Wall Mask
↓
Skeletonization
↓
Thickness Recovery
↓
Junction Detection
↓
Wall Graph Builder
↓
Topology Validator
↓
Dimension Solver
↓
Room Discovery
↓
Room Labeling (LLM)
↓
Door/Window Placement
↓
3D Extrusion
↓
Three.js

FINAL RULE:

Wall Graph is the source of truth.

Everything else is derived from the Wall Graph.
