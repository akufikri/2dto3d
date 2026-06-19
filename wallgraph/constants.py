"""
Named constants for wallgraph pipeline.

All tunable thresholds and magic numbers are centralised here so they can be
changed in one place and documented consistently.
"""

# ---------------------------------------------------------------------------
# WallDetector defaults
# ---------------------------------------------------------------------------

#: Minimum pixel length a wall segment must have to be kept.
MIN_WALL_LENGTH: float = 20.0

#: Minimum area (px²) a connected component must have to survive noise removal.
MIN_COMPONENT_AREA: int = 50

#: Kernel size for morphological *open* (noise removal).
MORPH_OPEN_KERNEL: int = 3

#: Kernel size for *heavy* open used in text/dimension filtering.
#: Walls in typical floorplans are ≥5 px thick; text strokes are 1–2 px.
#: A 7×7 open removes text/dimensions while preserving wall bodies.
MORPH_OPEN_KERNEL_HEAVY: int = 7

#: Kernel size for the pre-Hough open pass.
#: Applied to the cleaned binary mask immediately before Hough line detection.
#: Removes thin symbols (stair lines, toilet arcs, furniture) that survived the
#: standard 3×3 open and dimension-line filter, without destroying wall bodies.
#: Increase for noisier/more annotated drawings; 0 disables the pass.
HOUGH_PRE_OPEN_KERNEL: int = 3

#: Kernel size for morphological *close* (gap filling).
MORPH_CLOSE_KERNEL: int = 3

#: Number of morphological close iterations.
MORPH_CLOSE_ITERATIONS: int = 2

#: ``approxPolyDP`` epsilon expressed as a ratio of contour perimeter.
APPROX_EPSILON_RATIO: float = 0.02

#: Ratio tolerance used to classify a segment as horizontal or vertical.
#: A segment is H if ``|dy| < |dx| * ANGLE_TOLERANCE``.
ANGLE_TOLERANCE: float = 0.3

#: Maximum perpendicular distance (px) between two parallel walls
#: for them to be candidates for centerline merging.
#: Keep small (8) to preserve thick wall faces as separate segments
#: rather than collapsing to a single centerline.
PARALLEL_MERGE_DIST: float = 8.0

#: Maximum gap (px) between two collinear wall segments for them
#: to be merged into one continuous wall. Small values (15) keep
#: walls split at door openings rather than merging through gaps.
COLLINEAR_MERGE_GAP: float = 15.0

#: Curvature ratio above which a contour section is considered an arc.
#: At ARC_CURVATURE_THRESHOLD=0.05 and window=7 on CHAIN_APPROX_NONE
#: contours (pixel spacing), a 50 px radius arc generates curvature
#: values of ~0.04–0.06, while straight sections are <0.01.
ARC_CURVATURE_THRESHOLD: float = 0.05

#: Contours whose area is below this threshold AND whose aspect ratio is
#: below ``SQUARE_MAX_ASPECT`` are classified as small squares (door/window
#: symbols) and discarded.
SQUARE_MIN_AREA: int = 150

#: Aspect ratio limit for the small-square symbol filter.
SQUARE_MAX_ASPECT: float = 2.0

#: Hough line threshold (minimum votes).
HOUGH_THRESHOLD: int = 35

#: Hough minimum line length (px).
HOUGH_MIN_LINE_LENGTH: float = 30.0

#: Hough maximum gap between collinear points (px).
HOUGH_MAX_LINE_GAP: float = 15.0

# ---------------------------------------------------------------------------
# WallGraphBuilder defaults
# ---------------------------------------------------------------------------

#: Maximum distance (px) between two endpoints for them to be snapped
#: together into one shared node.
SNAP_DISTANCE: float = 8.0

#: Minimum wall length kept after graph construction.
GRAPH_MIN_WALL_LENGTH: float = 35.0

#: Maximum number of walls allowed for junction resolution.
#: Above this the step is skipped to avoid O(n²) latency.
JUNCTION_RESOLVE_MAX_WALLS: int = 400

#: Number of junction-resolution iterations.
JUNCTION_RESOLVE_ITERATIONS: int = 3

#: Minimum number of nodes a connected component must have to survive
#: post-build filtering.  Components below this threshold are small
#: isolated clusters (fixture outlines, symbols) not structural walls.
MIN_COMPONENT_NODES: int = 4

#: Angle tolerance for graph-level H/V wall filter.
#: A wall is horizontal if ``|dy| < |dx| * GRAPH_ANGLE_TOLERANCE``.
#: A wall is vertical if ``|dx| < |dy| * GRAPH_ANGLE_TOLERANCE``.
#: Diagonal walls (neither H nor V) are fixture/symbol noise in rectilinear plans.
#: Set to 0.0 to disable the filter.
GRAPH_ANGLE_TOLERANCE: float = 0.4

#: A wall is considered *noise* (isolated stub) when its length is below
#: ``GRAPH_MIN_WALL_LENGTH * NOISE_LENGTH_RATIO``.
NOISE_LENGTH_RATIO: float = 1.5

#: Maximum distance (px) a dangling (degree-1) endpoint can be extended to
#: reach a nearby wall.  Larger values close bigger gaps but risk false
#: connections.  Set to 0 to disable Layer-C gap closing.
DANGLING_ENDPOINT_MAX_GAP: float = 30.0

# ---------------------------------------------------------------------------
# WallGraphValidator defaults
# ---------------------------------------------------------------------------

#: Minimum arc angle span (radians) to keep a fitted arc wall.
ARC_MIN_ANGLE_SPAN: float = 0.785398  # π / 4 (45°)

#: Maximum arc angle span (radians) — near-full-circle arcs are noise.
ARC_MAX_ANGLE_SPAN: float = 4.71  # 3π / 2 (270°)

#: Minimum arc radius (px).
ARC_MIN_RADIUS: float = 20.0

#: Maximum arc radius (px) — arcs larger than this are likely noise.
ARC_MAX_RADIUS: float = 3000.0

#: Maximum distance (px) from an arc center to a vertex for the fillet check.
ARC_VERTEX_PROXIMITY: float = 30.0

# ---------------------------------------------------------------------------
# Door detection thresholds
# ---------------------------------------------------------------------------

#: Minimum radius (px) of a door swing arc.
#: Door swings in 1:100 scale plans are ~18-80 px radius.
DOOR_ARC_MIN_RADIUS: float = 18.0

#: Maximum radius (px) of a door swing arc.
DOOR_ARC_MAX_RADIUS: float = 80.0

#: Door swing arc angle span must be close to 90° (π/2).
#: Allow ±25° tolerance: 65°–115°.
DOOR_ARC_MIN_SPAN: float = 1.134  # 65° in radians
DOOR_ARC_MAX_SPAN: float = 2.007  # 115° in radians

#: Maximum perpendicular distance (px) from door arc center to nearest wall
#: for the arc to be classified as a door swing.
DOOR_WALL_PROXIMITY: float = 20.0

# ---------------------------------------------------------------------------
# Gap-based door detection
# ---------------------------------------------------------------------------

#: Minimum gap size (px) in a wall to be considered a door opening.
DOOR_GAP_MIN: float = 15.0

#: Maximum gap size (px) — larger openings are passages/arches, not doors.
DOOR_GAP_MAX: float = 120.0

#: Pixel tolerance when searching for content inside a wall gap.
DOOR_GAP_SEARCH_WIDTH: float = 30.0

#: Minimum arc pixels inside gap region to classify as swing door.
DOOR_GAP_ARC_MIN_PX: int = 8

#: Two parallel lines within this distance → sliding door.
DOOR_GAP_SLIDING_MAX_SEP: float = 25.0

# ---------------------------------------------------------------------------
# Dimension line detection thresholds
# ---------------------------------------------------------------------------

#: Minimum aspect ratio (max/min bounding dim) to classify as dimension line.
#: Dimension lines are very elongated; walls are moderate aspect.
DIM_LINE_MIN_ASPECT: float = 15.0

#: Maximum height (or width) in pixels for a thin element (dimension/text).
DIM_LINE_MAX_STROKE: int = 4

#: Minimum area (px²) for a component to be classified as text vs wall.
#: Components below this area in the heavy-open mask are likely text.
TEXT_AREA_THRESHOLD: int = 80

#: Maximum stroke width (px) for text/dimension elements.
#: Components thinner than this after light open are filtered as text.
TEXT_MAX_STROKE_WIDTH: int = 3

# ---------------------------------------------------------------------------
# Wall mask classification thresholds
# ---------------------------------------------------------------------------

#: Minimum bounding-box dimension (px) for a component to be classified
#: as a wall.  Walls are thick (≥5 px); text strokes are thin (1–2 px).
WALL_MIN_DIM: int = 5

#: Minimum solidity (area / bounding-box area) for wall classification.
#: Walls are solid rectangles (solidity ≈ 0.4–1.0); text/symbols are sparse
#: (solidity ≈ 0.05–0.15).  A threshold of 0.15 lets thin walls pass while
#: blocking merged text clusters.
WALL_SOLIDITY: float = 0.15

#: Maximum aspect ratio (max_dim / min_dim) for wall classification.
#: Walls are moderate aspect (≤10); dimension lines are extreme (>>10).
WALL_MAX_ASPECT: float = 10.0

#: Area multiplier for unconditional wall classification.
#: Components with area ≥ min_component_area * this multiplier are always
#: walls regardless of solidity/aspect — prevents false-negatives on large
#: legitimate wall clusters that may have low solidity from holes.
WALL_LARGE_AREA_MULT: int = 20
