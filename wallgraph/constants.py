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
SNAP_DISTANCE: float = 5.0

#: Minimum wall length kept after graph construction.
GRAPH_MIN_WALL_LENGTH: float = 30.0

#: Maximum number of walls allowed for junction resolution.
#: Above this the step is skipped to avoid O(n²) latency.
JUNCTION_RESOLVE_MAX_WALLS: int = 300

#: Number of junction-resolution iterations.
JUNCTION_RESOLVE_ITERATIONS: int = 3

#: A wall is considered *noise* (isolated stub) when its length is below
#: ``GRAPH_MIN_WALL_LENGTH * NOISE_LENGTH_RATIO``.
NOISE_LENGTH_RATIO: float = 0.7

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

#: Minimum area (px²) for a component to be classified as text vs wall.
#: Components below this area in the heavy-open mask are likely text.
TEXT_AREA_THRESHOLD: int = 80

#: Maximum stroke width (px) for text/dimension elements.
#: Components thinner than this after light open are filtered as text.
TEXT_MAX_STROKE_WIDTH: int = 3
