"""
WallDetector: detects wall segments from a floorplan image.

Detection is fully deterministic (OpenCV only — no ML).  Three strategies are
available via the ``method`` parameter of :meth:`detect`:

* ``"binary"`` *(default)* — Otsu threshold + morphology + contour tracing.
  Works well for clean architectural drawings.
* ``"hough"`` — skeletonise the binary mask then apply probabilistic Hough
  transform.  Produces fewer but more axis-aligned segments.
* ``"canny"`` — Canny edge detection + Hough transform.  Useful when wall
  boundaries are defined by edges rather than filled regions.
* ``"contour"`` — Alias for ``"binary"`` kept for backwards compatibility.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

from .models import Wall, Point, Door
from .constants import (
    MIN_WALL_LENGTH,
    MIN_COMPONENT_AREA,
    MORPH_OPEN_KERNEL,
    MORPH_OPEN_KERNEL_HEAVY,
    MORPH_CLOSE_KERNEL,
    MORPH_CLOSE_ITERATIONS,
    APPROX_EPSILON_RATIO,
    ANGLE_TOLERANCE,
    PARALLEL_MERGE_DIST,
    COLLINEAR_MERGE_GAP,
    ARC_CURVATURE_THRESHOLD,
    SQUARE_MIN_AREA,
    SQUARE_MAX_ASPECT,
    HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LENGTH,
    HOUGH_MAX_LINE_GAP,
    ARC_MIN_ANGLE_SPAN,
    ARC_MAX_ANGLE_SPAN,
    ARC_MIN_RADIUS,
    ARC_MAX_RADIUS,
    ARC_VERTEX_PROXIMITY,
    TEXT_AREA_THRESHOLD,
    WALL_MIN_DIM,
    WALL_SOLIDITY,
    WALL_MAX_ASPECT,
    WALL_LARGE_AREA_MULT,
    DOOR_ARC_MIN_RADIUS,
    DOOR_ARC_MAX_RADIUS,
    DOOR_ARC_MIN_SPAN,
    DOOR_ARC_MAX_SPAN,
    DOOR_WALL_PROXIMITY,
    DIM_LINE_MIN_ASPECT,
    DIM_LINE_MAX_STROKE,
    DOOR_GAP_MIN,
    DOOR_GAP_MAX,
    DOOR_GAP_SEARCH_WIDTH,
    DOOR_GAP_ARC_MIN_PX,
    DOOR_GAP_SLIDING_MAX_SEP,
    HOUGH_PRE_OPEN_KERNEL,
)


class WallDetector:
    """Detect wall segments from a floorplan image using OpenCV.

    Parameters
    ----------
    params:
        Optional dict of parameter overrides.  Any key that matches a
        ``constants.*`` name can be supplied here.  Unknown keys raise
        ``ValueError``.
    """

    _VALID_PARAMS = {
        "min_wall_length",
        "min_component_area",
        "open_kernel",
        "open_kernel_heavy",
        "close_kernel",
        "close_iterations",
        "approx_epsilon_ratio",
        "angle_tolerance",
        "parallel_merge_dist",
        "arc_curvature_threshold",
        "min_square_area",
        "max_square_aspect",
        "min_wall_half_thick",
        "hough_threshold",
        "hough_min_line_length",
        "hough_max_line_gap",
        "text_area_threshold",
        "arc_vertex_proximity",
        "wall_min_dim",
        "wall_solidity",
        "wall_max_aspect",
        "wall_large_area_mult",
        "hough_pre_open_kernel",
    }

    def __init__(self, params: dict | None = None) -> None:
        self.params: dict = {
            "min_wall_length": MIN_WALL_LENGTH,
            "min_component_area": MIN_COMPONENT_AREA,
            "open_kernel": MORPH_OPEN_KERNEL,
            "open_kernel_heavy": MORPH_OPEN_KERNEL_HEAVY,
            "close_kernel": MORPH_CLOSE_KERNEL,
            "close_iterations": MORPH_CLOSE_ITERATIONS,
            "approx_epsilon_ratio": APPROX_EPSILON_RATIO,
            "angle_tolerance": ANGLE_TOLERANCE,
            "parallel_merge_dist": PARALLEL_MERGE_DIST,
            "arc_curvature_threshold": ARC_CURVATURE_THRESHOLD,
            "min_square_area": SQUARE_MIN_AREA,
            "max_square_aspect": SQUARE_MAX_ASPECT,
            "min_wall_half_thick": 1.5,
            "hough_threshold": HOUGH_THRESHOLD,
            "hough_min_line_length": HOUGH_MIN_LINE_LENGTH,
            "hough_max_line_gap": HOUGH_MAX_LINE_GAP,
            "text_area_threshold": TEXT_AREA_THRESHOLD,
            "arc_vertex_proximity": ARC_VERTEX_PROXIMITY,
            "wall_min_dim": WALL_MIN_DIM,
            "wall_solidity": WALL_SOLIDITY,
            "wall_max_aspect": WALL_MAX_ASPECT,
            "wall_large_area_mult": WALL_LARGE_AREA_MULT,
            "hough_pre_open_kernel": HOUGH_PRE_OPEN_KERNEL,
        }
        if params:
            unknown = set(params) - self._VALID_PARAMS
            if unknown:
                raise ValueError(f"Unknown WallDetector params: {unknown}")
            self.params.update(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[Wall]:
        """Detect walls using binary threshold + morphology + contour method.

        This is the recommended method for clean architectural drawings.
        """
        walls, _ = self._detect_binary(image_path)
        return walls

    def detect_with_doors(self, image_path: str) -> tuple[list[Wall], list[Door]]:
        """Detect walls and doors. Returns (walls, doors)."""
        return self._detect_binary(image_path)

    def detect_hough(self, image_path: str) -> list[Wall]:
        """Detect walls using skeletonisation + probabilistic Hough transform."""
        img = self._load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cleaned = self._remove_small_components(binary)
        skel = self._skeletonize(cleaned)

        raw_lines = self._hough_lines(skel)
        snapped = self._snap_to_axis(raw_lines)

        walls: list[Wall] = []
        for idx, ((x1, y1), (x2, y2)) in enumerate(snapped):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < self.params["min_wall_length"]:
                continue
            walls.append(
                Wall(
                    id=f"W{idx:04d}",
                    start=Point(x=x1, y=y1),
                    end=Point(x=x2, y=y2),
                )
            )

        walls = self._merge_parallel_pairs(walls)
        walls = self._filter_small_walls(walls)
        return walls

    def detect_canny(self, image_path: str) -> list[Wall]:
        """Detect walls using Canny edges + probabilistic Hough transform.

        Useful when wall boundaries are defined by sharp edges rather than
        filled regions (e.g. line-drawing style floorplans).
        """
        img = self._load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        raw_lines = self._hough_lines(edges)
        snapped = self._snap_to_axis(raw_lines)

        walls: list[Wall] = []
        for idx, ((x1, y1), (x2, y2)) in enumerate(snapped):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < self.params["min_wall_length"]:
                continue
            walls.append(
                Wall(
                    id=f"W{idx:04d}",
                    start=Point(x=x1, y=y1),
                    end=Point(x=x2, y=y2),
                )
            )

        walls = self._merge_parallel_pairs(walls)
        walls = self._filter_small_walls(walls)
        return walls

    def detect_contour(self, image_path: str) -> list[Wall]:
        """Alias for :meth:`detect` (binary threshold + contour method).

        Kept for backwards compatibility.
        """
        return self._detect_binary(image_path)

    def detect_from_mask(self, mask_path: str) -> tuple[list[Wall], list[Door]]:
        """Detect walls from a pre-computed binary wall mask (white=wall).

        Designed for ML-generated masks (CubiCasa5K API output).  The mask
        already shows only walls — no furniture / annotation noise — so the
        pipeline is much simpler than ``detect_with_doors``:

        1. Load mask, binarize at 127.
        2. Remove small noise components.
        3. Skeletonize → 1-pixel-wide wall centrelines.
        4. Prune skeleton spurs (short branches from mask edges / T-junctions).
        5. Probabilistic Hough on skeleton with mask-tuned params.
        6. Snap to H/V axis, merge collinear segments.
        7. Gap-based door detection on original binary mask.

        Parameters
        ----------
        mask_path:
            Path to binary mask PNG (white=wall pixels, black=background).
        """
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load mask: {mask_path}")

        # Binarize (mask may have anti-aliasing values between 0-255)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Remove isolated speckle noise
        cleaned = self._remove_small_components(binary)

        # Skeletonize: reduce thick wall bands → 1-px centrelines
        skel = self._skeletonize(cleaned)

        # Prune short spurs (4 iterations ≈ remove 4-px stubs at T-junctions)
        # Keep low so valid short walls are not erased on small masks (600 px)
        skel = self._fast_prune_skeleton(skel, iterations=4)

        # Skeleton lines are 1-px sparse — use lower threshold / min-length than
        # the global defaults which are tuned for thick raw-image content.
        # maxLineGap=25 bridges small skeleton interruptions at T-junctions.
        saved = {k: self.params[k] for k in
                 ("hough_threshold", "hough_min_line_length", "hough_max_line_gap")}
        self.params["hough_threshold"]      = 15
        self.params["hough_min_line_length"] = 15
        self.params["hough_max_line_gap"]    = 25
        try:
            raw_lines = self._hough_lines(skel)
        finally:
            self.params.update(saved)

        snapped = self._snap_to_axis(raw_lines)

        walls: list[Wall] = []
        for idx, ((x1, y1), (x2, y2)) in enumerate(snapped):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < self.params["min_wall_length"]:
                continue
            walls.append(
                Wall(
                    id=f"W{idx:04d}",
                    start=Point(x=x1, y=y1),
                    end=Point(x=x2, y=y2),
                )
            )

        walls = self._merge_nearby_walls(walls)

        # Skip _filter_outside_boundary: ML mask already contains only wall
        # pixels — every Hough line found on it is a real wall segment.
        # Boundary / isolation filtering would incorrectly remove valid walls
        # whose perpendicular neighbour happens to end just short of them.

        walls = self._merge_nearby_walls(walls)

        # Door detection from wall gaps in original binary mask
        gap_doors = self._detect_doors_from_gaps(walls, binary)

        walls = self._filter_small_walls(walls)
        return walls, gap_doors

    def visualize(
        self,
        image_path: str,
        walls: list[Wall],
        output_path: str | None = None,
        doors: list[Door] | None = None,
    ) -> np.ndarray:
        """Draw detected walls and doors on the source image.

        Straight walls = green, arc walls = cyan, doors = yellow arc.
        """
        img = self._load_image(image_path)
        for w in walls:
            p1 = (int(w.start.x), int(w.start.y))
            p2 = (int(w.end.x), int(w.end.y))
            if w.is_arc and w.center is not None:
                center = (int(w.center.x), int(w.center.y))
                radius = int(w.radius)
                angle1 = int(np.degrees(w.start_angle))
                angle2 = int(np.degrees(w.end_angle))
                if angle2 < angle1:
                    angle2 += 360
                cv2.ellipse(img, center, (radius, radius), 0, angle1, angle2, (0, 255, 255), 3)
            else:
                cv2.line(img, p1, p2, (0, 255, 0), 3)
            cv2.circle(img, p1, 5, (255, 0, 0), -1)
            cv2.circle(img, p2, 5, (0, 0, 255), -1)

        for d in (doors or []):
            if d.center is not None:
                center = (int(d.center.x), int(d.center.y))
                # Color by door type
                color_map = {
                    "swing":        (0, 215, 255),   # yellow
                    "double_swing": (0, 140, 255),   # orange
                    "sliding":      (255, 180, 0),   # cyan-blue
                    "bifold":       (180, 0, 255),   # purple
                    "unknown":      (128, 128, 128), # gray
                }
                color = color_map.get(d.door_type, (0, 215, 255))

                if d.radius > 0:
                    radius = int(d.radius)
                    angle1 = int(np.degrees(d.start_angle)) if d.start_angle else 0
                    angle2 = int(np.degrees(d.end_angle)) if d.end_angle else 90
                    if angle2 < angle1:
                        angle2 += 360
                    cv2.ellipse(img, center, (radius, radius), 0, angle1, angle2, color, 3)
                else:
                    # No arc geometry — draw opening indicator
                    cv2.circle(img, center, 8, color, 2)

                cv2.circle(img, center, 5, color, -1)
                label = f"{d.door_type[0].upper()}{d.id[-3:]}"
                cv2.putText(img, label, (center[0] + 6, center[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if output_path:
            cv2.imwrite(output_path, img)
        return img

    # ------------------------------------------------------------------
    # Internal: binary/contour detection
    # ------------------------------------------------------------------

    def _is_door_arc(self, wall: Wall) -> bool:
        """Return True if arc wall looks like a door swing (quarter-circle).

        Door swing criteria:
        - Radius 18–80 px (typical door width at 1:100 scale)
        - Angle span 70°–115° (quarter-circle ± 25° tolerance)

        Structural rounded wall corners have larger radii or smaller spans.
        """
        if not wall.is_arc:
            return False
        span_deg = math.degrees(abs(wall.end_angle - wall.start_angle))
        return (
            DOOR_ARC_MIN_RADIUS <= wall.radius <= DOOR_ARC_MAX_RADIUS
            and math.degrees(DOOR_ARC_MIN_SPAN) <= span_deg <= math.degrees(DOOR_ARC_MAX_SPAN)
        )

    def _filter_dimension_lines(self, binary: np.ndarray) -> np.ndarray:
        """Remove dimension lines and isolated text from binary mask.

        Strategy per connected component:
        - **Dimension line**: bounding box aspect ≥ DIM_LINE_MIN_ASPECT AND
          min bounding dim ≤ DIM_LINE_MAX_STROKE. These are the ruler-style
          lines with arrows outside the floorplan boundary.
        - **Text blob**: small area AND thin stroke. Single-character and
          word-level text clusters are typically <400 px² with min_dim ≤ 4 px.

        Wall components that happen to be long are NOT removed here because
        they also have a large min_dim (walls are thick ≥ 5 px).
        """
        result = binary.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            min_dim = min(w, h)
            max_dim = max(w, h)
            aspect = max_dim / max(min_dim, 1)

            is_dim_line = (
                aspect >= DIM_LINE_MIN_ASPECT
                and min_dim <= DIM_LINE_MAX_STROKE
            )
            is_text = (
                area < 400
                and min_dim <= DIM_LINE_MAX_STROKE
            )

            if is_dim_line or is_text:
                result[labels == i] = 0

        return result

    def _detect_doors(
        self, binary: np.ndarray, walls: list[Wall]
    ) -> list[Door]:
        """Detect door swing arcs (quarter-circle) from binary mask.

        A door swing arc has:
        - Radius 35–150 px (1:100 scale drawings)
        - Angle span ~90° (65°–115°)
        - Arc center within DOOR_WALL_PROXIMITY px of at least one wall endpoint
          (the hinge point is always at a wall end)

        Process:
        1. Find thin contours (stroke ≤ 4 px) — door arcs are thin lines.
        2. For each arc-like contour, fit circle and check constraints.
        3. Match to nearest wall endpoint (hinge point).
        """
        doors: list[Door] = []
        seen_centers: list[tuple[float, float]] = []

        # Build list of all wall endpoints for proximity check
        wall_endpoints: list[tuple[float, float, str]] = []
        for w in walls:
            wall_endpoints.append((w.start.x, w.start.y, w.id))
            wall_endpoints.append((w.end.x, w.end.y, w.id))

        # Find components that are thin (potential door arcs)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

        for i in range(1, num_labels):
            w_bb = stats[i, cv2.CC_STAT_WIDTH]
            h_bb = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            min_dim = min(w_bb, h_bb)
            max_dim = max(w_bb, h_bb)

            # Door arc bounding box should be roughly square (swing covers a quadrant)
            if max_dim < DOOR_ARC_MIN_RADIUS * 1.2:
                continue
            if max_dim > DOOR_ARC_MAX_RADIUS * 2.5:
                continue
            # Arc bounding box aspect ratio: quarter-circle fits in ~square bbox
            if max_dim / max(min_dim, 1) > 2.5:
                continue
            # Must be thin stroke (not a filled wall region)
            if min_dim > DIM_LINE_MAX_STROKE * 3:
                continue
            # Area vs bounding box — arc is sparse
            solidity = area / max(w_bb * h_bb, 1)
            if solidity > 0.25:
                continue

            # Extract contour points for this component
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            pts = contours[0].reshape(-1, 2).astype(np.float64)
            if len(pts) < 10:
                continue

            # Fit circle
            x = pts[:, 0]
            y = pts[:, 1]
            A = np.column_stack([x, y, np.ones(len(x))])
            b_vec = x ** 2 + y ** 2
            try:
                res, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue

            cx = res[0] / 2
            cy = res[1] / 2
            r_sq = res[2] + cx ** 2 + cy ** 2
            if r_sq <= 0:
                continue
            radius = math.sqrt(r_sq)

            if radius < DOOR_ARC_MIN_RADIUS or radius > DOOR_ARC_MAX_RADIUS:
                continue

            # Compute residuals — arc should have low fit error
            dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            rms_error = float(np.sqrt(np.mean((dists - radius) ** 2)))
            if rms_error > radius * 0.15:
                continue

            # Check angle span
            angles = np.arctan2(y - cy, x - cx)
            span = float(angles.max() - angles.min())
            if span < 0:
                span += 2 * math.pi
            if span < DOOR_ARC_MIN_SPAN or span > DOOR_ARC_MAX_SPAN:
                continue

            # Deduplicate nearby centers
            duplicate = False
            for scx, scy in seen_centers:
                if math.hypot(cx - scx, cy - scy) < radius * 0.5:
                    duplicate = True
                    break
            if duplicate:
                continue

            # Match to nearest wall endpoint (hinge)
            best_wall_id = ""
            best_dist = DOOR_WALL_PROXIMITY
            for ex, ey, wid in wall_endpoints:
                d = math.hypot(cx - ex, cy - ey)
                if d < best_dist:
                    best_dist = d
                    best_wall_id = wid

            door_id = f"D{len(doors):04d}"
            doors.append(Door(
                id=door_id,
                wall_id=best_wall_id,
                width=round(radius * math.sqrt(2), 1),  # approx opening width
                center=Point(x=round(cx, 1), y=round(cy, 1)),
                radius=round(radius, 1),
                start_angle=round(float(angles.min()), 4),
                end_angle=round(float(angles.max()), 4),
            ))
            seen_centers.append((cx, cy))

        return doors

    def _detect_doors_from_gaps(
        self,
        walls: list[Wall],
        binary: np.ndarray,
    ) -> list[Door]:
        """Detect doors from wall gaps using two complementary strategies.

        Strategy A — Profile scan (gaps within a wall segment):
          Scan along each wall axis using the binary mask. Consecutive zero
          pixels wider than DOOR_GAP_MIN = door opening.

        Strategy B — Colinear pair gaps (gaps between adjacent segments):
          Two colinear wall segments with a gap between them = door.

        Both strategies classify gap content to determine door type.
        """
        doors: list[Door] = []
        seen: set[tuple[int, int]] = set()
        img_h, img_w = binary.shape
        COLLINEAR_TOL = 8.0

        straight = [w for w in walls if not w.is_arc]

        def add_door(gap_start_abs: int, gap_end_abs: int,
                     perp: int, axis: str) -> None:
            gap_size = gap_end_abs - gap_start_abs
            if gap_size < DOOR_GAP_MIN or gap_size > DOOR_GAP_MAX:
                return
            gap_mid = (gap_start_abs + gap_end_abs) // 2
            key = (gap_mid // 8 * 8, perp // 8 * 8)
            if key in seen:
                return
            seen.add(key)

            sw = int(DOOR_GAP_SEARCH_WIDTH)
            if axis == "h":
                rx0 = max(0, gap_start_abs - 5)
                rx1 = min(img_w, gap_end_abs + 5)
                ry0 = max(0, perp - sw)
                ry1 = min(img_h, perp + sw)
                cx, cy = float(gap_mid), float(perp)
            else:
                ry0 = max(0, gap_start_abs - 5)
                ry1 = min(img_h, gap_end_abs + 5)
                rx0 = max(0, perp - sw)
                rx1 = min(img_w, perp + sw)
                cx, cy = float(perp), float(gap_mid)

            region = binary[ry0:ry1, rx0:rx1]
            door_type, center, radius = self._classify_gap_region(
                region, rx0, ry0, gap_size, axis == "h"
            )
            if door_type == "unknown":
                door_type = "swing"

            doors.append(Door(
                id=f"D{len(doors):04d}",
                wall_id="",
                width=float(gap_size),
                center=center if center else Point(x=cx, y=cy),
                radius=radius,
                door_type=door_type,
            ))

        # Strategy A: profile scan inside each wall
        for wall in straight:
            is_h = wall.is_horizontal()
            is_v = wall.is_vertical()
            if not is_h and not is_v:
                continue

            if is_h:
                a_s = int(min(wall.start.x, wall.end.x))
                a_e = int(max(wall.start.x, wall.end.x))
                pc = int(wall.start.y)
            else:
                a_s = int(min(wall.start.y, wall.end.y))
                a_e = int(max(wall.start.y, wall.end.y))
                pc = int(wall.start.x)

            if a_e - a_s < DOOR_GAP_MIN * 2:
                continue

            half = 5
            profile = []
            for pos in range(a_s, a_e + 1):
                if is_h:
                    col = binary[max(0, pc - half):min(img_h, pc + half + 1), pos]
                else:
                    col = binary[pos, max(0, pc - half):min(img_w, pc + half + 1)]
                profile.append(1 if np.any(col > 0) else 0)

            # Find zero runs = gaps
            in_gap = False
            g_start = 0
            for i, v in enumerate(profile):
                if v == 0 and not in_gap:
                    g_start = i
                    in_gap = True
                elif v == 1 and in_gap:
                    add_door(a_s + g_start, a_s + i, pc, "h" if is_h else "v")
                    in_gap = False
            if in_gap:
                add_door(a_s + g_start, a_e, pc, "h" if is_h else "v")

        # Strategy B: colinear segment pairs
        for axis, group in [("h", [w for w in straight if w.is_horizontal()]),
                             ("v", [w for w in straight if w.is_vertical()])]:
            if len(group) < 2:
                continue

            # Build clusters by position on perpendicular axis
            grouped: dict[int, list[Wall]] = {}
            for w in group:
                pos = int(round((w.start.y if axis == "h" else w.start.x) / COLLINEAR_TOL) * COLLINEAR_TOL)
                grouped.setdefault(pos, []).append(w)

            for pos_key, cluster in grouped.items():
                if len(cluster) < 2:
                    continue
                perp = int(sum(w.start.y if axis == "h" else w.start.x for w in cluster) / len(cluster))
                if axis == "h":
                    cluster.sort(key=lambda w: min(w.start.x, w.end.x))
                else:
                    cluster.sort(key=lambda w: min(w.start.y, w.end.y))

                for k in range(len(cluster) - 1):
                    wa, wb = cluster[k], cluster[k + 1]
                    if axis == "h":
                        g_s = int(max(wa.start.x, wa.end.x))
                        g_e = int(min(wb.start.x, wb.end.x))
                    else:
                        g_s = int(max(wa.start.y, wa.end.y))
                        g_e = int(min(wb.start.y, wb.end.y))
                    add_door(g_s, g_e, perp, axis)

        return doors

    def _classify_gap_region(
        self,
        region: np.ndarray,
        rx0: int,
        ry0: int,
        gap_size: int,
        is_horizontal: bool,
    ) -> tuple[str, "Point | None", float]:
        """Classify what's inside a wall gap as door type.

        Returns (door_type, center_point, radius).
        door_type: "swing" | "sliding" | "double_swing" | "bifold" | "unknown"
        """
        if region.size == 0:
            return "unknown", None, 0.0

        rh, rw = region.shape

        # --- Count non-zero pixels ---
        px_count = int(np.sum(region > 0))
        if px_count < DOOR_GAP_ARC_MIN_PX:
            return "unknown", None, 0.0

        # --- Fit circle to pixels (swing arc detection) ---
        ys, xs = np.where(region > 0)
        if len(xs) < 8:
            return "unknown", None, 0.0

        pts_x = xs.astype(np.float64)
        pts_y = ys.astype(np.float64)

        A = np.column_stack([pts_x, pts_y, np.ones(len(pts_x))])
        b_vec = pts_x ** 2 + pts_y ** 2
        try:
            res, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return "unknown", None, 0.0

        lcx = res[0] / 2
        lcy = res[1] / 2
        r_sq = res[2] + lcx ** 2 + lcy ** 2
        if r_sq <= 0:
            return "unknown", None, 0.0
        radius = math.sqrt(r_sq)

        # Check fit quality
        dists = np.sqrt((pts_x - lcx) ** 2 + (pts_y - lcy) ** 2)
        rms = float(np.sqrt(np.mean((dists - radius) ** 2)))

        # Good arc fit: radius matches gap_size, low RMS error
        # Also require low pixel density (arc is sparse, not filled region)
        bbox_area = rw * rh
        density = px_count / max(bbox_area, 1)
        arc_match = (
            gap_size * 0.5 <= radius <= gap_size * 1.8
            and rms < radius * 0.3
            and 8 <= radius <= 90
            and density < 0.4  # arc stroke is sparse, not a filled shape
        )

        if arc_match:
            angles = np.arctan2(pts_y - lcy, pts_x - lcx)
            span = float(angles.max() - angles.min())

            # Near-360° span = two wall ends forming a ring = just a gap, classify swing
            # True double_swing: span 180°-270°, high pixel count, low rms
            if math.pi * 0.9 < span < math.pi * 1.5 and px_count > 50 and rms < radius * 0.2:
                door_type = "double_swing"
            else:
                door_type = "swing"

            center = Point(
                x=round(rx0 + lcx, 1),
                y=round(ry0 + lcy, 1),
            )
            return door_type, center, round(radius, 1)

        # --- Sliding door: two parallel lines in region ---
        # Project onto perpendicular axis and look for two density peaks
        if is_horizontal:
            proj = np.sum(region > 0, axis=1)  # sum across columns → row profile
        else:
            proj = np.sum(region > 0, axis=0)  # sum across rows → col profile

        peaks = []
        threshold = max(2, proj.max() * 0.3)
        in_peak = False
        for i, v in enumerate(proj):
            if v >= threshold and not in_peak:
                peaks.append(i)
                in_peak = True
            elif v < threshold:
                in_peak = False

        if len(peaks) >= 2:
            sep = peaks[-1] - peaks[0]
            if sep <= DOOR_GAP_SLIDING_MAX_SEP:
                return "sliding", None, 0.0

        # Has pixels but no clear pattern → generic swing (gap + some content)
        if px_count >= DOOR_GAP_ARC_MIN_PX * 2:
            return "swing", None, 0.0

        return "unknown", None, 0.0

    def _detect_binary(self, image_path: str) -> tuple[list[Wall], list[Door]]:
        img = self._load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Adaptive threshold (better for varying contrast) ---
        h, w_img = gray.shape
        block_size = min(51, max(11, w_img // 10 | 1))
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 10,
        )

        # Morphological close (fill small gaps in walls)
        k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.params["close_kernel"],) * 2
        )
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, k_close,
            iterations=self.params["close_iterations"],
        )

        # Morphological open (remove thin noise: text strokes, dim lines)
        k_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.params["open_kernel"],) * 2
        )
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

        # Remove small noise components
        cleaned = self._remove_small_components(opened)

        # --- Filter dimension lines and text before wall detection ---
        cleaned = self._filter_dimension_lines(cleaned)

        # --- Optional pre-Hough open: removes residual thin symbols ---
        # (stair lines, furniture, toilet arcs, etc.) that survive dimension-line
        # filtering.  Skipped when hough_pre_open_kernel == 0.
        hough_input = cleaned
        pre_k = self.params["hough_pre_open_kernel"]
        if pre_k > 0:
            k_pre = cv2.getStructuringElement(cv2.MORPH_RECT, (pre_k, pre_k))
            hough_input = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_pre)

        # --- Hough on (possibly pre-opened) mask for wall detection ---
        raw_lines = self._hough_lines(hough_input)
        snapped = self._snap_to_axis(raw_lines)

        walls: list[Wall] = []
        for idx, ((x1, y1), (x2, y2)) in enumerate(snapped):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < self.params["min_wall_length"]:
                continue
            walls.append(
                Wall(
                    id=f"W{idx:04d}",
                    start=Point(x=x1, y=y1),
                    end=Point(x=x2, y=y2),
                )
            )

        walls = self._merge_nearby_walls(walls)
        walls = self._filter_outside_boundary(walls, cleaned)
        walls = self._merge_nearby_walls(walls)

        # --- Gap-based door detection on raw merged walls ---
        # Run before arc detection so we use the full wall segments.
        gap_doors = self._detect_doors_from_gaps(walls, closed)

        # --- Arc detection on thickness-filtered mask ---
        thick_mask = self._thickness_filter(cleaned)
        thick_mask = self._remove_small_components(thick_mask)
        thick_contours, _ = cv2.findContours(
            thick_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        arc_walls: list[Wall] = []
        doors: list[Door] = []
        if thick_contours:
            arc_walls = self._detect_arcs_from_contour(thick_contours, len(walls))

        # Classify arc walls: structural rounded corner vs door-like arc.
        # Arcs from thick_mask are thick-stroke features (walls, not thin door swings).
        # Discriminate by endpoint connectivity: structural rounded corners have BOTH
        # endpoints near existing straight wall endpoints; door arcs touch only ONE
        # endpoint (the hinge). When both endpoints connect, snap them to the nearest
        # straight wall endpoint for clean graph topology.
        straight_walls_for_arc_check = [w for w in walls if not w.is_arc]
        for arc in arc_walls:
            near_start, near_end = self._arc_endpoint_connections(
                arc, straight_walls_for_arc_check
            )
            if near_start and near_end:
                # Both endpoints near wall endpoints → structural rounded corner
                arc = self._snap_arc_endpoints_to_walls(arc, straight_walls_for_arc_check)
                walls.append(arc)
            elif self._is_door_arc(arc):
                doors.append(Door(
                    id=arc.id.replace("_arc", "_door"),
                    wall_id="",
                    width=round(arc.radius * math.sqrt(2), 1),
                    center=arc.center,
                    radius=arc.radius,
                    start_angle=arc.start_angle,
                    end_angle=arc.end_angle,
                    door_type="swing",
                ))
            else:
                walls.append(arc)

        # Remove straight walls that overlap with detected arc walls
        walls = self._remove_arc_overlaps(walls)

        walls = self._merge_nearby_walls(walls)
        walls = self._filter_small_walls(walls)

        # Merge arc doors + gap doors, dedup by position
        all_doors = list(doors)
        for gd in gap_doors:
            duplicate = any(
                gd.center is not None
                and d.center is not None
                and math.hypot(gd.center.x - d.center.x, gd.center.y - d.center.y) < 20
                for d in all_doors
            )
            if not duplicate:
                all_doors.append(gd)

        return walls, all_doors

    # ------------------------------------------------------------------
    # Internal: image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return img

    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than ``min_component_area``."""
        min_area = self.params["min_component_area"]
        cleaned = np.zeros_like(binary)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask = (labels == i).astype(np.uint8) * 255
                cleaned = cv2.bitwise_or(cleaned, mask)
        return cleaned

    def _thickness_filter(self, binary: np.ndarray) -> np.ndarray:
        """Remove thin foreground regions via morphological opening.

        A ``MORPH_OPEN`` with kernel size ``open_kernel`` removes pixel
        structures thinner than the kernel (e.g. 5 px → all structures
        thinner than ~5 px are removed).  This cleanly removes dimension
        lines, text, and door arcs while preserving wall bodies.
        """
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.params["open_kernel"],) * 2
        )
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    def _heavy_open(self, binary: np.ndarray) -> np.ndarray:
        """Aggressive morphological open to isolate thick wall bodies.

        Uses a 7×7 kernel — removes text strokes (~1–2 px), dimension lines,
        door arcs, and other thin artifacts while preserving wall bodies
        (typically ≥5 px thick).
        """
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.params["open_kernel_heavy"],) * 2
        )
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    def _build_wall_mask(self, cleaned: np.ndarray) -> np.ndarray:
        """Build wall-only mask by classifying connected components.

        Strategy: instead of aggressive morphological open (which destroys
        thin walls), classify each connected component by bounding-box
        geometry. Three criteria must all pass for wall classification:

        1. **Thickness**: min_dim ≥ wall_min_dim (default 5). Walls are
           thick; text strokes are thin (1–2 px).
        2. **Solidity**: area / bbox_area ≥ wall_solidity (default 0.15).
           Walls are solid rectangles; text/symbols are sparse clusters.
        3. **Aspect**: max_dim / min_dim ≤ wall_max_aspect (default 10).
           Walls are moderate aspect; dimension lines are extreme.

        Exception: components with area ≥ min_component_area *
        wall_large_area_mult are always walls regardless of the above —
        prevents false-negatives on large legitimate wall clusters.
        """
        min_dim_threshold = self.params["wall_min_dim"]
        solidity_threshold = self.params["wall_solidity"]
        max_aspect = self.params["wall_max_aspect"]
        large_area_threshold = self.params["min_component_area"] * self.params["wall_large_area_mult"]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, 8
        )

        wall_mask = np.zeros_like(cleaned)

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            min_dim = min(w, h)
            max_dim = max(w, h)
            bbox_area = w * h
            solidity = comp_area / bbox_area if bbox_area > 0 else 0.0
            aspect = max_dim / max(min_dim, 1)

            # Large components are always walls (prevent false-negatives)
            if comp_area >= large_area_threshold:
                mask = (labels == i).astype(np.uint8) * 255
                wall_mask = cv2.bitwise_or(wall_mask, mask)
                continue

            # All three criteria must pass
            is_wall = (
                min_dim >= min_dim_threshold
                and solidity >= solidity_threshold
                and aspect <= max_aspect
            )

            if is_wall:
                mask = (labels == i).astype(np.uint8) * 255
                wall_mask = cv2.bitwise_or(wall_mask, mask)

        return wall_mask

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Reduce binary wall mask to 1-pixel-wide skeleton via thinning.

        Uses OpenCV's ``ximgproc.thinning`` when available; falls back to an
        iterative morphological approach otherwise.
        """
        img = (binary > 0).astype(np.uint8) * 255
        try:
            # Prefer XIMGPROC thinning (faster, higher quality)
            skel = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback: iterative erosion-based skeletonisation
            skel = np.zeros_like(img)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = img.copy()
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.dilate(eroded, element)
                subset = cv2.subtract(temp, opened)
                skel = cv2.bitwise_or(skel, subset)
                temp = eroded.copy()
                if cv2.countNonZero(temp) == 0:
                    break
        return skel

    def _fast_prune_skeleton(self, skel: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Remove endpoint spurs from skeleton using vectorized neighbor counting.

        Each iteration removes all pixels with exactly 1 neighbor (endpoint
        pixels = tips of short branches).  Repeated N times removes spurs up
        to N pixels long.  Much faster than the Python-loop
        :meth:`_prune_skeleton` for large images.
        """
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # exclude center pixel
        result = (skel > 0).astype(np.uint8)
        for _ in range(iterations):
            neighbor_count = cv2.filter2D(result, -1, kernel)
            # Endpoint: pixel is on AND has exactly 1 neighbor
            endpoints = (result == 1) & (neighbor_count == 1)
            result[endpoints] = 0
        return result * 255

    def _prune_skeleton(self, skel: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Remove endpoint branches from skeleton (short spurs from noise).

        Iteratively removes pixels that have only one neighbour (endpoints
        of short branches). After *iterations* passes, only structurally
        significant skeleton paths remain.
        """
        pruned = skel.copy()
        h, w = pruned.shape
        for _ in range(iterations):
            endpoints = np.zeros_like(pruned, dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    if pruned[y, x] == 0:
                        continue
                    y0 = max(0, y - 1)
                    y1 = min(h, y + 2)
                    x0 = max(0, x - 1)
                    x1 = min(w, x + 2)
                    neighbours = int(np.sum(pruned[y0:y1, x0:x1] > 0)) - 1
                    if neighbours == 1:
                        endpoints[y, x] = 255
            pruned = cv2.subtract(pruned, endpoints)
        return pruned

    def _snap_to_axis(
        self,
        lines: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Snap near-horizontal and near-vertical lines to exact H/V axis.

        Lines that are neither H nor V within ``angle_tolerance`` are kept
        as-is (diagonal walls are preserved, not silently dropped).
        """
        tol = self.params["angle_tolerance"]
        result = []
        for (x1, y1), (x2, y2) in lines:
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) < abs(dx) * tol:
                # Horizontal — average y
                y = (y1 + y2) / 2
                xs, xe = (min(x1, x2), max(x1, x2))
                result.append(((xs, y), (xe, y)))
            elif abs(dx) < abs(dy) * tol:
                # Vertical — average x
                x = (x1 + x2) / 2
                ys, ye = (min(y1, y2), max(y1, y2))
                result.append(((x, ys), (x, ye)))
            else:
                # Diagonal — keep original
                result.append(((x1, y1), (x2, y2)))
        return result

    def _hough_lines(
        self, img: np.ndarray
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Run probabilistic Hough on *img* and return line endpoint pairs."""
        lines = cv2.HoughLinesP(
            img,
            rho=1,
            theta=np.pi / 180,
            threshold=int(self.params["hough_threshold"]),
            minLineLength=int(self.params["hough_min_line_length"]),
            maxLineGap=int(self.params["hough_max_line_gap"]),
        )
        if lines is None:
            return []
        return [
            ((float(x1), float(y1)), (float(x2), float(y2)))
            for x1, y1, x2, y2 in lines[:, 0]
        ]

    # ------------------------------------------------------------------
    # Internal: contour-based wall extraction
    # ------------------------------------------------------------------

    def _is_small_square(self, cnt: np.ndarray) -> bool:
        """Return True if *cnt* looks like a small square symbol (door/window)."""
        area = cv2.contourArea(cnt)
        if area > self.params["min_square_area"] * 4:
            return False
        rect = cv2.minAreaRect(cnt)
        _, (rw, rh), _ = rect
        if rw < 1 or rh < 1:
            return True
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        return (
            aspect < self.params["max_square_aspect"]
            and area < self.params["min_square_area"]
        )

    def _extract_straight_walls(
        self, cnt: np.ndarray, base_idx: int
    ) -> list[Wall]:
        """Extract straight wall segments from a single contour.

        No fillet detection — arcs are handled separately by
        :meth:`_detect_arcs_from_contour`.
        """
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            return []

        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            return []

        epsilon = self.params["approx_epsilon_ratio"] * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 2:
            return []

        approx_pts = [tuple(p[0]) for p in approx]
        walls: list[Wall] = []
        tol = self.params["angle_tolerance"]

        for k in range(len(approx_pts) - 1):
            p1 = approx_pts[k]
            p2 = approx_pts[k + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length < self.params["min_wall_length"]:
                continue

            is_h = abs(dy) < abs(dx) * tol
            is_v = abs(dx) < abs(dy) * tol

            if is_h:
                y = (p1[1] + p2[1]) / 2
                xs, xe = min(p1[0], p2[0]), max(p1[0], p2[0])
                walls.append(
                    Wall(
                        id=f"W{base_idx + k:04d}",
                        start=Point(x=float(xs), y=float(y)),
                        end=Point(x=float(xe), y=float(y)),
                    )
                )
            elif is_v:
                x = (p1[0] + p2[0]) / 2
                ys, ye = min(p1[1], p2[1]), max(p1[1], p2[1])
                walls.append(
                    Wall(
                        id=f"W{base_idx + k:04d}",
                        start=Point(x=float(x), y=float(ys)),
                        end=Point(x=float(x), y=float(ye)),
                    )
                )

        return walls

    def _remove_arc_overlaps(self, walls: list[Wall]) -> list[Wall]:
        """Remove short straight walls whose midpoint is within an arc's radius.

        Long structural walls (>2× arc diameter) passing through an arc are
        kept — the arc is a small rounded corner at one end of the wall,
        not a replacement for the entire wall.  Only short walls whose
        entire length falls within the arc region are removed.
        """
        arcs = [w for w in walls if w.is_arc]
        if not arcs:
            return walls

        result: list[Wall] = []
        for w in walls:
            if w.is_arc:
                result.append(w)
                continue

            mid = w.midpoint()
            overlaps = False
            for arc in arcs:
                if arc.center is None:
                    continue
                d = mid.distance_to(arc.center)
                arc_diameter = arc.radius * 2
                # Long walls extend far beyond the arc — keep them
                if w.length() > arc_diameter * 2:
                    continue
                # Short walls whose midpoint is within arc radius are
                # likely fillet artifacts (the arc replaces this segment)
                if d < arc.radius * 1.5 + 5:
                    overlaps = True
                    break

            if not overlaps:
                result.append(w)

        return result

    # ------------------------------------------------------------------
    # Internal: arc detection
    # ------------------------------------------------------------------

    def _raw_contour_for(
        self, cnt: np.ndarray, cleaned: np.ndarray
    ) -> np.ndarray | None:
        raw_contours, _ = cv2.findContours(
            cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        br = cv2.boundingRect(cnt)
        best: np.ndarray | None = None
        best_iou = 0.3
        for rc in raw_contours:
            rbr = cv2.boundingRect(rc)
            x1 = max(br[0], rbr[0]); y1 = max(br[1], rbr[1])
            x2 = min(br[0]+br[2], rbr[0]+rbr[2]); y2 = min(br[1]+br[3], rbr[1]+rbr[3])
            if x2 <= x1 or y2 <= y1: continue
            inter = (x2-x1)*(y2-y1)
            union = br[2]*br[3] + rbr[2]*rbr[3] - inter
            iou = inter/max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best = rc
        return best.reshape(-1, 2) if best is not None else None

    def _detect_fillets(
        self,
        approx_pts: list[tuple[int, int]],
        raw_none: np.ndarray,
    ) -> list[Wall]:
        fillets: list[Wall] = []
        seen: set[tuple[float, float]] = set()
        n = len(approx_pts)

        for k in range(n):
            p1 = approx_pts[(k - 1) % n]
            p2 = approx_pts[k]
            p3 = approx_pts[(k + 1) % n]

            dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
            dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
            len1, len2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
            if len1 < 20 or len2 < 20:
                continue

            dot = dx1 * dx2 + dy1 * dy2
            mag = len1 * len2
            cos_angle = dot / mag
            angle = math.acos(max(-1.0, min(1.0, cos_angle)))

            if angle < 0.524 or angle > math.pi - 0.524:
                continue

            nearest = self._closest_point_idx(raw_none, p2)

            wall: Wall | None = None
            for half in (8, 10, 12, 15):
                seg = self._circular_slice(raw_none, nearest - half, nearest + half + 1)
                if len(seg) < 5:
                    continue
                w = self._fit_arc_wall(seg, 0)
                if w is not None:
                    wall = w
                    break

            if wall is None:
                continue

            key = (round(wall.center.x, 0), round(wall.center.y, 0))
            if key in seen:
                continue

            cx, cy = wall.center.x, wall.center.y

            # Arc center must be near a wall intersection
            near_vertex = False
            for vx, vy in approx_pts:
                d = math.hypot(cx - vx, cy - vy)
                if d < wall.radius * 1.2 + 10:
                    near_vertex = True
                    break
            if not near_vertex:
                continue

            fillets.append(wall)
            seen.add(key)

        return fillets

    def _detect_arcs_from_contour(
        self,
        contours: list[np.ndarray],
        base_idx: int,
    ) -> list[Wall]:
        """Detect arc/rounded walls from wall_mask contours.

        Two detection strategies:
        1. **Fillet detection**: at each approxPolyDP vertex where the
           angle is in the 45-135 degree range, fit a circle through the
           nearby raw contour points.
        2. **Curved segment detection**: find contour sections whose
           curvature exceeds the threshold and fit arcs through them.

        Uses wall_mask contour (text-free). Supports larger arc radii
        for structural rounded walls.
        """
        arcs: list[Wall] = []
        seen: set[tuple[float, float]] = set()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.params["min_component_area"] * 2:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue

            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                continue

            epsilon = self.params["approx_epsilon_ratio"] * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            approx_pts = [tuple(p[0]) for p in approx]

            # Strategy 1: fillet detection at vertices
            n = len(approx_pts)
            proximity = self.params["arc_vertex_proximity"]

            for k in range(n):
                p1 = approx_pts[(k - 1) % n]
                p2 = approx_pts[k]
                p3 = approx_pts[(k + 1) % n]

                dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
                dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
                len1, len2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
                if len1 < 15 or len2 < 15:
                    continue

                dot = dx1 * dx2 + dy1 * dy2
                mag = len1 * len2
                cos_angle = dot / mag
                angle = math.acos(max(-1.0, min(1.0, cos_angle)))

                if angle < 0.524 or angle > math.pi - 0.524:
                    continue

                nearest = self._closest_point_idx(pts, p2)

                wall: Wall | None = None
                for half in (10, 12, 15, 20, 25):
                    seg = self._circular_slice(
                        pts, nearest - half, nearest + half + 1
                    )
                    if len(seg) < 5:
                        continue
                    w = self._fit_arc_wall(seg, 0)
                    if w is not None and w.radius >= ARC_MIN_RADIUS and w.radius <= ARC_MAX_RADIUS:
                        span = abs(w.end_angle - w.start_angle)
                        if span >= ARC_MIN_ANGLE_SPAN and span <= ARC_MAX_ANGLE_SPAN:
                            wall = w
                            break

                if wall is None:
                    continue

                key = (round(wall.center.x, 0), round(wall.center.y, 0))
                if key in seen:
                    continue

                cx, cy = wall.center.x, wall.center.y

                # Arc center must be near a wall vertex
                near_vertex = False
                for vx, vy in approx_pts:
                    d = math.hypot(cx - vx, cy - vy)
                    if d < wall.radius * 1.2 + proximity:
                        near_vertex = True
                        break
                if not near_vertex:
                    continue

                wall.id = f"W{base_idx:04d}_arc"
                base_idx += 1
                arcs.append(wall)
                seen.add(key)

            # Strategy 2: curved segment detection on raw contour
            if len(pts) >= 7:
                curv_arcs = self._detect_curved_segments(pts, base_idx, approx_pts)
                for arc in curv_arcs:
                    key = (round(arc.center.x, 0), round(arc.center.y, 0))
                    if key not in seen:
                        arcs.append(arc)
                        seen.add(key)
                        base_idx += 1

        return arcs

    def _detect_curved_segments(
        self,
        raw_pts: np.ndarray,
        base_idx: int,
        approx_pts: list[tuple[int, int]],
    ) -> list[Wall]:
        """Find sustained curved sections in raw contour by curvature analysis.

        Scans contour with sliding window. Where local curvature exceeds
        arc_curvature_threshold, collects curved region and fits arc.
        """
        walls: list[Wall] = []
        n = len(raw_pts)
        if n < 7:
            return walls

        window = 7
        threshold = self.params["arc_curvature_threshold"]
        proximity = self.params["arc_vertex_proximity"]

        # Compute curvature at each point
        in_curve = np.zeros(n, dtype=bool)
        for i in range(n):
            i_prev = (i - window) % n
            i_next = (i + window) % n
            p_prev = raw_pts[i_prev].astype(float)
            p_curr = raw_pts[i].astype(float)
            p_next = raw_pts[i_next].astype(float)

            d_prev = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            d_next = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            if d_prev < 1 or d_next < 1:
                continue

            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            d_angle = abs(angle2 - angle1)
            if d_angle > math.pi:
                d_angle = 2 * math.pi - d_angle

            curvature = d_angle / (d_prev + d_next) * 2
            if curvature > threshold:
                in_curve[i] = True

        # Cluster consecutive curved points into segments
        if not np.any(in_curve):
            return walls

        segments: list[list[int]] = []
        current_seg: list[int] = []
        for i in range(n):
            if in_curve[i]:
                current_seg.append(i)
            else:
                if len(current_seg) >= 5:
                    segments.append(current_seg)
                current_seg = []
        if len(current_seg) >= 5:
            segments.append(current_seg)

        for seg_indices in segments:
            pts_seg = raw_pts[seg_indices].astype(np.float64)
            if len(pts_seg) < 5:
                continue

            w = self._fit_arc_wall(pts_seg, 0)
            if w is None:
                continue

            if w.radius < ARC_MIN_RADIUS or w.radius > ARC_MAX_RADIUS:
                continue

            span = abs(w.end_angle - w.start_angle)
            if span < ARC_MIN_ANGLE_SPAN or span > ARC_MAX_ANGLE_SPAN:
                continue

            # Arc must be near a wall vertex (not floating text/dimension)
            cx, cy = w.center.x, w.center.y
            near_vertex = False
            for vx, vy in approx_pts:
                d = math.hypot(cx - vx, cy - vy)
                if d < w.radius * 1.5 + proximity:
                    near_vertex = True
                    break
            if not near_vertex:
                continue

            w.id = f"W{base_idx:04d}_arc"
            walls.append(w)

        return walls

    def _overlaps_fillet(
        self, seg_pts: np.ndarray, fillets: list[Wall]
    ) -> bool:
        if len(seg_pts) < 2 or not fillets:
            return False
        seg_center = seg_pts.astype(float).mean(axis=0)
        for w in fillets:
            if w.center is None:
                continue
            ac = np.array([w.center.x, w.center.y])
            dist = np.linalg.norm(seg_center - ac)
            if dist < w.radius * 0.8:
                return True
        return False

    def _get_raw_points_between(
        self,
        raw_pts: np.ndarray,
        p1: tuple,
        p2: tuple,
    ) -> np.ndarray:
        """Return the subset of *raw_pts* lying between vertices *p1* and *p2*."""
        i1 = self._closest_point_idx(raw_pts, p1)
        i2 = self._closest_point_idx(raw_pts, p2)
        if i1 > i2:
            i1, i2 = i2, i1
        forward = raw_pts[i1 : i2 + 1]
        backward = np.vstack([raw_pts[i2:], raw_pts[: i1 + 1]])
        return forward if len(forward) <= len(backward) else backward

    @staticmethod
    def _match_contour(
        cnt: np.ndarray,
        candidates: list[np.ndarray],
        candidate_rects: list[tuple[int, int, int, int]],
        min_iou: float = 0.25,
    ) -> np.ndarray | None:
        br = cv2.boundingRect(cnt)
        best_match: np.ndarray | None = None
        best_iou = min_iou
        for cand, cbr in zip(candidates, candidate_rects):
            x1 = max(br[0], cbr[0])
            y1 = max(br[1], cbr[1])
            x2 = min(br[0] + br[2], cbr[0] + cbr[2])
            y2 = min(br[1] + br[3], cbr[1] + cbr[3])
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            union = br[2] * br[3] + cbr[2] * cbr[3] - inter
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_match = cand
        if best_match is not None:
            return best_match.reshape(-1, 2)
        return None

    @staticmethod
    def _contour_segment(
        pts: np.ndarray, i1: int, i2: int
    ) -> np.ndarray:
        n = len(pts)
        if i1 > i2:
            i1, i2 = i2, i1
        fwd = pts[i1:i2+1]
        bwd = np.vstack([pts[i2:], pts[:i1+1]])
        return fwd if len(fwd) <= len(bwd) else bwd

    @staticmethod
    def _circular_slice(
        arr: np.ndarray, start: int, end: int
    ) -> np.ndarray:
        n = len(arr)
        if start < 0:
            start += n
        if end > n:
            end -= n
        if start <= end:
            return arr[start:end]
        return np.vstack([arr[start:], arr[:end]])

    @staticmethod
    def _closest_point_idx(pts: np.ndarray, target: tuple) -> int:
        """Return the index of the point in *pts* closest to *target*."""
        """Return the index of the point in *pts* closest to *target*."""
        dists = np.sum((pts - np.array(target)) ** 2, axis=1)
        return int(np.argmin(dists))

    def _fit_arc_wall(self, pts: np.ndarray, base_idx: int) -> Wall | None:
        """Fit a circle through *pts* and return a :class:`Wall` arc, or None."""
        if len(pts) < 3:
            return None

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)

        A = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2

        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        cx = result[0] / 2
        cy = result[1] / 2
        r_sq = result[2] + cx ** 2 + cy ** 2
        if r_sq <= 0:
            return None
        radius = math.sqrt(r_sq)

        if radius < ARC_MIN_RADIUS or radius > ARC_MAX_RADIUS:
            return None

        angles = np.arctan2(y - cy, x - cx)
        start_angle = float(angles[0])
        end_angle = float(angles[-1])

        if end_angle < start_angle:
            start_angle, end_angle = end_angle, start_angle

        if end_angle - start_angle > math.pi:
            start_angle, end_angle = end_angle, start_angle + 2 * math.pi

        angle_span = abs(end_angle - start_angle)
        if angle_span < ARC_MIN_ANGLE_SPAN or angle_span > ARC_MAX_ANGLE_SPAN:
            return None

        arc_len = angle_span * radius
        if arc_len < self.params["min_wall_length"]:
            return None

        start_pt = Point(
            x=round(cx + radius * math.cos(start_angle), 1),
            y=round(cy + radius * math.sin(start_angle), 1),
        )
        end_pt = Point(
            x=round(cx + radius * math.cos(end_angle), 1),
            y=round(cy + radius * math.sin(end_angle), 1),
        )

        return Wall(
            id=f"W{base_idx:04d}_arc",
            start=start_pt,
            end=end_pt,
            center=Point(x=round(cx, 1), y=round(cy, 1)),
            radius=round(radius, 1),
            start_angle=round(start_angle, 4),
            end_angle=round(end_angle, 4),
        )

    # ------------------------------------------------------------------
    # Internal: post-processing / merging
    # ------------------------------------------------------------------

    def _merge_nearby_walls(self, walls: list[Wall]) -> list[Wall]:
        """Merge collinear, near-axis-aligned walls into single segments."""
        if len(walls) < 2:
            return walls

        arcs = [w for w in walls if w.is_arc]
        straight = [w for w in walls if not w.is_arc]
        h_walls = [w for w in straight if w.is_horizontal()]
        v_walls = [w for w in straight if w.is_vertical()]
        diag_walls = [w for w in straight if not w.is_horizontal() and not w.is_vertical()]

        merged: list[Wall] = []
        merged.extend(self._merge_axis(h_walls, "h"))
        merged.extend(self._merge_axis(v_walls, "v"))
        merged.extend(arcs)
        merged.extend(diag_walls)
        merged.extend(diag_walls)  # pass-through unchanged
        return merged

    def _merge_axis(self, walls: list[Wall], axis: str) -> list[Wall]:
        """Merge collinear walls along one axis, tolerating small offsets."""
        if not walls:
            return []

        sort_tol = self.params["parallel_merge_dist"]
        # Collinear proximity tolerance — walls must be at the same
        # axis position (within this tolerance) to merge.  Kept
        # small (5 px) to prevent chain-merging across the image:
        # e.g. left wall at x=68 merging through x=70→x=114→x=118
        # into the internal wall at x=119.
        collinear_tol = 8.0

        if axis == "h":
            walls.sort(
                key=lambda w: (round(w.start.y / sort_tol, 0), w.start.x)
            )
        else:
            walls.sort(
                key=lambda w: (round(w.start.x / sort_tol, 0), w.start.y)
            )

        merged: list[Wall] = []
        used = [False] * len(walls)

        for i, w in enumerate(walls):
            if used[i]:
                continue
            current = w
            # Remember the original axis position to prevent drift
            ref_pos = current.start.y if axis == "h" else current.start.x
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                other = walls[j]
                other_pos = other.start.y if axis == "h" else other.start.x
                if abs(ref_pos - other_pos) > collinear_tol:
                    continue
                if axis == "h":
                    cr = (
                        min(current.start.x, current.end.x),
                        max(current.start.x, current.end.x),
                    )
                    or_ = (
                        min(other.start.x, other.end.x),
                        max(other.start.x, other.end.x),
                    )
                else:
                    cr = (
                        min(current.start.y, current.end.y),
                        max(current.start.y, current.end.y),
                    )
                    or_ = (
                        min(other.start.y, other.end.y),
                        max(other.start.y, other.end.y),
                    )

                # Merge if overlapping or within gap tolerance
                gap = COLLINEAR_MERGE_GAP
                if cr[0] <= or_[1] + gap and or_[0] <= cr[1] + gap:
                    new_min = min(cr[0], or_[0])
                    new_max = max(cr[1], or_[1])
                    if axis == "h":
                        y = round((current.start.y + other.start.y) / 2)
                        current = Wall(
                            id=current.id,
                            start=Point(x=float(new_min), y=float(y)),
                            end=Point(x=float(new_max), y=float(y)),
                        )
                    else:
                        x = round((current.start.x + other.start.x) / 2)
                        current = Wall(
                            id=current.id,
                            start=Point(x=float(x), y=float(new_min)),
                            end=Point(x=float(x), y=float(new_max)),
                        )
                    used[j] = True
            merged.append(current)
        return merged

    def _merge_parallel_pairs(self, walls: list[Wall]) -> list[Wall]:
        """Merge pairs of closely-spaced parallel walls into a single centreline.

        This handles the common case where a thick wall is detected as two
        parallel lines (one for each face of the wall).
        """
        max_dist = self.params["parallel_merge_dist"]
        used = [False] * len(walls)
        result: list[Wall] = []

        for i, wa in enumerate(walls):
            if wa.is_arc:
                if not used[i]:
                    result.append(wa)
                    used[i] = True
                continue

            if used[i]:
                continue

            best_j, best_dist = -1, float("inf")
            for j, wb in enumerate(walls):
                if i == j or used[j] or wb.is_arc:
                    continue
                if wa.is_horizontal() and wb.is_horizontal():
                    dist = abs(wa.start.y - wb.start.y)
                elif wa.is_vertical() and wb.is_vertical():
                    dist = abs(wa.start.x - wb.start.x)
                else:
                    continue
                if dist < max_dist and dist < best_dist:
                    overlap = self._parallel_overlap(wa, wb)
                    if overlap > min(wa.length(), wb.length()) * 0.3:
                        best_j, best_dist = j, dist

            if best_j >= 0:
                used[best_j] = True
                result.append(self._make_centerline(wa, walls[best_j]))
            else:
                result.append(wa)

        return result

    def _parallel_overlap(self, a: Wall, b: Wall) -> float:
        """Return the overlapping length between two parallel walls."""
        if a.is_horizontal():
            s = max(min(a.start.x, a.end.x), min(b.start.x, b.end.x))
            e = min(max(a.start.x, a.end.x), max(b.start.x, b.end.x))
        else:
            s = max(min(a.start.y, a.end.y), min(b.start.y, b.end.y))
            e = min(max(a.start.y, a.end.y), max(b.start.y, b.end.y))
        return max(0.0, e - s)

    def _make_centerline(self, a: Wall, b: Wall) -> Wall:
        """Create a centreline wall from two parallel walls *a* and *b*."""
        if a.is_horizontal():
            y = round((a.start.y + b.start.y) / 2)
            xs = round(min(a.start.x, a.end.x, b.start.x, b.end.x))
            xe = round(max(a.start.x, a.end.x, b.start.x, b.end.x))
            return Wall(
                id=f"{a.id}_c",
                start=Point(x=float(xs), y=float(y)),
                end=Point(x=float(xe), y=float(y)),
            )
        else:
            x = round((a.start.x + b.start.x) / 2)
            ys = round(min(a.start.y, a.end.y, b.start.y, b.end.y))
            ye = round(max(a.start.y, a.end.y, b.start.y, b.end.y))
            return Wall(
                id=f"{a.id}_c",
                start=Point(x=float(x), y=float(ys)),
                end=Point(x=float(x), y=float(ye)),
            )

    _BOUNDARY_MARGIN: float = 30.0
    _ISOLATED_WALL_DIST: float = 15.0
    _ARC_SNAP: float = 30.0  # snap threshold for arc-to-straight-wall endpoint matching

    def _filter_outside_boundary(
        self, walls: list[Wall], mask: np.ndarray
    ) -> list[Wall]:
        """Remove walls that are isolated noise outside the floorplan structure.

        Two heuristics:
        1. **Mask boundary**: walls whose midpoint falls far outside the
           mask content area are removed. Uses the thick-wall mask (less
           noisy) to establish a tighter boundary.
        2. **Structural isolation**: walls that have no perpendicular
           intersection with any other wall AND no endpoint near any other
           wall are noise (text/dimension lines, etc.).
        """
        if len(walls) < 4:
            return walls

        # Use thickness-filtered mask for tighter boundary estimation
        thick = self._thickness_filter(mask)
        ys_thick, xs_thick = np.where(thick > 0)
        if len(xs_thick) > 20:
            x_min = float(np.percentile(xs_thick, 1))
            x_max = float(np.percentile(xs_thick, 99))
            y_min = float(np.percentile(ys_thick, 1))
            y_max = float(np.percentile(ys_thick, 99))
        else:
            # Fallback to cleaned mask
            ys_mask, xs_mask = np.where(mask > 0)
            if len(xs_mask) == 0:
                return walls
            x_min = float(np.percentile(xs_mask, 3))
            x_max = float(np.percentile(xs_mask, 97))
            y_min = float(np.percentile(ys_mask, 3))
            y_max = float(np.percentile(ys_mask, 97))

        margin = self._BOUNDARY_MARGIN

        # Pre-compute intersection info for structural isolation check
        has_intersection: dict[int, bool] = {}
        for i, w in enumerate(walls):
            if w.is_arc:
                has_intersection[i] = True
                continue
            intersected = False
            for j, other in enumerate(walls):
                if i == j or other.is_arc:
                    continue
                if w.is_horizontal() and other.is_vertical():
                    # H wall at y, V wall at x — intersect if x in H range and y in V range
                    h_xs = (min(w.start.x, w.end.x), max(w.start.x, w.end.x))
                    v_ys = (min(other.start.y, other.end.y), max(other.start.y, other.end.y))
                    if other.start.x >= h_xs[0] - 5 and other.start.x <= h_xs[1] + 5 and w.start.y >= v_ys[0] - 5 and w.start.y <= v_ys[1] + 5:
                        intersected = True
                        break
                elif w.is_vertical() and other.is_horizontal():
                    v_ys = (min(w.start.y, w.end.y), max(w.start.y, w.end.y))
                    h_xs = (min(other.start.x, other.end.x), max(other.start.x, other.end.x))
                    if w.start.x >= h_xs[0] - 5 and w.start.x <= h_xs[1] + 5 and other.start.y >= v_ys[0] - 5 and other.start.y <= v_ys[1] + 5:
                        intersected = True
                        break
            has_intersection[i] = intersected

        # Check endpoint proximity for walls without intersections
        snap_d = self._ISOLATED_WALL_DIST
        # Diagonal walls use a larger snap distance — their endpoints may not
        # align exactly with H/V wall endpoints but are nearby
        snap_d_diag = snap_d * 3
        structural_indices = set(i for i, v in has_intersection.items() if v)
        for i, w in enumerate(walls):
            if has_intersection[i] or w.is_arc:
                continue
            is_diag = not w.is_horizontal() and not w.is_vertical()
            snap = snap_d_diag if is_diag else snap_d
            near_structural = False
            for j in structural_indices:
                other = walls[j]
                for ep in [(w.start.x, w.start.y), (w.end.x, w.end.y)]:
                    for op in [(other.start.x, other.start.y), (other.end.x, other.end.y)]:
                        if math.hypot(ep[0]-op[0], ep[1]-op[1]) < snap:
                            near_structural = True
                            break
                    if near_structural:
                        break
                if near_structural:
                    break
            has_intersection[i] = near_structural

        result: list[Wall] = []
        for i, w in enumerate(walls):
            mid = w.midpoint()
            outside_boundary = (
                mid.x < x_min - margin
                or mid.x > x_max + margin
                or mid.y < y_min - margin
                or mid.y > y_max + margin
            )
            isolated = not has_intersection[i]
            if outside_boundary or isolated:
                continue
            result.append(w)

        return result

    def _arc_endpoint_connections(
        self,
        arc: Wall,
        straight_walls: list[Wall],
    ) -> tuple[bool, bool]:
        """Return (start_near, end_near): whether each arc endpoint is within
        ``_ARC_SNAP`` pixels of any straight wall endpoint.

        Both True → structural rounded corner connecting two walls.
        Only one True → likely a door arc with one hinge at a wall endpoint.
        """
        start_near = False
        end_near = False
        snap = self._ARC_SNAP
        for w in straight_walls:
            for ep in [w.start, w.end]:
                if math.hypot(arc.start.x - ep.x, arc.start.y - ep.y) < snap:
                    start_near = True
                if math.hypot(arc.end.x - ep.x, arc.end.y - ep.y) < snap:
                    end_near = True
            if start_near and end_near:
                break
        return start_near, end_near

    def _snap_arc_endpoints_to_walls(
        self,
        arc: Wall,
        straight_walls: list[Wall],
    ) -> Wall:
        """Snap arc start/end to the nearest straight wall endpoint within ``_ARC_SNAP``.

        Ensures clean graph topology: arc shares exact endpoint coordinates with
        adjacent straight walls so ``WallGraphBuilder._snap_endpoints`` sees them
        as the same node.
        """
        snap = self._ARC_SNAP
        best_start, best_start_d = arc.start, snap
        best_end, best_end_d = arc.end, snap
        for w in straight_walls:
            for ep in [w.start, w.end]:
                ds = math.hypot(arc.start.x - ep.x, arc.start.y - ep.y)
                de = math.hypot(arc.end.x - ep.x, arc.end.y - ep.y)
                if ds < best_start_d:
                    best_start_d = ds
                    best_start = ep
                if de < best_end_d:
                    best_end_d = de
                    best_end = ep
        return Wall(
            id=arc.id,
            start=best_start,
            end=best_end,
            thickness=arc.thickness,
            exterior=arc.exterior,
            center=arc.center,
            radius=arc.radius,
            start_angle=arc.start_angle,
            end_angle=arc.end_angle,
        )

    # Minimum absolute pixel length used by _filter_small_walls (non-isolated)
    _KEEP_LENGTH_HARD: float = 35.0
    # A wall shorter than this is kept only if it has neighbours on the same axis
    _KEEP_LENGTH_SOFT: float = 25.0
    # Neighbour proximity on the perpendicular axis
    _NEIGHBOUR_TOL: float = 15.0

    def _filter_small_walls(self, walls: list[Wall]) -> list[Wall]:
        """Remove short isolated wall segments likely caused by fixtures/noise."""
        if len(walls) < 3:
            return walls

        result: list[Wall] = []
        min_len = self.params["min_wall_length"]

        for w in walls:
            if w.is_arc:
                result.append(w)
                continue

            length = w.length()
            if length >= self._KEEP_LENGTH_HARD:
                result.append(w)
                continue

            # Check if the wall has any near-parallel neighbour
            has_neighbour = False
            for other in walls:
                if other is w:
                    continue
                if w.is_horizontal() and other.is_horizontal():
                    if abs(w.start.y - other.start.y) < self._NEIGHBOUR_TOL:
                        has_neighbour = True
                        break
                elif w.is_vertical() and other.is_vertical():
                    if abs(w.start.x - other.start.x) < self._NEIGHBOUR_TOL:
                        has_neighbour = True
                        break

            if has_neighbour or length >= self._KEEP_LENGTH_SOFT:
                result.append(w)

        return [w for w in result if w.length() >= min_len]
