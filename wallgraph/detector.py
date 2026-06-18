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

from .models import Wall, Point
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
        return self._detect_binary(image_path)

    def detect_hough(self, image_path: str) -> list[Wall]:
        """Detect walls using skeletonisation + probabilistic Hough transform.

        Produces fewer, more axis-aligned line segments.  Suitable for
        drawings where walls are thick filled regions.
        """
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

    def visualize(
        self,
        image_path: str,
        walls: list[Wall],
        output_path: str | None = None,
    ) -> np.ndarray:
        """Draw detected walls on the source image and optionally save it.

        Straight walls are drawn in green, arcs in cyan.  Start endpoints
        are blue, end endpoints are red.

        Parameters
        ----------
        image_path:
            Path to the original floorplan image.
        walls:
            Wall segments to draw.
        output_path:
            If provided, save the annotated image to this path.

        Returns
        -------
        np.ndarray
            The annotated BGR image array.
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

        if output_path:
            cv2.imwrite(output_path, img)
        return img

    # ------------------------------------------------------------------
    # Internal: binary/contour detection
    # ------------------------------------------------------------------

    def _detect_binary(self, image_path: str) -> list[Wall]:
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

        # Morphological open (remove thin noise)
        k_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.params["open_kernel"],) * 2
        )
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

        # Remove small noise components
        cleaned = self._remove_small_components(opened)

        # --- Hough directly on cleaned mask for straight walls ---
        # Using mask (not skeleton) preserves thin 3-5px perimeter walls
        # that skeleton+pruning would destroy. Duplicate detections from
        # thick walls are collapsed by merge logic below.
        raw_lines = self._hough_lines(cleaned)
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

        # --- Arc detection on thickness-filtered mask ---
        # Use 5×5 open mask (text-free) for arc detection — preserves
        # wall body shape including fillets at corners.
        thick_mask = self._thickness_filter(cleaned)
        thick_mask = self._remove_small_components(thick_mask)
        thick_contours, _ = cv2.findContours(
            thick_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        raw_pts: np.ndarray | None = None
        if thick_contours:
            raw_pts = max(thick_contours, key=cv2.contourArea).reshape(-1, 2)

        if raw_pts is not None:
            arc_walls = self._detect_arcs_from_contour(
                thick_contours, raw_pts, len(walls)
            )
            walls.extend(arc_walls)

        # Remove straight walls that overlap with detected arcs
        walls = self._remove_arc_overlaps(walls)

        walls = self._merge_nearby_walls(walls)
        walls = self._filter_small_walls(walls)
        return walls

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
        thin walls), classify each connected component by its bounding-box
        minimum dimension. Walls are thick (min_dim ≥ threshold); text,
        dimension lines, and door arcs are thin (min_dim < threshold).

        Classification rules:
        - **Wall**: min_dim ≥ 15 OR area ≥ min_component_area * 4
          (large components are always walls, regardless of min_dim)
        - **Text/dimension**: min_dim < 15 AND area < min_component_area * 4
          (thin, small components are text labels, dimension lines, symbols)

        This preserves ALL wall pixels (including thin 3–5 px walls) while
        precisely removing text, dimension lines, door arcs, and symbols.
        """
        area_threshold = self.params["min_component_area"] * 4

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, 8
        )

        wall_mask = np.zeros_like(cleaned)

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            min_dim = min(w, h)

            # Wall: thick component OR large-area component
            if min_dim >= 15 or comp_area >= area_threshold:
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
        raw_pts: np.ndarray,
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

                nearest = self._closest_point_idx(raw_pts, p2)

                wall: Wall | None = None
                for half in (10, 12, 15, 20, 25):
                    seg = self._circular_slice(
                        raw_pts, nearest - half, nearest + half + 1
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
            if len(raw_pts) >= 7:
                curv_arcs = self._detect_curved_segments(raw_pts, base_idx, approx_pts)
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

        merged: list[Wall] = []
        merged.extend(self._merge_axis(h_walls, "h"))
        merged.extend(self._merge_axis(v_walls, "v"))
        merged.extend(arcs)
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

    def _filter_outside_boundary(
        self, walls: list[Wall], mask: np.ndarray
    ) -> list[Wall]:
        """Remove walls that are isolated noise outside the floorplan structure.

        Two heuristics:
        1. **Mask boundary**: walls whose midpoint falls far outside the
           mask content area are removed.
        2. **Structural isolation**: walls that have no perpendicular
           intersection with any other wall AND no endpoint near any other
           wall are noise (text/dimension lines, etc.).
        """
        if len(walls) < 4:
            return walls

        # Mask-based boundary
        ys_mask, xs_mask = np.where(mask > 0)
        if len(xs_mask) == 0:
            return walls

        # Use 95th percentile for boundary — excludes outlier text areas
        x_min = float(np.percentile(xs_mask, 5))
        x_max = float(np.percentile(xs_mask, 95))
        y_min = float(np.percentile(ys_mask, 5))
        y_max = float(np.percentile(ys_mask, 95))
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
        # Only count proximity to STRUCTURAL walls (walls that have
        # intersections), not to other isolated walls.  Two isolated
        # text lines near each other are still noise.
        snap_d = self._ISOLATED_WALL_DIST
        structural_indices = set(i for i, v in has_intersection.items() if v)
        for i, w in enumerate(walls):
            if has_intersection[i] or w.is_arc:
                continue
            near_structural = False
            for j in structural_indices:
                other = walls[j]
                for ep in [(w.start.x, w.start.y), (w.end.x, w.end.y)]:
                    for op in [(other.start.x, other.start.y), (other.end.x, other.end.y)]:
                        if math.hypot(ep[0]-op[0], ep[1]-op[1]) < snap_d:
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
