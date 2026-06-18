"""
Geometry utility functions shared across the wallgraph package.

All functions operate on plain ``(float, float)`` tuples so they can be used
without importing any wallgraph models.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------


def angle_between(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Return the angle (radians) of the vector from *p1* to *p2*."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def distance(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def point_line_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Perpendicular distance from point *p* to the infinite line through *a* and *b*."""
    denom = distance(a, b)
    if denom < 1e-10:
        return distance(p, a)
    return (
        abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
        / denom
    )


# ---------------------------------------------------------------------------
# Segment / intersection helpers
# ---------------------------------------------------------------------------


def point_on_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tol: float = 1.0,
) -> bool:
    """Return ``True`` if point *(px, py)* lies on segment *(x1,y1)–(x2,y2)* within *tol*.

    Uses the projection-onto-segment approach: the point is projected onto the
    segment's supporting line and checked that the projection falls within
    [0, 1] and is within *tol* pixels of the point.
    """
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        # Degenerate segment (single point)
        return distance((px, py), (x1, y1)) < tol
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0 or t > 1:
        return False
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return distance((px, py), (proj_x, proj_y)) < tol


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Return the intersection of infinite lines *p1–p2* and *p3–p4*, or ``None`` if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Return the intersection of segments *p1–p2* and *p3–p4*, or ``None``.

    Unlike :func:`line_intersection`, this verifies that the intersection lies
    within both segment bounding boxes (with a small 1e-10 tolerance).
    """
    pt = line_intersection(p1, p2, p3, p4)
    if pt is None:
        return None
    x, y = pt
    eps = 1e-10
    in_seg1 = (
        min(p1[0], p2[0]) - eps <= x <= max(p1[0], p2[0]) + eps
        and min(p1[1], p2[1]) - eps <= y <= max(p1[1], p2[1]) + eps
    )
    in_seg2 = (
        min(p3[0], p4[0]) - eps <= x <= max(p3[0], p4[0]) + eps
        and min(p3[1], p4[1]) - eps <= y <= max(p3[1], p4[1]) + eps
    )
    return pt if (in_seg1 and in_seg2) else None


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------


def merge_collinear_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    angle_tol: float = 0.05,
    dist_tol: float = 5.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Merge overlapping collinear segments into their bounding spans.

    Two segments are considered collinear when:

    * Their angle difference is within *angle_tol* radians (mod π), **and**
    * Both endpoints of one segment are within *dist_tol* pixels of the
      supporting line of the other.

    Parameters
    ----------
    segments:
        List of ``((x1, y1), (x2, y2))`` segment tuples.
    angle_tol:
        Maximum allowed angle difference (radians) for two segments to be
        considered parallel/collinear.
    dist_tol:
        Maximum perpendicular distance (px) for a segment to be merged into
        an existing group.

    Returns
    -------
    list of ``((x1, y1), (x2, y2))`` tuples, one per merged group.
    """
    if not segments:
        return []

    used = [False] * len(segments)
    merged: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for i, seg1 in enumerate(segments):
        if used[i]:
            continue
        p1, p2 = seg1
        ang = angle_between(p1, p2)
        group = [seg1]
        used[i] = True

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            p3, p4 = segments[j]
            ang2 = angle_between(p3, p4)
            ang_diff = abs(ang - ang2)
            # Account for segments that are antiparallel (differ by π)
            if ang_diff > angle_tol and abs(ang_diff - math.pi) > angle_tol:
                continue
            d1 = point_line_distance(p3, p1, p2)
            d2 = point_line_distance(p4, p1, p2)
            if d1 > dist_tol or d2 > dist_tol:
                continue
            group.append(segments[j])
            used[j] = True

        # Project all group points onto the group's principal axis and take extremes
        all_pts: list[tuple[float, float]] = []
        for (a, b), (c, d) in group:  # type: ignore[misc]
            all_pts.extend([(a, b), (c, d)])  # type: ignore[list-item]

        proj = [
            (pt[0] * math.cos(ang) + pt[1] * math.sin(ang), pt)
            for pt in all_pts
        ]
        proj.sort(key=lambda x: x[0])
        merged.append((proj[0][1], proj[-1][1]))

    return merged
