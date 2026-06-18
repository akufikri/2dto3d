"""Tests for wallgraph.utils — geometry helper functions."""

from __future__ import annotations

import math
import pytest

from wallgraph.utils import (
    angle_between,
    distance,
    point_line_distance,
    point_on_segment,
    line_intersection,
    segment_intersection,
    merge_collinear_segments,
)


class TestDistance:
    def test_zero_distance(self):
        assert distance((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_pythagorean_triple(self):
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert distance((-1, -1), (2, 3)) == pytest.approx(5.0)


class TestAngleBetween:
    def test_east(self):
        assert angle_between((0, 0), (1, 0)) == pytest.approx(0.0)

    def test_north(self):
        # Positive y is downward in image coords, but math is the same
        assert angle_between((0, 0), (0, 1)) == pytest.approx(math.pi / 2)

    def test_southwest(self):
        ang = angle_between((0, 0), (-1, -1))
        assert ang == pytest.approx(-3 * math.pi / 4)


class TestPointLineDistance:
    def test_point_on_line(self):
        assert point_line_distance((1, 0), (0, 0), (2, 0)) == pytest.approx(0.0)

    def test_perpendicular(self):
        # Point (1, 1) above line y=0
        assert point_line_distance((1, 1), (0, 0), (2, 0)) == pytest.approx(1.0)

    def test_degenerate_line(self):
        # a == b → distance from point to a
        assert point_line_distance((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)


class TestPointOnSegment:
    def test_midpoint(self):
        assert point_on_segment(5, 0, 0, 0, 10, 0, tol=1.0)

    def test_start_point(self):
        assert point_on_segment(0, 0, 0, 0, 10, 0, tol=1.0)

    def test_end_point(self):
        assert point_on_segment(10, 0, 0, 0, 10, 0, tol=1.0)

    def test_off_segment_extension(self):
        # Point at (15, 0) — beyond end
        assert not point_on_segment(15, 0, 0, 0, 10, 0, tol=1.0)

    def test_off_segment_lateral(self):
        # Point at (5, 5) — too far from horizontal segment
        assert not point_on_segment(5, 5, 0, 0, 10, 0, tol=1.0)

    def test_within_tolerance(self):
        assert point_on_segment(5, 0.5, 0, 0, 10, 0, tol=1.0)

    def test_degenerate_segment_hit(self):
        assert point_on_segment(0, 0, 0, 0, 0, 0, tol=1.0)

    def test_degenerate_segment_miss(self):
        assert not point_on_segment(5, 0, 0, 0, 0, 0, tol=1.0)


class TestLineIntersection:
    def test_perpendicular_cross(self):
        pt = line_intersection((0, 5), (10, 5), (5, 0), (5, 10))
        assert pt is not None
        assert pt[0] == pytest.approx(5.0)
        assert pt[1] == pytest.approx(5.0)

    def test_parallel_lines(self):
        assert line_intersection((0, 0), (10, 0), (0, 1), (10, 1)) is None

    def test_collinear_lines(self):
        # Collinear lines → determinant ≈ 0 → None
        assert line_intersection((0, 0), (10, 0), (5, 0), (15, 0)) is None


class TestSegmentIntersection:
    def test_crossing_segments(self):
        pt = segment_intersection((0, 5), (10, 5), (5, 0), (5, 10))
        assert pt is not None
        assert pt[0] == pytest.approx(5.0)
        assert pt[1] == pytest.approx(5.0)

    def test_non_crossing_segments(self):
        # Segments would cross if extended, but don't overlap
        pt = segment_intersection((0, 0), (3, 0), (5, 0), (8, 0))
        assert pt is None

    def test_t_junction(self):
        # Horizontal segment, vertical endpoint exactly on it
        pt = segment_intersection((0, 5), (10, 5), (5, 0), (5, 5))
        assert pt is not None
        assert pt[0] == pytest.approx(5.0)
        assert pt[1] == pytest.approx(5.0)


class TestMergeCollinearSegments:
    def test_two_overlapping_horizontal(self):
        segs = [((0.0, 0.0), (5.0, 0.0)), ((3.0, 0.0), (10.0, 0.0))]
        result = merge_collinear_segments(segs, angle_tol=0.05, dist_tol=1.0)
        assert len(result) == 1
        xs = sorted([result[0][0][0], result[0][1][0]])
        assert xs[0] == pytest.approx(0.0, abs=0.1)
        assert xs[1] == pytest.approx(10.0, abs=0.1)

    def test_non_collinear_segments_kept_separate(self):
        segs = [
            ((0.0, 0.0), (10.0, 0.0)),  # horizontal
            ((0.0, 5.0), (10.0, 5.0)),  # parallel but offset by 5px
        ]
        result = merge_collinear_segments(segs, angle_tol=0.05, dist_tol=1.0)
        assert len(result) == 2

    def test_empty_input(self):
        assert merge_collinear_segments([]) == []

    def test_single_segment_unchanged(self):
        segs = [((1.0, 2.0), (5.0, 2.0))]
        result = merge_collinear_segments(segs)
        assert len(result) == 1
