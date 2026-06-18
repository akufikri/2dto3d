"""Tests for wallgraph.models — data model correctness."""

from __future__ import annotations

import math
import pytest

from wallgraph.models import Point, Wall, Door, Window, WallGraph


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------


class TestPoint:
    def test_distance_to_self(self):
        p = Point(x=3, y=4)
        assert p.distance_to(p) == pytest.approx(0.0)

    def test_distance_to_other(self):
        p1 = Point(x=0, y=0)
        p2 = Point(x=3, y=4)
        assert p1.distance_to(p2) == pytest.approx(5.0)

    def test_as_tuple(self):
        p = Point(x=1.5, y=2.5)
        assert p.as_tuple() == (1.5, 2.5)

    def test_quantize_snaps_to_grid(self):
        p = Point(x=13.2, y=27.9)
        q = p.quantize(grid=5.0)
        assert q.x == pytest.approx(15.0)
        assert q.y == pytest.approx(30.0)

    def test_quantize_zero_stays_zero(self):
        p = Point(x=0, y=0)
        assert p.quantize().x == pytest.approx(0.0)
        assert p.quantize().y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Wall (straight)
# ---------------------------------------------------------------------------


class TestWallStraight:
    def test_is_not_arc(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0))
        assert not w.is_arc

    def test_length_horizontal(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0))
        assert w.length() == pytest.approx(100.0)

    def test_length_diagonal(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=3, y=4))
        assert w.length() == pytest.approx(5.0)

    def test_is_horizontal_true(self):
        w = Wall(id="W0", start=Point(x=0, y=10), end=Point(x=100, y=10))
        assert w.is_horizontal()

    def test_is_horizontal_false(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=0, y=100))
        assert not w.is_horizontal()

    def test_is_vertical_true(self):
        w = Wall(id="W0", start=Point(x=50, y=0), end=Point(x=50, y=100))
        assert w.is_vertical()

    def test_is_vertical_false(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0))
        assert not w.is_vertical()

    def test_midpoint(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0))
        mid = w.midpoint()
        assert mid.x == pytest.approx(50.0)
        assert mid.y == pytest.approx(0.0)

    def test_angle_horizontal(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0))
        assert w.angle() == pytest.approx(0.0)

    def test_angle_vertical(self):
        w = Wall(id="W0", start=Point(x=0, y=0), end=Point(x=0, y=100))
        assert w.angle() == pytest.approx(math.pi / 2)

    def test_as_tuple(self):
        w = Wall(id="W0", start=Point(x=1, y=2), end=Point(x=3, y=4))
        assert w.as_tuple() == ((1.0, 2.0), (3.0, 4.0))


# ---------------------------------------------------------------------------
# Wall (arc)
# ---------------------------------------------------------------------------


class TestWallArc:
    @pytest.fixture
    def arc_wall(self):
        return Wall(
            id="ARC0",
            start=Point(x=100, y=0),
            end=Point(x=0, y=100),
            center=Point(x=0, y=0),
            radius=100.0,
            start_angle=0.0,
            end_angle=math.pi / 2,
        )

    def test_is_arc(self, arc_wall):
        assert arc_wall.is_arc

    def test_arc_length(self, arc_wall):
        # Quarter circle: length = r * π/2
        expected = 100 * math.pi / 2
        assert arc_wall.length() == pytest.approx(expected, rel=1e-3)

    def test_arc_midpoint(self, arc_wall):
        # Mid angle = π/4 → (cos(π/4)*100, sin(π/4)*100)
        mid = arc_wall.midpoint()
        assert mid.x == pytest.approx(100 * math.cos(math.pi / 4), rel=1e-3)
        assert mid.y == pytest.approx(100 * math.sin(math.pi / 4), rel=1e-3)


# ---------------------------------------------------------------------------
# WallGraph
# ---------------------------------------------------------------------------


class TestWallGraph:
    def test_empty_graph(self):
        g = WallGraph()
        assert g.walls == []
        assert g.doors == []
        assert g.windows == []

    def test_graph_with_walls(self, simple_graph):
        assert len(simple_graph.walls) == 4

    def test_serialise_roundtrip(self, simple_graph):
        json_str = simple_graph.model_dump_json()
        restored = WallGraph.model_validate_json(json_str)
        assert len(restored.walls) == len(simple_graph.walls)
        for a, b in zip(simple_graph.walls, restored.walls):
            assert a.id == b.id
            assert a.start.x == pytest.approx(b.start.x)
            assert a.end.y == pytest.approx(b.end.y)
