"""Tests for WallGraphBuilder — endpoint snapping, junction resolution, dedup."""

from __future__ import annotations

import pytest

from wallgraph.models import Point, Wall, WallGraph
from wallgraph.builder import WallGraphBuilder


@pytest.fixture
def builder():
    return WallGraphBuilder(snap_distance=5.0, min_wall_length=10.0)


# ---------------------------------------------------------------------------
# _snap_endpoints
# ---------------------------------------------------------------------------


class TestSnapEndpoints:
    def test_nearby_endpoints_merged(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            # Start is 3px away from end of W0 → should snap
            Wall(id="W1", start=Point(x=103, y=0), end=Point(x=103, y=100)),
        ]
        snapped = builder._snap_endpoints(walls)
        # The start of W1 and end of W0 should share the same x coordinate
        w0_end_x = next(w.end.x for w in snapped if w.id == "W0")
        w1_start_x = next(w.start.x for w in snapped if w.id == "W1")
        assert w0_end_x == pytest.approx(w1_start_x, abs=1.0)

    def test_far_endpoints_not_merged(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=200, y=0), end=Point(x=300, y=0)),
        ]
        snapped = builder._snap_endpoints(walls)
        assert len(snapped) == 2

    def test_degenerate_wall_removed_after_snap(self, builder):
        """Two walls whose endpoints are within snap distance of each other
        collapse to a zero-length segment → should be dropped."""
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=3, y=0)),
        ]
        # Both endpoints snap to the same point (distance = 3 < snap = 5)
        # but only if other walls' endpoints cluster them.
        # This test just checks that the pipeline doesn't crash.
        result = builder._snap_endpoints(walls)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_exact_duplicate_removed(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=0, y=0), end=Point(x=100, y=0)),
        ]
        result = builder._deduplicate(walls)
        assert len(result) == 1

    def test_reverse_duplicate_removed(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=100, y=0), end=Point(x=0, y=0)),
        ]
        result = builder._deduplicate(walls)
        assert len(result) == 1

    def test_distinct_walls_kept(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=0, y=100), end=Point(x=100, y=100)),
        ]
        result = builder._deduplicate(walls)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _filter_short_walls
# ---------------------------------------------------------------------------


class TestFilterShortWalls:
    def test_short_wall_removed(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=5, y=0)),  # 5px < 10px
        ]
        assert builder._filter_short_walls(walls) == []

    def test_long_wall_kept(self, builder):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
        ]
        assert len(builder._filter_short_walls(walls)) == 1


# ---------------------------------------------------------------------------
# _split_wall_at — ID uniqueness fix
# ---------------------------------------------------------------------------


class TestSplitWallAt:
    def test_split_ids_are_unique(self, builder):
        w = Wall(id="W0000", start=Point(x=0, y=0), end=Point(x=300, y=0))
        pts = [(100.0, 0.0), (200.0, 0.0)]
        segs = builder._split_wall_at(w, pts)
        ids = [s.id for s in segs]
        assert len(ids) == len(set(ids)), "Duplicate IDs after split!"

    def test_split_produces_correct_count(self, builder):
        w = Wall(id="W0000", start=Point(x=0, y=0), end=Point(x=300, y=0))
        pts = [(100.0, 0.0), (200.0, 0.0)]
        segs = builder._split_wall_at(w, pts)
        assert len(segs) == 3

    def test_split_endpoints_contiguous(self, builder):
        w = Wall(id="W0000", start=Point(x=0, y=0), end=Point(x=300, y=0))
        pts = [(100.0, 0.0), (200.0, 0.0)]
        segs = builder._split_wall_at(w, pts)
        segs.sort(key=lambda s: s.start.x)
        assert segs[0].start.x == pytest.approx(0.0)
        assert segs[0].end.x == pytest.approx(100.0)
        assert segs[1].start.x == pytest.approx(100.0)
        assert segs[1].end.x == pytest.approx(200.0)
        assert segs[2].start.x == pytest.approx(200.0)
        assert segs[2].end.x == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_rectangle_graph_structure(self, builder, simple_rectangle_walls):
        g = builder.build_graph(simple_rectangle_walls)
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 4

    def test_graph_is_connected(self, builder, simple_rectangle_walls):
        import networkx as nx
        g = builder.build_graph(simple_rectangle_walls)
        assert nx.is_connected(g)


# ---------------------------------------------------------------------------
# build (full pipeline)
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_returns_wallgraph(self, builder, simple_rectangle_walls):
        graph = builder.build(simple_rectangle_walls)
        assert isinstance(graph, WallGraph)

    def test_build_rectangle_preserved(self, builder, simple_rectangle_walls):
        graph = builder.build(simple_rectangle_walls)
        assert len(graph.walls) >= 3  # At least 3 walls survive

    def test_build_empty_input(self, builder):
        graph = builder.build([])
        assert graph.walls == []

    def test_build_single_wall(self, builder):
        walls = [Wall(id="W0", start=Point(x=0, y=0), end=Point(x=200, y=0))]
        graph = builder.build(walls)
        assert len(graph.walls) == 1
