"""
WallGraphBuilder: converts raw wall segments into a clean, topologically
consistent :class:`~wallgraph.models.WallGraph`.

Pipeline
--------
1. :meth:`_snap_endpoints`     — cluster nearby endpoints → shared nodes
2. :meth:`_resolve_junctions`  — split walls at T/X intersections
3. :meth:`_filter_short_walls` — discard walls below ``min_wall_length``
4. :meth:`_deduplicate`        — remove exact-duplicate segments
5. :meth:`_snap_endpoints`     — second pass after dedup
6. :meth:`_filter_noise`       — remove isolated degree-1 stubs via graph
"""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx

from .models import Wall, Point, WallGraph
from .utils import point_on_segment, segment_intersection, distance
from .constants import (
    SNAP_DISTANCE,
    GRAPH_MIN_WALL_LENGTH,
    JUNCTION_RESOLVE_MAX_WALLS,
    JUNCTION_RESOLVE_ITERATIONS,
    NOISE_LENGTH_RATIO,
    MIN_COMPONENT_NODES,
    GRAPH_ANGLE_TOLERANCE,
    DANGLING_ENDPOINT_MAX_GAP,
)


class WallGraphBuilder:
    """Build a topologically consistent :class:`WallGraph` from raw wall segments.

    Parameters
    ----------
    snap_distance:
        Maximum distance (px) between two endpoints for them to be snapped
        into one shared graph node.
    min_wall_length:
        Minimum wall length to keep after graph construction.
    """

    def __init__(
        self,
        snap_distance: float = SNAP_DISTANCE,
        min_wall_length: float = GRAPH_MIN_WALL_LENGTH,
        min_component_nodes: int = MIN_COMPONENT_NODES,
        angle_tolerance: float = GRAPH_ANGLE_TOLERANCE,
        dangling_max_gap: float = DANGLING_ENDPOINT_MAX_GAP,
    ) -> None:
        self.snap = snap_distance
        self.min_wall_len = min_wall_length
        self.min_component_nodes = min_component_nodes
        self.angle_tolerance = angle_tolerance
        self.dangling_max_gap = dangling_max_gap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, walls: list[Wall]) -> WallGraph:
        """Run the full builder pipeline and return a :class:`WallGraph`.

        Parameters
        ----------
        walls:
            Raw wall segments (output of :class:`~wallgraph.detector.WallDetector`).

        Returns
        -------
        WallGraph
            Cleaned, deduplicated, topology-resolved wall graph.
        """
        walls = self._snap_endpoints(walls)
        walls = self._resolve_junctions(walls)
        walls = self._filter_short_walls(walls)
        walls = self._deduplicate(walls)
        walls = self._snap_endpoints(walls)
        walls = self._filter_noise(walls)
        walls = self._close_dangling_endpoints(walls)
        walls = self._filter_diagonal_walls(walls)
        walls = self._filter_small_components(walls)
        return WallGraph(walls=walls)

    def build_graph(self, walls: Iterable[Wall]) -> nx.Graph:
        """Build a NetworkX graph from *walls* for topology analysis.

        Nodes are ``(x, y)`` tuples; edges carry ``wall_id`` and ``length``
        attributes.
        """
        G: nx.Graph = nx.Graph()
        for w in walls:
            s = w.start.as_tuple()
            e = w.end.as_tuple()
            G.add_node(s)
            G.add_node(e)
            G.add_edge(s, e, wall_id=w.id, length=w.length())
        return G

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _snap_endpoints(self, walls: list[Wall]) -> list[Wall]:
        """Cluster endpoints within ``snap_distance`` and replace with centroid.

        Two-pass algorithm:
        1. Gather all endpoints.
        2. For each unprocessed point, grow a cluster of nearby points and
           compute their centroid.  All cluster members map to that centroid.
        """
        if not walls:
            return walls

        all_pts: list[tuple[float, float]] = []
        for w in walls:
            all_pts.append(w.start.as_tuple())
            all_pts.append(w.end.as_tuple())

        merged_pt: dict[tuple[float, float], Point] = {}

        for pt in all_pts:
            if pt in merged_pt:
                continue
            cluster: list[tuple[float, float]] = [pt]
            for other in all_pts:
                if other not in merged_pt and other != pt:
                    if distance(pt, other) <= self.snap:
                        cluster.append(other)

            avg_x = round(sum(p[0] for p in cluster) / len(cluster), 1)
            avg_y = round(sum(p[1] for p in cluster) / len(cluster), 1)
            merged = Point(x=avg_x, y=avg_y)
            for p in cluster:
                merged_pt[p] = merged

        new_walls: list[Wall] = []
        for w in walls:
            s = merged_pt.get(w.start.as_tuple(), w.start)
            e = merged_pt.get(w.end.as_tuple(), w.end)
            if distance(s.as_tuple(), e.as_tuple()) < 1:
                continue
            new_walls.append(
                Wall(
                    id=w.id,
                    start=s,
                    end=e,
                    thickness=w.thickness,
                    exterior=w.exterior,
                    center=w.center,
                    radius=w.radius,
                    start_angle=w.start_angle,
                    end_angle=w.end_angle,
                )
            )
        return new_walls

    def _resolve_junctions(self, walls: list[Wall]) -> list[Wall]:
        """Split walls at T-junctions and cross-intersections.

        Iterates up to ``JUNCTION_RESOLVE_ITERATIONS`` passes.  Skipped when
        wall count exceeds ``JUNCTION_RESOLVE_MAX_WALLS`` to bound O(n²) cost.

        Arc walls are never split.
        """
        if len(walls) > JUNCTION_RESOLVE_MAX_WALLS:
            return walls

        result = list(walls)
        changed = True
        iteration = 0

        while changed and iteration < JUNCTION_RESOLVE_ITERATIONS:
            changed = False
            iteration += 1
            i = 0
            while i < len(result):
                # Guard: stop if wall count grew beyond the O(n²) safety limit
                if len(result) > JUNCTION_RESOLVE_MAX_WALLS:
                    changed = False
                    break
                w = result[i]
                if w.is_arc:
                    i += 1
                    continue

                split_points: set[tuple[float, float]] = set()

                for j, other in enumerate(result):
                    if i == j or other.is_arc:
                        continue

                    # T-junction: endpoint of *other* lies on body of *w*
                    for ep in [
                        (other.start.x, other.start.y),
                        (other.end.x, other.end.y),
                    ]:
                        if point_on_segment(
                            ep[0], ep[1],
                            w.start.x, w.start.y,
                            w.end.x, w.end.y,
                            tol=self.snap,
                        ):
                            d_s = distance(ep, w.start.as_tuple())
                            d_e = distance(ep, w.end.as_tuple())
                            if d_s > self.snap and d_e > self.snap:
                                split_points.add(ep)

                    # Cross-intersection
                    inter = segment_intersection(
                        w.start.as_tuple(), w.end.as_tuple(),
                        other.start.as_tuple(), other.end.as_tuple(),
                    )
                    if inter:
                        d_s = distance(inter, w.start.as_tuple())
                        d_e = distance(inter, w.end.as_tuple())
                        if d_s > self.snap and d_e > self.snap:
                            split_points.add(inter)

                if split_points:
                    new_segs = self._split_wall_at(w, list(split_points))
                    new_segs = [s for s in new_segs if s.length() >= self.snap]
                    if new_segs:
                        result[i] = new_segs[0]
                        result.extend(new_segs[1:])
                    changed = True
                i += 1

        return result

    def _split_wall_at(
        self, wall: Wall, points: list[tuple[float, float]]
    ) -> list[Wall]:
        """Split *wall* at one or more interior *points*.

        Each sub-segment receives a unique ID derived from the parent wall ID
        and its position index, ensuring no two segments share the same ID.
        """
        along = sorted(
            [(distance(p, wall.start.as_tuple()), p) for p in points],
            key=lambda x: x[0],
        )

        segs: list[Wall] = []
        prev = wall.start.as_tuple()
        for seg_idx, (_, pt) in enumerate(along):
            segs.append(
                Wall(
                    id=f"{wall.id}_s{seg_idx}",
                    start=Point(x=prev[0], y=prev[1]),
                    end=Point(x=pt[0], y=pt[1]),
                    thickness=wall.thickness,
                )
            )
            prev = pt
        segs.append(
            Wall(
                id=f"{wall.id}_s{len(along)}",
                start=Point(x=prev[0], y=prev[1]),
                end=Point(x=wall.end.x, y=wall.end.y),
                thickness=wall.thickness,
            )
        )
        return segs

    def _filter_short_walls(self, walls: list[Wall]) -> list[Wall]:
        """Discard walls shorter than ``min_wall_length``.

        Arc walls use half the threshold since their chord length is shorter
        than an equivalent straight wall spanning the same region.
        """
        arc_min = max(self.min_wall_len * 0.5, 15.0)
        return [
            w for w in walls
            if (w.is_arc and w.length() >= arc_min)
            or (not w.is_arc and w.length() >= self.min_wall_len)
        ]

    def _filter_noise(self, walls: list[Wall]) -> list[Wall]:
        """Remove isolated floating stubs using the topology graph.

        A wall is classified as noise when both its endpoints have degree 1
        in the wall graph *and* the wall is shorter than
        ``min_wall_length * NOISE_LENGTH_RATIO``.

        Arc walls are never classified as noise — they represent curved
        structural features or door swings which may not connect to a grid.
        """
        if len(walls) < 3:
            return walls

        g = self.build_graph(walls)
        noise_ids: set[str] = set()
        noise_threshold = self.min_wall_len * NOISE_LENGTH_RATIO

        for w in walls:
            if w.is_arc:
                continue  # Never remove arc walls as noise
            s = w.start.as_tuple()
            e = w.end.as_tuple()
            s_deg = g.degree(s)
            e_deg = g.degree(e)
            if s_deg == 1 and e_deg == 1 and w.length() < noise_threshold:
                noise_ids.add(w.id)

        return [w for w in walls if w.id not in noise_ids]

    def _filter_diagonal_walls(self, walls: list[Wall]) -> list[Wall]:
        """Remove diagonal (non-H/V) straight walls.

        Structural walls in rectilinear floor plans are axis-aligned.
        Diagonal segments are typically fixture outlines (toilets, sinks,
        stair treads) that survived earlier filters.

        A wall is kept when:
        - It is an arc wall (curved structural features are always kept).
        - ``angle_tolerance == 0`` (filter disabled).
        - ``|dy| < |dx| * angle_tolerance``  → horizontal
        - ``|dx| < |dy| * angle_tolerance``  → vertical
        """
        if not self.angle_tolerance:
            return walls

        kept: list[Wall] = []
        for w in walls:
            if w.is_arc:
                kept.append(w)
                continue
            dx = abs(w.end.x - w.start.x)
            dy = abs(w.end.y - w.start.y)
            is_h = dy < dx * self.angle_tolerance
            is_v = dx < dy * self.angle_tolerance
            if is_h or is_v:
                kept.append(w)
        return kept

    def _filter_small_components(self, walls: list[Wall]) -> list[Wall]:
        """Remove isolated small connected components from the wall graph.

        Components with fewer than ``min_component_nodes`` nodes are typically
        fixture outlines (toilet, sink, stair symbols) rather than structural
        walls.  Arc walls inside a kept component are preserved; arc walls in
        dropped components are also dropped.
        """
        if len(walls) < self.min_component_nodes:
            return walls

        g = self.build_graph(walls)
        import networkx as nx
        kept_nodes: set[tuple[float, float]] = set()
        for comp in nx.connected_components(g):
            if len(comp) >= self.min_component_nodes:
                kept_nodes.update(comp)

        return [
            w for w in walls
            if w.start.as_tuple() in kept_nodes or w.end.as_tuple() in kept_nodes
        ]

    def _close_dangling_endpoints(self, walls: list[Wall]) -> list[Wall]:
        """Layer C gap fix: extend degree-1 endpoints to connect nearby walls.

        After noise filtering, some endpoints may "dangle" just short of
        intersecting another wall because of ML mask gaps.  For each dangling
        (degree-1) node this method:

        1. Casts a probe ray from the endpoint along the wall's direction.
        2. Finds the nearest wall segment that the probe intersects within
           ``dangling_max_gap`` pixels.
        3. Adds a short bridge wall from the endpoint to the intersection.

        The bridge walls are passed through :meth:`_snap_endpoints` and
        :meth:`_resolve_junctions` so the hit wall is properly split at the
        new T-junction.

        Arc walls are never extended.
        """
        if not walls or self.dangling_max_gap <= 0:
            return walls

        g = self.build_graph(walls)
        wall_by_id: dict[str, Wall] = {w.id: w for w in walls}

        # Collect dangling endpoints: degree-1 nodes only
        dangling: list[tuple[tuple[float, float], tuple[float, float], str]] = []
        for node in list(g.nodes()):
            if g.degree(node) != 1:
                continue
            neighbor = next(iter(g.neighbors(node)))
            edge_data = g.get_edge_data(node, neighbor) or {}
            wall_id = edge_data.get("wall_id", "")
            parent = wall_by_id.get(wall_id)
            if parent and parent.is_arc:
                continue
            dangling.append((node, neighbor, wall_id))

        if not dangling:
            return walls

        bridges: list[Wall] = []
        bridge_counter = 0

        for (dang_pt, conn_pt, wall_id) in dangling:
            dx = dang_pt[0] - conn_pt[0]
            dy = dang_pt[1] - conn_pt[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1:
                continue
            ux, uy = dx / seg_len, dy / seg_len  # unit vector away from graph

            # Probe segment: dangling point → max_gap ahead in same direction
            probe_end = (
                dang_pt[0] + ux * self.dangling_max_gap,
                dang_pt[1] + uy * self.dangling_max_gap,
            )

            best_dist = self.dangling_max_gap + 1
            best_pt: tuple[float, float] | None = None

            for w in walls:
                if w.is_arc or w.id == wall_id:
                    continue
                # Skip walls already sharing the connected node
                if w.start.as_tuple() == conn_pt or w.end.as_tuple() == conn_pt:
                    continue

                inter = segment_intersection(
                    dang_pt, probe_end,
                    w.start.as_tuple(), w.end.as_tuple(),
                )
                if inter is not None:
                    d = distance(dang_pt, inter)
                    if self.snap < d < best_dist:
                        best_dist = d
                        best_pt = inter

            if best_pt is not None:
                bridge_counter += 1
                parent = wall_by_id.get(wall_id)
                thick = parent.thickness if parent else 0
                bridges.append(
                    Wall(
                        id=f"bridge_{bridge_counter}",
                        start=Point(x=round(dang_pt[0], 1), y=round(dang_pt[1], 1)),
                        end=Point(x=round(best_pt[0], 1), y=round(best_pt[1], 1)),
                        thickness=thick,
                    )
                )

        if not bridges:
            return walls

        # Re-run snap + junction resolve so hit walls split at new T-junctions
        combined = list(walls) + bridges
        combined = self._snap_endpoints(combined)
        combined = self._resolve_junctions(combined)
        return combined

    def _deduplicate(self, walls: list[Wall]) -> list[Wall]:
        """Remove exact duplicate wall segments (considering both directions)."""
        seen: set[tuple[float, float, float, float]] = set()
        unique: list[Wall] = []
        for w in walls:
            x1, y1 = round(w.start.x, 0), round(w.start.y, 0)
            x2, y2 = round(w.end.x, 0), round(w.end.y, 0)
            key1 = (x1, y1, x2, y2)
            key2 = (x2, y2, x1, y1)
            if key1 not in seen and key2 not in seen:
                seen.add(key1)
                unique.append(w)
        return unique
