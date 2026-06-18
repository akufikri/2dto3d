"""
WallGraphValidator: validates topology and geometry of a :class:`WallGraph`.

Usage
-----
::

    from wallgraph import WallGraphValidator, WallGraph

    validator = WallGraphValidator()
    report = validator.validate(graph)

    if report.passed:
        print("Graph is valid")
    else:
        for err in report.errors:
            print("ERROR:", err)

    for warn in report.warnings:
        print("WARN:", warn)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .models import WallGraph
from .builder import WallGraphBuilder
from .constants import SNAP_DISTANCE


@dataclass
class ValidationReport:
    """Result of a :class:`WallGraphValidator` run.

    Attributes
    ----------
    passed:
        ``True`` when no errors were found (warnings are allowed).
    errors:
        List of error messages.  Any error sets ``passed = False``.
    warnings:
        List of non-fatal warning messages.
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record an error and mark the report as failed."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = ["=== Validation Report ==="]
        lines.append(f"Passed: {self.passed}")
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not self.errors and not self.warnings:
            lines.append("  ✓ No issues found")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class WallGraphValidator:
    """Validate the topology and geometry of a :class:`WallGraph`.

    Parameters
    ----------
    snap_distance:
        The endpoint snap distance used when building the graph (forwarded to
        the internal :class:`~wallgraph.builder.WallGraphBuilder` used for
        graph construction).
    """

    def __init__(self, snap_distance: float = SNAP_DISTANCE) -> None:
        self._builder = WallGraphBuilder(snap_distance=snap_distance)

    def validate(self, graph: WallGraph) -> ValidationReport:
        """Run all checks and return a :class:`ValidationReport`.

        Checks (in order):

        1. **Empty graph** — error if no walls at all.
        2. **Zero-length walls** — error for each degenerate wall.
        3. **Graph connectivity** — warning if disconnected components found.
        4. **Floating walls** — warning for degree-1 graph nodes.
        5. **Duplicate segments** — warning for exact duplicates.
        """
        report = ValidationReport()

        if not graph.walls:
            report.add_error("No walls in graph")
            return report

        g = self._builder.build_graph(graph.walls)

        self._check_zero_length(graph, report)
        self._check_connectivity(g, graph, report)
        self._check_floating_walls(g, report)
        self._check_duplicates(graph, report)

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_zero_length(
        self, graph: WallGraph, report: ValidationReport
    ) -> None:
        """Flag walls with length < 1 px as errors."""
        for w in graph.walls:
            if w.length() < 1:
                report.add_error(f"Zero-length wall: {w.id}")

    def _check_connectivity(
        self,
        g: nx.Graph,
        graph: WallGraph,
        report: ValidationReport,
    ) -> None:
        """Warn if the wall graph is not fully connected."""
        if not nx.is_connected(g):
            components = list(nx.connected_components(g))
            report.add_warning(
                f"Wall graph has {len(components)} disconnected components"
            )
            for i, comp in enumerate(components):
                report.add_warning(
                    f"  Component {i + 1}: {len(comp)} nodes"
                )

    def _check_floating_walls(
        self, g: nx.Graph, report: ValidationReport
    ) -> None:
        """Warn about degree-1 nodes, which indicate floating wall ends."""
        floating = [n for n in g.nodes if g.degree(n) == 1]
        if floating:
            report.add_warning(
                f"{len(floating)} degree-1 nodes (possible floating walls)"
            )

    def _check_duplicates(
        self, graph: WallGraph, report: ValidationReport
    ) -> None:
        """Warn about exact duplicate wall segments (both directions checked)."""
        seen: set[str] = set()
        for w in graph.walls:
            key = f"{w.start.x},{w.start.y},{w.end.x},{w.end.y}"
            rev = f"{w.end.x},{w.end.y},{w.start.x},{w.start.y}"
            if key in seen or rev in seen:
                report.add_warning(f"Duplicate wall: {w.id}")
            seen.add(key)
