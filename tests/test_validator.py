"""Tests for WallGraphValidator — topology checks."""

from __future__ import annotations

import pytest

from wallgraph.models import Point, Wall, WallGraph
from wallgraph.validator import WallGraphValidator, ValidationReport


@pytest.fixture
def validator():
    return WallGraphValidator(snap_distance=5.0)


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_defaults_passed(self):
        r = ValidationReport()
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_marks_failed(self):
        r = ValidationReport()
        r.add_error("something went wrong")
        assert not r.passed
        assert len(r.errors) == 1

    def test_add_warning_keeps_passed(self):
        r = ValidationReport()
        r.add_warning("be careful")
        assert r.passed  # warnings don't fail the report

    def test_str_contains_error(self):
        r = ValidationReport()
        r.add_error("bad wall")
        assert "bad wall" in str(r)
        assert "✗" in str(r)

    def test_str_contains_warning(self):
        r = ValidationReport()
        r.add_warning("floating wall")
        assert "floating wall" in str(r)
        assert "⚠" in str(r)

    def test_str_success_message(self):
        r = ValidationReport()
        assert "No issues" in str(r)

    def test_as_dict(self):
        r = ValidationReport()
        r.add_error("err1")
        r.add_warning("warn1")
        d = r.as_dict()
        assert d["passed"] is False
        assert "err1" in d["errors"]
        assert "warn1" in d["warnings"]


# ---------------------------------------------------------------------------
# WallGraphValidator
# ---------------------------------------------------------------------------


class TestValidateEmptyGraph:
    def test_empty_graph_fails(self, validator):
        report = validator.validate(WallGraph())
        assert not report.passed
        assert any("No walls" in e for e in report.errors)


class TestValidateValidGraph:
    def test_valid_rectangle_passes(self, validator, simple_graph):
        report = validator.validate(simple_graph)
        assert report.passed, f"Expected pass, got errors: {report.errors}"

    def test_valid_rectangle_no_errors(self, validator, simple_graph):
        report = validator.validate(simple_graph)
        assert report.errors == []


class TestValidateZeroLengthWall:
    def test_zero_length_wall_is_error(self, validator):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=0, y=0)),  # degenerate
        ]
        report = validator.validate(WallGraph(walls=walls))
        assert not report.passed
        assert any("Zero-length" in e for e in report.errors)


class TestValidateDisconnected:
    def test_disconnected_graph_warns(self, validator):
        # Two separate walls with no shared endpoints
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=500, y=0), end=Point(x=600, y=0)),
        ]
        report = validator.validate(WallGraph(walls=walls))
        assert any("disconnected" in w.lower() for w in report.warnings)


class TestValidateDuplicates:
    def test_duplicate_wall_warns(self, validator):
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=100, y=0)),
            Wall(id="W1", start=Point(x=0, y=0), end=Point(x=100, y=0)),
        ]
        report = validator.validate(WallGraph(walls=walls))
        assert any("Duplicate" in w for w in report.warnings)


class TestValidateFloatingWalls:
    def test_floating_wall_warns(self, validator):
        # T-shape: centre wall not connected at one end
        walls = [
            Wall(id="W0", start=Point(x=0, y=0), end=Point(x=300, y=0)),
            Wall(id="W1", start=Point(x=150, y=0), end=Point(x=150, y=100)),
        ]
        # W1's end (150,100) has degree 1 → floating
        report = validator.validate(WallGraph(walls=walls))
        # Either the floating wall or the disconnected warning should fire
        all_msgs = report.errors + report.warnings
        assert any("degree-1" in m or "floating" in m.lower() for m in all_msgs)
