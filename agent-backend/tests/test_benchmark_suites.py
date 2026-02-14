"""Tests for benchmark suite parsers and registry (no network required)."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from benchwarmer.utils.benchmark_suites import (
    parse_dimacs,
    parse_biqmac,
    parse_edge_list,
    list_suites,
    list_instances,
    _cache_path,
    fetch_instance,
    CACHE_DIR,
)


# ──────────────────────────────────────────────────────────────
# DIMACS parser
# ──────────────────────────────────────────────────────────────

class TestParseDIMACS:
    def test_basic_graph(self):
        text = """\
c Example DIMACS graph
p edge 5 4
e 1 2
e 2 3
e 3 4
e 4 5
"""
        result = parse_dimacs(text)
        assert len(result["nodes"]) == 5
        assert result["nodes"] == [0, 1, 2, 3, 4]
        assert len(result["edges"]) == 4
        # Verify 1-indexed → 0-indexed
        assert result["edges"][0] == {"source": 0, "target": 1, "weight": 1.0}
        assert result["edges"][3] == {"source": 3, "target": 4, "weight": 1.0}
        assert result["metadata"]["generator"] == "dimacs"

    def test_weighted_edges(self):
        text = """\
p edge 3 2
e 1 2 3.5
e 2 3 1.2
"""
        result = parse_dimacs(text)
        assert result["edges"][0]["weight"] == 3.5
        assert result["edges"][1]["weight"] == 1.2

    def test_with_comments(self):
        text = """\
c This is a comment
c Another comment
p edge 2 1
c Edge below:
e 1 2
"""
        result = parse_dimacs(text)
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_col_format(self):
        """DIMACS coloring format uses same structure."""
        text = """\
c queen5_5.col
p edge 25 160
e 1 2
e 1 6
"""
        result = parse_dimacs(text)
        assert len(result["nodes"]) == 25
        assert len(result["edges"]) == 2


# ──────────────────────────────────────────────────────────────
# Biq Mac parser
# ──────────────────────────────────────────────────────────────

class TestParseBiqMac:
    def test_basic(self):
        text = """\
5 4
1 2 1.0
2 3 -1.0
3 4 2.5
4 5 0.5
"""
        result = parse_biqmac(text)
        assert len(result["nodes"]) == 5
        assert result["nodes"] == [0, 1, 2, 3, 4]
        assert len(result["edges"]) == 4
        # Check 1-indexed → 0-indexed
        assert result["edges"][0] == {"source": 0, "target": 1, "weight": 1.0}
        assert result["edges"][1] == {"source": 1, "target": 2, "weight": -1.0}
        assert result["metadata"]["generator"] == "biqmac"

    def test_large_header(self):
        text = """\
100 3
1 50 1.0
50 100 2.0
1 100 3.0
"""
        result = parse_biqmac(text)
        assert len(result["nodes"]) == 100
        assert len(result["edges"]) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_biqmac("")


# ──────────────────────────────────────────────────────────────
# Edge list parser
# ──────────────────────────────────────────────────────────────

class TestParseEdgeList:
    def test_basic(self):
        text = """\
# Comment
0\t1
1\t2
2\t3
"""
        result = parse_edge_list(text)
        assert len(result["nodes"]) == 4
        assert len(result["edges"]) == 3

    def test_skips_self_loops(self):
        text = "0 1\n1 1\n1 2\n"
        result = parse_edge_list(text)
        assert len(result["edges"]) == 2  # self-loop removed

    def test_reindexes_nodes(self):
        """Non-contiguous node IDs get remapped to 0..n-1."""
        text = "10 20\n20 30\n"
        result = parse_edge_list(text)
        assert result["nodes"] == [0, 1, 2]
        assert result["edges"][0]["source"] == 0
        assert result["edges"][0]["target"] == 1
        assert result["edges"][1]["source"] == 1
        assert result["edges"][1]["target"] == 2

    def test_handles_percent_comments(self):
        text = "% header\n0 1\n1 2\n"
        result = parse_edge_list(text)
        assert len(result["edges"]) == 2

    def test_space_separated(self):
        text = "0 1\n1 2\n2 3\n"
        result = parse_edge_list(text)
        assert len(result["edges"]) == 3


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_all_suites(self):
        suites = list_suites()
        assert len(suites) >= 3
        names = {s["key"] for s in suites}
        assert "biqmac" in names
        assert "dimacs" in names
        assert "snap" in names

    def test_filter_by_problem_class(self):
        mc_suites = list_suites(problem_class="maximum_cut")
        for s in mc_suites:
            assert "maximum_cut" in s["problem_classes"]

    def test_list_instances(self):
        instances = list_instances("biqmac")
        assert len(instances) > 0
        assert all("name" in i for i in instances)
        assert all("url" in i for i in instances)

    def test_unknown_suite_raises(self):
        with pytest.raises(ValueError, match="Unknown suite"):
            list_instances("nonexistent")

    def test_cache_path(self):
        path = _cache_path("biqmac", "g05_60.0")
        assert "biqmac" in path
        assert "g05_60.0" in path
        # Verify it points to benchmarks/
        assert "benchmarks" in path
