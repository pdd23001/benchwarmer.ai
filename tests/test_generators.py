"""Tests for instance generators."""

import pytest

from benchwarmer.generators import (
    GENERATOR_REGISTRY,
    get_generator,
    ErdosRenyiGenerator,
    BarabasiAlbertGenerator,
    Grid2DGenerator,
    PlanarRandomGenerator,
    PlantedPartitionGenerator,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _assert_valid_graph(graph: dict, expected_size: int):
    """Verify a graph dict has the right structure."""
    assert "nodes" in graph
    assert "edges" in graph
    assert "metadata" in graph
    assert len(graph["nodes"]) == expected_size

    for edge in graph["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert edge["source"] in graph["nodes"]
        assert edge["target"] in graph["nodes"]
        assert edge["source"] != edge["target"]  # no self-loops


# ── Registry ─────────────────────────────────────────────────────────

def test_registry_lists_all():
    assert len(GENERATOR_REGISTRY) == 5
    assert "erdos_renyi" in GENERATOR_REGISTRY
    assert "barabasi_albert" in GENERATOR_REGISTRY
    assert "grid_2d" in GENERATOR_REGISTRY
    assert "planar_random" in GENERATOR_REGISTRY
    assert "planted_partition" in GENERATOR_REGISTRY


def test_get_generator_unknown():
    with pytest.raises(ValueError, match="Unknown generator"):
        get_generator("nope")


# ── Erdős-Rényi ──────────────────────────────────────────────────────

class TestErdosRenyi:
    def test_basic(self):
        gen = ErdosRenyiGenerator()
        g = gen.generate(50, p=0.3, seed=42)
        _assert_valid_graph(g, 50)
        assert g["metadata"]["generator"] == "erdos_renyi"

    def test_weighted(self):
        gen = ErdosRenyiGenerator()
        g = gen.generate(30, p=0.5, weighted=True, seed=42)
        _assert_valid_graph(g, 30)
        for edge in g["edges"]:
            assert "weight" in edge
            assert 0.0 <= edge["weight"] <= 1.0

    def test_deterministic(self):
        gen = ErdosRenyiGenerator()
        g1 = gen.generate(40, p=0.3, seed=123)
        g2 = gen.generate(40, p=0.3, seed=123)
        assert g1["edges"] == g2["edges"]


# ── Barabási-Albert ──────────────────────────────────────────────────

class TestBarabasiAlbert:
    def test_basic(self):
        gen = BarabasiAlbertGenerator()
        g = gen.generate(80, m=3, seed=42)
        _assert_valid_graph(g, 80)
        assert g["metadata"]["generator"] == "barabasi_albert"

    def test_small_graph(self):
        gen = BarabasiAlbertGenerator()
        g = gen.generate(5, m=10, seed=42)  # m clamped to 4
        _assert_valid_graph(g, 5)


# ── Grid 2D ──────────────────────────────────────────────────────────

class TestGrid2D:
    def test_perfect_square(self):
        gen = Grid2DGenerator()
        g = gen.generate(25, seed=42)
        _assert_valid_graph(g, 25)
        assert g["metadata"]["params"]["rows"] == 5
        assert g["metadata"]["params"]["cols"] == 5

    def test_non_square(self):
        gen = Grid2DGenerator()
        g = gen.generate(36, seed=42)
        _assert_valid_graph(g, 36)
        assert g["metadata"]["params"]["rows"] * g["metadata"]["params"]["cols"] == 36


# ── Planar Random ────────────────────────────────────────────────────

class TestPlanarRandom:
    def test_basic(self):
        gen = PlanarRandomGenerator()
        g = gen.generate(50, seed=42)
        _assert_valid_graph(g, 50)
        assert g["metadata"]["generator"] == "planar_random"

    def test_weighted(self):
        gen = PlanarRandomGenerator()
        g = gen.generate(30, weighted=True, seed=42)
        for edge in g["edges"]:
            assert "weight" in edge
            assert edge["weight"] > 0  # Euclidean distances are positive


# ── Planted Partition ────────────────────────────────────────────────

class TestPlantedPartition:
    def test_basic(self):
        gen = PlantedPartitionGenerator()
        g = gen.generate(40, num_communities=4, seed=42)
        _assert_valid_graph(g, 40)
        assert g["metadata"]["generator"] == "planted_partition"

    def test_uneven_sizes(self):
        gen = PlantedPartitionGenerator()
        g = gen.generate(43, num_communities=4, seed=42)
        _assert_valid_graph(g, 43)
