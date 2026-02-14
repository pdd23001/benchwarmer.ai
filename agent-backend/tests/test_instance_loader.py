"""Tests for custom instance loader."""

import json
import os

import pytest

from benchwarmer.utils.instance_loader import load_instances


@pytest.fixture
def single_instance(tmp_path):
    """Create a valid single-instance JSON file."""
    data = {
        "nodes": [0, 1, 2],
        "edges": [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 1, "target": 2, "weight": 1.0},
        ],
    }
    path = tmp_path / "single.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def batch_instances(tmp_path):
    """Create a valid batch JSON file with multiple instances."""
    data = [
        {
            "nodes": [0, 1, 2],
            "edges": [{"source": 0, "target": 1}],
        },
        {
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 1}, {"source": 2, "target": 3}],
        },
    ]
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestInstanceLoader:
    def test_load_single(self, single_instance):
        instances = load_instances(single_instance)
        assert len(instances) == 1
        assert instances[0]["nodes"] == [0, 1, 2]
        assert len(instances[0]["edges"]) == 2
        assert "metadata" in instances[0]
        assert instances[0]["instance_name"] == "custom_0"

    def test_load_batch(self, batch_instances):
        instances = load_instances(batch_instances)
        assert len(instances) == 2
        assert len(instances[0]["nodes"]) == 3
        assert len(instances[1]["nodes"]) == 4

    def test_metadata_injected(self, single_instance):
        instances = load_instances(single_instance)
        meta = instances[0]["metadata"]
        assert meta["generator"] == "custom"
        assert meta["size"] == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_instances("/nonexistent/file.json")

    def test_missing_nodes(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"edges": []}))
        with pytest.raises(ValueError, match="missing required 'nodes'"):
            load_instances(str(path))

    def test_missing_edges(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"nodes": [0, 1]}))
        with pytest.raises(ValueError, match="missing required 'edges'"):
            load_instances(str(path))

    def test_invalid_type(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('"just a string"')
        with pytest.raises(ValueError, match="Expected a JSON object or array"):
            load_instances(str(path))
