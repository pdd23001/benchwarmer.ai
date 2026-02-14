"""
Benchmark suite integration — fetch and parse graph instances from 
established benchmarking libraries.

Supported suites:
- **Biq Mac**: Max-Cut instances (rudy, ising, beasley)
- **DIMACS**: Vertex cover, clique, coloring instances
- **SNAP**: Real-world network edge lists

All instances are downloaded via HTTP, cached locally in 
``<project_root>/benchmarks/<suite>/``, and parsed into the standard 
``{nodes, edges, metadata}`` format used by the benchmark engine.
"""

from __future__ import annotations

import gzip
import logging
import os
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache relative to project root: Benchwarmer.AI/benchmarks/
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # .../benchwarmer
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT) # .../Benchwarmer.AI
CACHE_DIR = os.path.join(PROJECT_ROOT, "benchmarks")


# ──────────────────────────────────────────────────────────────
# Parsers — convert raw text to {nodes, edges, metadata}
# ──────────────────────────────────────────────────────────────

def parse_dimacs(text: str) -> dict[str, Any]:
    """
    Parse the DIMACS graph format.

    Format::

        c comment line
        p edge <n_nodes> <n_edges>
        e <u> <v>            (1-indexed)
        e <u> <v> <weight>   (optional weight)

    Returns our standard instance dict.
    """
    nodes_set: set[int] = set()
    edges: list[dict[str, Any]] = []
    n_nodes = 0

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue

        parts = line.split()

        if parts[0] == "p":
            # p edge <n> <m>  or  p col <n> <m>
            n_nodes = int(parts[2])
            continue

        if parts[0] == "e":
            # 1-indexed → 0-indexed
            try:
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
            except ValueError:
                continue # Skip malformed lines
            weight = float(parts[3]) if len(parts) > 3 else 1.0
            nodes_set.update([u, v])
            edges.append({"source": u, "target": v, "weight": weight})

    # Build full node list (some nodes may have no edges)
    all_nodes = list(range(max(n_nodes, max(nodes_set) + 1 if nodes_set else 0)))

    return {
        "nodes": all_nodes,
        "edges": edges,
        "metadata": {
            "generator": "dimacs",
            "size": len(all_nodes),
            "params": {"format": "dimacs"},
        },
    }


def parse_biqmac(text: str) -> dict[str, Any]:
    """
    Parse the Biq Mac Library sparse format.

    Format::

        <n_nodes> <n_edges>
        <u> <v> <weight>     (1-indexed)
        ...

    Returns our standard instance dict.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty Biq Mac file")

    header = lines[0].split()
    n_nodes = int(header[0])
    # n_edges = int(header[1])  # not needed

    edges: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        # 1-indexed → 0-indexed
        try:
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            weight = float(parts[2])
        except ValueError:
            continue
        edges.append({"source": u, "target": v, "weight": weight})

    return {
        "nodes": list(range(n_nodes)),
        "edges": edges,
        "metadata": {
            "generator": "biqmac",
            "size": n_nodes,
            "params": {"format": "biqmac_sparse"},
        },
    }


def parse_edge_list(text: str) -> dict[str, Any]:
    """
    Parse a simple edge list (tab or space separated).

    Handles SNAP-style files::

        # Comment lines
        <u>\\t<v>

    Nodes are auto-detected and re-indexed to 0..n-1.
    """
    nodes_set: set[int] = set()
    raw_edges: list[tuple[int, int]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if u == v:
            continue  # skip self-loops
        nodes_set.update([u, v])
        raw_edges.append((u, v))

    # Re-index nodes to 0..n-1
    sorted_nodes = sorted(nodes_set)
    node_map = {old: new for new, old in enumerate(sorted_nodes)}

    edges = [
        {"source": node_map[u], "target": node_map[v], "weight": 1.0}
        for u, v in raw_edges
    ]

    return {
        "nodes": list(range(len(sorted_nodes))),
        "edges": edges,
        "metadata": {
            "generator": "edge_list",
            "size": len(sorted_nodes),
            "params": {"original_node_count": len(sorted_nodes)},
        },
    }


# ──────────────────────────────────────────────────────────────
# Suite Registry
# ──────────────────────────────────────────────────────────────

SUITE_REGISTRY: dict[str, dict[str, Any]] = {
    "biqmac": {
        "name": "Biq Mac Library",
        "description": "Max-Cut benchmark instances (20-500 nodes)",
        "problem_classes": ["maximum_cut"],
        "parser": parse_biqmac,
        "instances": [
            # Rudy random graphs — popular for max-cut benchmarking
            {"name": "g05_60.0", "url": "https://biqmac.aau.at/library/mac/rudy/g05_60.0", "nodes": 60, "description": "Rudy random 60-node, density 0.5"},
            {"name": "g05_80.0", "url": "https://biqmac.aau.at/library/mac/rudy/g05_80.0", "nodes": 80, "description": "Rudy random 80-node, density 0.5"},
            {"name": "g05_100.0", "url": "https://biqmac.aau.at/library/mac/rudy/g05_100.0", "nodes": 100, "description": "Rudy random 100-node, density 0.5"},
            {"name": "pm1s_80.0", "url": "https://biqmac.aau.at/library/mac/rudy/pm1s_80.0", "nodes": 80, "description": "Rudy ±1 weights, 80-node"},
            {"name": "pm1s_100.0", "url": "https://biqmac.aau.at/library/mac/rudy/pm1s_100.0", "nodes": 100, "description": "Rudy ±1 weights, 100-node"},
            {"name": "w01_100.0", "url": "https://biqmac.aau.at/library/mac/rudy/w01_100.0", "nodes": 100, "description": "Rudy 0/1 weights, 100-node"},
            # Ising spin glass instances
            {"name": "t2g10_5555", "url": "https://biqmac.aau.at/library/mac/ising/t2g10_5555", "nodes": 100, "description": "Ising 2D grid 10×10"},
            {"name": "t2g15_5555", "url": "https://biqmac.aau.at/library/mac/ising/t2g15_5555", "nodes": 225, "description": "Ising 2D grid 15×15"},
            {"name": "t2g20_5555", "url": "https://biqmac.aau.at/library/mac/ising/t2g20_5555", "nodes": 400, "description": "Ising 2D grid 20×20"},
        ],
    },
    "dimacs": {
        "name": "DIMACS Benchmarks",
        "description": "Classic graph optimization instances (coloring, clique, cover)",
        "problem_classes": ["minimum_vertex_cover", "maximum_cut"],
        "parser": parse_dimacs,
        "instances": [
            {"name": "myciel3", "url": "https://mat.tepper.cmu.edu/COLOR/instances/myciel3.col", "nodes": 11, "description": "Mycielski graph, χ=4"},
            {"name": "myciel4", "url": "https://mat.tepper.cmu.edu/COLOR/instances/myciel4.col", "nodes": 23, "description": "Mycielski graph, χ=5"},
            {"name": "myciel5", "url": "https://mat.tepper.cmu.edu/COLOR/instances/myciel5.col", "nodes": 47, "description": "Mycielski graph, χ=6"},
            {"name": "myciel6", "url": "https://mat.tepper.cmu.edu/COLOR/instances/myciel6.col", "nodes": 95, "description": "Mycielski graph, χ=7"},
            {"name": "queen5_5", "url": "https://mat.tepper.cmu.edu/COLOR/instances/queen5_5.col", "nodes": 25, "description": "Queen graph 5×5"},
            {"name": "queen6_6", "url": "https://mat.tepper.cmu.edu/COLOR/instances/queen6_6.col", "nodes": 36, "description": "Queen graph 6×6"},
            {"name": "queen7_7", "url": "https://mat.tepper.cmu.edu/COLOR/instances/queen7_7.col", "nodes": 49, "description": "Queen graph 7×7"},
            {"name": "queen8_8", "url": "https://mat.tepper.cmu.edu/COLOR/instances/queen8_8.col", "nodes": 64, "description": "Queen graph 8×8"},
            {"name": "jean", "url": "https://mat.tepper.cmu.edu/COLOR/instances/jean.col", "nodes": 80, "description": "Jean graph"},
            {"name": "anna", "url": "https://mat.tepper.cmu.edu/COLOR/instances/anna.col", "nodes": 138, "description": "Anna graph"},
            {"name": "huck", "url": "https://mat.tepper.cmu.edu/COLOR/instances/huck.col", "nodes": 74, "description": "Huck graph"},
            {"name": "david", "url": "https://mat.tepper.cmu.edu/COLOR/instances/david.col", "nodes": 87, "description": "David graph"},
        ],
    },
    "snap": {
        "name": "SNAP (Stanford Networks)",
        "description": "Real-world network datasets",
        "problem_classes": ["maximum_cut", "minimum_vertex_cover"],
        "parser": parse_edge_list,
        "compressed": True,
        "instances": [
            {"name": "ca-GrQc", "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz", "nodes": 5242, "description": "Arxiv GR-QC collaboration"},
            {"name": "ca-HepTh", "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz", "nodes": 9877, "description": "Arxiv HEP-TH collaboration"},
            {"name": "ca-HepPh", "url": "https://snap.stanford.edu/data/ca-HepPh.txt.gz", "nodes": 12008, "description": "Arxiv HEP-PH collaboration"},
            {"name": "ca-AstroPh", "url": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz", "nodes": 18772, "description": "Arxiv Astro-PH collaboration"},
            {"name": "facebook_combined", "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz", "nodes": 4039, "description": "Facebook ego networks"},
            {"name": "email-Eu-core", "url": "https://snap.stanford.edu/data/email-Eu-core.txt.gz", "nodes": 1005, "description": "EU research institution email"},
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def list_suites(
    problem_class: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return metadata for all suites, optionally filtered by problem class."""
    results = []
    for key, suite in SUITE_REGISTRY.items():
        if problem_class and problem_class not in suite["problem_classes"]:
            continue
        results.append({
            "key": key,
            "name": suite["name"],
            "description": suite["description"],
            "problem_classes": suite["problem_classes"],
            "instance_count": len(suite["instances"]),
        })
    return results


def list_instances(suite_key: str) -> list[dict[str, Any]]:
    """Return the available instances for a suite."""
    if suite_key not in SUITE_REGISTRY:
        raise ValueError(
            f"Unknown suite '{suite_key}'. "
            f"Available: {', '.join(SUITE_REGISTRY.keys())}"
        )
    return SUITE_REGISTRY[suite_key]["instances"]


def fetch_instance(
    suite_key: str,
    instance_name: str,
    force_download: bool = False,
) -> dict[str, Any]:
    """
    Download (or load from cache) and parse a benchmark instance.

    Parameters
    ----------
    suite_key : str
        Suite identifier (e.g. "biqmac", "dimacs", "snap").
    instance_name : str
        Instance name within the suite.
    force_download : bool
        If True, bypass the cache and re-download.

    Returns
    -------
    dict
        Parsed instance in ``{nodes, edges, metadata}`` format.
    """
    if suite_key not in SUITE_REGISTRY:
        raise ValueError(f"Unknown suite '{suite_key}'")

    suite = SUITE_REGISTRY[suite_key]

    # Find instance
    inst_meta = None
    for inst in suite["instances"]:
        if inst["name"] == instance_name:
            inst_meta = inst
            break

    if inst_meta is None:
        available = [i["name"] for i in suite["instances"]]
        raise ValueError(
            f"Unknown instance '{instance_name}' in suite '{suite_key}'. "
            f"Available: {available}"
        )

    # Check cache
    cache_path = _cache_path(suite_key, instance_name)

    if os.path.exists(cache_path) and not force_download:
        logger.info("Loading cached: %s/%s", suite_key, instance_name)
        with open(cache_path) as f:
            raw = f.read()
    else:
        logger.info("Downloading: %s", inst_meta["url"])
        try:
            raw = _download(inst_meta["url"], compressed=suite.get("compressed", False))
        except Exception as e:
            logger.error("Failed to download %s: %s", inst_meta["url"], e)
            raise e
        # Cache it
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(raw)
        logger.info("Cached to: %s", cache_path)

    # Parse
    parser = suite["parser"]
    try:
        instance = parser(raw)
    except Exception as e:
        logger.error("Failed to parse %s: %s", instance_name, e)
        raise e

    # Add suite metadata
    instance["metadata"]["suite"] = suite_key
    instance["metadata"]["instance_name"] = instance_name
    instance["metadata"]["description"] = inst_meta.get("description", "")
    instance["instance_name"] = f"{suite_key}_{instance_name}"

    return instance


def fetch_multiple(
    suite_key: str,
    instance_names: list[str],
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Fetch multiple instances from the same suite."""
    return [
        fetch_instance(suite_key, name, force_download)
        for name in instance_names
    ]


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _cache_path(suite_key: str, instance_name: str) -> str:
    """Return the local cache file path for an instance: PROJECT_ROOT/benchmarks/<suite>/<name>.txt"""
    return os.path.join(CACHE_DIR, suite_key, f"{instance_name}.txt")


def _download(url: str, compressed: bool = False, timeout: int = 30) -> str:
    """Download a URL and return the text content."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BenchwarmerAI/1.0"},
    )

    response = urllib.request.urlopen(req, timeout=timeout)
    data = response.read()

    if compressed or url.endswith(".gz"):
        data = gzip.decompress(data)

    return data.decode("utf-8", errors="replace")
