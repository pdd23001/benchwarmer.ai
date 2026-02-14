"""Random planar graph generator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from benchwarmer.generators.base import BaseGenerator


class PlanarRandomGenerator(BaseGenerator):
    """
    Generates random planar graphs via Delaunay triangulation of
    random 2-D points.

    Useful for modelling road networks, floor plans, and other
    spatially-embedded structures.

    Parameters
    ----------
    weighted : bool, default False
        If True, edge weights are set to Euclidean distance between
        endpoints (rounded to 4 decimals).
    seed : int | None
        Random seed for reproducibility.
    """

    name = "planar_random"

    def generate(self, size: int, **params: Any) -> dict:
        weighted = params.get("weighted", False)
        seed = params.get("seed", None)

        import numpy as np

        rng = np.random.default_rng(seed)
        points = rng.random((size, 2))

        # Build planar graph via Delaunay triangulation
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points)
            G = nx.Graph()
            G.add_nodes_from(range(size))
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u, v = int(simplex[i]), int(simplex[j])
                        if not G.has_edge(u, v):
                            if weighted:
                                dist = float(np.linalg.norm(points[u] - points[v]))
                                G.add_edge(u, v, weight=round(dist, 4))
                            else:
                                G.add_edge(u, v)
        except ImportError:
            # Fallback: random geometric graph (approximately planar for
            # suitable radius choices)
            radius = 2.0 / (size ** 0.5)
            G = nx.random_geometric_graph(size, radius, seed=seed)
            # Relabel to simple ints
            G = nx.convert_node_labels_to_integers(G)

        return self._nx_to_dict(
            G, self.name, size, {"weighted": weighted},
        )
