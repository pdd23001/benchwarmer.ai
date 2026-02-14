"""Barabási-Albert preferential attachment graph generator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from benchwarmer.generators.base import BaseGenerator


class BarabasiAlbertGenerator(BaseGenerator):
    """
    Generates scale-free graphs using the Barabási-Albert model.

    New nodes attach preferentially to high-degree existing nodes, producing
    power-law degree distributions common in social and internet networks.

    Parameters
    ----------
    m : int, default 3
        Number of edges to attach from a new node to existing nodes.
    weighted : bool, default False
        If True, assign uniform-random weights in [0, 1] to edges.
    seed : int | None
        Random seed for reproducibility.
    """

    name = "barabasi_albert"

    def generate(self, size: int, **params: Any) -> dict:
        m = params.get("m", 3)
        weighted = params.get("weighted", False)
        seed = params.get("seed", None)

        # m must be < size
        m = min(m, max(1, size - 1))
        G = nx.barabasi_albert_graph(size, m, seed=seed)

        if weighted:
            import random as _random
            rng = _random.Random(seed)
            for u, v in G.edges():
                G[u][v]["weight"] = round(rng.random(), 4)

        return self._nx_to_dict(G, self.name, size, {"m": m, "weighted": weighted})
