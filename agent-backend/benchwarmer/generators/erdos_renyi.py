"""Erdős-Rényi G(n, p) random graph generator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from benchwarmer.generators.base import BaseGenerator


class ErdosRenyiGenerator(BaseGenerator):
    """
    Generates random graphs using the Erdős-Rényi G(n, p) model.

    Each pair of nodes is connected independently with probability *p*.

    Parameters
    ----------
    p : float, default 0.3
        Edge probability.
    weighted : bool, default False
        If True, assign uniform-random weights in [0, 1] to edges.
    seed : int | None
        Random seed for reproducibility.
    """

    name = "erdos_renyi"

    def generate(self, size: int, **params: Any) -> dict:
        p = params.get("p", 0.3)
        weighted = params.get("weighted", False)
        seed = params.get("seed", None)

        G = nx.erdos_renyi_graph(size, p, seed=seed)

        if weighted:
            import random as _random
            rng = _random.Random(seed)
            for u, v in G.edges():
                G[u][v]["weight"] = round(rng.random(), 4)

        return self._nx_to_dict(G, self.name, size, {"p": p, "weighted": weighted})
