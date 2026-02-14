"""2-D grid / lattice graph generator."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from benchwarmer.generators.base import BaseGenerator


class Grid2DGenerator(BaseGenerator):
    """
    Generates 2-D grid (lattice) graphs.

    Useful for modelling building floor-plans, road grids, and similar
    regular spatial layouts.

    The ``size`` parameter is interpreted as the *total* number of nodes.
    The grid dimensions are chosen as close to square as possible:
    ``rows × cols`` where ``rows * cols == size``.

    Parameters
    ----------
    weighted : bool, default False
        If True, assign uniform-random weights in [0, 1] to edges.
    seed : int | None
        Random seed for reproducibility (only affects weights).
    """

    name = "grid_2d"

    def generate(self, size: int, **params: Any) -> dict:
        weighted = params.get("weighted", False)
        seed = params.get("seed", None)

        # Find the most-square factorisation of *size*
        rows = int(math.isqrt(size))
        while rows > 0 and size % rows != 0:
            rows -= 1
        if rows == 0:
            rows = 1
        cols = size // rows

        G = nx.grid_2d_graph(rows, cols)

        # Relabel nodes from (i, j) tuples to plain integers
        mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)

        if weighted:
            import random as _random
            rng = _random.Random(seed)
            for u, v in G.edges():
                G[u][v]["weight"] = round(rng.random(), 4)

        return self._nx_to_dict(
            G, self.name, size, {"rows": rows, "cols": cols, "weighted": weighted},
        )
