"""Abstract base class for all instance generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    """
    Base class for graph instance generators.

    Every generator produces a standardised graph dict::

        {
            "nodes": [0, 1, 2, ...],
            "edges": [
                {"source": 0, "target": 1, "weight": 1.0},
                ...
            ],
            "metadata": {
                "generator": "erdos_renyi",
                "size": 100,
                "params": {"p": 0.3},
            }
        }
    """

    name: str = "base"

    @abstractmethod
    def generate(self, size: int, **params: Any) -> dict:
        """
        Generate a graph instance.

        Parameters
        ----------
        size : int
            Number of nodes in the generated graph.
        **params
            Generator-specific parameters.

        Returns
        -------
        dict
            A graph dict with keys ``nodes``, ``edges``, ``metadata``.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all generators
    # ------------------------------------------------------------------

    @staticmethod
    def _nx_to_dict(
        G,  # noqa: N803  (networkx convention)
        generator_name: str,
        size: int,
        params: dict[str, Any],
    ) -> dict:
        """Convert a ``networkx.Graph`` to the standard dict format."""
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
            })
        return {
            "nodes": list(G.nodes()),
            "edges": edges,
            "metadata": {
                "generator": generator_name,
                "size": size,
                "params": params,
            },
        }
