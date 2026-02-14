"""Planted partition (stochastic block model) graph generator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from benchwarmer.generators.base import BaseGenerator


class PlantedPartitionGenerator(BaseGenerator):
    """
    Generates graphs with planted community structure using the
    stochastic block model.

    Useful for testing algorithms on graphs with known cluster structure
    (social networks with communities, etc.).

    Parameters
    ----------
    num_communities : int, default 4
        Number of communities.
    p_in : float, default 0.7
        Edge probability within a community.
    p_out : float, default 0.05
        Edge probability between communities.
    weighted : bool, default False
        If True, assign uniform-random weights in [0, 1] to edges.
    seed : int | None
        Random seed for reproducibility.
    """

    name = "planted_partition"

    def generate(self, size: int, **params: Any) -> dict:
        num_communities = params.get("num_communities", 4)
        p_in = params.get("p_in", 0.7)
        p_out = params.get("p_out", 0.05)
        weighted = params.get("weighted", False)
        seed = params.get("seed", None)

        # Distribute nodes as evenly as possible across communities
        base, remainder = divmod(size, num_communities)
        community_sizes = [
            base + (1 if i < remainder else 0)
            for i in range(num_communities)
        ]

        G = nx.planted_partition_graph(
            l=num_communities,
            k=base,     # nx uses uniform size; we rebuild below if uneven
            p_in=p_in,
            p_out=p_out,
            seed=seed,
        )

        # planted_partition_graph produces l*k nodes. If size != l*k we
        # trim or revert to stochastic_block_model for exact sizing.
        if G.number_of_nodes() != size:
            # Use the more flexible stochastic_block_model
            p_matrix = [
                [p_in if i == j else p_out for j in range(num_communities)]
                for i in range(num_communities)
            ]
            G = nx.stochastic_block_model(
                community_sizes, p_matrix, seed=seed,
            )
            # Remove the 'block' node attribute to keep output clean
            for node in G.nodes():
                G.nodes[node].pop("block", None)

        # Relabel to contiguous integers
        G = nx.convert_node_labels_to_integers(G)

        if weighted:
            import random as _random
            rng = _random.Random(seed)
            for u, v in G.edges():
                G[u][v]["weight"] = round(rng.random(), 4)

        return self._nx_to_dict(
            G,
            self.name,
            size,
            {
                "num_communities": num_communities,
                "p_in": p_in,
                "p_out": p_out,
                "weighted": weighted,
            },
        )
