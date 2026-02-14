"""Instance generators for benchmark graph problems."""

from benchwarmer.generators.base import BaseGenerator
from benchwarmer.generators.erdos_renyi import ErdosRenyiGenerator
from benchwarmer.generators.barabasi_albert import BarabasiAlbertGenerator
from benchwarmer.generators.grid_2d import Grid2DGenerator
from benchwarmer.generators.planar_random import PlanarRandomGenerator
from benchwarmer.generators.planted_partition import PlantedPartitionGenerator

# Registry: name → class
GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    "erdos_renyi": ErdosRenyiGenerator,
    "barabasi_albert": BarabasiAlbertGenerator,
    "grid_2d": Grid2DGenerator,
    "planar_random": PlanarRandomGenerator,
    "planted_partition": PlantedPartitionGenerator,
}


def get_generator(name: str) -> type[BaseGenerator]:
    """Look up a generator class by name."""
    if name not in GENERATOR_REGISTRY:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(f"Unknown generator '{name}'. Available: {available}")
    return GENERATOR_REGISTRY[name]


def list_generators() -> list[str]:
    """Return the names of all available generators."""
    return sorted(GENERATOR_REGISTRY.keys())


__all__ = [
    "BaseGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
    "list_generators",
    "ErdosRenyiGenerator",
    "BarabasiAlbertGenerator",
    "Grid2DGenerator",
    "PlanarRandomGenerator",
    "PlantedPartitionGenerator",
]
