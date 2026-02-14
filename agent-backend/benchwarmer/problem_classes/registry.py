"""Auto-discovery registry for problem class modules."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

# Cache of discovered classes
_REGISTRY: dict[str, Any] | None = None


def _discover() -> dict[str, Any]:
    """Import every sibling module and collect ``ProblemClass`` objects."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    _REGISTRY = {}
    package_dir = Path(__file__).resolve().parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_") or module_info.name == "registry":
            continue
        mod = importlib.import_module(f"benchwarmer.problem_classes.{module_info.name}")
        cls = getattr(mod, "ProblemClass", None)
        if cls is not None:
            _REGISTRY[cls.name] = cls

    return _REGISTRY


def list_problem_classes() -> list[dict[str, Any]]:
    """Return a list of dicts summarising every registered problem class."""
    registry = _discover()
    return [
        {
            "name": cls.name,
            "objective": cls.objective,
            "description": cls.description,
            "keywords": cls.keywords,
            "generators": cls.available_generators(),
        }
        for cls in registry.values()
    ]


def get_problem_class(name: str) -> Any:
    """Look up a ``ProblemClass`` by name."""
    registry = _discover()
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown problem class '{name}'. Available: {available}")
    return registry[name]
