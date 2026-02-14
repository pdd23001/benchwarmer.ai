"""Abstract base class for algorithm wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AlgorithmWrapper(ABC):
    """
    Base class that every benchmarked algorithm must implement.

    Subclass this, set ``name``, and implement :meth:`solve`.

    Example
    -------
    >>> class MyGreedy(AlgorithmWrapper):
    ...     name = "my_greedy"
    ...     def solve(self, instance, timeout=60.0):
    ...         # ... your algorithm logic ...
    ...         return {"solution": {"vertices": [0, 2, 5]}, "metadata": {}}
    """

    name: str = "unnamed"

    @abstractmethod
    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        """
        Solve the given problem instance.

        Parameters
        ----------
        instance : dict
            A graph dict (``nodes``, ``edges``, ``metadata``).
        timeout : float
            Maximum wall-clock seconds allowed.

        Returns
        -------
        dict
            Must contain at least:

            - ``"solution"`` – problem-specific answer
              (e.g. ``{"vertices": [...]}`` for vertex cover)
            - ``"metadata"`` – optional extra info (iterations, etc.)
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
