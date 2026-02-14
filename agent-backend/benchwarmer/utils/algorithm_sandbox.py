"""
Algorithm Sandbox — executes LLM-generated algorithm code and extracts
the AlgorithmWrapper subclass.

Provides a restricted execution environment that allows common algorithm
imports (random, collections, itertools, heapq, math) while blocking
dangerous operations. After executing the code, it finds the
AlgorithmWrapper subclass and runs a smoke test on a tiny graph instance.
"""

from __future__ import annotations

import io
import traceback
from typing import Any, Optional

from benchwarmer.algorithms.base import AlgorithmWrapper


def execute_algorithm_code(
    code: str,
    problem_class: str,
    smoke_instance: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Execute generated algorithm code and extract the AlgorithmWrapper subclass.

    Parameters
    ----------
    code : str
        Python code defining a class that extends AlgorithmWrapper.
    problem_class : str
        The problem class name (for context in error messages).
    smoke_instance : dict | None
        A small graph instance for smoke testing. If None, a default
        5-node triangle graph is used.

    Returns
    -------
    dict
        On success: ``{"success": True, "algorithm": <instance>, "name": "...", "smoke_result": {...}}``
        On failure: ``{"success": False, "error": "...", "traceback": "..."}``
    """
    import collections
    import heapq
    import itertools
    import math
    import random

    if smoke_instance is None:
        smoke_instance = _default_smoke_instance()

    # Safe builtins — more permissive than plot sandbox since algorithms
    # need more standard library functionality.
    # __build_class__ is required by Python internally for class definitions
    # inside exec().
    import builtins as _builtins
    from abc import abstractmethod, ABC

    safe_builtins = {
        # Python internals needed for class definitions
        "__build_class__": _builtins.__build_class__,
        "__name__": "__algorithm_sandbox__",
        # Types
        "True": True,
        "False": False,
        "None": None,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        "bytearray": bytearray,
        "object": object,
        "type": type,
        # ABC support
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        # Functions
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "min": min,
        "max": max,
        "round": round,
        "abs": abs,
        "sum": sum,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "callable": callable,
        "iter": iter,
        "next": next,
        "hash": hash,
        "id": id,
        "repr": repr,
        "super": super,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "Exception": Exception,
    }

    # Build execution namespace with common algorithm imports
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        # Base class for subclassing
        "AlgorithmWrapper": AlgorithmWrapper,
        # Common algorithm imports
        "random": random,
        "math": math,
        "collections": collections,
        "itertools": itertools,
        "heapq": heapq,
        # Commonly used directly
        "defaultdict": collections.defaultdict,
        "deque": collections.deque,
        "Counter": collections.Counter,
        "inf": float("inf"),
    }

    # Step 1: Execute the code
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return {
            "success": False,
            "error": f"Code execution failed: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step 2: Find the AlgorithmWrapper subclass
    algo_class = None
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if (
            isinstance(obj, type)
            and issubclass(obj, AlgorithmWrapper)
            and obj is not AlgorithmWrapper
        ):
            algo_class = obj
            break  # Take the first one found

    if algo_class is None:
        return {
            "success": False,
            "error": (
                "No AlgorithmWrapper subclass found in the generated code. "
                "The code must define a class that extends AlgorithmWrapper "
                "with a `name` attribute and a `solve()` method."
            ),
            "traceback": "",
        }

    # Step 3: Instantiate
    try:
        algo_instance = algo_class()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to instantiate {algo_class.__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step 4: Verify it has a name
    if not hasattr(algo_instance, "name") or algo_instance.name == "unnamed":
        return {
            "success": False,
            "error": (
                f"Class {algo_class.__name__} must set a `name` attribute "
                f"(got '{getattr(algo_instance, 'name', 'missing')}')."
            ),
            "traceback": "",
        }

    # Step 5: Smoke test — run on tiny instance
    try:
        result = algo_instance.solve(smoke_instance, timeout=10.0)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Smoke test failed — solve() raised: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step 6: Validate smoke test result format
    if not isinstance(result, dict):
        return {
            "success": False,
            "error": f"solve() must return a dict, got {type(result).__name__}",
            "traceback": "",
        }
    if "solution" not in result:
        return {
            "success": False,
            "error": (
                "solve() must return a dict with a 'solution' key. "
                f"Got keys: {list(result.keys())}"
            ),
            "traceback": "",
        }

    # Attach source code for Modal serialization (inspect.getsource fails
    # on classes created via exec, so we preserve the original string).
    algo_instance._source_code = code

    return {
        "success": True,
        "algorithm": algo_instance,
        "name": algo_instance.name,
        "smoke_result": result,
        "code": code,  # preserve for Modal serialization
    }


def _default_smoke_instance() -> dict:
    """Return a small 5-node graph for smoke testing."""
    return {
        "nodes": [0, 1, 2, 3, 4],
        "edges": [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 1, "target": 2, "weight": 1.0},
            {"source": 2, "target": 3, "weight": 1.0},
            {"source": 3, "target": 4, "weight": 1.0},
            {"source": 4, "target": 0, "weight": 1.0},
            {"source": 0, "target": 2, "weight": 1.0},
        ],
        "metadata": {
            "generator": "smoke_test",
            "size": 5,
            "params": {},
        },
    }
