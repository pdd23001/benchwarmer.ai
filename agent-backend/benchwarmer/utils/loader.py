"""
Utility to load custom AlgorithmWrapper subclasses from a file path.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from benchwarmer.algorithms.base import AlgorithmWrapper


def load_algorithm_from_file(file_path: str) -> AlgorithmWrapper:
    """
    Load an AlgorithmWrapper subclass from a Python file.
    
    The file must define exactly one class that inherits from AlgorithmWrapper.
    The class will be instantiated with no arguments.

    Parameters
    ----------
    file_path : str
        Path to the python file (e.g., "my_algo.py").

    Returns
    -------
    AlgorithmWrapper
        An instance of the loaded algorithm class.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ImportError
        If file cannot be imported.
    ValueError
        If no suitable class is found or multiple are found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {file_path}: {e}")

    # Find AlgorithmWrapper subclasses defined in this module
    candidates = []
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, AlgorithmWrapper)
            and obj is not AlgorithmWrapper
            and obj.__module__ == module_name  # Ensure it's defined here, not imported
        ):
            candidates.append(obj)

    if not candidates:
        raise ValueError(
            f"No `AlgorithmWrapper` subclass found in {file_path}. "
            "Make sure your class inherits from `benchwarmer.algorithms.base.AlgorithmWrapper`."
        )

    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Multiple algorithm classes found: {names}. "
            "Please define only one main algorithm class per file."
        )

    # Instantiate
    try:
        instance = candidates[0]()
        # Ensure it has a name
        if not hasattr(instance, "name") or instance.name == "unnamed":
             instance.name = module_name
        return instance
    except Exception as e:
        raise ValueError(f"Failed to instantiate {candidates[0].__name__}: {e}")
