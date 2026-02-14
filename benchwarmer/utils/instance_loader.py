"""
Utility to load custom graph instances from JSON files.

Supports single-instance and batch (list of instances) JSON files.
Each instance must have at minimum ``nodes`` and ``edges`` keys.
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_instances(path: str) -> list[dict[str, Any]]:
    """
    Load graph instances from a JSON file.

    The file may contain either:
    - A single instance dict (with ``nodes`` and ``edges``)
    - A list of instance dicts

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    list[dict]
        List of validated graph instance dicts.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the JSON structure is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Normalize to list
    if isinstance(data, dict):
        instances = [data]
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError(
            f"Expected a JSON object or array, got {type(data).__name__}"
        )

    # Validate each instance
    validated = []
    for i, inst in enumerate(instances):
        if not isinstance(inst, dict):
            raise ValueError(f"Instance {i} is not a dict: {type(inst).__name__}")

        if "nodes" not in inst:
            raise ValueError(
                f"Instance {i} missing required 'nodes' key. "
                f"Expected format: {{\"nodes\": [...], \"edges\": [...]}}"
            )
        if "edges" not in inst:
            raise ValueError(
                f"Instance {i} missing required 'edges' key. "
                f"Expected format: {{\"nodes\": [...], \"edges\": [...]}}"
            )

        # Ensure metadata exists
        if "metadata" not in inst:
            inst["metadata"] = {
                "generator": "custom",
                "size": len(inst["nodes"]),
                "params": {},
            }

        # Assign an instance name if missing
        if "instance_name" not in inst:
            inst["instance_name"] = f"custom_{i}"

        validated.append(inst)

    return validated
