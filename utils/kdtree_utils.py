from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Sequence


def build_tree(points: np.ndarray) -> cKDTree:
    """Build a KD-tree from the given point array."""
    return cKDTree(points)


def query_tree(tree: cKDTree, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Query KD-tree for nearest neighbours.

    Returns distances and indices for the ``k`` closest points.
    """
    return tree.query(query, k=k)
