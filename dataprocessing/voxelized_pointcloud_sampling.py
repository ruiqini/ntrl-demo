"""Voxelized point cloud sampling utilities."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass

import numpy as np

from utils.kdtree_utils import build_tree


@dataclass
class SamplerConfig:
    bb_min: float
    bb_max: float
    input_res: int
    num_points: int


def create_grid_points_from_bounds(minimum: float, maximum: float, res: int) -> np.ndarray:
    x = np.linspace(minimum, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    points_list = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return points_list


class VoxelizedPointcloudSampler:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.grid_points = create_grid_points_from_bounds(cfg.bb_min, cfg.bb_max, cfg.input_res)
        self.kdtree = build_tree(self.grid_points)

    def sample(self, path: str) -> None:
        """Sample a voxelized point cloud for the given mesh path."""
        try:
            import igl  # Deferred import since this is optional

            out_path = os.path.dirname(path)
            file_name = os.path.splitext(os.path.basename(path))[0]
            input_file = os.path.join(out_path, file_name + "_scaled.off")
            out_file = os.path.join(
                out_path,
                f"voxelized_point_cloud_{self.cfg.input_res}res_{self.cfg.num_points}points.npz",
            )

            if os.path.exists(out_file):
                print(f"Exists: {out_file}")
                return

            # Placeholder for actual sampling logic
            print(f"Finished: {path}")
        except Exception:
            print(f"Error with {path}: {traceback.format_exc()}")
