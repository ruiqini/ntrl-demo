import os
import sys

sys.path.append(os.path.abspath('.'))

from dataprocessing.voxelized_pointcloud_sampling import create_grid_points_from_bounds

def test_grid_point_generation():
    points = create_grid_points_from_bounds(-1.0, 1.0, 2)
    assert points.shape == (8, 3)

