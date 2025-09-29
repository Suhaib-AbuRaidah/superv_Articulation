import numpy as np
from scipy.ndimage import distance_transform_edt


def points_to_occ_grid(points_occ_start,points_occ_target, occ_list_start, occ_list_target, grid_res=64):
    """
    Turn sampled points + occupancy labels into a voxel grid.
    """
    # # Normalize coordinates to [0, 1] cube

    occ_start = occ_list_start[1]
    occ_target = occ_list_target[1]    
    # Scale to grid
    idxs_start = (points_occ_start * (grid_res - 1)).astype(int)
    idxs_target = (points_occ_target * (grid_res - 1)).astype(int)
    # Build occupancy grid
    occ_grid_start = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    for i, occ in zip(idxs_start, occ_start):
        x, y, z = i
        occ_grid_start[x, y, z] = max(occ_grid_start[x, y, z], occ)  # mark occupied if any point inside

    occ_grid_target = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    for i, occ in zip(idxs_target, occ_target):
        x, y, z = i
        occ_grid_target[x, y, z] = max(occ_grid_target[x, y, z], occ)  # mark occupied if any point inside

    return occ_grid_start,occ_grid_target


def occ_to_tsdf(occ_grid, voxel_size=0.1, trunc_dist=500):
    """
    Convert occupancy grid to TSDF.
    """
    # Distance to nearest occupied voxel (outside surface)
    outside_dist = distance_transform_edt(occ_grid == 0) * voxel_size
    # Distance to nearest free voxel (inside surface)
    inside_dist = distance_transform_edt(occ_grid == 1) * voxel_size

    tsdf = outside_dist - inside_dist  # positive outside, negative inside
    
    # Truncate
    tsdf = np.clip(tsdf, -trunc_dist * voxel_size, trunc_dist * voxel_size)

    return tsdf

def voxel_index_to_world(i, j, k, mins, maxs, grid_res=64):
    # step size along each axis
    step = (maxs - mins) / (grid_res - 1)
    return mins + step * np.array([i, j, k])

def tsdf_of_point(points, tsdf, grid_res=64):
    lst = []
    points = points*(grid_res-1)
    points = np.clip(points, 0, grid_res-1)
    for u in range(points.shape[0]):
        i, j, k = points[u].astype(int)
        lst.append(tsdf[i, j, k])
    return np.array(lst)


