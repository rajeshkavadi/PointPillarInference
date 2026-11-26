"""Deterministic CPU pillar generation compatible with OpenPCDet/spconv.

spconv's ``VoxelGenerator`` emits:

    voxels    [M, max_points_per_voxel, C]   (zero padded, first points kept)
    coords    [M, 4] int32                   (batch, z, y, x)
    num_points [M] int32

This module reproduces that layout without CUDA.  Points are processed in their
input order, so truncation to ``max_points_per_voxel`` keeps the same points
that a deterministic spconv run keeps.  When more than ``max_voxels`` pillars
exist, the first ``max_voxels`` in that order are kept (no random resampling).
"""

import numpy as np


def voxelize(points, point_cloud_range, voxel_size, grid_size,
             max_points_per_voxel=32, max_voxels=40000):
    """Convert an ``N x (3+C)`` float array into pillars.

    Returns:
        voxels: float32 [M, T, C]
        num_points: int32 [M]
        coords: int32 [M, 4] with columns (batch=0, z, y, x)
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be N x (>=3), got %r" % (points.shape,))
    if points.shape[1] == 3:
        points = np.concatenate(
            [points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
    n_features = points.shape[1]

    empty_voxels = np.zeros((0, int(max_points_per_voxel), n_features), dtype=np.float32)
    empty_counts = np.zeros((0,), dtype=np.int32)
    empty_coords = np.zeros((0, 4), dtype=np.int32)
    if points.shape[0] == 0:
        return empty_voxels, empty_counts, empty_coords

    gx, gy, gz = [int(v) for v in grid_size]
    pcr_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    vsize = np.asarray(voxel_size, dtype=np.float32)

    coords_xyz = np.floor((points[:, :3] - pcr_min[None, :]) / vsize[None, :]).astype(np.int32)
    valid = ((coords_xyz[:, 0] >= 0) & (coords_xyz[:, 0] < gx) &
             (coords_xyz[:, 1] >= 0) & (coords_xyz[:, 1] < gy) &
             (coords_xyz[:, 2] >= 0) & (coords_xyz[:, 2] < gz))
    if not valid.any():
        return empty_voxels, empty_counts, empty_coords

    coords_xyz = coords_xyz[valid]
    points = points[valid]

    # Stable sort by (z, y, x) preserves original point order inside each pillar.
    flat = ((coords_xyz[:, 2].astype(np.int64) * gy +
             coords_xyz[:, 1].astype(np.int64)) * gx +
            coords_xyz[:, 0].astype(np.int64))
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    sorted_points = points[order]

    starts = np.flatnonzero(np.r_[True, sorted_flat[1:] != sorted_flat[:-1]])
    counts = np.diff(np.r_[starts, sorted_flat.shape[0]])
    n_groups = starts.shape[0]

    if n_groups > max_voxels:
        starts = starts[:max_voxels]
        counts = counts[:max_voxels]
        n_groups = max_voxels

    group_ids = np.repeat(np.arange(n_groups, dtype=np.int64), counts)
    ranks = np.arange(sorted_points.shape[0], dtype=np.int64) - np.repeat(starts, counts)
    keep = ranks < int(max_points_per_voxel)
    if n_groups < len(np.flatnonzero(np.r_[True, sorted_flat[1:] != sorted_flat[:-1]])):
        # Only points belonging to the retained groups are valid.
        keep &= group_ids < n_groups

    rows = group_ids[keep]
    point_ranks = ranks[keep]

    voxels = np.zeros((n_groups, int(max_points_per_voxel), n_features), dtype=np.float32)
    voxels[rows, point_ranks] = sorted_points[keep]
    num_points = np.bincount(rows, minlength=n_groups).astype(np.int32)

    group_first = starts
    cell = coords_xyz[order[group_first]]  # (x, y, z) of each retained pillar
    out_coords = np.zeros((n_groups, 4), dtype=np.int32)
    out_coords[:, 0] = 0
    out_coords[:, 1] = cell[:, 2]
    out_coords[:, 2] = cell[:, 1]
    out_coords[:, 3] = cell[:, 0]
    return voxels, num_points, out_coords
