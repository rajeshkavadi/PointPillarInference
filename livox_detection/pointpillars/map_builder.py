"""Persistent world-frame map for the web viewer.

The detector only ever sees the *current* LiDAR frame, but the viewer wants a
map that keeps growing while the sensor moves (like a navigation map instead of
a radar sweep).  This module keeps the accumulated cloud in a world frame and
only emits the points that are new to the map.

There is no odometry topic in the demo bags, so the motion between consecutive
frames is estimated from the scans themselves:

  1. project each frame onto a coarse top-down (BEV) occupancy grid,
  2. cross-correlate it with the previous frame's grid (FFT, integer cell
     resolution), which gives the apparent translation of the scene,
  3. move the sensor pose by the opposite of that translation (a static world
     point moves by ``-d`` in the sensor frame when the sensor moves by ``+d``).

The estimate is only translation (no yaw) and is deliberately conservative: a
low correlation score, or a step larger than ``max_step``, keeps the previous
pose.  The map keeps one point per ``voxel_size`` cube, so revisiting an area
does not duplicate it and the buffer stays bounded.
"""

import numpy as np


class TrajectoryMap(object):
    """Accumulates frames into a bounded, de-duplicated world-frame map."""

    def __init__(self, voxel_size=0.4, max_points=1500000, max_range=70.0,
                 bev_cell=0.5, bev_half_extent=32.0, max_step=3.0,
                 min_score=0.05):
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.max_range = float(max_range)
        self.bev_cell = float(bev_cell)
        self.bev_half_extent = float(bev_half_extent)
        self.max_step = float(max_step)
        self.min_score = float(min_score)

        self.pose = (0.0, 0.0)
        self.epoch = 0          # bumped on reset; the viewer drops its old map
        self._prev_bev = None
        self._velocity = (0.0, 0.0)

        self._voxels = set()
        self._capacity = 1 << 16
        self._points = np.zeros((self._capacity, 4), dtype=np.float32)
        self._count = 0
        self.full = False
        self._mask = (1 << 21) - 1
        self._zmask = (1 << 10) - 1

    # ------------------------------------------------------------------
    # map contents
    # ------------------------------------------------------------------
    @property
    def total(self):
        return self._count

    def snapshot(self):
        return {
            "map_points": int(self._count),
            "map_full": bool(self.full),
            "pose": {"x": round(self.pose[0], 3), "y": round(self.pose[1], 3)},
            "velocity": [round(self._velocity[0], 3), round(self._velocity[1], 3)],
        }

    def reset(self):
        self.pose = (0.0, 0.0)
        self.epoch += 1
        self._velocity = (0.0, 0.0)
        self._prev_bev = None
        self._voxels = set()
        self._count = 0
        self.full = False

    # ------------------------------------------------------------------
    # per-frame update
    # ------------------------------------------------------------------
    def update(self, points):
        """Add one sensor-frame cloud and return ``(pose, new_world_points)``.

        ``pose`` is the estimated sensor position ``(x, y)`` in the map frame
        after this frame.  ``new_world_points`` is an ``N x 4`` float32 view of
        the points that were not already covered by the map.
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return self.pose, self._points[0:0]
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), np.float32)], axis=1)
        pts = pts[:, :4]
        finite = np.isfinite(pts[:, :3]).all(axis=1)
        if not finite.all():
            pts = pts[finite]

        bev = self._bev(pts)
        if bev is not None and self._prev_bev is not None:
            dx, dy, score = self._estimate_motion(self._prev_bev, bev)
            if score >= self.min_score:
                self._velocity = (dx, dy)
                self.pose = (self.pose[0] + dx, self.pose[1] + dy)
            # A weak match means "no confirmed motion": keep the pose.
        self._prev_bev = bev

        if self.full or pts.shape[0] == 0:
            return self.pose, self._points[0:0]

        world = pts.copy()
        world[:, 0] += self.pose[0]
        world[:, 1] += self.pose[1]

        codes = self._voxel_codes(world)
        codes, first = np.unique(codes, return_index=True)
        seen = self._voxels
        fresh = [(code, row) for code, row in zip(codes.tolist(), first.tolist())
                 if code not in seen]
        if not fresh:
            return self.pose, self._points[0:0]

        rows = np.fromiter((row for _code, row in fresh), dtype=np.int64,
                           count=len(fresh))
        new = world[rows]
        if self._count + new.shape[0] > self.max_points:
            new = new[:self.max_points - self._count]
            self.full = True
        self._append(new)
        self._voxels.update(code for code, _row in fresh)
        return self.pose, self._points[self._count - new.shape[0]:self._count]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _append(self, new):
        need = self._count + new.shape[0]
        if need > self._capacity:
            capacity = self._capacity
            while capacity < need:
                capacity = min(capacity * 2, self.max_points)
            grown = np.zeros((max(capacity, need), 4), dtype=np.float32)
            grown[:self._count] = self._points[:self._count]
            self._points = grown
            self._capacity = grown.shape[0]
        self._points[self._count:need] = new
        self._count = need

    def _voxel_codes(self, world):
        inv = 1.0 / self.voxel_size
        ix = np.floor(world[:, 0] * inv).astype(np.int64) & self._mask
        iy = np.floor(world[:, 1] * inv).astype(np.int64) & self._mask
        iz = np.floor(world[:, 2] * inv).astype(np.int64) & self._zmask
        return ix | (iy << 21) | (iz << 42)

    def _bev(self, pts):
        half = self.bev_half_extent
        cell = self.bev_cell
        size = int(round(2.0 * half / cell))
        r2 = self.max_range * self.max_range
        xy = pts[:, :2]
        inside = (xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1]) <= r2
        if not inside.any():
            return None
        xy = xy[inside]
        idx = np.floor((xy + half) / cell).astype(np.int64)
        keep = ((idx[:, 0] >= 0) & (idx[:, 0] < size) &
                (idx[:, 1] >= 0) & (idx[:, 1] < size))
        idx = idx[keep]
        if idx.shape[0] == 0:
            return None
        flat = idx[:, 0] * size + idx[:, 1]
        grid = np.bincount(flat, minlength=size * size).astype(np.float32)
        grid = grid.reshape(size, size)
        np.clip(grid, 0.0, 4.0, out=grid)
        return grid

    def _estimate_motion(self, prev, cur):
        """Return ``(dx, dy, score)`` in metres for this frame's pose change."""
        if prev.shape != cur.shape:
            return 0.0, 0.0, 0.0
        a = prev - prev.mean()
        b = cur - cur.mean()
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 1e-9:
            return 0.0, 0.0, 0.0
        corr = np.fft.irfft2(np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b),
                             s=a.shape)
        peak = int(np.argmax(corr))
        score = float(corr.flat[peak] / denom)
        rows, cols = a.shape            # rows index x, cols index y (see _bev)
        ix, iy = divmod(peak, cols)
        # The correlation peak is the apparent scene shift (cur = prev + s),
        # so the sensor moved the opposite way.
        step = self.bev_cell
        sx = ix if ix <= rows // 2 else ix - rows
        sy = iy if iy <= cols // 2 else iy - cols
        dx = -sx * step
        dy = -sy * step
        if abs(dx) > self.max_step or abs(dy) > self.max_step:
            return 0.0, 0.0, 0.0
        return float(dx), float(dy), score
