"""End-to-end CPU PointPillars inference for a single LiDAR frame."""

import time

import numpy as np
import torch

from .model import PointPillar
from .postprocess import postprocess
from .voxelizer import voxelize


def load_checkpoint(path):
    """Load an OpenPCDet training checkpoint on CPU and return its state dict."""
    path = str(path)
    try:
        ckpt = torch.load(path, map_location=torch.device("cpu"))
    except Exception as exc:
        raise RuntimeError("could not load checkpoint %s on CPU: %s" % (path, exc))
    if not isinstance(ckpt, dict):
        raise RuntimeError("unexpected checkpoint format in %s: %s" % (path, type(ckpt)))

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no state_dict")

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    meta = {k: ckpt[k] for k in ("epoch", "it", "version") if k in ckpt}
    return cleaned, meta


class PointPillarDetector(object):
    """CPU-only detector.  Device other than 'cpu'/'auto' is rejected."""

    def __init__(self, cfg, checkpoint, overrides=None):
        self.cfg = cfg
        self.class_names = list(cfg["class_names"])
        device = str(overrides.get("device", cfg["inference"]["device"])
                     if overrides else cfg["inference"]["device"]).lower()
        if device not in ("cpu", "auto"):
            raise RuntimeError("CPU-only project: refusing device=%r" % device)
        self.device = torch.device("cpu")

        self.model = PointPillar(cfg)
        state, self.meta = load_checkpoint(checkpoint)
        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(state.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint/architecture mismatch (no weights were modified).\n"
                "  missing in checkpoint: %s\n  unexpected in checkpoint: %s"
                % (missing[:15], unexpected[:15]))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()  # BatchNorm uses running statistics, no updating.
        self.checkpoint_status = "strict load OK (%d tensors)" % len(state)

        inf = cfg["inference"]
        self.intensity_mode = inf["intensity_mode"]
        self.intensity_value = inf["intensity_value"]
        self.intensity_scale = inf["intensity_scale"]
        self.z_offset = inf["z_offset"]
        self.z_rotation = np.deg2rad(inf["z_rotation_deg"])
        self._warned_intensity = False

        expected = cfg["head"]["num_anchors_per_location"] * cfg["num_class"]
        actual = self.model.dense_head.conv_cls.out_channels
        if actual != expected:
            raise RuntimeError("head mismatch: conv_cls has %d channels, config expects %d"
                               % (actual, expected))

    # ------------------------------------------------------------------
    def _intensity_factor(self, intensity):
        scale = self.intensity_scale
        if isinstance(scale, (int, float)):
            return float(scale) if float(scale) != 0 else 1.0
        if intensity.size == 0:
            return 1.0
        vmax = float(np.max(intensity))
        if vmax <= 1.5:
            return 1.0
        return 255.0 if vmax <= 255.0 else vmax

    def preprocess(self, points, has_intensity=None):
        """Return (voxels, num_points, coords, filtered_point_count).

        ``points`` is an N x 3/4 array.  ``has_intensity`` tells whether the 4th
        column really came from the sensor; when it did not, filling it with
        zeros is an explicit, logged decision (never silent).
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be N x (>=3), got %r" % (pts.shape,))
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), np.float32)], axis=1)
        else:
            pts = pts[:, :4].copy()

        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]

        if self.intensity_mode == "zero":
            if has_intensity is False or pts.shape[1] < 4:
                if not self._warned_intensity:
                    print("[INTENSITY WARNING] input has no intensity field; filling the "
                          "4th VFE feature with 0.0. The checkpoint was trained with "
                          "intensity, so accuracy will be degraded. This is NOT a "
                          "model/config change -- provide a bag with intensity for "
                          "faithful inference.")
                    self._warned_intensity = True
                pts[:, 3] = 0.0
        elif self.intensity_mode == "constant":
            pts[:, 3] = self.intensity_value
        else:  # field
            if has_intensity is False:
                raise RuntimeError("INTENSITY_MODE=field but the point cloud has no "
                                   "intensity field; refusing to fabricate features")

        factor = self._intensity_factor(pts[:, 3])
        if factor != 1.0:
            pts[:, 3] = np.clip(pts[:, 3] / factor, 0.0, 1.0)

        if self.z_rotation != 0.0 and pts.shape[0]:
            c, s = np.cos(self.z_rotation), np.sin(self.z_rotation)
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            pts[:, 0] = c * x - s * y
            pts[:, 1] = s * x + c * y
        if self.z_offset != 0.0:
            pts[:, 2] += self.z_offset

        pcr = self.cfg["point_cloud_range"]
        mask = ((pts[:, 0] >= pcr[0]) & (pts[:, 0] <= pcr[3]) &
                (pts[:, 1] >= pcr[1]) & (pts[:, 1] <= pcr[4]))
        pts = pts[mask]

        voxels, num_points, coords = voxelize(
            pts,
            point_cloud_range=pcr,
            voxel_size=self.cfg["voxel_size"],
            grid_size=self.cfg["grid_size"],
            max_points_per_voxel=self.cfg["max_points_per_voxel"],
            max_voxels=self.cfg["max_voxels"],
        )
        return voxels, num_points, coords, int(pts.shape[0])

    def _to_tensor(self, arr, dtype):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)

    def _boxes_back(self, boxes):
        if boxes.shape[0] == 0:
            return boxes
        boxes = boxes.copy()
        if self.z_offset != 0.0:
            boxes[:, 2] -= self.z_offset
        if self.z_rotation != 0.0:
            c, s = np.cos(-self.z_rotation), np.sin(-self.z_rotation)
            x, y = boxes[:, 0].copy(), boxes[:, 1].copy()
            boxes[:, 0] = c * x - s * y
            boxes[:, 1] = s * x + c * y
            boxes[:, 6] -= self.z_rotation
        return boxes

    # ------------------------------------------------------------------
    def detect(self, points, has_intensity=None):
        """Run inference on one frame and return boxes/scores/labels/timings."""
        timings = {}
        t0 = time.perf_counter()
        voxels, num_points, coords, n_filtered = self.preprocess(points, has_intensity)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        empty = {
            "boxes": np.zeros((0, 7), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "num_original_points": int(np.asarray(points).shape[0]),
            "num_filtered_points": n_filtered,
            "num_voxels": 0,
            "timings": timings,
        }
        if voxels.shape[0] == 0:
            timings["voxelization_ms"] = 0.0
            timings["network_ms"] = 0.0
            timings["decode_ms"] = 0.0
            timings["nms_ms"] = 0.0
            timings["total_ms"] = timings["preprocess_ms"]
            return empty

        t1 = time.perf_counter()
        voxels_t = self._to_tensor(voxels, torch.float32)
        counts_t = self._to_tensor(num_points, torch.int32)
        coords_t = self._to_tensor(coords, torch.int32)
        timings["voxelization_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        with torch.no_grad():
            cls, box, direction, anchors = self.model.extract_features(
                voxels_t, counts_t, coords_t)
        timings["network_ms"] = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        with torch.no_grad():
            cls, boxes = self.model.dense_head.decode(cls, box, direction, anchors)
        timings["decode_ms"] = (time.perf_counter() - t3) * 1000.0

        t4 = time.perf_counter()
        with torch.no_grad():
            post = self.cfg["post"]
            out_boxes, out_scores, out_labels = postprocess(
                cls, boxes,
                score_thresh=post["score_thresh"],
                nms_thresh=post["nms_thresh"],
                nms_pre_maxsize=post["nms_pre_maxsize"],
                nms_post_maxsize=post["nms_post_maxsize"],
            )
        timings["nms_ms"] = (time.perf_counter() - t4) * 1000.0

        boxes_np = out_boxes.detach().cpu().numpy().astype(np.float32)
        empty.update({
            "boxes": self._boxes_back(boxes_np),
            "scores": out_scores.detach().cpu().numpy().astype(np.float32),
            "labels": out_labels.detach().cpu().numpy().astype(np.int64),
            "num_voxels": int(voxels.shape[0]),
        })
        timings["total_ms"] = sum(timings[k] for k in (
            "preprocess_ms", "voxelization_ms", "network_ms", "decode_ms", "nms_ms"))
        return empty

    def describe(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        lines = [
            "device                : %s (CPU only; CUDA is never used)" % self.device,
            "checkpoint            : %s" % self.checkpoint_status,
            "checkpoint meta       : %s" % (self.meta or "none"),
            "class names           : %s" % self.class_names,
            "parameters            : %d (%.3f M)" % (n_params, n_params / 1e6),
            "architecture          : PillarVFE -> PointPillarScatter -> BaseBEVBackbone "
            "-> AnchorHeadSingle",
            "  vfe linear          : %s" % (tuple(self.model.vfe.pfn_layers[0].linear.weight.shape),),
            "  backbone blocks     : %s" % [len(b) for b in self.model.backbone_2d.blocks],
            "  head conv outputs   : cls=%d box=%d dir=%s" % (
                self.model.dense_head.conv_cls.out_channels,
                self.model.dense_head.conv_box.out_channels,
                (self.model.dense_head.conv_dir_cls.out_channels
                 if self.model.dense_head.conv_dir_cls else 0)),
            "point cloud range     : %s" % self.cfg["point_cloud_range"],
            "voxel size            : %s" % self.cfg["voxel_size"],
            "BEV grid              : %s" % self.cfg["grid_size"],
            "anchors / location    : %d" % self.cfg["head"]["num_anchors_per_location"],
        ]
        return "\n".join(lines)
