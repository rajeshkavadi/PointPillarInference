"""Load and validate the OpenPCDet PointPillars inference configuration.

The YAML keeps the original OpenPCDet parameter names.  This module only reads
them; it never guesses or substitutes a different model parameter.
"""

import os

import numpy as np
import yaml


class ConfigError(RuntimeError):
    pass


def _get(d, *path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise ConfigError("missing config key: %s" % ".".join(path))
        cur = cur[key]
    return cur


def _numbers(value, name, count=None):
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ConfigError("%s must be a list of numbers, got %r" % (name, value))
    if count is not None and len(out) != count:
        raise ConfigError("%s must have %d values, got %d (%r)" % (name, count, len(out), value))
    return out


def load_config(path, overrides=None):
    """Read ``pointpillars.yaml`` and derive grid/model settings from it.

    Args:
        path: path to the OpenPCDet-style YAML.
        overrides: optional flat dict of INFERENCE overrides (from CLI flags).

    Returns:
        dict with validated configuration used by model/pipeline/ROS scripts.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ConfigError("config file not found: %s" % path)
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping: %s" % path)

    class_names = [str(c) for c in _get(raw, "CLASS_NAMES")]
    if not class_names:
        raise ConfigError("CLASS_NAMES is empty")

    pcr = _numbers(_get(raw, "DATA_CONFIG", "POINT_CLOUD_RANGE"),
                   "DATA_CONFIG.POINT_CLOUD_RANGE", 6)
    if not (pcr[3] > pcr[0] and pcr[4] > pcr[1] and pcr[5] > pcr[2]):
        raise ConfigError("invalid POINT_CLOUD_RANGE: %r" % (pcr,))

    voxel_cfg = None
    for processor in _get(raw, "DATA_CONFIG", "DATA_PROCESSOR"):
        if processor.get("NAME") == "transform_points_to_voxels":
            voxel_cfg = processor
            break
    if voxel_cfg is None:
        raise ConfigError("DATA_CONFIG.DATA_PROCESSOR has no transform_points_to_voxels entry")

    voxel_size = _numbers(voxel_cfg["VOXEL_SIZE"], "VOXEL_SIZE", 3)
    if min(voxel_size) <= 0:
        raise ConfigError("VOXEL_SIZE must be positive: %r" % (voxel_size,))
    max_points_per_voxel = int(voxel_cfg["MAX_POINTS_PER_VOXEL"])
    max_voxels = int(voxel_cfg["MAX_NUMBER_OF_VOXELS"]["test"])

    grid = np.round((np.asarray(pcr[3:6]) - np.asarray(pcr[0:3])) /
                    np.asarray(voxel_size)).astype(np.int64)
    grid_size = [int(v) for v in grid]
    if min(grid_size[:2]) <= 0 or grid_size[2] < 1:
        raise ConfigError("degenerate grid %r from range %r / voxel %r"
                          % (grid_size, pcr, voxel_size))
    if grid_size[0] % 16 or grid_size[1] % 16:
        raise ConfigError("BEV grid %dx%d is not divisible by 16" % tuple(grid_size[:2]))

    model = raw["MODEL"]
    vfe_cfg = model["VFE"]
    backbone_cfg = model["BACKBONE_2D"]
    head_cfg = model["DENSE_HEAD"]
    post_cfg = model["POST_PROCESSING"]
    nms_cfg = post_cfg["NMS_CONFIG"]

    anchor_cfg = []
    for item in head_cfg["ANCHOR_GENERATOR_CONFIG"]:
        anchor_cfg.append({
            "class_name": str(item["class_name"]),
            "anchor_sizes": [list(map(float, s)) for s in item["anchor_sizes"]],
            "anchor_rotations": [float(a) for a in item["anchor_rotations"]],
            "anchor_bottom_heights": [float(h) for h in item["anchor_bottom_heights"]],
            "align_center": bool(item.get("align_center", False)),
            "feature_map_stride": int(item.get("feature_map_stride", 2)),
        })
    if len(anchor_cfg) != len(class_names):
        raise ConfigError("got %d anchor configs but %d class names"
                          % (len(anchor_cfg), len(class_names)))

    num_anchors_per_location = 0
    for item in anchor_cfg:
        count = (len(item["anchor_rotations"]) * len(item["anchor_sizes"]) *
                 len(item["anchor_bottom_heights"]))
        fms = item["feature_map_stride"]
        if grid_size[0] % fms or grid_size[1] % fms:
            raise ConfigError("grid %r not divisible by feature_map_stride %d"
                              % (grid_size, fms))
        num_anchors_per_location += count

    infer = dict(raw.get("INFERENCE", {}))
    if overrides:
        infer.update({k: v for k, v in overrides.items() if v is not None})

    device = str(infer.get("DEVICE", "cpu")).lower()
    if device not in ("cpu", "auto"):
        raise ConfigError("CPU-only build: DEVICE must be 'cpu' (or 'auto'), got %r"
                          % infer.get("DEVICE"))

    intensity_mode = str(infer.get("INTENSITY_MODE", "zero")).lower()
    if intensity_mode not in ("zero", "constant", "field"):
        raise ConfigError("INTENSITY_MODE must be zero|constant|field, got %r" % intensity_mode)
    scale = infer.get("INTENSITY_SCALE", "auto")
    if not isinstance(scale, str):
        scale = float(scale)
    elif scale.lower() != "auto":
        try:
            scale = float(scale)
        except ValueError:
            raise ConfigError("INTENSITY_SCALE must be 'auto' or a number, got %r" % scale)

    cfg = {
        "config_path": path,
        "class_names": class_names,
        "num_class": len(class_names),
        "point_cloud_range": pcr,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        "max_points_per_voxel": max_points_per_voxel,
        "max_voxels": max_voxels,
        "num_point_features": 4,  # fixed by the checkpoint's VFE: linear.weight (64, 10)
        "vfe": {
            "use_norm": bool(vfe_cfg["USE_NORM"]),
            "with_distance": bool(vfe_cfg["WITH_DISTANCE"]),
            "use_absolute_xyz": bool(vfe_cfg["USE_ABSLOTE_XYZ"]),
            "num_filters": [int(v) for v in vfe_cfg["NUM_FILTERS"]],
        },
        "num_bev_features": int(model["MAP_TO_BEV"]["NUM_BEV_FEATURES"]),
        "backbone": {
            "layer_nums": [int(v) for v in backbone_cfg["LAYER_NUMS"]],
            "layer_strides": [int(v) for v in backbone_cfg["LAYER_STRIDES"]],
            "num_filters": [int(v) for v in backbone_cfg["NUM_FILTERS"]],
            "upsample_strides": [int(v) for v in backbone_cfg["UPSAMPLE_STRIDES"]],
            "num_upsample_filters": [int(v) for v in backbone_cfg["NUM_UPSAMPLE_FILTERS"]],
        },
        "head": {
            "anchor_generator_config": anchor_cfg,
            "num_anchors_per_location": num_anchors_per_location,
            "use_direction_classifier": bool(head_cfg["USE_DIRECTION_CLASSIFIER"]),
            "dir_offset": float(head_cfg["DIR_OFFSET"]),
            "dir_limit_offset": float(head_cfg["DIR_LIMIT_OFFSET"]),
            "num_dir_bins": int(head_cfg["NUM_DIR_BINS"]),
        },
        "post": {
            "score_thresh": float(post_cfg["SCORE_THRESH"]),
            "nms_type": str(nms_cfg["NMS_TYPE"]),
            "multi_classes_nms": bool(nms_cfg["MULTI_CLASSES_NMS"]),
            "nms_thresh": float(nms_cfg["NMS_THRESH"]),
            "nms_pre_maxsize": int(nms_cfg["NMS_PRE_MAXSIZE"]),
            "nms_post_maxsize": int(nms_cfg["NMS_POST_MAXSIZE"]),
        },
        "inference": {
            "device": device,
            "lidar_topic": str(infer.get("LIDAR_TOPIC", "/livox/lidar")),
            "points_topic": str(infer.get("POINTS_TOPIC", "/pointcloud")),
            "box_topic": str(infer.get("BOX_TOPIC", "/detect_box3d")),
            "text_topic": str(infer.get("TEXT_TOPIC", "/text_det")),
            "json_topic": str(infer.get("JSON_TOPIC", "/detect_json")),
            "intensity_mode": intensity_mode,
            "intensity_value": float(infer.get("INTENSITY_VALUE", 0.0)),
            "intensity_scale": scale,
            "z_offset": float(infer.get("Z_OFFSET", 0.0)),
            "z_rotation_deg": float(infer.get("Z_ROTATION_DEG", 0.0)),
            "frame_id": str(infer.get("FRAME_ID", "") or ""),
            "print_every": int(infer.get("PRINT_EVERY", 10)),
        },
    }

    if cfg["post"]["nms_type"] != "nms_gpu":
        raise ConfigError("this checkpoint was trained with NMS_TYPE=nms_gpu (3D IoU NMS); "
                          "got %r" % cfg["post"]["nms_type"])
    if cfg["post"]["multi_classes_nms"]:
        raise ConfigError("this checkpoint expects class-agnostic NMS (MULTI_CLASSES_NMS=False)")
    return cfg


def model_config(cfg):
    """Subset of ``cfg`` needed to construct the torch model."""
    keys = (
        "class_names", "num_class", "point_cloud_range", "voxel_size", "grid_size",
        "num_point_features", "vfe", "num_bev_features", "backbone", "head",
    )
    return {key: cfg[key] for key in keys}
