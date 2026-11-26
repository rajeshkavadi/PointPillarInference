#!/usr/bin/env python3
"""CPU-only PointPillars checkpoint validation and single-frame inference.

Stages (run in this order):

  Test 1 -- checkpoint validation
      python3 tools/pointpillars_cpu_inference.py --validate-only

  Test 2 -- single frame from a ROS bag (also saves the frame)
      python3 tools/pointpillars_cpu_inference.py \
          --bag ../data/demo1.bag --frame 0 \
          --save-frame ../data/demo1_frame0.npy

  Test 3 -- independent Open3D visualisation (optional)
      python3 tools/pointpillars_cpu_inference.py \
          --points ../data/demo1_frame0.npy --open3d

  Benchmark -- several frames
      python3 tools/pointpillars_cpu_inference.py --bag ../data/demo1.bag --frames 5
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pointpillars.bag_frame import find_cloud_topic, inspect_bag, read_frame  # noqa: E402
from pointpillars.config import ConfigError, load_config  # noqa: E402
from pointpillars.pipeline import PointPillarDetector  # noqa: E402
from pointpillars.postprocess import boxes_to_corners_3d  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and run an OpenPCDet PointPillars checkpoint on CPU")
    parser.add_argument("--ckpt", default=os.path.join(PROJECT_ROOT, "checkpoints",
                                                       "pointpillars_checkpoint.pth"),
                        help="trained .pth checkpoint")
    parser.add_argument("--cfg_file", default=os.path.join(PROJECT_ROOT, "configs",
                                                           "pointpillars.yaml"),
                        help="original OpenPCDet-style config")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto"],
                        help="CPU only (auto also resolves to CPU here)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Test 1: load/validate checkpoint, then exit")
    parser.add_argument("--bag", default=None, help="ROS bag to read a frame from")
    parser.add_argument("--frame", type=int, default=0, help="frame index inside the bag")
    parser.add_argument("--frames", type=int, default=0,
                        help="benchmark N frames (0 = just one frame)")
    parser.add_argument("--topic", default=None, help="PointCloud2 topic (auto-detected)")
    parser.add_argument("--points", default=None, help=".npy/.bin point frame (N x 3/4)")
    parser.add_argument("--save-frame", default=None, help="save extracted frame to .npy")
    parser.add_argument("--open3d", action="store_true",
                        help="show point cloud + boxes in Open3D (no ROS needed)")
    parser.add_argument("--intensity-mode", default=None, choices=["zero", "constant", "field"])
    parser.add_argument("--intensity-scale", default=None)
    parser.add_argument("--z-offset", type=float, default=None)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def load_points_file(path):
    if path.endswith(".npy"):
        points = np.load(path)
    elif path.endswith(".bin"):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    else:
        raise RuntimeError("unsupported point file %s (use .npy or .bin)" % path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError("expected N x 3/4 points in %s, got %r" % (path, points.shape))
    return points


def save_frame(points, path, meta):
    path = os.path.abspath(os.path.expanduser(path))
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    np.save(path, points.astype(np.float32))
    sidecar = os.path.splitext(path)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("saved frame: %s (%s)" % (path, sidecar))


def print_detections(det, class_names, max_print=20):
    boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
    print("\ndetections: %d" % len(boxes))
    for i in range(min(max_print, len(boxes))):
        b = boxes[i]
        name = class_names[int(labels[i]) - 1] if 0 < int(labels[i]) <= len(class_names) else "?"
        print("  [%2d] %-11s score=%.4f  x=%8.2f y=%8.2f z=%7.2f  "
              "dx=%5.2f dy=%5.2f dz=%5.2f  yaw=%6.3f"
              % (i, name, scores[i], b[0], b[1], b[2], b[3], b[4], b[5], b[6]))
    if len(boxes) > max_print:
        print("  ... %d more" % (len(boxes) - max_print))


def print_timing(det):
    t = det["timings"]
    print("\ntiming breakdown (ms):")
    print("  preprocessing        : %8.2f" % t.get("preprocess_ms", 0.0))
    print("  pillar generation    : %8.2f" % t.get("voxelization_ms", 0.0))
    print("  network inference    : %8.2f" % t.get("network_ms", 0.0))
    print("  box decoding         : %8.2f" % t.get("decode_ms", 0.0))
    print("  NMS / post-process   : %8.2f" % t.get("nms_ms", 0.0))
    print("  total                : %8.2f  (%.2f FPS)"
          % (t.get("total_ms", 0.0), 1000.0 / max(t.get("total_ms", 1e-6), 1e-6)))


def visualize_open3d(points, det, class_names):
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed. Install with: pip3 install open3d==0.15.2 "
              "(Python 3.6); skipping visualization.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3], dtype=np.float64))
    geometries = [pcd]
    lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    palette = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0.5, 0]], dtype=float)
    corners = boxes_to_corners_3d(det["boxes"])
    for i in range(corners.shape[0]):
        label = int(det["labels"][i]) - 1
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(corners[i], dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        color = palette[label % len(palette)]
        line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
        geometries.append(line_set)
    print("opening Open3D window (close it to continue)")
    o3d.visualization.draw_geometries(geometries)


def main():
    args = parse_args()
    overrides = {"device": args.device}
    if args.intensity_mode:
        overrides["INTENSITY_MODE"] = args.intensity_mode
    if args.intensity_scale:
        overrides["INTENSITY_SCALE"] = args.intensity_scale
    if args.z_offset is not None:
        overrides["Z_OFFSET"] = args.z_offset

    try:
        cfg = load_config(args.cfg_file, overrides)
    except ConfigError as exc:
        print("CONFIG ERROR: %s" % exc)
        return 2
    if args.score_thresh is not None:
        cfg["post"]["score_thresh"] = args.score_thresh
    if args.nms_thresh is not None:
        cfg["post"]["nms_thresh"] = args.nms_thresh

    print("torch: %s | CUDA available: %s (never used)" %
          (torch.__version__, torch.cuda.is_available()))
    try:
        detector = PointPillarDetector(cfg, args.ckpt, overrides)
    except RuntimeError as exc:
        print("CHECKPOINT ERROR: %s" % exc)
        return 2

    print("\n" + detector.describe())
    if args.validate_only:
        print("\nCHECKPOINT VALIDATION PASSED")
        return 0

    frames = []
    if args.points:
        pts = load_points_file(args.points)
        frames.append({"points": pts, "has_intensity": None,
                       "fields": ["npy"], "frame_id": "-", "stamp": 0.0,
                       "index": 0, "topic": args.points})
    elif args.bag:
        try:
            info = inspect_bag(args.bag)
            topic = args.topic or find_cloud_topic(args.bag)[0]
            print("\nbag: %s" % os.path.abspath(args.bag))
            print("topics: %s" % info["topics"])
            print("using topic: %s (messages=%d, duration=%.2fs)"
                  % (topic, info["topics"][topic]["messages"], info["duration"]))
        except Exception as exc:
            print("BAG ERROR: %s" % exc)
            return 2
        n = max(1, args.frames)
        for idx in range(args.frame, args.frame + n):
            try:
                frames.append(read_frame(args.bag, idx, topic))
            except RuntimeError as exc:
                print("BAG ERROR: %s" % exc)
                break
    else:
        print("ERROR: provide --bag FILE or --points FILE (or use --validate-only)")
        return 2

    # Warm up once so benchmark numbers reflect steady state.
    detector.detect(frames[0]["points"], frames[0]["has_intensity"])

    all_times = []
    for frame in frames:
        meta = {"index": frame["index"], "topic": frame["topic"],
                "frame_id": frame["frame_id"], "stamp": frame["stamp"],
                "fields": frame["fields"], "has_intensity": frame["has_intensity"]}
        if args.save_frame and args.bag and frame["index"] == args.frame:
            save_frame(frame["points"], args.save_frame, meta)

        det = detector.detect(frame["points"], frame["has_intensity"])
        all_times.append(det["timings"]["total_ms"])
        if args.print_every > 0 and frame["index"] % args.print_every == 0:
            print("\n=== frame %d (stamp %.3f, frame_id=%r, fields=%s) ==="
                  % (frame["index"], frame["stamp"], frame["frame_id"], frame["fields"]))
            print("original points   : %d" % det["num_original_points"])
            print("filtered points   : %d (OpenPCDet point_cloud_range)" %
                  det["num_filtered_points"])
            print("pillars           : %d" % det["num_voxels"])
            print_detections(det, cfg["class_names"])
            print_timing(det)
            if args.open3d:
                visualize_open3d(frame["points"], det, cfg["class_names"])

    if len(all_times) > 1:
        steady = all_times[1:]
        print("\nbenchmark over %d frames: mean=%.1f ms (%.2f FPS), min=%.1f ms, max=%.1f ms"
              % (len(steady), np.mean(steady), 1000.0 / max(np.mean(steady), 1e-6),
                 np.min(steady), np.max(steady)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
