#!/usr/bin/env python3
"""ROS1 node: CPU-only OpenPCDet PointPillars on a Livox PointCloud2 stream.

Topics (same names as upstream ``livox_detection/tools/vis_ros.py``):

    /livox/lidar     in   sensor_msgs/PointCloud2
    /pointcloud      out  sensor_msgs/PointCloud2  (x,y,z,intensity)
    /detect_box3d    out  visualization_msgs/MarkerArray (wireframes + heading)
    /text_det        out  visualization_msgs/MarkerArray (class/score labels)

The incoming header stamp and frame_id are preserved (``~frame_id`` overrides
only the frame_id).  Inference is CPU-only.  Because one CPU pass takes seconds
on this host, the subscriber keeps only the newest message and the main loop
processes frames as fast as it can, explicitly skipping stale ones; this is
preferable to blocking the ROS callback queue for 4-5 s per callback.
"""

import argparse
import json
import os
import sys
import threading
import time

try:  # works both from the source tree and from a catkin-deployed script
    import rospkg
    PROJECT_ROOT = rospkg.RosPack().get_path("livox_detection")
except Exception:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import rospy  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402
import std_msgs.msg  # noqa: E402

from pointpillars.config import ConfigError, load_config  # noqa: E402
from pointpillars.pipeline import PointPillarDetector  # noqa: E402
from pointpillars.ros_utils import (  # noqa: E402
    build_marker_arrays, make_pointcloud2, pointcloud2_to_numpy,
)
from pointpillars.web_server import detections_payload  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="CPU PointPillars ROS node for Livox PointCloud2")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint (default: ROS param ~ckpt or bundled checkpoint)")
    parser.add_argument("--cfg_file", default=None, help="OpenPCDet-style YAML")
    parser.add_argument("--lidar_topic", default=None)
    parser.add_argument("--points_topic", default=None)
    parser.add_argument("--box_topic", default=None)
    parser.add_argument("--text_topic", default=None)
    parser.add_argument("--json_topic", default=None,
                        help="compact detection JSON for the web viewer")
    parser.add_argument("--frame_id", default=None, help="override incoming frame_id")
    parser.add_argument("--intensity-mode", dest="intensity_mode", default=None,
                        choices=["zero", "constant", "field"])
    parser.add_argument("--z-offset", dest="z_offset", type=float, default=None)
    parser.add_argument("--score-thresh", dest="score_thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", dest="nms_thresh", type=float, default=None)
    parser.add_argument("--queue-size", dest="queue_size", type=int, default=None)
    parser.add_argument("--log-every", dest="log_every", type=int, default=None)
    parser.add_argument("--no-republish-points", dest="republish_points",
                        action="store_false", default=True)
    return parser.parse_args(argv)


def param(name, default):
    return rospy.get_param("~" + name, default)


def main(argv=None):
    rospy.init_node("pointpillars_cpu", anonymous=True)
    args = parse_args(rospy.myargv(argv)[1:])

    default_cfg = os.path.join(PROJECT_ROOT, "configs", "pointpillars.yaml")
    default_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pointpillars_checkpoint.pth")
    cfg_file = args.cfg_file or param("cfg_file", default_cfg)
    ckpt = args.ckpt or param("ckpt", default_ckpt)

    overrides = {"DEVICE": "cpu"}
    for attr, key in (("intensity_mode", "INTENSITY_MODE"),
                      ("z_offset", "Z_OFFSET")):
        value = getattr(args, attr)
        if value is None:
            value = param(key.lower(), None)
        if value is not None:
            overrides[key] = value

    try:
        cfg = load_config(cfg_file, overrides)
    except ConfigError as exc:
        rospy.logfatal("configuration error: %s", exc)
        return 2
    if args.score_thresh is not None:
        cfg["post"]["score_thresh"] = float(args.score_thresh)
    if args.nms_thresh is not None:
        cfg["post"]["nms_thresh"] = float(args.nms_thresh)

    inf = cfg["inference"]
    lidar_topic = args.lidar_topic or param("lidar_topic", inf["lidar_topic"])
    points_topic = args.points_topic or param("points_topic", inf["points_topic"])
    box_topic = args.box_topic or param("box_topic", inf["box_topic"])
    text_topic = args.text_topic or param("text_topic", inf["text_topic"])
    json_topic = args.json_topic or param("json_topic", inf["json_topic"])
    frame_id = args.frame_id or param("frame_id", inf["frame_id"]) or ""
    queue_size = int(args.queue_size if args.queue_size is not None
                     else param("queue_size", 1))
    log_every = int(args.log_every if args.log_every is not None
                    else param("log_every", inf["print_every"]))

    try:
        detector = PointPillarDetector(cfg, ckpt, overrides)
    except RuntimeError as exc:
        rospy.logfatal("checkpoint/model error: %s", exc)
        return 2

    points_pub = rospy.Publisher(points_topic, PointCloud2, queue_size=10) \
        if args.republish_points else None
    boxes_pub = rospy.Publisher(box_topic, MarkerArray, queue_size=10)
    text_pub = rospy.Publisher(text_topic, MarkerArray, queue_size=10)
    json_pub = rospy.Publisher(json_topic, String, queue_size=10)

    lock = threading.Lock()
    state = {"counter": 0, "msg": None}

    def on_cloud(msg):
        with lock:
            state["counter"] += 1
            state["msg"] = (state["counter"], msg)

    rospy.Subscriber(lidar_topic, PointCloud2, on_cloud, queue_size=queue_size)
    rospy.loginfo("CPU PointPillars ready: %s -> %s / %s / %s / %s",
                  lidar_topic, points_topic, box_topic, text_topic, json_topic)
    rospy.loginfo("checkpoint: %s", detector.checkpoint_status)
    rospy.loginfo("CPU-only inference: ~4-5 s/frame on this host; stale frames are skipped.")

    last_processed = -1
    processed = 0
    skipped = 0
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        with lock:
            item = state["msg"]
        if item is None or item[0] == last_processed:
            rate.sleep()
            continue
        counter, msg = item
        if last_processed >= 0:
            skipped += max(0, counter - last_processed - 1)
        last_processed = counter

        started = time.perf_counter()
        try:
            points, has_intensity, _fields = pointcloud2_to_numpy(msg)
            det = detector.detect(points, has_intensity)
        except Exception as exc:  # keep the node alive after one bad frame
            rospy.logerr("inference failed on frame %d: %s", counter, exc)
            continue

        header = std_msgs.msg.Header()
        header.stamp = msg.header.stamp           # preserved from the bag/driver
        header.frame_id = frame_id or msg.header.frame_id

        if points_pub is not None:
            points_pub.publish(make_pointcloud2(points, header))
        boxes_arr, text_arr = build_marker_arrays(
            det["boxes"], det["scores"], det["labels"], cfg["class_names"], header)
        boxes_pub.publish(boxes_arr)
        text_pub.publish(text_arr)
        json_pub.publish(String(json.dumps({
            "frame_id": header.frame_id,
            "stamp": header.stamp.to_sec(),
            "boxes": detections_payload(det["boxes"], det["scores"],
                                        det["labels"], cfg["class_names"]),
            "meta": {"infer_ms": round(det["timings"].get("total_ms", 0.0), 1),
                     "points_in": int(det["num_original_points"]),
                     "voxels": int(det.get("num_voxels", 0))},
        }, separators=(",", ":"))))

        processed += 1
        wall_ms = (time.perf_counter() - started) * 1000.0
        if log_every > 0 and (processed - 1) % log_every == 0:
            rospy.loginfo(
                "frame %d | stamp=%.3f frame=%s | points=%d boxes=%d | infer=%.0f ms "
                "wall=%.0f ms | skipped=%d",
                counter, header.stamp.to_sec(), header.frame_id,
                det["num_original_points"], len(det["boxes"]),
                det["timings"].get("network_ms", 0.0), wall_ms, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
