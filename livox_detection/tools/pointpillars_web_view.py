#!/usr/bin/env python3
"""Local three.js viewer for CPU PointPillars -- no RViz required.

Two data sources feed the same browser page:

  bag mode (default, completely ROS-free)
      python3 tools/pointpillars_web_view.py --bag data/demo1.bag
      Reads LiDAR frames straight from a ROS1 bag, runs the CPU detector in
      this process and streams points + boxes to http://127.0.0.1:8080.
      The bag replays in a loop until Ctrl+C, and a Cloudflare quick tunnel
      (https://<random>.trycloudflare.com) is opened for remote viewers.

While the sensor moves, the viewer builds a persistent map: every frame is
placed in a world frame using a scan-to-scan motion estimate
(``pointpillars/map_builder.py``) and only the new points are streamed to the
browser, which never drops what it has already drawn.  Detection still runs on
the current frame only.  ``--no-map`` restores the old single-frame view.

  ROS mode (replaces RViz when the detector is already running as a node)
      python3 tools/pointpillars_web_view.py --ros
      Subscribes to ``/pointcloud`` and ``/detect_json`` and mirrors them to
      the same page.

The page draws the cloud colourised by range, concentric 5 m radar rings with
distance labels, and the 3D boxes.  Everything it needs is vendored in
``webapp/`` so the viewer works offline.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser

import numpy as np


def _project_root():
    """Works from the source tree, a catkin install space and any machine."""
    env = os.environ.get("LIVOX_DETECTION_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(here, "webapp")):
        return here
    try:
        import rospkg
        found = rospkg.RosPack().get_path("livox_detection")
        if found:
            return found
    except Exception:
        pass
    return here


PROJECT_ROOT = _project_root()
sys.path.insert(0, PROJECT_ROOT)

from pointpillars.config import ConfigError, load_config  # noqa: E402
from pointpillars.map_builder import TrajectoryMap  # noqa: E402
from pointpillars.tunnel import CloudflareTunnel  # noqa: E402
from pointpillars.web_server import (  # noqa: E402
    DEFAULT_WEBAPP_DIR, FrameHub, WebViewerServer, detections_payload, encode_frame,
)


def _existing(path):
    return path if path and os.path.isfile(os.path.expanduser(path)) else None


def resolve_bag(value):
    """Find a bag without hard-coding anyone's home directory."""
    candidates = [
        value,
        os.environ.get("LIVOX_BAG"),
        os.path.join(PROJECT_ROOT, "data", "demo1.bag"),
        os.path.join(os.path.dirname(PROJECT_ROOT), "demo1.bag"),
        os.path.join(os.getcwd(), "demo1.bag"),
    ]
    for candidate in candidates:
        found = _existing(candidate)
        if found:
            return os.path.abspath(os.path.expanduser(found))
    raise SystemExit(
        "bag not found. Pass --bag /path/to/file.bag (looked in: %s)"
        % ", ".join(os.path.abspath(os.path.expanduser(c))
                    for c in candidates if c))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Local three.js PointPillars viewer (bag or ROS source)")
    parser.add_argument("--bag", default=None, help="ROS1 bag to replay")
    parser.add_argument("--topic", default=None,
                        help="PointCloud2 topic in the bag (auto-detected)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = run forever)")
    parser.add_argument("--no-loop", dest="loop", action="store_false", default=True,
                        help="stop at the end of the bag instead of replaying")
    parser.add_argument("--play-rate", type=float, default=1.0,
                        help="pacing for --no-infer replay (1.0 = bag speed)")
    parser.add_argument("--ros", action="store_true",
                        help="mirror /pointcloud + /detect_json instead of a bag")
    parser.add_argument("--points-topic", default=None,
                        help="ROS mode: point cloud topic (default /pointcloud)")
    parser.add_argument("--json-topic", default=None,
                        help="ROS mode: detection JSON topic (default /detect_json)")
    parser.add_argument("--no-infer", dest="infer", action="store_false",
                        default=True, help="bag mode: show the cloud without detections")
    parser.add_argument("--ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "pointpillars_checkpoint.pth"))
    parser.add_argument("--cfg_file", default=os.path.join(
        PROJECT_ROOT, "configs", "pointpillars.yaml"))
    parser.add_argument("--intensity-mode", dest="intensity_mode", default=None,
                        choices=["zero", "constant", "field"])
    parser.add_argument("--z-offset", dest="z_offset", type=float, default=None)
    parser.add_argument("--score-thresh", dest="score_thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", dest="nms_thresh", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (localhost only by default)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--webapp-dir", default=None,
                        help="override the webapp directory (default: packaged one)")
    parser.add_argument("--no-share", dest="share", action="store_false", default=True,
                        help="keep the viewer local instead of opening a "
                             "Cloudflare quick tunnel")
    parser.add_argument("--cloudflared", default=None,
                        help="path to the cloudflared binary (default: found on PATH)")
    parser.add_argument("--no-map", dest="build_map", action="store_false", default=True,
                        help="single-frame view: do not accumulate the persistent map")
    parser.add_argument("--map-voxel", type=float, default=0.4,
                        help="map resolution in metres: one point per cube (default 0.4)")
    parser.add_argument("--map-max-points", type=int, default=1500000,
                        help="upper bound on accumulated map points (default 1500000)")
    parser.add_argument("--no-open", dest="open_browser", action="store_false",
                        default=True, help="do not try to open a browser window")
    return parser.parse_args(argv)


def empty_detections(n_points):
    return {
        "boxes": np.zeros((0, 7), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "timings": {},
        "num_original_points": int(n_points),
    }


def publish_frame(hub, detector, frame, seq, class_names, mapping=None,
                  map_reset=False):
    """Run inference (if any) on one frame, place it on the map and publish it.

    Detection runs on the current frame only; ``mapping`` keeps the accumulated
    map.  In map mode the streamed cloud and boxes are in map coordinates and
    the payload also carries the points that are new to the map.
    """
    points = frame["points"]
    pose = None
    map_delta = None
    if mapping is not None:
        pose, map_delta = mapping.update(points)
        points = points.copy()
        points[:, 0] += pose[0]
        points[:, 1] += pose[1]

    started = time.perf_counter()
    if detector is not None:
        det = detector.detect(frame["points"], frame["has_intensity"])
    else:
        det = empty_detections(len(points))
    infer_ms = (time.perf_counter() - started) * 1000.0
    boxes = detections_payload(det["boxes"], det["scores"], det["labels"],
                               class_names, pose)
    meta = {
        "infer_ms": round(infer_ms, 1),
        "points_in": int(len(frame["points"])),
        "voxels": int(det.get("num_voxels", 0)),
        "intensity": bool(frame.get("has_intensity")),
        "index": int(frame.get("index", seq)),
    }
    if mapping is not None:
        meta["map_points"] = int(mapping.total)
        meta["map_full"] = bool(mapping.full)
        meta["map_epoch"] = int(mapping.epoch)
        meta["pose"] = {"x": round(pose[0], 3), "y": round(pose[1], 3)}
        if map_reset:
            meta["map_reset"] = True
    payload = encode_frame(seq, frame.get("stamp", time.time()),
                           frame.get("frame_id", "livox_frame"), points, boxes, meta,
                           map_points=map_delta, pose=pose)
    hub.publish(payload, seq, len(points), len(boxes),
                frame.get("stamp", time.time()), frame.get("frame_id", ""), meta)
    return det, boxes


def run_bag(args, hub, stop_event, mapping=None):
    from pointpillars.bag_frame import find_cloud_topic, iter_frames

    bag = resolve_bag(args.bag)
    topic = args.topic or find_cloud_topic(bag)[0]
    print("[web-view] bag:   %s" % bag)
    print("[web-view] topic: %s%s" % (topic, "" if args.infer else " (detections off)"))

    class_names = ["Car", "Pedestrian", "Cyclist"]
    detector = None
    if args.infer:
        overrides = {"DEVICE": "cpu"}
        if args.intensity_mode:
            overrides["INTENSITY_MODE"] = args.intensity_mode
        if args.z_offset is not None:
            overrides["Z_OFFSET"] = args.z_offset
        cfg = load_config(args.cfg_file, overrides)
        if args.score_thresh is not None:
            cfg["post"]["score_thresh"] = float(args.score_thresh)
        if args.nms_thresh is not None:
            cfg["post"]["nms_thresh"] = float(args.nms_thresh)
        class_names = list(cfg["class_names"])
        from pointpillars.pipeline import PointPillarDetector  # imports torch only here
        detector = PointPillarDetector(cfg, args.ckpt, overrides)
        print("[web-view] checkpoint: %s (%s)"
              % (os.path.basename(args.ckpt), detector.checkpoint_status))

    seq = 0
    reset_pending = False
    while not stop_event.is_set():
        produced = 0
        frame_started = None
        for frame in iter_frames(bag, topic, 0):
            if stop_event.is_set():
                return
            if frame["index"] < args.start_frame:
                continue
            if args.max_frames and seq >= args.max_frames:
                return
            frame_started = time.perf_counter()
            det, boxes = publish_frame(hub, detector, frame, seq, class_names, mapping,
                                       map_reset=reset_pending)
            reset_pending = False
            seq += 1
            produced += 1
            if args.max_frames and seq >= args.max_frames:
                return
            if produced % 5 == 0 or produced == 1:
                extra = ""
                if mapping is not None:
                    extra = " | map=%d pose=(%.1f, %.1f)" % (
                        mapping.total, mapping.pose[0], mapping.pose[1])
                print("[web-view] frame %d | points=%d boxes=%d | %.0f ms%s"
                      % (seq, len(frame["points"]), len(boxes),
                         (time.perf_counter() - frame_started) * 1000.0, extra),
                      flush=True)
            if detector is None and args.play_rate > 0:
                # Bag frames are 10 Hz; keep a plain cloud view readable.
                target = 0.1 / args.play_rate
                elapsed = time.perf_counter() - frame_started
                if elapsed < target:
                    time.sleep(target - elapsed)
        if not args.loop:
            if mapping is not None and not stop_event.is_set():
                print("[web-view] end of bag; the built map stays on screen. "
                      "Ctrl+C to exit.", flush=True)
                stop_event.wait()
            return
        print("[web-view] end of bag, replaying from the start", flush=True)
        if mapping is not None:
            # A replayed bag restarts at the first frame, which is far from the
            # pose the map ended at: rebuild instead of smearing both runs.
            mapping.reset()
            reset_pending = True
            print("[web-view] map reset for the new pass", flush=True)


def run_ros(args, hub, stop_event, mapping=None):
    import rospy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from pointpillars.ros_utils import pointcloud2_to_numpy

    rospy.init_node("pointpillars_web_view", anonymous=True, disable_signals=True)
    points_topic = args.points_topic or rospy.get_param("~points_topic", "/pointcloud")
    json_topic = args.json_topic or rospy.get_param("~json_topic", "/detect_json")
    print("[web-view] ROS in: %s + %s" % (points_topic, json_topic))

    lock = threading.Lock()
    state = {"frame": None, "seq": 0, "boxes": [], "meta": {}, "json_count": 0}
    pending = {"frame": None}

    def on_cloud(msg):
        try:
            points, has_intensity, _fields = pointcloud2_to_numpy(msg)
        except Exception as exc:  # one bad frame must not kill the viewer
            rospy.logerr("web view: bad point cloud: %s", exc)
            return
        with lock:
            state["seq"] += 1
            state["frame"] = {
                "points": points,
                "has_intensity": has_intensity,
                "stamp": msg.header.stamp.to_sec(),
                "frame_id": msg.header.frame_id or "livox_frame",
                "index": state["seq"] - 1,
            }
            pending["frame"] = state["frame"]

    def on_json(msg):
        try:
            doc = json.loads(msg.data)
        except ValueError:
            return
        with lock:
            state["boxes"] = doc.get("boxes") or []
            state["meta"] = doc.get("meta") or {}
            state["json_count"] += 1

    rospy.Subscriber(points_topic, PointCloud2, on_cloud, queue_size=1)
    rospy.Subscriber(json_topic, String, on_json, queue_size=10)

    published_seq = -1
    rate = rospy.Rate(50)
    while not stop_event.is_set() and not rospy.is_shutdown():
        with lock:
            frame = pending["frame"]
            boxes = list(state["boxes"])
            meta = dict(state["meta"])
        if frame is None or frame["index"] <= published_seq:
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            continue
        published_seq = frame["index"]
        points = frame["points"]
        pose = None
        map_delta = None
        if mapping is not None:
            pose, map_delta = mapping.update(points)
            points = points.copy()
            points[:, 0] += pose[0]
            points[:, 1] += pose[1]
            moved = []
            for box in boxes:
                box = dict(box)
                box["x"] = round(float(box.get("x", 0.0)) + pose[0], 3)
                box["y"] = round(float(box.get("y", 0.0)) + pose[1], 3)
                moved.append(box)
            boxes = moved
            meta["map_points"] = int(mapping.total)
            meta["map_epoch"] = int(mapping.epoch)
            meta["pose"] = {"x": round(pose[0], 3), "y": round(pose[1], 3)}
        meta.update({"points_in": int(len(frame["points"])),
                     "intensity": bool(frame["has_intensity"]),
                     "index": frame["index"]})
        payload = encode_frame(frame["index"], frame["stamp"], frame["frame_id"],
                               points, boxes, meta, map_points=map_delta, pose=pose)
        hub.publish(payload, frame["index"], len(points), len(boxes),
                    frame["stamp"], frame["frame_id"], meta)


INSTALL_HINT = (
    "python3 -m pip install -r requirements.txt \\\n"
    "    --extra-index-url https://download.pytorch.org/whl/cpu"
)


def preflight(args):
    """Fail fast with an actionable message instead of a late traceback."""
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as exc:
        raise SystemExit("missing Python dependency: %s\n  install with:\n    %s"
                         % (exc, INSTALL_HINT))
    if not args.ros:
        try:
            import rosbag  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "bag mode needs the ROS1 rosbag Python module (%s).\n"
                "  On a ROS machine: source /opt/ros/<distro>/setup.bash and install\n"
                "  the Python 3 helpers with:\n    %s\n"
                "  or run the viewer against live ROS topics with --ros."
                % (exc, INSTALL_HINT))
        if args.infer:
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise SystemExit(
                    "detection needs torch on CPU (%s).\n  install with:\n    %s\n"
                    "  (or pass --no-infer to view the cloud only)"
                    % (exc, INSTALL_HINT))
    else:
        try:
            import rospy  # noqa: F401
            from sensor_msgs.msg import PointCloud2  # noqa: F401
            from std_msgs.msg import String  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--ros needs a sourced ROS 1 environment (%s).\n"
                "  source /opt/ros/<distro>/setup.bash and retry, or drop --ros\n"
                "  to read a bag directly (no ROS required)." % exc)
        # rospy.init_node() retries a missing master forever instead of
        # failing, which looks like "nothing is displayed": check it first.
        try:
            from rosgraph import Master
            Master("/pointpillars_web_view").getPid()
        except Exception as exc:
            raise SystemExit(
                "no ROS master at %s (%s).\n"
                "  start `roscore` (or source the machine that runs it) and retry,\n"
                "  or drop --ros to read a bag directly (no ROS required)."
                % (os.environ.get("ROS_MASTER_URI", "http://localhost:11311"), exc))


def open_browser(url):
    """Best-effort browser launch; headless machines just keep the URL."""
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def start_share(port, host, binary, holder):
    """Open the Cloudflare tunnel in the background so frames start at once."""
    def worker():
        try:
            tunnel = CloudflareTunnel(port, host=host, binary=binary)
            holder["tunnel"] = tunnel
            tunnel.start()
            print("[web-view] share: %s" % tunnel.url, flush=True)
            print("[web-view]        public Cloudflare link", flush=True)
        except RuntimeError as exc:
            print("[web-view] share unavailable: %s" % exc, file=sys.stderr,
                  flush=True)

    thread = threading.Thread(target=worker, name="cloudflare-share")
    thread.daemon = True
    thread.start()
    return thread


def main(argv=None):
    args = parse_args(argv)
    preflight(args)
    if not args.ros:
        if args.infer and not os.path.isfile(args.ckpt):
            raise SystemExit("checkpoint not found: %s" % args.ckpt)
        if not os.path.isfile(args.cfg_file):
            raise SystemExit("config not found: %s" % args.cfg_file)

    mapping = None
    if args.build_map:
        mapping = TrajectoryMap(voxel_size=args.map_voxel,
                                max_points=args.map_max_points)

    hub = FrameHub(source="ros" if args.ros else "bag")
    server = WebViewerServer(hub, host=args.host, port=args.port,
                            webapp_dir=args.webapp_dir or DEFAULT_WEBAPP_DIR)
    try:
        server.start()
    except OSError as exc:
        raise SystemExit("cannot listen on %s:%d (%s); try --port 8081"
                         % (args.host, args.port, exc))

    print("", flush=True)
    print("==============================================", flush=True)
    print("   POINTPILLARS WEB VIEWER  (three.js)", flush=True)
    print("==============================================", flush=True)
    print("  open:  %s" % server.url, flush=True)
    print("  source: %s" % ("ROS topics" if args.ros else "bag replay"), flush=True)
    if mapping is not None:
        print("  map:   on (%.2f m voxels, max %d pts; inference is per frame)"
              % (args.map_voxel, args.map_max_points), flush=True)
    else:
        print("  map:   off (single-frame view)", flush=True)
    print("  rings: 5 m steps, cloud colourised by range", flush=True)
    if not args.ros and args.loop:
        print("  replay: loop until Ctrl+C", flush=True)
    if not args.ros and args.infer:
        print("  note:  the first frame takes ~5-10 s on CPU; the page "
              "stays on its loading overlay meanwhile", flush=True)
    print("==============================================", flush=True)
    print("", flush=True)

    holder = {}
    if args.share:
        print("[web-view] opening a Cloudflare quick tunnel in the background "
              "(localhost works without it) ...", flush=True)
        start_share(args.port, args.host, args.cloudflared, holder)
    if args.open_browser and open_browser(server.url):
        print("[web-view] opened a browser window", flush=True)

    stop_event = threading.Event()
    try:
        if args.ros:
            try:
                run_ros(args, hub, stop_event, mapping)
            except Exception as exc:  # roscore missing, bad topic, ...
                print("[web-view] ROS mode failed: %s" % exc, file=sys.stderr)
                return 2
        else:
            run_bag(args, hub, stop_event, mapping)
    except ConfigError as exc:
        print("[web-view] configuration error: %s" % exc, file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print("[web-view] error: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        tunnel = holder.get("tunnel")
        if tunnel is not None:
            try:
                tunnel.stop()
            except Exception:
                pass
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
