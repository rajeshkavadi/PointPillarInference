#!/usr/bin/env python3
"""Extract one LiDAR frame from a ROS bag to .npy (+ .json metadata).

The .npy file is N x 4 float32: x, y, z, and either real intensity (when the
PointCloud2 has an intensity-like field) or 0.0.  The sidecar JSON records the
original topic, frame_id, stamp and field names so nothing is silently lost.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from pointpillars.bag_frame import find_cloud_topic, inspect_bag, read_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Extract a PointCloud2 frame from a ROS bag")
    parser.add_argument("bag", help="ROS bag file")
    parser.add_argument("--frame", type=int, default=0, help="zero-based message index")
    parser.add_argument("--topic", default=None, help="PointCloud2 topic (auto-detected if omitted)")
    parser.add_argument("--out", default=None, help="output .npy path (default: <bag>_frame<N>.npy)")
    parser.add_argument("--list-topics", action="store_true", help="list topics and exit")
    args = parser.parse_args()

    try:
        info = inspect_bag(args.bag)
    except Exception as exc:
        print("BAG ERROR: %s" % exc)
        return 2

    print("bag: %s" % os.path.abspath(args.bag))
    for name, meta in info["topics"].items():
        print("  %-40s %s (%d msgs)" % (name, meta["msg_type"], meta["messages"]))
    if args.list_topics:
        return 0

    topic = args.topic or find_cloud_topic(args.bag)[0]
    try:
        frame = read_frame(args.bag, args.frame, topic)
    except RuntimeError as exc:
        print("BAG ERROR: %s" % exc)
        return 2

    out = args.out or (os.path.splitext(os.path.abspath(args.bag))[0]
                       + "_frame%d.npy" % args.frame)
    out = os.path.abspath(os.path.expanduser(out))
    folder = os.path.dirname(out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    np.save(out, frame["points"].astype(np.float32))
    sidecar = os.path.splitext(out)[0] + ".json"
    meta = {
        "index": frame["index"],
        "topic": frame["topic"],
        "frame_id": frame["frame_id"],
        "stamp": frame["stamp"],
        "fields": frame["fields"],
        "has_intensity": frame["has_intensity"],
    }
    with open(sidecar, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote %s (%d points)" % (out, len(frame["points"])))
    print("wrote %s" % sidecar)
    if not frame["has_intensity"]:
        print("NOTE: this frame has no intensity field; the 4th column is 0.0 in the .npy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
