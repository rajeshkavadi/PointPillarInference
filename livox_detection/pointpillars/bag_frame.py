"""Read PointCloud2 frames from a ROS1 bag using Python 3.

ROS Melodic's rosbag is a Python 2 package, but it imports under Python 3 once
``pycryptodomex``, ``python-gnupg`` and ``rospkg`` are installed.  Bags shipped
with livox_detection are uncompressed, so no LZ4 support is needed.
"""

import os

import numpy as np

from .ros_utils import pointcloud2_to_numpy


def inspect_bag(bag_path):
    """Return topic information for a bag (verified topic name before use)."""
    import rosbag
    bag = rosbag.Bag(os.path.expanduser(str(bag_path)), "r")
    try:
        info = bag.get_type_and_topic_info()
        topics = {}
        for name, meta in info.topics.items():
            topics[name] = {
                "msg_type": meta.msg_type,
                "messages": int(meta.message_count),
            }
        return {
            "topics": topics,
            "duration": bag.get_end_time() - bag.get_start_time(),
            "start_time": bag.get_start_time(),
            "end_time": bag.get_end_time(),
        }
    finally:
        bag.close()


def find_cloud_topic(bag_path, preferred="/livox/lidar"):
    info = inspect_bag(bag_path)
    topics = info["topics"]
    if preferred in topics and topics[preferred]["msg_type"] == "sensor_msgs/PointCloud2":
        return preferred, info
    clouds = [name for name, meta in topics.items()
              if meta["msg_type"] == "sensor_msgs/PointCloud2"]
    if not clouds:
        raise RuntimeError("no sensor_msgs/PointCloud2 topic in %s; topics=%s"
                           % (bag_path, list(topics)))
    return clouds[0], info


def read_frame(bag_path, frame_index=0, topic=None):
    """Read one frame.

    Returns a dict with:
        points: N x 4 float32 (4th column zero if bag has no intensity)
        has_intensity: bool
        fields: list of PointField names in the message
        frame_id, stamp, topic, index
    """
    import rosbag
    bag_path = os.path.expanduser(str(bag_path))
    if not os.path.isfile(bag_path):
        raise RuntimeError("bag not found: %s" % bag_path)
    topic = topic or find_cloud_topic(bag_path)[0]
    bag = rosbag.Bag(bag_path, "r")
    try:
        for index, (msg_topic, msg, stamp) in enumerate(bag.read_messages(topics=[topic])):
            if index != frame_index:
                continue
            points, has_intensity, fields = pointcloud2_to_numpy(msg)
            return {
                "points": points,
                "has_intensity": has_intensity,
                "fields": fields,
                "frame_id": msg.header.frame_id,
                "stamp": stamp.to_sec(),
                "topic": msg_topic,
                "index": index,
                "width": int(msg.width),
                "height": int(msg.height),
            }
    finally:
        bag.close()
    raise RuntimeError("frame %d not found on topic %s in %s"
                       % (frame_index, topic, bag_path))


def iter_frames(bag_path, topic=None, max_frames=0):
    """Yield the same dicts as read_frame for every frame in the bag."""
    import rosbag
    bag_path = os.path.expanduser(str(bag_path))
    topic = topic or find_cloud_topic(bag_path)[0]
    bag = rosbag.Bag(bag_path, "r")
    try:
        for index, (msg_topic, msg, stamp) in enumerate(bag.read_messages(topics=[topic])):
            if max_frames and index >= max_frames:
                break
            points, has_intensity, fields = pointcloud2_to_numpy(msg)
            yield {
                "points": points,
                "has_intensity": has_intensity,
                "fields": fields,
                "frame_id": msg.header.frame_id,
                "stamp": stamp.to_sec(),
                "topic": msg_topic,
                "index": index,
                "width": int(msg.width),
                "height": int(msg.height),
            }
    finally:
        bag.close()
