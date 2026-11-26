"""ROS1 message helpers (no ros_numpy dependency).

ROS imports are kept local so the inference CLI can run without a ROS master.
The PointCloud2 parser handles padded rows and missing intensity fields.
"""

import copy
import math
import numpy as np

from .postprocess import boxes_to_corners_3d

_POINT_FIELD_DTYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2",
    5: "i4", 6: "u4", 7: "f4", 8: "f8",
}
_INTENSITY_NAMES = ("intensity", "i", "reflectivity", "reflectance", "intensities")

# Upstream livox_detection colour map (tools/vis_ros.py), extended with KITTI names.
_COLORS = {
    "car": (0.0, 1.0, 1.0), "Car": (0.0, 1.0, 1.0),
    "vehicle": (0.0, 1.0, 1.0), "truck": (0.0, 1.0, 1.0),
    "pedestrian": (1.0, 1.0, 0.0), "Pedestrian": (1.0, 1.0, 0.0),
    "cyclist": (0.0, 1.0, 0.0), "Cyclist": (0.0, 1.0, 0.0),
    "bicycle": (0.0, 1.0, 0.0), "barrier": (1.0, 1.0, 1.0),
    "traffic_cone": (1.0, 1.0, 1.0),
}
_FALLBACK_COLORS = [(0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                    (1.0, 0.5, 0.0), (1.0, 0.0, 1.0), (0.5, 0.5, 1.0)]

_BOX_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def pointcloud2_to_numpy(msg):
    """Convert PointCloud2 -> (points[N,4] float32, has_intensity, field_names).

    The 4th column is 0.0 when no intensity-like field exists; ``has_intensity``
    lets the caller report this explicitly instead of fabricating a feature
    silently.  Non-finite points are dropped.
    """
    fields = list(msg.fields or [])
    by_name = {}
    for field in fields:
        if field.name not in by_name:
            by_name[field.name] = field
    if not all(name in by_name for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 missing x/y/z (fields: %s)"
                         % [f.name for f in fields])

    intensity_name = None
    for name in _INTENSITY_NAMES:
        if name in by_name:
            intensity_name = name
            break
    names = ["x", "y", "z"] + ([intensity_name] if intensity_name else [])
    endian = ">" if msg.is_bigendian else "<"
    formats, offsets = [], []
    for name in names:
        field = by_name[name]
        base = _POINT_FIELD_DTYPES.get(int(field.datatype))
        if base is None:
            raise ValueError("unsupported PointField datatype %s for '%s'"
                             % (field.datatype, name))
        formats.append(endian + base if int(field.count) <= 1 else
                       (endian + base, int(field.count)))
        offsets.append(int(field.offset))

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(msg.point_step),
    })
    width, height = int(msg.width), int(msg.height)
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if height > 1 and row_step != width * point_step:
        rows = raw[:height * row_step].reshape(height, row_step)
        raw = rows[:, :width * point_step].copy().reshape(-1)
    elif height > 1:
        raw = raw[:height * row_step]
    arr = np.frombuffer(raw.tobytes(), dtype=dtype, count=width * height)

    points = np.empty((arr.shape[0], 4), dtype=np.float32)
    points[:, 0] = arr["x"]
    points[:, 1] = arr["y"]
    points[:, 2] = arr["z"]
    if intensity_name:
        val = arr[intensity_name]
        points[:, 3] = val[:, 0] if val.ndim > 1 else val
    else:
        points[:, 3] = 0.0
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
    return points, intensity_name is not None, [f.name for f in fields]


def make_pointcloud2(points, header):
    """Create a PointCloud2 with x/y/z/intensity from an N x 4 array."""
    from sensor_msgs.msg import PointCloud2, PointField

    pts = np.asarray(points, dtype=np.float32)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np.ascontiguousarray(pts).tobytes()
    return msg


def _yaw_quaternion(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x, q.y = 0.0, 0.0
    q.z = math.sin(float(yaw) * 0.5)
    q.w = math.cos(float(yaw) * 0.5)
    return q


def build_marker_array(boxes, scores, labels, class_names, header, line_width=0.06):
    """MarkerArray with one wireframe box + heading arrow + label per object.

    Marker IDs are unique (3 per detection).  Reuses the upstream topics and
    marker conventions (/detect_box3d, /text_det) so the original RViz setup
    keeps working.
    """
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    if boxes is None or len(boxes) == 0:
        return arr

    corners = boxes_to_corners_3d(boxes)
    for i in range(corners.shape[0]):
        label = int(labels[i]) - 1
        color = _COLORS.get(class_names[label] if 0 <= label < len(class_names) else "",
                            _FALLBACK_COLORS[label % len(_FALLBACK_COLORS)])
        name = class_names[label] if 0 <= label < len(class_names) else "obj"

        box_marker = Marker()
        box_marker.header = header
        box_marker.ns = "pointpillars_boxes"
        box_marker.id = i * 3
        box_marker.type = Marker.LINE_LIST
        box_marker.action = Marker.ADD
        box_marker.pose.orientation.w = 1.0
        box_marker.scale.x = line_width
        box_marker.color.r, box_marker.color.g, box_marker.color.b = color
        box_marker.color.a = 0.9
        box_marker.lifetime = rospy_duration()
        for a, b in _BOX_LINES:
            for idx in (a, b):
                pt = corners[i, idx]
                box_marker.points.append(Point(float(pt[0]), float(pt[1]), float(pt[2])))
        arr.markers.append(box_marker)

        arrow = Marker()
        arrow.header = header
        arrow.ns = "pointpillars_heading"
        arrow.id = i * 3 + 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = float(boxes[i, 0])
        arrow.pose.position.y = float(boxes[i, 1])
        arrow.pose.position.z = float(boxes[i, 2])
        arrow.pose.orientation = _yaw_quaternion(boxes[i, 6])
        arrow.scale.x = max(0.5, float(boxes[i, 3]) * 0.8)
        arrow.scale.y = 0.12
        arrow.scale.z = 0.12
        arrow.color.r, arrow.color.g, arrow.color.b = color
        arrow.color.a = 0.8
        arrow.lifetime = rospy_duration()
        arr.markers.append(arrow)

        text = Marker()
        text.header = header
        text.ns = "pointpillars_labels"
        text.id = i * 3 + 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(boxes[i, 0])
        text.pose.position.y = float(boxes[i, 1])
        text.pose.position.z = float(boxes[i, 2] + boxes[i, 5] * 0.5 + 0.5)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.6
        text.color.r, text.color.g, text.color.b = color
        text.color.a = 1.0
        x, y, z, yaw = (float(boxes[i, 0]), float(boxes[i, 1]),
                        float(boxes[i, 2]), float(boxes[i, 6]))
        text.text = "%s %.2f  [%.1f, %.1f, %.1f] yaw=%.2f" % (
            name, float(scores[i]), x, y, z, yaw)
        text.lifetime = rospy_duration()
        arr.markers.append(text)
    return arr


def build_marker_arrays(boxes, scores, labels, class_names, header, line_width=0.06):
    """Return (boxes/arrows MarkerArray, text MarkerArray).

    Matching upstream ``tools/vis_ros.py``, wireframes + headings go to
    ``/detect_box3d`` and labels go to ``/text_det``.  Each array carries its
    own DELETEALL marker so RViz does not retain stale objects.
    """
    from visualization_msgs.msg import MarkerArray

    combined = build_marker_array(boxes, scores, labels, class_names, header, line_width)
    boxes_arr, text_arr = MarkerArray(), MarkerArray()
    boxes_arr.markers.append(copy.copy(combined.markers[0]))
    text_arr.markers.append(copy.copy(combined.markers[0]))
    for marker in combined.markers[1:]:
        if marker.ns == "pointpillars_labels":
            text_arr.markers.append(marker)
        else:
            boxes_arr.markers.append(marker)
    return boxes_arr, text_arr


def rospy_duration(seconds=0):
    """Zero duration = markers persist until explicitly deleted."""
    import rospy
    return rospy.Duration(seconds)
