#!/usr/bin/env python3
"""Task 2 waypoint provider.

Default ``auto`` mode consumes the one transient-local arena frame, detects red
triangular (critical) survivors, maps them to grid intersections and plans a
minimum-length Manhattan route from F6.  ``task2b`` is deliberately an explicit
compatibility mode for validating the prescribed Task 2B route:

    ros2 run swift_pico waypoint_service --ros-args -p route_mode:=task2b
"""

import itertools
import json
import math
import threading

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from waypoint_navigation.srv import GetWaypoints


GRID_COLUMNS = 'ABCDEFGHIJK'
GRID_ROWS = tuple(range(1, 12))
START_LABEL = 'F6'
ALTITUDE = 13.0
TASK2B_WAYPOINTS = (
    ('wp1', -2.41, -2.41),
    ('wp2', 2.41, 1.20),
    ('wp3', 1.20, -2.41),
    ('wp4', -1.20, 1.20),
)


def make_pose(x, y, z=ALTITUDE):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def label_to_index(label):
    return GRID_COLUMNS.index(label[0]), int(label[1:]) - 1


def index_to_label(column, row):
    return f'{GRID_COLUMNS[column]}{row + 1}'


def manhattan(left, right):
    lx, ly = label_to_index(left)
    rx, ry = label_to_index(right)
    return abs(lx - rx) + abs(ly - ry)


def best_visit_order(survivors):
    """Exact TSP path (not a greedy approximation) for the small survivor set."""
    if not survivors:
        return []
    return list(min(
        itertools.permutations(sorted(set(survivors))),
        key=lambda order: manhattan(START_LABEL, order[0]) +
        sum(manhattan(order[i], order[i + 1]) for i in range(len(order) - 1))))


def append_grid_segment(route, start, finish, survivor_labels):
    """Append a Manhattan segment; x/column is traversed before y/row."""
    column, row = label_to_index(start)
    finish_column, finish_row = label_to_index(finish)
    while column != finish_column:
        column += 1 if finish_column > column else -1
        label = index_to_label(column, row)
        route.append((label, label in survivor_labels))
    while row != finish_row:
        row += 1 if finish_row > row else -1
        label = index_to_label(column, row)
        route.append((label, label in survivor_labels))


def make_grid_route(survivors):
    """Return ``(grid-label, hover)`` pairs, including non-hover intersections."""
    order = best_visit_order(survivors)
    route = []
    current = START_LABEL
    survivor_set = set(order)
    for target in order:
        append_grid_segment(route, current, target, survivor_set)
        current = target
    # If a future survivor was crossed while travelling, the final arrival has
    # still to be a hover waypoint.  Coalesce only adjacent duplicates.
    compact = []
    for label, is_survivor in route:
        if compact and compact[-1][0] == label:
            compact[-1] = (label, compact[-1][1] or is_survivor)
        else:
            compact.append((label, is_survivor))
    return compact


def _cluster(values, separation=4.0):
    clusters = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > separation:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(cluster)) for cluster in clusters]


def _arena_grid_bounds(image):
    """Infer the outer 12-cell grid rectangle from long Hough line segments."""
    height, width = image.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                            minLineLength=int(min(height, width) * 0.45), maxLineGap=16)
    if lines is None:
        raise ValueError('arena grid lines were not detected')
    vertical = []
    horizontal = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x1 - x2) <= 4 and abs(y2 - y1) >= height * 0.45:
            vertical.append((x1 + x2) / 2.0)
        if abs(y1 - y2) <= 4 and abs(x2 - x1) >= width * 0.45:
            horizontal.append((y1 + y2) / 2.0)
    vertical = _cluster(vertical)
    horizontal = _cluster(horizontal)
    if len(vertical) < 2 or len(horizontal) < 2:
        raise ValueError('insufficient arena grid lines')
    # The outside grid border is always the widest pair.  The inner eleven
    # intersections are equally spaced at 1/12 .. 11/12 of this rectangle.
    left, right = min(vertical), max(vertical)
    top, bottom = min(horizontal), max(horizontal)
    if right - left < width * 0.35 or bottom - top < height * 0.35:
        raise ValueError('invalid arena grid geometry')
    return left, top, right, bottom


def detect_critical_labels(image):
    """Return nearest grid labels for red triangular critical-survivor tokens.

    This performs detection from the supplied frame; it does not contain arena
    positions or survivor labels.  HSV avoids treating the blue/green terrain
    texture as a survivor, and the polygon test separates critical triangles
    from yellow circular stable survivors.
    """
    if image is None or image.size == 0:
        raise ValueError('empty arena image')
    left, top, right, bottom = _arena_grid_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 105, 70), (12, 255, 255))
    red |= cv2.inRange(hsv, (165, 105, 70), (179, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(vertices) != 3:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        x = moments['m10'] / moments['m00']
        y = moments['m01'] / moments['m00']
        if not (left <= x <= right and top <= y <= bottom):
            continue
        # A..K occupy the inner 11 intersections of a 12-cell grid.
        column = int(round((x - (left + (right - left) / 12.0)) / ((right - left) / 12.0)))
        row = int(round((y - (top + (bottom - top) / 12.0)) / ((bottom - top) / 12.0)))
        if 0 <= column < len(GRID_COLUMNS) and 0 <= row < len(GRID_ROWS):
            labels.append(index_to_label(column, row))
    return sorted(set(labels), key=label_to_index)


def image_to_bgr(message):
    height, width = message.height, message.width
    if height == 0 or width == 0:
        raise ValueError('received empty ROS image')
    encoding = message.encoding.lower()
    channels = 3 if encoding in ('bgr8', 'rgb8') else 4 if encoding in ('bgra8', 'rgba8') else 1
    raw = np.frombuffer(message.data, dtype=np.uint8).reshape(height, message.step)
    image = raw[:, :width * channels].reshape(height, width, channels)
    if encoding == 'bgr8':
        return image.copy()
    if encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == 'bgra8':
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ('mono8', '8uc1'):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported image encoding: {message.encoding}')


class WayPoints(Node):
    def __init__(self):
        super().__init__('waypoints_service')
        self._route_condition = threading.Condition()
        self._route = None
        self._route_ids = None
        self._planning = False
        self.image_sub = None
        self.route_mode = self.declare_parameter('route_mode', 'auto').value
        group = ReentrantCallbackGroup()
        self.service = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback,
                                           callback_group=group)
        if self.route_mode == 'task2b':
            self._set_task2b_route()
        elif self.route_mode == 'auto':
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos,
                                                      callback_group=group)
            self.get_logger().info('Waiting for the latched /image_raw arena frame')
        else:
            raise ValueError("route_mode must be 'auto' or 'task2b'")

    def _set_task2b_route(self):
        with self._route_condition:
            self._route = [make_pose(x, y) for _, x, y in TASK2B_WAYPOINTS]
            self._route_ids = [name for name, _, _ in TASK2B_WAYPOINTS]
            self._route_condition.notify_all()
        self.get_logger().info('Task 2B route loaded: wp1 -> wp2 -> wp3 -> wp4')

    def image_callback(self, message):
        # A transient-local publisher can deliver once immediately after this
        # subscriber is created.  Ignore a duplicate callback after planning.
        with self._route_condition:
            if self._route is not None or self._planning:
                return
            self._planning = True
        try:
            labels = detect_critical_labels(image_to_bgr(message))
            if not labels:
                raise ValueError('no critical triangular survivors detected')
            route = make_grid_route(labels)
            coordinates_path = get_package_share_directory('swift_pico') + '/scripts/grid_whycode_coordinates.json'
            with open(coordinates_path, encoding='utf-8') as coordinate_file:
                coordinates = json.load(coordinate_file)
            poses = [make_pose(coordinates[label]['x'], coordinates[label]['y']) for label, _ in route]
            marker_ids = [('survivor:' if hover else 'transit:') + label for label, hover in route]
        except Exception as error:
            with self._route_condition:
                self._planning = False
                self._route_condition.notify_all()
            self.get_logger().error(f'Could not plan route from arena frame: {error}')
            return

        with self._route_condition:
            self._route = poses
            self._route_ids = marker_ids
            self._planning = False
            self._route_condition.notify_all()
        self.get_logger().info(
            f'Detected critical survivors {labels}; planned {len(poses)} grid waypoints from {START_LABEL}')
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            response.waypoints = PoseArray()
            response.marker_ids = []
            return response
        # Do not answer early with an empty route.  A multi-threaded executor
        # keeps the image callback alive while the service request waits.
        with self._route_condition:
            while self._route is None and rclpy.ok():
                self._route_condition.wait(timeout=0.25)
            poses = [] if self._route is None else list(self._route)
            marker_ids = [] if self._route_ids is None else list(self._route_ids)
        response.waypoints = PoseArray()
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.waypoints.header.frame_id = 'whycode'
        response.waypoints.poses = poses
        response.marker_ids = marker_ids
        self.get_logger().info(f'Returning {len(poses)} planned waypoints')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WayPoints()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
