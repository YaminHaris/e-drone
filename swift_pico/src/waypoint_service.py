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
TASK1A_REQUIRED_IDS = {80, 85, 90, 95}


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


def grid_distance(left, right):
    """Euclidean distance between labelled grid intersections."""
    lx, ly = label_to_index(left)
    rx, ry = label_to_index(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def best_visit_order(survivors):
    """Exact TSP path (not a greedy approximation) for the small survivor set."""
    if not survivors:
        return []
    return list(min(
        itertools.permutations(sorted(set(survivors))),
        key=lambda order: grid_distance(START_LABEL, order[0]) +
        sum(grid_distance(order[i], order[i + 1]) for i in range(len(order) - 1))))


def append_grid_segment(route, start, finish, survivor_labels):
    """Append the shortest 8-connected grid segment.

    Each generated point is a grid intersection.  A diagonal step is allowed
    when both coordinates change, which produces the sparse, natural route
    shown in the task figure instead of an arbitrary L-shaped detour.
    """
    column, row = label_to_index(start)
    finish_column, finish_row = label_to_index(finish)
    while column != finish_column or row != finish_row:
        if column != finish_column:
            column += 1 if finish_column > column else -1
        if row != finish_row:
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


def task1a_detect_aruco(gray):
    """Task 1A's version-compatible 4x4/250 ArUco detection."""
    if hasattr(cv2.aruco, 'ArucoDetector'):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return detector.detectMarkers(gray)[:2]
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def task1a_arena_corners(corners, ids):
    """Task 1A's inner-corner selection, ordered TL, TR, BR, BL."""
    if ids is None:
        raise ValueError('No marker detected')
    marker_ids = ids.flatten().astype(int)
    if not TASK1A_REQUIRED_IDS.issubset(set(marker_ids)):
        raise ValueError('One or More Ids are missing')
    marker_corners = {marker_id: corners[index][0]
                      for index, marker_id in enumerate(marker_ids)}
    marker_centres = {marker_id: np.mean(points, axis=0)
                      for marker_id, points in marker_corners.items()}
    arena_middle = np.mean(list(marker_centres.values()), axis=0)
    inner_corner = {
        marker_id: points[np.argmin([np.linalg.norm(point - arena_middle) for point in points])]
        for marker_id, points in marker_corners.items()
    }
    by_y = sorted(marker_centres.items(), key=lambda item: item[1][1])
    top = sorted(by_y[:2], key=lambda item: item[1][0])
    bottom = sorted(by_y[2:], key=lambda item: item[1][0])
    tl_id, tr_id = top[0][0], top[1][0]
    bl_id, br_id = bottom[0][0], bottom[1][0]
    return np.float32([inner_corner[tl_id], inner_corner[tr_id],
                       inner_corner[br_id], inner_corner[bl_id]])


def task1a_nearest_grid_label(cx, cy):
    """Task 1A's nearest-intersection mapping on its 900 px rectified arena."""
    column = int(np.clip(round(cx / 75.0), 1, 11))
    row = int(np.clip(round(cy / 75.0), 1, 11))
    return f"{chr(ord('A') + column - 1)}{row}"


def task1a_contour_centres(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width > 0 and height > 0:
            centres.append((x + width / 2.0, y + height / 2.0))
    return centres


def task1a_critical_survivors(image):
    """Embedded Task 1A pipeline: rectify, segment red, label intersections."""
    if image is None or image.size == 0:
        raise ValueError('Could not read image')
    corners, ids = task1a_detect_aruco(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    source = task1a_arena_corners(corners, ids)
    destination = np.float32([[0, 0], [900, 0], [900, 900], [0, 900]])
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (900, 900))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 150, 100]), np.array([179, 255, 255])))
    return sorted({task1a_nearest_grid_label(x, y)
                   for x, y in task1a_contour_centres(red_mask)})


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
            labels = task1a_critical_survivors(image_to_bgr(message))
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
