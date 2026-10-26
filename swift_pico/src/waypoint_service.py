#!/usr/bin/env python3

"""
Task 2C waypoint service.

Receives an arena image, detects critical survivors, plans a route
through the grid, and provides the resulting waypoints to the client.
"""

import json
import itertools
import threading

import cv2
import numpy as np
import rclpy

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from waypoint_navigation.srv import GetWaypoints


# The arena has columns A-K and rows 1-11.
GRID_COLUMNS = "ABCDEFGHIJK"
START_LABEL = "F6"
ALTITUDE = 13.0

# These four ArUco markers identify the corners of the arena.
REQUIRED_MARKER_IDS = {80, 85, 90, 95}


def make_pose(x, y, z=ALTITUDE):
    """Create a ROS pose at the given coordinates."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def label_to_index(label):
    """Convert a grid label such as F6 into zero-based indices."""
    column = GRID_COLUMNS.index(label[0])
    row = int(label[1:]) - 1
    return column, row


def index_to_label(column, row):
    """Convert grid indices back into a label such as F6."""
    return f"{GRID_COLUMNS[column]}{row + 1}"


def grid_distance(start, end):
    """Calculate the distance between two grid intersections."""
    x1, y1 = label_to_index(start)
    x2, y2 = label_to_index(end)

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def best_visit_order(survivors):
    """Find the shortest visiting order for the detected survivors."""
    if not survivors:
        return []

    survivors = sorted(set(survivors))

    return list(min(
        itertools.permutations(survivors),
        key=lambda order: (
            grid_distance(START_LABEL, order[0])
            + sum(
                grid_distance(order[i], order[i + 1])
                for i in range(len(order) - 1)
            )
        ),
    ))


def append_grid_segment(route, start, end, survivors):
    """Add grid intersections along a diagonal or straight path."""
    column, row = label_to_index(start)
    end_column, end_row = label_to_index(end)

    while column != end_column or row != end_row:
        if column != end_column:
            column += 1 if end_column > column else -1

        if row != end_row:
            row += 1 if end_row > row else -1

        label = index_to_label(column, row)
        route.append((label, label in survivors))


def make_grid_route(survivors):
    """
    Build the complete route from F6 through the survivors.

    Each entry contains a grid label and whether the drone should hover
    there. Survivor locations require a hover; transit locations do not.
    """
    visit_order = best_visit_order(survivors)
    route = []
    current = START_LABEL
    survivor_set = set(visit_order)

    for target in visit_order:
        append_grid_segment(
            route, current, target, survivor_set
        )
        current = target

    # Remove consecutive duplicate locations while preserving hover flags.
    compact_route = []

    for label, is_survivor in route:
        if compact_route and compact_route[-1][0] == label:
            previous_label, previous_hover = compact_route[-1]
            compact_route[-1] = (
                previous_label,
                previous_hover or is_survivor,
            )
        else:
            compact_route.append((label, is_survivor))

    return compact_route


def detect_aruco_markers(gray_image):
    """Detect the arena markers across supported OpenCV versions."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            cv2.aruco.DetectorParameters(),
        )
        corners, ids, _ = detector.detectMarkers(gray_image)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray_image,
            dictionary,
            parameters=parameters,
        )

    return corners, ids


def find_arena_corners(corners, ids):
    """Find the four inner arena corners in TL, TR, BR, BL order."""
    if ids is None:
        raise ValueError("No ArUco markers detected")

    marker_ids = ids.flatten().astype(int)

    if not REQUIRED_MARKER_IDS.issubset(set(marker_ids)):
        raise ValueError("One or more required arena markers are missing")

    marker_corners = {
        marker_id: corners[i][0]
        for i, marker_id in enumerate(marker_ids)
    }

    # Find the centre of each marker and the centre of the arena.
    centres = {
        marker_id: np.mean(points, axis=0)
        for marker_id, points in marker_corners.items()
    }
    arena_centre = np.mean(list(centres.values()), axis=0)

    # The corner closest to the arena centre is the inner corner.
    inner_corners = {
        marker_id: points[
            np.argmin([
                np.linalg.norm(point - arena_centre)
                for point in points
            ])
        ]
        for marker_id, points in marker_corners.items()
    }

    # Sort markers by vertical position, then horizontal position.
    sorted_by_y = sorted(centres.items(), key=lambda item: item[1][1])
    top = sorted(sorted_by_y[:2], key=lambda item: item[1][0])
    bottom = sorted(sorted_by_y[2:], key=lambda item: item[1][0])

    top_left = top[0][0]
    top_right = top[1][0]
    bottom_left = bottom[0][0]
    bottom_right = bottom[1][0]

    return np.float32([
        inner_corners[top_left],
        inner_corners[top_right],
        inner_corners[bottom_right],
        inner_corners[bottom_left],
    ])


def nearest_grid_label(x, y):
    """Map a point in the rectified image to its nearest grid intersection."""
    column = int(np.clip(round(x / 75.0), 1, 11))
    row = int(np.clip(round(y / 75.0), 1, 11))

    return f"{chr(ord('A') + column - 1)}{row}"


def find_contour_centres(mask):
    """Find the centres of sufficiently large red regions."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    centres = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width > 0 and height > 0:
            centres.append((
                x + width / 2.0,
                y + height / 2.0,
            ))

    return centres


def detect_critical_survivors(image):
    """Rectify the arena image and locate red survivors on the grid."""
    if image is None or image.size == 0:
        raise ValueError("Received an empty image")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_markers(gray_image)
    source_corners = find_arena_corners(corners, ids)

    # Transform the arena into a fixed 900 x 900 image.
    destination_corners = np.float32([
        [0, 0],
        [900, 0],
        [900, 900],
        [0, 900],
    ])

    transform = cv2.getPerspectiveTransform(
        source_corners,
        destination_corners,
    )
    warped = cv2.warpPerspective(image, transform, (900, 900))

    # Red is split into two ranges because it wraps around the HSV hue scale.
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(
        hsv,
        np.array([0, 150, 100]),
        np.array([10, 255, 255]),
    )
    upper_red = cv2.inRange(
        hsv,
        np.array([170, 150, 100]),
        np.array([179, 255, 255]),
    )

    red_mask = cv2.bitwise_or(lower_red, upper_red)

    return sorted({
        nearest_grid_label(x, y)
        for x, y in find_contour_centres(red_mask)
    })


def image_to_bgr(message):
    """Convert a ROS image message into an OpenCV BGR image."""
    height = message.height
    width = message.width
    encoding = message.encoding.lower()

    if height == 0 or width == 0:
        raise ValueError("Received an empty ROS image")

    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }.get(encoding)

    if channels is None:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    raw = raw.reshape(height, message.step)
    image = raw[:, :width * channels].reshape(
        height, width, channels
    )

    if encoding == "bgr8":
        return image.copy()

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class WayPoints(Node):

    def __init__(self):
        super().__init__("waypoints_service")

        # The route is planned once an arena image arrives.
        self.route_condition = threading.Condition()
        self.route = None
        self.marker_ids = None
        self.planning = False

        callback_group = ReentrantCallbackGroup()

        # The simulator provides the arena image on this topic.
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            image_qos,
            callback_group=callback_group,
        )

        # The client requests the planned route through this service.
        self.service = self.create_service(
            GetWaypoints,
            "waypoints",
            self.waypoint_callback,
            callback_group=callback_group,
        )

        self.get_logger().info(
            "Waiting for the arena image to plan the Task 2C route"
        )

    def image_callback(self, message):
        # Ignore duplicate frames while a route is being planned or is ready.
        with self.route_condition:
            if self.route is not None or self.planning:
                return

            self.planning = True

        try:
            image = image_to_bgr(message)
            survivors = detect_critical_survivors(image)

            if not survivors:
                raise ValueError("No critical survivors detected")

            grid_route = make_grid_route(survivors)

            # Convert grid labels into real WhyCode coordinates.
            package_path = get_package_share_directory("swift_pico")
            coordinates_path = (
                package_path + "/scripts/grid_whycode_coordinates.json"
            )

            with open(coordinates_path, encoding="utf-8") as file:
                coordinates = json.load(file)

            poses = [
                make_pose(
                    coordinates[label]["x"],
                    coordinates[label]["y"],
                )
                for label, _ in grid_route
            ]

            # The client uses these IDs to decide whether to hover.
            marker_ids = [
                ("survivor:" if is_survivor else "transit:") + label
                for label, is_survivor in grid_route
            ]

        except Exception as error:
            with self.route_condition:
                self.planning = False
                self.route_condition.notify_all()

            self.get_logger().error(f"Route planning failed: {error}")
            return

        with self.route_condition:
            self.route = poses
            self.marker_ids = marker_ids
            self.planning = False
            self.route_condition.notify_all()

        self.get_logger().info(
            f"Detected survivors: {survivors}. "
            f"Planned {len(poses)} waypoints from {START_LABEL}."
        )

        # The image is no longer needed after planning succeeds.
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            response.waypoints = PoseArray()
            response.marker_ids = []
            return response

        # If the client asks early, wait for image processing to finish.
        with self.route_condition:
            while self.route is None and rclpy.ok():
                self.route_condition.wait(timeout=0.25)

            poses = [] if self.route is None else list(self.route)
            marker_ids = (
                [] if self.marker_ids is None
                else list(self.marker_ids)
            )

        response.waypoints = PoseArray()
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.waypoints.header.frame_id = "whycode"
        response.waypoints.poses = poses
        response.marker_ids = marker_ids

        self.get_logger().info(
            f"Returning {len(poses)} planned waypoints"
        )

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


if __name__ == "__main__":
    main()
