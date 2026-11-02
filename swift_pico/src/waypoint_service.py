#!/usr/bin/env python3

"""
Task 3A waypoint service.

From one arena image this node finds the critical survivors, the stable
survivors and the obstacles, then plans routes that stay clear of the
obstacles. It offers two services:

    waypoints     the critical route, starting from F6
    stable_route  the stable route from the drone's position, ending at F6
"""

import heapq
import itertools
import json
import math
import os
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
from waypoint_navigation.srv import GetStableRoute, GetWaypoints


HOME_LABEL = "F6"
FLIGHT_Z = 13.0
# Z is measured down from the camera, so 15 is the ground. A goal at this
# height tells pico_server to hover at FLIGHT_Z and then land.
LANDING_Z = 15.0

# The four ArUco markers at the corners of the arena.
REQUIRED_MARKER_IDS = {80, 85, 90, 95}

# The rectified arena image: 900 x 900 pixels, 75 pixels per grid square.
ARENA_SIZE = 900
GRID_SPACING_PX = 75.0
# A1 is at about x = y = 6.12 and sits 375 pixels from the image centre.
PIXELS_PER_UNIT = 375.0 / 6.12

# An obstacle must be bigger than the drone, which is also dark and sits on
# F6 in the camera frame. The cubes are about 18000 square pixels.
MIN_OBSTACLE_AREA_PX = 4000

# Obstacles are nearly black. Dark trees start just above this brightness.
OBSTACLE_MAX_BRIGHTNESS = 35

# How far to stay from an obstacle, in pixels (61 pixels is 1 unit). The drone
# reaches 0.33 units from its centre, can cut a corner by 0.56 and may drift.
CLEARANCE_PX = 75

# A survivor can sit closer to an obstacle than that. Only within this distance
# of the start or end of a path may the drone use the smaller clearance.
TIGHT_CLEARANCE_PX = 35
TIGHT_ZONE_RADIUS_PX = 90

# The planner times every candidate path with this model of the drone. These
# match pico_server: top speed, how hard it speeds up and brakes, and how far
# a curve may cut a corner, which sets the speed through each turn.
MAX_SPEED = 3.2           # units per second
ACCELERATION = 1.2        # units per second squared
TURN_ACCELERATION = 1.0
MAX_CORNER_CUT = 0.5
MIN_CORNER_SPEED = 0.6
SECONDS_PER_CORNER = 0.1  # the small delay of swapping to the next goal

# A path may turn at this many corners. More would take longer to search and
# is not needed to get around three obstacles.
MAX_CORNERS = 3

# Corners sit just outside the off-limits areas. Their outlines are simplified
# to this accuracy, in pixels, and the corners are placed this far beyond them
# so that straight lines between neighbouring corners stay clear.
OUTLINE_ACCURACY_PX = 12
CORNER_MARGIN_PX = 14

# Corners must stay where the camera can see the drone.
ARENA_LIMIT = 6.2

# Settling, the Visit, the descent and the Save at one survivor.
SECONDS_PER_SAVE = 6.0

# Reconnect to the camera if no usable frame has arrived after this long.
FRAME_TIMEOUT = 8.0  # seconds

# The critical route should finish this long after take-off (the limit is 55).
CRITICAL_TIME_BUDGET = 50.0


def make_pose(x, y, z=FLIGHT_Z):
    """Create a ROS pose at the given coordinates."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def distance(a, b):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def corner_speed(start, corner, after):
    """The speed the drone can hold through a corner without cutting too far."""
    incoming = (corner[0] - start[0], corner[1] - start[1])
    outgoing = (after[0] - corner[0], after[1] - corner[1])
    cosine = (
        (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
        / (math.hypot(*incoming) * math.hypot(*outgoing))
    )
    half_turn = math.acos(max(-1.0, min(1.0, cosine))) / 2.0

    cut_per_radius = 1.0 / max(math.cos(half_turn), 0.05) - 1.0
    radius = MAX_CORNER_CUT / max(cut_per_radius, 1e-3)
    speed = math.sqrt(TURN_ACCELERATION * radius)
    return max(MIN_CORNER_SPEED, min(MAX_SPEED, speed))


def segment_time(length, start_speed, end_speed):
    """Time for a straight line: speed up, maybe cruise, then slow down."""
    # The peak speed if the drone speeds up and then slows down with no cruise.
    peak = math.sqrt(
        ACCELERATION * length + (start_speed ** 2 + end_speed ** 2) / 2.0
    )
    peak = min(peak, MAX_SPEED)

    speeding_up = (peak - start_speed) / ACCELERATION
    slowing_down = (peak - end_speed) / ACCELERATION
    ramp_length = (
        (peak + start_speed) / 2.0 * speeding_up
        + (peak + end_speed) / 2.0 * slowing_down
    )
    cruising = max(0.0, length - ramp_length) / MAX_SPEED
    return speeding_up + cruising + slowing_down


def path_flight_time(points):
    """Estimate the time to fly a path, starting and ending at rest."""
    # The speed allowed at each point: zero at the ends, limited by each turn.
    speeds = [0.0]
    for before, corner, after in zip(points, points[1:], points[2:]):
        speeds.append(corner_speed(before, corner, after))
    speeds.append(0.0)

    # The drone can only change speed so fast, so a short leg may limit the
    # speed at its ends. Pass over the speeds forwards, then backwards.
    lengths = [distance(a, b) for a, b in zip(points, points[1:])]
    for i, length in enumerate(lengths):
        reachable = math.sqrt(speeds[i] ** 2 + 2 * ACCELERATION * length)
        speeds[i + 1] = min(speeds[i + 1], reachable)
    for i in reversed(range(len(lengths))):
        reachable = math.sqrt(speeds[i + 1] ** 2 + 2 * ACCELERATION * lengths[i])
        speeds[i] = min(speeds[i], reachable)

    total = SECONDS_PER_CORNER * (len(points) - 2)
    for i, length in enumerate(lengths):
        total += segment_time(length, speeds[i], speeds[i + 1])
    return total


# ----------------------------------------------------------------- Detection


def detect_aruco_markers(gray_image):
    """Detect the arena markers across supported OpenCV versions."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray_image)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray_image, dictionary, parameters=parameters
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
        marker_id: corners[i][0] for i, marker_id in enumerate(marker_ids)
    }
    centres = {
        marker_id: np.mean(points, axis=0)
        for marker_id, points in marker_corners.items()
    }
    arena_centre = np.mean(list(centres.values()), axis=0)

    # The corner closest to the arena centre is the inner corner.
    inner_corners = {}
    for marker_id, points in marker_corners.items():
        gaps = [np.linalg.norm(point - arena_centre) for point in points]
        inner_corners[marker_id] = points[np.argmin(gaps)]

    # Sort markers by vertical position, then horizontal position.
    sorted_by_y = sorted(centres.items(), key=lambda item: item[1][1])
    top = sorted(sorted_by_y[:2], key=lambda item: item[1][0])
    bottom = sorted(sorted_by_y[2:], key=lambda item: item[1][0])

    return np.float32([
        inner_corners[top[0][0]],
        inner_corners[top[1][0]],
        inner_corners[bottom[1][0]],
        inner_corners[bottom[0][0]],
    ])


def rectify_arena(image):
    """Warp the camera image so the arena fills a square, top-down image."""
    if image is None or image.size == 0:
        raise ValueError("Received an empty image")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_markers(gray_image)
    source_corners = find_arena_corners(corners, ids)

    size = ARENA_SIZE
    destination = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    transform = cv2.getPerspectiveTransform(source_corners, destination)
    return cv2.warpPerspective(image, transform, (size, size))


def nearest_grid_label(x, y):
    """Map a point in the rectified image to its nearest grid intersection."""
    column = int(np.clip(round(x / GRID_SPACING_PX), 1, 11))
    row = int(np.clip(round(y / GRID_SPACING_PX), 1, 11))
    return f"{chr(ord('A') + column - 1)}{row}"


def find_survivor_labels(mask):
    """Return the grid labels of the coloured regions in a mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    labels = set()
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        labels.add(nearest_grid_label(x + width / 2.0, y + height / 2.0))

    return sorted(labels)


def find_critical_survivors(hsv):
    """Critical survivors are red, and red wraps around the HSV hue scale."""
    low_reds = cv2.inRange(hsv, (0, 150, 100), (10, 255, 255))
    high_reds = cv2.inRange(hsv, (170, 150, 100), (179, 255, 255))
    return find_survivor_labels(cv2.bitwise_or(low_reds, high_reds))


def find_stable_survivors(hsv):
    """Stable survivors are yellow."""
    return find_survivor_labels(
        cv2.inRange(hsv, (20, 150, 150), (35, 255, 255))
    )


def find_obstacles(hsv):
    """Return a mask of the black obstacles, ignoring small dark patches."""
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, OBSTACLE_MAX_BRIGHTNESS))

    # Opening removes thin dark details such as shadows and tree gaps.
    kernel = np.ones((15, 15), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    obstacles = [
        contour for contour in contours
        if cv2.contourArea(contour) >= MIN_OBSTACLE_AREA_PX
    ]

    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, obstacles, -1, 255, cv2.FILLED)
    return mask, len(obstacles)


def image_to_bgr(message):
    """Convert a ROS image message into an OpenCV BGR image."""
    height = message.height
    width = message.width
    encoding = message.encoding.lower()

    if height == 0 or width == 0:
        raise ValueError("Received an empty ROS image")

    channels = {
        "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1,
    }.get(encoding)

    if channels is None:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    raw = raw.reshape(height, message.step)
    image = raw[:, :width * channels].reshape(height, width, channels)

    if encoding == "bgr8":
        return image.copy()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ Planning


def to_point(column, row):
    """Convert a rectified image pixel into a WhyCode (x, y) point."""
    centre = ARENA_SIZE / 2.0
    return (
        (centre - column) / PIXELS_PER_UNIT,
        (centre - row) / PIXELS_PER_UNIT,
    )


def to_pixel(point):
    """Convert a WhyCode (x, y) point into rectified image (column, row)."""
    centre = ARENA_SIZE / 2.0
    return (
        centre - point[0] * PIXELS_PER_UNIT,
        centre - point[1] * PIXELS_PER_UNIT,
    )


class RoutePlanner:
    """
    Finds the fastest obstacle-free paths. A path only needs to turn at the
    corners of the off-limits areas, so it tries every combination of those
    corners and times each one, including slowing down for its turns.
    """

    def __init__(self, coordinates, obstacle_mask):
        self.points = {
            label: (value["x"], value["y"])
            for label, value in coordinates.items()
        }
        self.paths = {}

        self.off_limits = self.grow(obstacle_mask, CLEARANCE_PX)
        self.tight_off_limits = self.grow(obstacle_mask, TIGHT_CLEARANCE_PX)
        self.corners = self.find_corners()

    def find_corners(self):
        """The places a path may turn: just outside each off-limits area."""
        outer = self.grow(self.off_limits, CORNER_MARGIN_PX)
        contours, _ = cv2.findContours(
            outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        corners = []
        for contour in contours:
            outline = cv2.approxPolyDP(contour, OUTLINE_ACCURACY_PX, True)
            for vertex in outline:
                point = to_point(vertex[0][0], vertex[0][1])
                if max(abs(point[0]), abs(point[1])) <= ARENA_LIMIT:
                    corners.append(point)
        return corners

    def grow(self, mask, clearance):
        """Grow the white areas of a mask outwards by a clearance."""
        size = 2 * clearance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask, kernel)

    def mask_for_path(self, start, goal):
        """Full clearance everywhere, except close to the two path ends."""
        mask = self.off_limits.copy()
        for point in (start, goal):
            column, row = to_pixel(point)
            cv2.circle(
                mask, (int(column), int(row)), TIGHT_ZONE_RADIUS_PX, 0,
                cv2.FILLED,
            )
        return cv2.bitwise_or(mask, self.tight_off_limits)

    def is_clear(self, start, end, mask):
        """Check every few pixels along a straight line against a mask."""
        start_x, start_y = to_pixel(start)
        end_x, end_y = to_pixel(end)
        samples = int(math.hypot(end_x - start_x, end_y - start_y) / 4) + 2

        columns = np.linspace(start_x, end_x, samples).astype(int)
        rows = np.linspace(start_y, end_y, samples).astype(int)
        columns = np.clip(columns, 0, ARENA_SIZE - 1)
        rows = np.clip(rows, 0, ARENA_SIZE - 1)

        return not np.any(mask[rows, columns])

    def fastest_path(self, start, goal, mask):
        """Try every way through the corners and keep the fastest."""
        best_path = None
        best_time = math.inf

        def extend(path, length):
            nonlocal best_path, best_time

            if self.is_clear(path[-1], goal, mask):
                candidate = path + [goal]
                time_needed = path_flight_time(candidate)
                if time_needed < best_time:
                    best_path = candidate
                    best_time = time_needed

            if len(path) > MAX_CORNERS:
                return

            for corner in self.corners:
                if corner in path:
                    continue

                # Even at top speed, this could not beat the best path so far.
                new_length = length + distance(path[-1], corner)
                shortest = new_length + distance(corner, goal)
                if shortest / MAX_SPEED >= best_time:
                    continue

                if self.is_clear(path[-1], corner, mask):
                    extend(path + [corner], new_length)

        extend([start], 0.0)
        return best_path

    def path(self, start, goal):
        """Return the points from start to goal, clear of the obstacles."""
        if start == goal:
            return [start]

        if (start, goal) in self.paths:
            return self.paths[(start, goal)]

        path = self.fastest_path(start, goal, self.mask_for_path(start, goal))

        # Only possible if a point is inside an obstacle. A straight line
        # keeps the mission going, where failing would strand the drone.
        if path is None:
            path = [start, goal]

        self.paths[(start, goal)] = path
        return path

    def path_time(self, start, goal):
        path = self.path(start, goal)
        return path_flight_time(path) if len(path) > 1 else 0.0

    def tour_time(self, start, labels, end_label=None):
        """Flight time from start through labels, in that order."""
        stops = [self.points[label] for label in labels]
        if end_label is not None:
            stops.append(self.points[end_label])

        total = 0.0
        current = start
        for stop in stops:
            total += self.path_time(current, stop)
            current = stop
        return total

    def best_stable_order(self, start, stable):
        """The visiting order giving the fastest flight, ending at home."""
        return min(
            itertools.permutations(stable),
            key=lambda order: self.tour_time(start, order, HOME_LABEL),
        )

    def best_critical_order(self, critical, stable):
        """
        The critical order that makes the whole mission fastest, including
        the stable route that follows it, as long as it fits the window.
        """
        home = self.points[HOME_LABEL]
        orders = list(itertools.permutations(critical))

        def critical_time(order):
            flight = self.tour_time(home, order)
            return flight + SECONDS_PER_SAVE * len(order)

        def mission_time(order):
            last = self.points[order[-1]]
            rest = self.best_stable_order(last, stable)
            return (
                self.tour_time(home, order)
                + self.tour_time(last, rest, HOME_LABEL)
            )

        in_time = [o for o in orders if critical_time(o) <= CRITICAL_TIME_BUDGET]
        if in_time:
            return min(in_time, key=mission_time)
        return min(orders, key=critical_time)

    def build_route(self, start, labels, end_at_home=False):
        """
        Turn a visiting order into (point, marker id) pairs. The marker id
        tells the client what to do there: transit, survivor or home.
        """
        route = []
        current = start

        for label in labels:
            target = self.points[label]
            route += self.transit_points(current, target)
            route.append((target, f"survivor:{label}"))
            current = target

        if end_at_home:
            home = self.points[HOME_LABEL]
            route += self.transit_points(current, home)
            route.append((home, f"home:{HOME_LABEL}"))

        return route

    def transit_points(self, start, goal):
        """The corners of a path, without its two end points."""
        return [
            (point, f"transit:{point[0]:.2f},{point[1]:.2f}")
            for point in self.path(start, goal)[1:-1]
        ]


def route_to_response(route, response, stamp):
    """Copy a planned route into a service response."""
    response.waypoints = PoseArray()
    response.waypoints.header.stamp = stamp
    response.waypoints.header.frame_id = "whycode"
    response.marker_ids = []

    for point, marker_id in route:
        height = LANDING_Z if marker_id.startswith("home:") else FLIGHT_Z
        response.waypoints.poses.append(make_pose(point[0], point[1], height))
        response.marker_ids.append(marker_id)

    return response


def draw_plan(arena, obstacle_mask, planner, route):
    """Draw the off-limits areas and a route, for the task checkpoints."""
    picture = arena.copy()

    for mask, colour in ((planner.off_limits, (0, 165, 255)),
                         (obstacle_mask, (255, 0, 255))):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(picture, contours, -1, colour, 2)

    pixels = [to_pixel(planner.points[HOME_LABEL])]
    pixels += [to_pixel(point) for point, _ in route]
    for start, end in zip(pixels, pixels[1:]):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.arrowedLine(picture, start, end, (255, 255, 255), 2, tipLength=0.03)

    return picture


# ---------------------------------------------------------------------- Node


class WayPoints(Node):

    def __init__(self):
        super().__init__("waypoints_service")

        # Set this to a file path to save a picture of the plan.
        self.declare_parameter("debug_image", "")

        # The services wait on this until the frame has been processed.
        self.plan_ready = threading.Condition()
        self.planner = None
        self.critical_route = None
        self.stable_labels = None
        self.planning = False

        callback_group = ReentrantCallbackGroup()
        self.image_group = callback_group

        self.subscribe_to_image()
        self.create_timer(
            FRAME_TIMEOUT, self.check_for_frame, callback_group=callback_group
        )

        self.create_service(
            GetWaypoints, "waypoints", self.waypoints_callback,
            callback_group=callback_group,
        )
        self.create_service(
            GetStableRoute, "stable_route", self.stable_route_callback,
            callback_group=callback_group,
        )

        self.get_logger().info("Waiting for the arena image to plan the route")

    def subscribe_to_image(self):
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.image_sub = self.create_subscription(
            Image, "/image_raw", self.image_callback, image_qos,
            callback_group=self.image_group,
        )

    def check_for_frame(self):
        """A large image occasionally never arrives, so connect again."""
        with self.plan_ready:
            if self.critical_route is not None or self.planning:
                return

        self.get_logger().warning("No arena image yet, reconnecting")
        self.destroy_subscription(self.image_sub)
        self.subscribe_to_image()

    def load_coordinates(self):
        share_path = get_package_share_directory("swift_pico")
        path = share_path + "/scripts/grid_whycode_coordinates.json"
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def image_callback(self, message):
        # Plan once. Other frames are only used if this one fails.
        with self.plan_ready:
            if self.critical_route is not None or self.planning:
                return
            self.planning = True

        try:
            self.plan_from_image(message)
        except Exception as error:
            self.get_logger().error(
                f"Planning failed, trying the next frame: {error}"
            )

        with self.plan_ready:
            self.planning = False
            self.plan_ready.notify_all()

    def plan_from_image(self, message):
        arena = rectify_arena(image_to_bgr(message))
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)

        critical = find_critical_survivors(hsv)
        stable = find_stable_survivors(hsv)
        obstacle_mask, obstacle_count = find_obstacles(hsv)

        self.get_logger().info(
            f"Critical: {critical}  Stable: {stable}  "
            f"Obstacles: {obstacle_count}"
        )
        if not critical:
            raise ValueError("No critical survivors detected")
        if len(critical) != 3 or len(stable) != 3 or obstacle_count != 3:
            self.get_logger().warning("Expected 3 of each, check the detection")

        planner = RoutePlanner(self.load_coordinates(), obstacle_mask)
        home = planner.points[HOME_LABEL]
        order = planner.best_critical_order(critical, stable)
        route = planner.build_route(home, order)

        self.get_logger().info(
            "Critical route: " + " ".join(marker for _, marker in route)
        )

        debug_image = self.get_parameter("debug_image").value
        if debug_image:
            last = planner.points[order[-1]]
            stable_order = planner.best_stable_order(last, stable)
            whole_mission = route + planner.build_route(
                last, stable_order, end_at_home=True
            )
            cv2.imwrite(
                debug_image,
                draw_plan(arena, obstacle_mask, planner, whole_mission),
            )

        with self.plan_ready:
            self.planner = planner
            self.critical_route = route
            self.stable_labels = stable

        # The image is no longer needed once the plan exists.
        self.destroy_subscription(self.image_sub)

    def wait_for_plan(self):
        """A request can arrive before the frame is processed, so wait."""
        with self.plan_ready:
            while self.critical_route is None and rclpy.ok():
                self.plan_ready.wait(timeout=0.25)

    def waypoints_callback(self, request, response):
        if not request.get_waypoints:
            return response

        self.wait_for_plan()

        self.get_logger().info(
            f"Sending the critical route ({len(self.critical_route)} waypoints)"
        )
        stamp = self.get_clock().now().to_msg()
        return route_to_response(self.critical_route, response, stamp)

    def stable_route_callback(self, request, response):
        self.wait_for_plan()

        start = (request.position.x, request.position.y)
        order = self.planner.best_stable_order(start, self.stable_labels)
        route = self.planner.build_route(start, order, end_at_home=True)

        self.get_logger().info(
            f"Stable route from [{start[0]:.2f}, {start[1]:.2f}]: "
            + " ".join(marker for _, marker in route)
        )
        stamp = self.get_clock().now().to_msg()
        return route_to_response(route, response, stamp)


def main(args=None):
    # The arena image is too big for the default shared memory, and over UDP
    # it may never arrive. The simulator's nodes use this profile for the same
    # reason. It has to be set before ROS starts.
    share_path = get_package_share_directory("swift_pico")
    os.environ.setdefault(
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        share_path + "/config/fastdds_profile.xml",
    )

    rclpy.init(args=args)

    node = WayPoints()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
