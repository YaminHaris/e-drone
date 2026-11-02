#!/usr/bin/env python3
"""
Task 3A waypoint action server using the Task 2A LQR controller.

A goal with hover=False is flown past. A goal with hover=True is held inside
a sphere for the Visit (Z = 13) or Save (Z = 14) time. A goal at ground height
(Z = 15) is the landing: hold at Z = 13, then descend and switch the motors
off. Cancelling a goal holds the drone where it is until the next goal.
"""

import math
import time
import threading

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Header
from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from waypoint_navigation.action import NavToWaypoint


# Controller settings from the Task 2A Pico controller.
CONTROL_PERIOD = 0.02
LOOP_PERIOD = 0.02  # how often a running goal checks its progress
HOVER_THRUST_N = 14.896
MAX_THRUST_N = 30.0

RC_CENTER = 1500.0
RC_TILT_SPAN = 85.0
RC_TILT_LIMIT_RAD = math.radians(4.5)
MAX_TILT_RAD = math.radians(8.0)

MARKER_TIMEOUT = 0.20
FILTER_ALPHA = 0.6
FILTER_BETA = 0.2

# How close to a waypoint counts as passing it, in the X-Y plane only. The
# controller slows down near a target, so a wider circle keeps the speed up.
TRANSIT_RADIUS = 0.7

# Holds count inside a 0.4 sphere for 1 or 2 seconds, judged from the recorded
# marker positions. The server reads the same positions, so a small margin is
# enough, and a tighter sphere only restarts holds on marker noise.
HOLD_RADIUS = 0.39
VISIT_HOLD_TIME = 1.05
SAVE_HOLD_TIME = 2.05

FLIGHT_Z = 13.0
SAVE_Z = 14.0

# Z is measured down from the camera. A goal at Z = 15 means land. The marker
# reads anywhere from 14.7 to 15.0 once the drone is resting on the ground, so
# the motors go off below that range, or after a time limit if it is never seen.
LANDING_Z = 15.0
LANDING_TARGET_Z = 15.3  # aim below the ground so the drone does not hover
TOUCHDOWN_Z = 14.55      # motors go off here, which stops any bounce
LANDING_EXTRA_TIME = 1.0  # seconds allowed after the descent should be over
DESCENT_RATE = 0.9       # whycode units per second

# LQR gain for the same model as Task 2A, tuned in flight tests for speed
# without wobble. X-Y weights: Q position 2.6, Q velocity 0.75, R tilt 100.
# Higher gains were faster on paper but wobbled and overshot.
K = np.array([
    [0.1581, 0.0, 0.0, 0.1986, 0.0, 0.0],
    [0.0, 0.1581, 0.0, 0.0, 0.1986, 0.0],
    [0.0, 0.0, -4.844368481057, 0.0, 0.0, -4.644725744029],
])

# The position error used for control is capped at this size. It limits the
# top speed, so the drone can always brake in time on a long leg.
MAX_POSITION_ERROR = 4.0

# Transit waypoints are flown through in a curve instead of nearly stopping.
# The speed comes from the turn: the sharper it is, the slower the drone goes,
# so that the curve cuts the corner by no more than MAX_CORNER_CUT. The planner
# keeps 1.22 units from obstacles, and the drone reaches 0.33 from its centre.
MAX_CORNER_CUT = 0.5
TURN_ACCELERATION = 1.0  # units/s^2, below the 1.38 that 8 degrees can give
BRAKING_ACCELERATION = 1.0
SPEED_PER_ERROR = K[0, 0] / K[0, 3]  # steady speed for each unit of error
MAX_SPEED = MAX_POSITION_ERROR * SPEED_PER_ERROR
MIN_CORNER_SPEED = 0.6
MAX_TURN_START_DISTANCE = 2.0

# WhyCode sometimes misreads the marker for a moment, and the position jumps.
# These stop such a jump from kicking the drone during a hold: one frame can
# only move the estimate so far, and near a hold point the tilt stays gentle.
MAX_FRAME_CORRECTION = 0.15  # units
HOLD_ZONE_RADIUS = 0.7       # units
HOLD_TILT_RAD = math.radians(3.0)
MAX_THRUST_CHANGE = 6.0      # newtons

# Any yaw command turns the drone at full rate (45 degrees per second), so
# noise made it twitch. Only large errors are corrected, and the tilt commands
# already allow for smaller ones. The reading is smoothed, but too much delay
# makes the drone overshoot and swing back and forth: at this smoothing it
# stops about 13 degrees past the deadband edge, well short of the other edge.
YAW_DEADBAND = 20.0  # degrees
YAW_SMOOTHING = 0.1
YAW_KP = 3.0
MAX_YAW_CORRECTION = 100.0
DESIRED_YAW = -0.5


def clamp(value, low, high):
    return max(low, min(value, high))


def limit_length(vector, limit):
    """Shorten a vector to a maximum length, keeping its direction."""
    length = np.linalg.norm(vector)
    if length > limit:
        return vector * limit / length
    return vector


def plan_corner(start, corner, after):
    """
    Plan a curve through a corner, from the three points in the X-Y plane.
    Returns the speed to hold through the turn, and how far before the corner
    to start turning.
    """
    incoming = corner - start
    outgoing = after - corner
    lengths = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    if lengths < 1e-6:
        return MIN_CORNER_SPEED, TRANSIT_RADIUS

    cosine = clamp(np.dot(incoming, outgoing) / lengths, -1.0, 1.0)
    half_turn = math.acos(cosine) / 2.0

    # For a turn of this radius, the corner is cut by radius * (sec - 1).
    cut_per_radius = 1.0 / max(math.cos(half_turn), 0.05) - 1.0
    radius = MAX_CORNER_CUT / max(cut_per_radius, 1e-3)

    speed = math.sqrt(TURN_ACCELERATION * radius)
    speed = clamp(speed, MIN_CORNER_SPEED, MAX_SPEED)

    # The turn starts where the curve leaves the straight line.
    radius = speed ** 2 / TURN_ACCELERATION
    turn_start = clamp(
        radius * math.tan(half_turn), TRANSIT_RADIUS, MAX_TURN_START_DISTANCE
    )

    # Fly at least half of the leg first. A corner close to the start must not
    # be skipped, because the line straight past it could cross an obstacle.
    return speed, min(turn_start, np.linalg.norm(incoming) / 2.0)


class WayPointServer(Node):

    def __init__(self):
        super().__init__("waypoint_server")

        # The marker callback and controller timer share the same state.
        self.lock = threading.Lock()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.measured = np.zeros(3)  # unfiltered, used to judge holds
        self.last_marker_time = None

        self.target = None
        self.corner_speed = None  # set while flying through a transit corner
        self.current_yaw = 0.0
        self.active_goal = False
        self.landed = False
        self.startup_ticks = 0

        self.callback_group = ReentrantCallbackGroup()

        # Only this node publishes the drone's flight commands.
        self.command_pub = self.create_publisher(
            SwiftMsgs, "/drone_command", 10
        )
        self.error_pub = self.create_publisher(
            Error, "/pos_error", 10
        )
        self.heartbeat_pub = self.create_publisher(
            Header, "/controller_heartbeat", 10
        )

        self.create_subscription(
            MarkerArray,
            "/whycode_node/markers",
            self.whycode_callback,
            1,
            callback_group=self.callback_group,
        )

        # The client sends each waypoint to this action server.
        self.action_server = ActionServer(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Keep the LQR controller running at 50 Hz.
        self.create_timer(
            CONTROL_PERIOD,
            self.control_step,
            callback_group=self.callback_group,
        )
        self.create_timer(0.1, self.publish_heartbeat)

        self.disarm()
        self.get_logger().info(
            "Waypoint server ready; LQR running at 50 Hz"
        )

    def publish_heartbeat(self):
        message = Header()
        message.stamp = self.get_clock().now().to_msg()
        message.frame_id = "pico_controller_active"
        self.heartbeat_pub.publish(message)

    def disarm(self):
        command = SwiftMsgs()
        command.rc_roll = 1000
        command.rc_pitch = 1000
        command.rc_yaw = 1000
        command.rc_throttle = 1000
        command.rc_aux4 = 1000
        self.command_pub.publish(command)

    def publish_command(self, pitch, roll, thrust, yaw):
        # Keep commands within the limits used by the Task 2A controller.
        pitch = clamp(pitch, -MAX_TILT_RAD, MAX_TILT_RAD)
        roll = clamp(roll, -MAX_TILT_RAD, MAX_TILT_RAD)
        thrust = clamp(thrust, 0.0, MAX_THRUST_N)

        # Convert thrust into the RC throttle range.
        if thrust >= HOVER_THRUST_N:
            thrust_range = MAX_THRUST_N - HOVER_THRUST_N
        else:
            thrust_range = HOVER_THRUST_N

        throttle = (
            RC_CENTER
            + (thrust - HOVER_THRUST_N) * 500.0 / thrust_range
        )

        command = SwiftMsgs()
        command.rc_pitch = round(
            RC_CENTER + pitch * RC_TILT_SPAN / RC_TILT_LIMIT_RAD
        )
        command.rc_roll = round(
            RC_CENTER + roll * RC_TILT_SPAN / RC_TILT_LIMIT_RAD
        )
        command.rc_yaw = round(
            clamp(RC_CENTER + yaw, 1000.0, 2000.0)
        )
        command.rc_throttle = round(
            clamp(throttle, 1000.0, 2000.0)
        )
        command.rc_aux4 = 2000
        self.command_pub.publish(command)

    def whycode_callback(self, message):
        if not message.markers:
            return

        marker = message.markers[0]
        point = marker.position.position

        measured = np.array(
            [point.x, point.y, point.z], dtype=float
        )

        if not np.all(np.isfinite(measured)):
            return

        now = time.monotonic()

        # Estimate velocity from successive marker measurements.
        with self.lock:
            self.measured = measured

            if self.last_marker_time is None:
                self.position = measured
                self.velocity = np.zeros(3)
            else:
                dt = now - self.last_marker_time

                if dt <= 0.001:
                    return

                if dt > MARKER_TIMEOUT:
                    self.position = measured
                    self.velocity = np.zeros(3)
                else:
                    predicted = self.position + self.velocity * dt
                    residual = np.clip(
                        measured - predicted,
                        -MAX_FRAME_CORRECTION,
                        MAX_FRAME_CORRECTION,
                    )

                    self.position = (
                        predicted + FILTER_ALPHA * residual
                    )
                    self.velocity = (
                        self.velocity + FILTER_BETA / dt * residual
                    )

            self.last_marker_time = now

            # The supplied boilerplate uses marker rotation Z for yaw.
            measured_yaw = math.degrees(marker.rotation.z)
            yaw_change = (measured_yaw - self.current_yaw + 180.0) % 360.0 - 180.0
            self.current_yaw += YAW_SMOOTHING * yaw_change

    def control_step(self):
        now = time.monotonic()

        with self.lock:
            fresh = (
                self.last_marker_time is not None
                and now - self.last_marker_time <= MARKER_TIMEOUT
            )
            position = self.position.copy()
            velocity = self.velocity.copy()
            target = None if self.target is None else self.target.copy()
            corner_speed = self.corner_speed
            yaw = self.current_yaw
            landed = self.landed

        # Once on the ground the motors stay off until a new goal arrives.
        if landed:
            self.disarm()
            return

        # Give the simulator a few disarmed control cycles at startup.
        if self.startup_ticks < 5:
            self.disarm()
            self.startup_ticks += 1
            return

        # Without a target or fresh position data, don't chase stale data.
        if not fresh or target is None:
            self.publish_command(0.0, 0.0, HOVER_THRUST_N, 0.0)
            return

        # LQR state: position error followed by velocity error.
        state = np.concatenate((position, velocity))
        reference = np.concatenate((target, np.zeros(3)))
        error = state - reference

        # Reuse the Task 2A gain matrix.
        # Limit lengths, not each axis, or diagonal legs would curve off the
        # planned straight line and towards the obstacles.
        capped_error = error.copy()
        capped_error[:2] = limit_length(error[:2], MAX_POSITION_ERROR)

        # Through a corner, set the error so the drone keeps its speed and only
        # brakes as much as the turn needs. This is the same control law.
        if corner_speed is not None:
            distance = np.linalg.norm(error[:2])
            speed = min(
                MAX_SPEED,
                math.sqrt(
                    corner_speed ** 2 + 2.0 * BRAKING_ACCELERATION * distance
                ),
            )
            if distance > 1e-3:
                capped_error[:2] = (
                    error[:2] / distance * speed / SPEED_PER_ERROR
                )

        control = -K @ capped_error

        if not np.all(np.isfinite(control)):
            self.publish_command(0.0, 0.0, HOVER_THRUST_N, 0.0)
            return

        tilt_limit = MAX_TILT_RAD
        if corner_speed is None and np.linalg.norm(error[:2]) < HOLD_ZONE_RADIUS:
            tilt_limit = HOLD_TILT_RAD

        pitch, roll = limit_length(control[:2], tilt_limit)
        thrust_change = clamp(control[2], -MAX_THRUST_CHANGE, MAX_THRUST_CHANGE)

        # Tilt commands act in the drone's own frame. Turning them by the
        # measured yaw keeps position control correct if the drone gets
        # spun, for example by a knock.
        yaw_rad = math.radians(yaw)
        pitch, roll = (
            math.cos(yaw_rad) * pitch + math.sin(yaw_rad) * roll,
            math.cos(yaw_rad) * roll - math.sin(yaw_rad) * pitch,
        )

        # Keep yaw close to its desired angle.
        yaw_error = (DESIRED_YAW - yaw + 180.0) % 360.0 - 180.0
        yaw_correction = 0.0
        if abs(yaw_error) > YAW_DEADBAND:
            yaw_correction = clamp(
                YAW_KP * yaw_error,
                -MAX_YAW_CORRECTION,
                MAX_YAW_CORRECTION,
            )

        self.publish_command(
            pitch,
            roll,
            HOVER_THRUST_N + thrust_change,
            yaw_correction,
        )

        # Publish position errors for monitoring and debugging.
        error_message = Error()
        error_message.pitch_error = float(error[0])
        error_message.roll_error = float(error[1])
        error_message.throttle_error = float(error[2])
        error_message.yaw_error = float(yaw_error)
        self.error_pub.publish(error_message)

    def goal_callback(self, goal_request):
        target = goal_request.waypoint.position

        values = [target.x, target.y, target.z]
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Rejecting invalid waypoint")
            return GoalResponse.REJECT

        with self.lock:
            if self.active_goal:
                self.get_logger().warning(
                    "A waypoint is already being executed"
                )
                return GoalResponse.REJECT

            self.active_goal = True
            self.landed = False

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def read_marker(self):
        """Return the latest filtered position, raw position and freshness."""
        with self.lock:
            fresh = (
                self.last_marker_time is not None
                and time.monotonic() - self.last_marker_time <= MARKER_TIMEOUT
            )
            return self.position.copy(), self.measured.copy(), fresh

    def send_feedback(self, goal_handle, position):
        """Send the current measured position back to the client."""
        feedback = NavToWaypoint.Feedback()
        pose = feedback.current_waypoint
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "whycode"
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.w = 1.0
        goal_handle.publish_feedback(feedback)

    def land(self, goal_handle):
        """Lower the target slowly until the drone reaches the ground."""
        with self.lock:
            target = self.target.copy()

        self.get_logger().info("Landing")

        descent_time = (LANDING_TARGET_Z - FLIGHT_Z) / DESCENT_RATE
        deadline = time.monotonic() + descent_time + LANDING_EXTRA_TIME

        while rclpy.ok():
            position, _, _ = self.read_marker()
            self.send_feedback(goal_handle, position)

            if position[2] >= TOUCHDOWN_Z or time.monotonic() >= deadline:
                break

            target[2] = min(
                target[2] + DESCENT_RATE * LOOP_PERIOD, LANDING_TARGET_Z
            )
            with self.lock:
                self.target = target.copy()

            time.sleep(LOOP_PERIOD)

        with self.lock:
            self.landed = True

        self.get_logger().info("Landed, motors off")

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        point = goal.waypoint.position

        # A goal at ground height is flown at altitude first, then landed.
        is_landing = point.z >= LANDING_Z
        height = FLIGHT_Z if is_landing else point.z
        target = np.array([point.x, point.y, height], dtype=float)

        if height >= SAVE_Z:
            hold_time = SAVE_HOLD_TIME
        else:
            hold_time = VISIT_HOLD_TIME

        # A transit goal with a known next waypoint is flown as a curve.
        orientation = goal.waypoint.orientation
        corner_speed = None
        pass_distance = TRANSIT_RADIUS
        position, _, _ = self.read_marker()
        incoming = target[:2] - position[:2]

        if not goal.hover and orientation.z > 0.5:
            after = np.array([orientation.x, orientation.y])
            corner_speed, pass_distance = plan_corner(
                position[:2], target[:2], after
            )

        with self.lock:
            self.target = target
            self.corner_speed = corner_speed

        started = time.monotonic()
        inside_since = None
        result = NavToWaypoint.Result()

        self.get_logger().info(
            f"Navigating to [{target[0]:.2f}, "
            f"{target[1]:.2f}, {target[2]:.2f}], hover={goal.hover}"
        )

        try:
            while rclpy.ok():
                now = time.monotonic()
                position, measured, fresh = self.read_marker()

                self.send_feedback(goal_handle, position)

                if goal_handle.is_cancel_requested:
                    # Hold where the drone is until the next goal arrives.
                    with self.lock:
                        self.target = position
                        self.corner_speed = None
                    goal_handle.canceled()
                    self.get_logger().info("Goal cancelled, holding position")
                    return result

                if goal.hover:
                    # Losing the marker or leaving the sphere restarts a hold.
                    gap = np.linalg.norm(measured - target)
                    inside = fresh and gap <= HOLD_RADIUS
                else:
                    # Reached when the turn should start, or if already past.
                    offset = position[:2] - target[:2]
                    near = np.linalg.norm(offset) <= pass_distance
                    passed = np.dot(offset, incoming) > 0.0
                    inside = fresh and (near or passed)

                if not inside:
                    inside_since = None
                elif inside_since is None:
                    inside_since = now

                held_long_enough = (
                    inside_since is not None
                    and now - inside_since >= hold_time
                )

                if inside and (not goal.hover or held_long_enough):
                    if is_landing:
                        self.land(goal_handle)

                    result.hov_time = int(round(now - started))
                    goal_handle.succeed()
                    return result

                time.sleep(LOOP_PERIOD)

        finally:
            # The target stays set, so the drone keeps holding this position
            # until the next goal. Only the action server is released here.
            with self.lock:
                self.active_goal = False

        goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)

    node = WayPointServer()
    executor = MultiThreadedExecutor(num_threads=4)
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
