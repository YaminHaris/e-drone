#!/usr/bin/env python3
"""Task 2B waypoint action server using the Task 2A LQR controller."""

import math
import time
import threading

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from waypoint_navigation.action import NavToWaypoint


# Controller settings from the Task 2A Pico controller.
CONTROL_PERIOD = 0.02
HOVER_THRUST_N = 14.896
MAX_THRUST_N = 30.0

RC_CENTER = 1500.0
RC_TILT_SPAN = 85.0
RC_TILT_LIMIT_RAD = math.radians(4.5)
MAX_TILT_RAD = math.radians(3.0)

MARKER_TIMEOUT = 0.20
FILTER_ALPHA = 0.6
FILTER_BETA = 0.2

POSITION_TOLERANCE = 0.4
HOVER_TIME = 3.0

# This is the same K matrix used by the Task 2A controller.
K = np.array([
    [0.098304489591, 0.0, 0.0, 0.172352727574, 0.0, 0.0],
    [0.0, 0.098304489591, 0.0, 0.0, 0.172352727574, 0.0],
    [0.0, 0.0, -4.844368481057, 0.0, 0.0, -4.644725744029],
])

# Basic yaw correction. Verify the direction and tune in simulation.
YAW_KP = 3.0
MAX_YAW_CORRECTION = 100.0
DESIRED_YAW = -0.5


def clamp(value, low, high):
    return max(low, min(value, high))


class WayPointServer(Node):

    def __init__(self):
        super().__init__("waypoint_server")

        # The marker callback and controller timer share the same state.
        self.lock = threading.Lock()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_marker_time = None

        self.target = None
        self.current_yaw = 0.0
        self.active_goal = False
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
            "Task 2B waypoint server ready; LQR running at 50 Hz"
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
                    residual = measured - predicted

                    self.position = (
                        predicted + FILTER_ALPHA * residual
                    )
                    self.velocity = (
                        self.velocity + FILTER_BETA / dt * residual
                    )

            self.last_marker_time = now

            # The supplied boilerplate uses marker rotation Z for yaw.
            self.current_yaw = math.degrees(marker.rotation.z)

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
            yaw = self.current_yaw

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
        control = -K @ error

        if not np.all(np.isfinite(control)):
            self.publish_command(0.0, 0.0, HOVER_THRUST_N, 0.0)
            return

        pitch, roll, thrust_change = control

        # Keep yaw close to its desired angle.
        yaw_error = (DESIRED_YAW - yaw + 180.0) % 360.0 - 180.0
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

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        point = goal.waypoint.position

        target = np.array(
            [point.x, point.y, point.z], dtype=float
        )

        with self.lock:
            self.target = target

        started = time.monotonic()
        inside_since = None
        result = NavToWaypoint.Result()

        self.get_logger().info(
            f"Navigating to [{target[0]:.2f}, "
            f"{target[1]:.2f}, {target[2]:.2f}]"
        )

        try:
            while rclpy.ok():
                now = time.monotonic()

                with self.lock:
                    position = self.position.copy()
                    fresh = (
                        self.last_marker_time is not None
                        and now - self.last_marker_time <= MARKER_TIMEOUT
                    )

                # Send the current measured position back to the client.
                feedback = NavToWaypoint.Feedback()
                feedback.current_waypoint = PoseStamped()
                feedback.current_waypoint.header.stamp = (
                    self.get_clock().now().to_msg()
                )
                feedback.current_waypoint.header.frame_id = "whycode"

                feedback.current_waypoint.pose.position.x = float(position[0])
                feedback.current_waypoint.pose.position.y = float(position[1])
                feedback.current_waypoint.pose.position.z = float(position[2])
                feedback.current_waypoint.pose.orientation.w = 1.0

                goal_handle.publish_feedback(feedback)

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return result

                # Every axis must be within +/- 0.4 WhyCode units.
                inside = fresh and bool(
                    np.all(np.abs(position - target) <= POSITION_TOLERANCE)
                )

                if inside:
                    if inside_since is None:
                        inside_since = now
                        self.get_logger().info(
                            "Waypoint reached; starting stabilisation timer"
                        )

                    # The task requires a continuous three-second hold.
                    if not goal.hover or now - inside_since >= HOVER_TIME:
                        result.hov_time = int(round(now - started))
                        goal_handle.succeed()
                        self.get_logger().info(
                            f"Waypoint complete in {result.hov_time} seconds"
                        )
                        return result
                else:
                    # Leaving the tolerance box resets the hold timer.
                    inside_since = None

                time.sleep(0.05)

        finally:
            # Release the action server for the next waypoint.
            with self.lock:
                self.target = None
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
