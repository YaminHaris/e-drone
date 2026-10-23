#!/usr/bin/env python3
"""Task 2 waypoint action server with the Task 2A discrete LQR outer loop.

The simulator already contains the inner attitude/thrust controller.  This node
therefore regulates WhyCode position and velocity, and converts the LQR output
to the RC interface used by ``rotors_swift_interface``.  A goal only changes
the reference; it never creates a second, competing controller.
"""

import math
import threading
import time

import numpy as np
import rclpy
from error_msg.msg import Error
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from swift_msgs.msg import SwiftMsgs
from waypoint_navigation.action import NavToWaypoint
from whycode_interfaces.msg import MarkerArray


MASS_KG = 1.5
GRAVITY = 9.81
DT = 0.02
HOVER_THRUST_N = 14.896
MAX_THRUST_N = 30.0
MAX_TILT_RAD = math.radians(4.5)
OPERATIONAL_MAX_TILT_RAD = math.radians(3.0)
RC_CENTER = 1500.0
RC_TILT_SPAN = 85.0
STATE_TIMEOUT_S = 0.20
HOVER_TIME_S = 3.0
TOLERANCE = 0.4


def clamp(value, low, high):
    return max(low, min(high, value))


def discrete_lqr_gain():
    """Solve the DARE for the same local model and weights as Task 2A."""
    a = np.eye(6)
    b = np.zeros((6, 3))
    for axis in range(3):
        a[axis, axis + 3] = DT
    # Inputs are [pitch, roll, delta_thrust].  WhyCode depth grows downward.
    b[0, 0] = 0.5 * GRAVITY * DT * DT
    b[1, 1] = 0.5 * GRAVITY * DT * DT
    b[2, 2] = -0.5 * DT * DT / MASS_KG
    b[3, 0] = GRAVITY * DT
    b[4, 1] = GRAVITY * DT
    b[5, 2] = -DT / MASS_KG

    q = np.diag([10.0, 10.0, 20.0, 8.0, 8.0, 6.0])
    r = np.diag([10.0, 10.0, 0.8])
    p = q.copy()
    for _ in range(20_000):
        middle = r + b.T @ p @ b
        p_next = a.T @ p @ a - a.T @ p @ b @ np.linalg.solve(middle, b.T @ p @ a) + q
        if np.max(np.abs(p_next - p)) < 1.0e-10:
            p = p_next
            break
        p = p_next
    else:
        raise RuntimeError("discrete Riccati iteration did not converge")
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    if not np.all(np.isfinite(gain)) or np.max(np.abs(np.linalg.eigvals(a - b @ gain))) >= 1.0:
        raise RuntimeError("invalid or unstable discrete LQR gain")
    return gain


class WayPointServer(Node):
    def __init__(self):
        super().__init__('waypoint_server')
        self._lock = threading.Lock()
        self._state = np.zeros(6, dtype=float)
        self._last_marker_at = None
        self._target = None
        self._active_goal = False
        self._last_stale_log = 0.0
        self._gain = discrete_lqr_gain()

        group = ReentrantCallbackGroup()
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)
        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1,
                                 callback_group=group)
        self._action_server = ActionServer(
            self, NavToWaypoint, 'waypoint_navigation', self.execute_callback,
            callback_group=group, goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.create_timer(DT, self.control_step, callback_group=group)
        self.create_timer(0.1, self.publish_heartbeat, callback_group=group)
        self.publish_heartbeat()
        self.arm()
        self.get_logger().info('Waypoint action server ready; discrete LQR at 50 Hz')

    def publish_heartbeat(self):
        message = Header()
        message.stamp = self.get_clock().now().to_msg()
        message.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(message)

    def _publish_command(self, pitch_rad, roll_rad, total_thrust_n):
        pitch = clamp(float(pitch_rad), -OPERATIONAL_MAX_TILT_RAD, OPERATIONAL_MAX_TILT_RAD)
        roll = clamp(float(roll_rad), -OPERATIONAL_MAX_TILT_RAD, OPERATIONAL_MAX_TILT_RAD)
        thrust = clamp(float(total_thrust_n), 0.0, MAX_THRUST_N)
        command = SwiftMsgs()
        command.rc_pitch = round(RC_CENTER + pitch * RC_TILT_SPAN / MAX_TILT_RAD)
        command.rc_roll = round(RC_CENTER + roll * RC_TILT_SPAN / MAX_TILT_RAD)
        command.rc_yaw = 1500
        command.rc_aux4 = 2000
        if thrust >= HOVER_THRUST_N:
            throttle = RC_CENTER + (thrust - HOVER_THRUST_N) * 500.0 / (MAX_THRUST_N - HOVER_THRUST_N)
        else:
            throttle = RC_CENTER + (thrust - HOVER_THRUST_N) * 500.0 / HOVER_THRUST_N
        command.rc_throttle = round(clamp(throttle, 1000.0, 2000.0))
        self.command_pub.publish(command)

    def arm(self):
        disarm = SwiftMsgs()
        disarm.rc_roll = disarm.rc_pitch = disarm.rc_yaw = disarm.rc_throttle = disarm.rc_aux4 = 1000
        self.command_pub.publish(disarm)
        self._publish_command(0.0, 0.0, HOVER_THRUST_N)

    def whycode_callback(self, message):
        if not message.markers:
            return
        point = message.markers[0].position.position
        measured = np.array([point.x, point.y, point.z], dtype=float)
        if not np.all(np.isfinite(measured)):
            return
        now = time.monotonic()
        with self._lock:
            if self._last_marker_at is not None:
                elapsed = now - self._last_marker_at
                if 0.001 < elapsed <= STATE_TIMEOUT_S:
                    raw_velocity = (measured - self._state[:3]) / elapsed
                    self._state[3:] = 0.30 * raw_velocity + 0.70 * self._state[3:]
                else:
                    self._state[3:] = 0.0
            self._state[:3] = measured
            self._last_marker_at = now

    def control_step(self):
        now = time.monotonic()
        with self._lock:
            fresh = self._last_marker_at is not None and now - self._last_marker_at <= STATE_TIMEOUT_S
            state = self._state.copy()
            target = None if self._target is None else self._target.copy()
        if not fresh or target is None or not np.all(np.isfinite(state)):
            self._publish_command(0.0, 0.0, HOVER_THRUST_N)
            if now - self._last_stale_log > 1.0:
                self.get_logger().warn('Holding level hover: goal or fresh WhyCode state unavailable')
                self._last_stale_log = now
            return
        reference = np.concatenate((target, np.zeros(3)))
        error = state - reference
        control = -self._gain @ error
        if not np.all(np.isfinite(control)):
            self.get_logger().error('Rejected non-finite LQR command')
            self._publish_command(0.0, 0.0, HOVER_THRUST_N)
            return
        self._publish_command(control[0], control[1], HOVER_THRUST_N + control[2])
        error_message = Error()
        error_message.pitch_error = float(error[0])
        error_message.roll_error = float(error[1])
        error_message.throttle_error = float(error[2])
        error_message.yaw_error = 0.0
        self.pos_error_pub.publish(error_message)

    def goal_callback(self, _goal_request):
        with self._lock:
            if self._active_goal:
                self.get_logger().warn('Rejecting concurrent waypoint goal')
                return GoalResponse.REJECT
            self._active_goal = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        target = np.array([goal.waypoint.position.x, goal.waypoint.position.y, goal.waypoint.position.z], dtype=float)
        if not np.all(np.isfinite(target)):
            goal_handle.abort()
            with self._lock:
                self._active_goal = False
            return NavToWaypoint.Result()

        with self._lock:
            self._target = target
        started_at = time.monotonic()
        inside_since = None
        result = NavToWaypoint.Result()
        self.get_logger().info(f'Navigating to [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}], hover={goal.hover}')
        try:
            while rclpy.ok():
                now = time.monotonic()
                with self._lock:
                    position = self._state[:3].copy()
                    fresh = self._last_marker_at is not None and now - self._last_marker_at <= STATE_TIMEOUT_S

                feedback = NavToWaypoint.Feedback()
                feedback.current_waypoint.header.stamp = self.get_clock().now().to_msg()
                feedback.current_waypoint.header.frame_id = 'whycode'
                feedback.current_waypoint.pose.position.x = float(position[0])
                feedback.current_waypoint.pose.position.y = float(position[1])
                feedback.current_waypoint.pose.position.z = float(position[2])
                feedback.current_waypoint.pose.orientation.w = 1.0
                goal_handle.publish_feedback(feedback)

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return result
                inside = fresh and bool(np.all(np.abs(position - target) <= TOLERANCE))
                if inside:
                    if not goal.hover:
                        result.hov_time = int(round(now - started_at))
                        goal_handle.succeed()
                        return result
                    if inside_since is None:
                        inside_since = now
                        self.get_logger().info('Waypoint tolerance entered; starting 3 s hover timer')
                    elif now - inside_since >= HOVER_TIME_S:
                        result.hov_time = int(round(now - started_at))
                        goal_handle.succeed()
                        self.get_logger().info(f'Waypoint held for {HOVER_TIME_S:.0f} seconds')
                        return result
                else:
                    inside_since = None
                time.sleep(0.10)
        finally:
            with self._lock:
                self._active_goal = False

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


if __name__ == '__main__':
    main()
