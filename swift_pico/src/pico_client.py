#!/usr/bin/env python3

"""
Task 3A mission client.

Flies the critical route, keeps the 55 second clock, decides when the Switch
happens, then fetches and flies the stable route that ends with the landing.
"""

import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetStableRoute, GetWaypoints


CRITICAL_WINDOW = 55.0  # seconds from take-off
VISIT_Z = 13.0
SAVE_Z = 14.0

# The mission phases, in order.
WAITING = "waiting"      # route not fetched or action server not up yet
CRITICAL = "critical"
SWITCHING = "switching"  # cancelling the active goal, fetching stable route
STABLE = "stable"
FINISHED = "finished"


def make_goal(pose, hover, z=None, next_pose=None):
    """
    Create a goal at a pose, optionally at a different height.

    The action only carries one pose, and its orientation is not otherwise
    used. So a transit goal stores the following waypoint there (x and y, with
    z = 1 to show it is set), which lets the server plan its turn in advance.
    """
    goal = NavToWaypoint.Goal()
    goal.waypoint = Pose()
    goal.waypoint.position.x = pose.position.x
    goal.waypoint.position.y = pose.position.y
    goal.waypoint.position.z = pose.position.z if z is None else z
    goal.waypoint.orientation.w = 1.0
    if next_pose is not None:
        goal.waypoint.orientation.x = next_pose.position.x
        goal.waypoint.orientation.y = next_pose.position.y
        goal.waypoint.orientation.z = 1.0
    goal.hover = hover
    return goal


def route_to_goals(poses, marker_ids):
    """
    Turn a route into action goals. Each survivor needs two: the Visit at
    Z = 13, then the Save at Z = 14. The home goal already has its landing
    height, and transit points are flown past without stopping.
    """
    goals = []

    for index, (pose, marker_id) in enumerate(zip(poses, marker_ids)):
        if marker_id.startswith("survivor:"):
            goals.append((make_goal(pose, True, VISIT_Z), marker_id + " visit"))
            goals.append((make_goal(pose, True, SAVE_Z), marker_id + " save"))
        elif marker_id.startswith("home:"):
            goals.append((make_goal(pose, True), marker_id + " land"))
        else:
            next_pose = poses[index + 1]
            goals.append(
                (make_goal(pose, False, next_pose=next_pose), marker_id)
            )

    return goals


class WayPointClient(Node):

    def __init__(self):
        super().__init__("waypoint_client")

        # Shorten this to test the Switch: -p critical_window:=5.0
        self.declare_parameter("critical_window", CRITICAL_WINDOW)
        self.critical_window = self.get_parameter("critical_window").value

        self.action_client = ActionClient(
            self, NavToWaypoint, "waypoint_navigation"
        )
        self.critical_client = self.create_client(GetWaypoints, "waypoints")
        self.stable_client = self.create_client(GetStableRoute, "stable_route")

        self.phase = WAITING
        self.goals = []
        self.goal_index = 0
        self.goal_handle = None     # the goal the server is flying
        self.goal_sent = False      # sent, but no reply from the server yet
        self.route_requested = False
        self.start_time = None
        self.position = None        # latest position from action feedback

        self.create_timer(0.05, self.mission_step)
        self.get_logger().info(
            f"Critical window is {self.critical_window:.0f} seconds"
        )

    def elapsed(self):
        return time.monotonic() - self.start_time

    def mission_step(self):
        """Runs 20 times a second and moves the mission between phases."""
        if self.phase == WAITING:
            self.request_critical_route()

            # The clock starts as the first goal goes out, because the drone
            # lifts off as soon as it receives that goal.
            if self.goals and self.action_client.server_is_ready():
                self.start_time = time.monotonic()
                self.phase = CRITICAL
                self.get_logger().info("Take-off: critical clock started")
                self.send_current_goal()

        elif self.phase == CRITICAL:
            if self.elapsed() >= self.critical_window:
                self.get_logger().warning("Critical window is over")
                self.start_switch()

    # ------------------------------------------------------- Critical route

    def request_critical_route(self):
        if self.route_requested or not self.critical_client.service_is_ready():
            return

        self.route_requested = True
        request = GetWaypoints.Request()
        request.get_waypoints = True
        future = self.critical_client.call_async(request)
        future.add_done_callback(self.critical_route_callback)
        self.get_logger().info("Waiting for the critical route")

    def critical_route_callback(self, future):
        response = future.result()
        self.goals = route_to_goals(
            response.waypoints.poses, response.marker_ids
        )
        self.goal_index = 0
        self.get_logger().info(f"Critical route has {len(self.goals)} goals")

    # ---------------------------------------------------------- Flying goals

    def send_current_goal(self):
        goal, name = self.goals[self.goal_index]
        position = goal.waypoint.position

        self.get_logger().info(
            f"[{self.elapsed():5.1f}s] {name} "
            f"[{position.x:.2f}, {position.y:.2f}, {position.z:.2f}]"
        )

        self.goal_sent = True
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, message):
        self.position = message.feedback.current_waypoint.pose.position

    def goal_response_callback(self, future):
        self.goal_sent = False
        handle = future.result()

        if not handle.accepted:
            if self.phase == SWITCHING:
                self.request_stable_route()
            else:
                self.get_logger().warning("Goal rejected, sending it again")
                self.send_current_goal()
            return

        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.result_callback)

        # The Switch can happen while this goal is on its way to the server.
        if self.phase == SWITCHING:
            handle.cancel_goal_async()

    def result_callback(self, future):
        self.goal_handle = None
        status = future.result().status

        # Whether the cancel won or the goal just finished, it is over now.
        if self.phase == SWITCHING:
            self.request_stable_route()
            return

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f"Goal failed (status {status}), retrying")
            self.send_current_goal()
            return

        self.goal_index += 1
        if self.goal_index < len(self.goals):
            self.send_current_goal()
        elif self.phase == CRITICAL:
            self.get_logger().info(
                f"[{self.elapsed():5.1f}s] All critical survivors saved"
            )
            self.start_switch()
        else:
            self.phase = FINISHED
            self.get_logger().info(
                f"[{self.elapsed():5.1f}s] Landed. Mission complete"
            )

    # ------------------------------------------------------------ The Switch

    def start_switch(self):
        self.phase = SWITCHING
        self.get_logger().info(f"[{self.elapsed():5.1f}s] The Switch")

        # result_callback or goal_response_callback carry on from the cancel.
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        elif not self.goal_sent:
            self.request_stable_route()

    def request_stable_route(self):
        request = GetStableRoute.Request()
        request.position = self.position
        future = self.stable_client.call_async(request)
        future.add_done_callback(self.stable_route_callback)

    def stable_route_callback(self, future):
        response = future.result()
        self.goals = route_to_goals(
            response.waypoints.poses, response.marker_ids
        )
        self.goal_index = 0
        self.phase = STABLE
        self.get_logger().info(f"Stable route has {len(self.goals)} goals")
        self.send_current_goal()


def main(args=None):
    rclpy.init(args=args)
    node = WayPointClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
