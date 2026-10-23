#!/usr/bin/env python3
"""Sequential Task 2 waypoint service/action client."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints


class WayPointClient(Node):
    def __init__(self):
        super().__init__('waypoint_client')
        self.action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        self.service_client = self.create_client(GetWaypoints, 'waypoints')
        self.goals = []
        self.marker_ids = []
        self.goal_index = 0
        self._requested = False
        self._sent_goal = False
        self._finished = False
        self._service_timer = self.create_timer(1.0, self.request_waypoints)
        self._server_timer = self.create_timer(0.5, self.try_start_navigation)

    def request_waypoints(self):
        if self._requested or not self.service_client.service_is_ready():
            return
        request = GetWaypoints.Request()
        request.get_waypoints = True
        self._requested = True
        future = self.service_client.call_async(request)
        future.add_done_callback(self.waypoints_callback)
        self.get_logger().info('Requested planned waypoints from service')

    def waypoints_callback(self, future):
        try:
            response = future.result()
        except Exception as error:  # ROS service errors are retryable.
            self.get_logger().warn(f'Waypoint request failed: {error}')
            self._requested = False
            return
        self.goals = list(response.waypoints.poses)
        self.marker_ids = list(response.marker_ids)
        if len(self.marker_ids) != len(self.goals):
            self.marker_ids = ['survivor:unknown'] * len(self.goals)
        if not self.goals:
            self.get_logger().error('Waypoint service returned an empty route; retrying')
            self._requested = False
            return
        self.get_logger().info(f'Received a {len(self.goals)}-waypoint route')

    def try_start_navigation(self):
        if not self.goals or self._sent_goal or self._finished:
            return
        if self.action_client.wait_for_server(timeout_sec=0.0):
            self.send_current_goal()

    def send_current_goal(self):
        if self.goal_index >= len(self.goals):
            if not self._finished:
                self._finished = True
                self.get_logger().info('All waypoints have been reached successfully')
            return
        pose = self.goals[self.goal_index]
        marker_id = self.marker_ids[self.goal_index]
        goal = NavToWaypoint.Goal()
        goal.waypoint = pose
        # Task 2C marks grid-only route vertices this way; only survivors hover.
        goal.hover = not marker_id.startswith('transit:')
        self._sent_goal = True
        self.get_logger().info(
            f'Sending {self.goal_index + 1}/{len(self.goals)}: {marker_id} '
            f'[{pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f}], hover={goal.hover}')
        future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Waypoint goal was rejected; retrying it')
            self._sent_goal = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        response = future.result()
        if response.status != 4:  # GoalStatus.STATUS_SUCCEEDED
            self.get_logger().error(f'Waypoint {self.goal_index + 1} did not succeed (status {response.status})')
            self._sent_goal = False
            return
        self.get_logger().info(f'Waypoint {self.goal_index + 1} settled in {response.result.hov_time} s')
        self.goal_index += 1
        self._sent_goal = False
        self.send_current_goal()

    def feedback_callback(self, feedback_message):
        point = feedback_message.feedback.current_waypoint.pose.position
        self.get_logger().debug(f'WhyCode feedback: [{point.x:.2f}, {point.y:.2f}, {point.z:.2f}]')


def main(args=None):
    rclpy.init(args=args)
    node = WayPointClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
