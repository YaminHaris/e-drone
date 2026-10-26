#!/usr/bin/env python3

"""
Simple Task 2B waypoint service.

Run using:
ros2 run swift_pico waypoint_service_task2b

This node sends four fixed waypoints whenever the drone requests them.
No image processing or route planning is needed for Task 2B.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from waypoint_navigation.srv import GetWaypoints


# These are the four waypoints the drone needs to visit, in order.
# Each waypoint contains a name, an x coordinate, and a y coordinate.
WAYPOINTS = [
    ("wp1", -2.41, -2.41),
    ("wp2",  2.41,  1.20),
    ("wp3",  1.20, -2.41),
    ("wp4", -1.20,  1.20),
]

# Keep the drone at the same altitude for all four waypoints.
ALTITUDE = 13.0


def create_pose(x, y):
    """Create a waypoint at the given position."""
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = ALTITUDE

    # No rotation is needed, so use the default orientation.
    pose.orientation.w = 1.0

    return pose


class WaypointService(Node):
    def __init__(self):
        super().__init__("waypoints_service")

        # Build the route once when the node starts.
        self.waypoints = PoseArray()
        self.waypoints.header.frame_id = "whycode"

        self.marker_ids = []

        for name, x, y in WAYPOINTS:
            self.waypoints.poses.append(create_pose(x, y))
            self.marker_ids.append(name)

        # The navigation system calls this service to request the route.
        self.service = self.create_service(
            GetWaypoints,
            "waypoints",
            self.waypoint_callback
        )

        self.get_logger().info("Task 2B waypoint service is ready.")

    def waypoint_callback(self, request, response):
        # If no waypoints were requested, return an empty response.
        if not request.get_waypoints:
            response.waypoints = PoseArray()
            response.marker_ids = []
            return response

        # Return the four hardcoded waypoints in their original order.
        response.waypoints = PoseArray()
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.waypoints.header.frame_id = "whycode"
        response.waypoints.poses = list(self.waypoints.poses)

        response.marker_ids = list(self.marker_ids)

        self.get_logger().info("Sent the four Task 2B waypoints.")
        return response


def main(args=None):
    rclpy.init(args=args)

    node = WaypointService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
