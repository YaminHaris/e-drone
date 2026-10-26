#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given drone.
This node publishes and subsribes the following topics:

        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command           /whycode_node/markers
        /pos_error
        /controller_heartbeat

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

import math
import time

import rclpy
from rclpy.node import Node
from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from std_msgs.msg import Header

# predefine constant for use

CONTROL_PERIOD = 0.02  # controller will run every 0.02sec or 50 Hz
HOVER_THRUST_N = 14.896 # amount of thrust to be stable at one position
MAX_THRUST_N = 30.0 # max thrust allowed
RC_TILT_LIMIT_RAD = math.radians(4.5)
MAX_TILT_RAD = math.radians(3.0)
RC_CENTER = 1500.0
RC_TILT_SPAN = 85.0
MARKER_TIMEOUT = 0.20
DISARM_TICKS = 5
FILTER_ALPHA = 0.6 # how much we can trust the new measure vale of position
FILTER_BETA = 0.2 # how much we can trust the new measure vale of velocity


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller') # node name

        self.desired_state = [1.5, -1.5, 13.0] # this is our desire position

        # this is LQR gain matrix calcualte by using A,B,Q,R - the best optimmal gain that is possible

        self.K = [
            [0.098304489591, 0.0,             0.0,              0.172352727574, 0.0,             0.0],
            [0.0,             0.098304489591, 0.0,              0.0,             0.172352727574, 0.0],
            [0.0,             0.0,            -4.844368481057,  0.0,             0.0,            -4.644725744029],
        ]

        self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # first three are position of drone and next three are velocity of each direction x,y,z

        self.position = [0.0, 0.0, 0.0] # inital  position of drone
        self.velocity = [0.0, 0.0, 0.0]
        self.tracking = False # means not get any marker at
        self.last_marker_time = None
        self.last_warning_time = 0.0
        self.startup_ticks = 0

        self.cmd = SwiftMsgs() # just initalizing the swift msg
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1000

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)# publice  roll,pitch ,  yaw,throttle value so drone can move
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10) # publice error = desire-current position
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)
        self.marker_sub = self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1) # subscribeer that recive /whycode_node/markers' msg in MarkerArray

        self.heartbeat_timer = self.create_timer(0.1, self.publish_heartbeat) # run 10 time per sec
        self.control_timer = self.create_timer(CONTROL_PERIOD, self.control_step)
        self.publish_heartbeat()
        self.get_logger().info(
            f'LQR controller started , Target={self.desired_state}, '
            f'control rate={1.0 / CONTROL_PERIOD:.0f} Hz'
        )

    def publish_heartbeat(self): # define the heartbeat function
        heartbeat = Header()
        heartbeat.stamp = self.get_clock().now().to_msg()
        heartbeat.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(heartbeat)

    def whycode_callback(self, msg): # run everytime when Maker array recieve

        if not msg.markers:
            return

        marker_position = msg.markers[0].position.position
        measured = [  # get all the position of the drone
            float(marker_position.x),
            float(marker_position.y),
            float(marker_position.z),
        ]

        if not all(math.isfinite(value) for value in measured):
            self.get_logger().warning('Ignoring non-finite WhyCode marker position')
            return

        now = time.monotonic()
        if not self.tracking: # this is to check if it's  first time getting marker vaalue

            self.position = measured
            self.velocity = [0.0, 0.0, 0.0]

        else:
            dt = now - self.last_marker_time  # calcualting time since previous marker

            # this is to aviod small interval or 0  division error
            if dt <= 1e-3:
                return

            # reset the velocity if marker data is not there for long time
            if dt > MARKER_TIMEOUT:
                self.position = measured
                self.velocity = [0.0, 0.0, 0.0]
            else:
                predicted = [
                    self.position[i] + self.velocity[i] * dt
                    for i in range(3)
                ]
                residual = [
                    measured[i] - predicted[i]
                    for i in range(3)
                ]
                self.position = [
                    predicted[i] + FILTER_ALPHA * residual[i]
                    for i in range(3)
                ]
                self.velocity = [
                    self.velocity[i] + (FILTER_BETA / dt) * residual[i]
                    for i in range(3)
                ]

        # Save the latest measurement time and state.
        self.last_marker_time = now
        self.tracking = True
        self.current_state = self.position + self.velocity

    def clamp(self, value, lower, upper):  # use to keep  values under allowed range
        return max(lower, min(value, upper))

    def publish_command(self, pitch_rad, roll_rad, thrust_n): # convert the requessted angle  or thrust into RC cammand and publice them

        pitch_rad = self.clamp(pitch_rad, -MAX_TILT_RAD, MAX_TILT_RAD) # limit pitch angle +-3 degree
        roll_rad = self.clamp(roll_rad, -MAX_TILT_RAD, MAX_TILT_RAD) # limit roll angle +-3 degree
        thrust_n = self.clamp(thrust_n, 0.0, MAX_THRUST_N) # limit the thrust max shoud be 30

        if thrust_n >= HOVER_THRUST_N:
            thrust_range = MAX_THRUST_N - HOVER_THRUST_N
        else:
            thrust_range = HOVER_THRUST_N

        # converting thrust to throttle
        throttle = RC_CENTER + (thrust_n - HOVER_THRUST_N) * 500.0 / thrust_range

        cmd = SwiftMsgs()
        cmd.rc_pitch = int(round(RC_CENTER + pitch_rad * RC_TILT_SPAN / RC_TILT_LIMIT_RAD))
        cmd.rc_roll = int(round(RC_CENTER + roll_rad * RC_TILT_SPAN / RC_TILT_LIMIT_RAD))
        cmd.rc_yaw = 1500
        cmd.rc_throttle = int(round(self.clamp(throttle, 1000.0, 2000.0)))
        cmd.rc_aux4 = 2000  # armed
        self.command_pub.publish(cmd)

    def publish_disarm(self): # define fuction to use for disarm command
        cmd = SwiftMsgs()
        cmd.rc_roll = 1000
        cmd.rc_pitch = 1000
        cmd.rc_yaw = 1000
        cmd.rc_throttle = 1000
        cmd.rc_aux4 = 1000
        self.command_pub.publish(cmd)

    def hold_neutral(self):  # it just hovering setting at a place
        self.publish_command(0.0, 0.0, HOVER_THRUST_N)

    def control_step(self):
        # first disarm the drone for a few cycles  this helps the simulator detect next arming command

        if self.startup_ticks < DISARM_TICKS:
            self.publish_disarm()
            self.startup_ticks += 1
            return

        # get the current time in second
        now = time.monotonic()

        # checking that did we receive marker data
        if not self.tracking or self.last_marker_time is None:
            self.hold_neutral()  # Send neutral pitch, roll and hover thrust
            return

        # checking how old our marker data is
        age = now - self.last_marker_time

        # if data is more than 0.2 sec old than we will not use it
        if age > MARKER_TIMEOUT:
            self.hold_neutral()

            if now - self.last_warning_time > 1.0:
                self.get_logger().warning(
                    'Holding neutral: no fresh WhyCode marker data'
                )
                self.last_warning_time = now

            return

        # predict the current X, Y and Z positions using velocity

        state = [
            self.current_state[i] + self.current_state[i + 3] * age
            for i in range(3)
        ] + self.current_state[3:6]  # add X, Y and Z velocity

        # calculate the error between current position and target
        # For velocity use the current velocity as the error
        error = [
            state[i] - self.desired_state[i] if i < 3 else state[i]
            for i in range(6)
        ]

        # calculating u = -K * error ,outputs are pitch, roll ,thrust

        control = [
            -sum(self.K[r][c] * error[c] for c in range(6))
            for r in range(3)
        ]

        # if calculation are uneven than pass hovering setting
        if not all(math.isfinite(u) for u in control):
            self.hold_neutral()
            return

        # separate the three control outputs
        pitch, roll, thrust = control

        # send the commands to the drone and add the extra thrust reqquired

        self.publish_command(
            pitch, roll, HOVER_THRUST_N + thrust
        )

        # create msg to send errors
        pos_error = Error()

        pos_error.pitch_error = float(error[0])
        pos_error.roll_error = float(error[1])
        pos_error.throttle_error = float(error[2])

        # we do not calculate yaw error
        pos_error.yaw_error = 0.0

        # publish  error message on the /pos_error topic
        self.pos_error_pub.publish(pos_error)


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
