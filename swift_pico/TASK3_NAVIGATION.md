# Task 3A mission

Run these in order, each in its own terminal:

```bash
ros2 launch swift_pico swift_pico_simulation.launch.py
ros2 run swift_pico waypoint_service
ros2 run swift_pico pico_client
ros2 bag record -o task_3a /whycode_node/markers /controller_heartbeat /ros_check
ros2 run swift_pico pico_server
```

Starting `pico_server` starts the mission, so start the bag just before it.

## How it works

- `waypoint_service` reads one frame, finds the survivors and obstacles, and
  times every route through the corners of the off-limits areas, allowing for
  acceleration, braking and turn angle, then picks the fastest. It offers the critical route on
  `waypoints` and plans the stable route on `stable_route` from the position
  it is given. The stable route ends at F6.
- `pico_client` starts the clock as it sends the first goal. The Switch happens
  when the last critical Save finishes or at 55 seconds. It sends two goals per
  survivor: the Visit at Z = 13 and the Save at Z = 14.
- `pico_server` runs the LQR controller. It holds the drone at the last target
  between goals. A goal at Z = 15 means land: hold at Z = 13, then descend.
  Transit goals also carry the next waypoint (in the unused orientation field),
  so the server flies through corners in a curve, slowing only as the turn needs.

## Useful checks

```bash
# See the plan: off-limits areas in orange, obstacles in pink, route in white.
ros2 run swift_pico waypoint_service --ros-args -p debug_image:=/tmp/plan.png

# Force an early Switch.
ros2 run swift_pico pico_client --ros-args -p critical_window:=5.0

# Ask for a stable route from any position.
ros2 service call /stable_route waypoint_navigation/srv/GetStableRoute \
  "{position: {x: 2.0, y: -3.0, z: 13.0}}"

# Land at F6 by hand.
ros2 action send_goal /waypoint_navigation waypoint_navigation/action/NavToWaypoint \
  "{waypoint: {position: {x: 0.0, y: 0.0, z: 15.0}}, hover: true}"
```

The Task 2B and 2C versions of these nodes are in commit `2926897`.
