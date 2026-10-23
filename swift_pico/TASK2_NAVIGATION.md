# Task 2 waypoint navigation

The workspace now provides `waypoint_navigation/NavToWaypoint` and
`waypoint_navigation/GetWaypoints`, plus three Swift Pico executables:

```bash
ros2 run swift_pico waypoint_service
ros2 run swift_pico pico_client
ros2 run swift_pico pico_server
```

Start the simulator first, then the service and client, and start the server
last.  The client deliberately waits for the action server and dispatches a
new goal only after the prior goal has succeeded.

`waypoint_service` defaults to Task 2C mode.  It subscribes once to the
transient-local `/image_raw` arena frame, detects red triangular critical
survivors, maps the nearest grid intersection through
`grid_whycode_coordinates.json`, and generates an exact shortest Manhattan
visit order from `F6`.  Grid-only route vertices are returned as `transit:*`
and have `hover=false`; survivor vertices are returned as `survivor:*` and
have `hover=true`.

For the specified Task 2B validation route, use:

```bash
ros2 run swift_pico waypoint_service --ros-args -p route_mode:=task2b
```

The action server includes the same 50 Hz discrete outer-loop LQR model as
`pico_controller_LQR`, with one controller owner at a time.  It requires all
three WhyCode axis errors to be within ±0.4 and holds a hover waypoint for
three continuous seconds before succeeding.  It rejects stale/non-finite state
and commands level hover in that case.
