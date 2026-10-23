# Swift Pico discrete LQR controller

`pico_controller_LQR` is the completed C++ Task 2 simulation controller. It is
an outer-loop position controller and deliberately retains the simulator's
existing attitude controller as the inner loop.

## Model and control law

`x = [x, y, z, x_dot, y_dot, z_dot]^T` and
`x_ref = [target_x, target_y, target_z, 0, 0, 0]^T`. The input is
`u = [pitch, roll, dT]^T`, with `dT` thrust above calibrated hover. These are
WhyCode camera-centric coordinates, not world-up coordinates: in this simulator
the top camera is at 15 m, WhyCode `y` is opposite world y, and WhyCode `z` is
camera depth. Thus around level flight `x_ddot = g*pitch`, `y_ddot = g*roll`,
and `z_ddot = -dT/m`. A physical 3 m hover is therefore `target_z:=12.0`.
The model is discretized at 50 Hz with a zero-order hold. The controller uses
`u = -K(x - x_ref)` and solves the discrete Riccati equation with Eigen, not a
hard-coded gain.

The default lateral tuning is `Q = diag(10, 10, 20, 8, 8, 6)` and
`R = diag(10, 10, 0.8)`. `q_xy_position`, `q_xy_velocity`, and `r_tilt` are
ROS parameters, so lateral Q/R candidates can be tested against recorded error
instead of guessed. The controller uses a 3 degree operational
tilt limit (inside the simulator's verified 4.5 degree interface limit). The
simulator declares 1.5 kg, 9.81 m/s², a hover map of 1500 RC
to 14.896 N, thrust limits 0--30 N, and +/-4.5 degrees at +/-85 RC units. The
controller works in radians/newtons, inverts this verified mapping, and limits
commands only to those interface limits. It is a local, level-flight model, not
a global nonlinear flight controller.


## Interface and safeguards

The node preserves `/drone_command`, `/pos_error`, `/controller_heartbeat`, and
`/whycode_node/markers`. It defaults to the supplied target `[1, -1, 13]` m;
`target_x`, `target_y`, and `target_z` are parameters. It rejects empty and
non-finite markers, estimates/filter velocities, and sends a neutral-level
hover command if state feedback is older than 0.20 s, a matrix fails, or a
command is non-finite. This is simulation software only, not hardware-ready.

## Build and test

```bash
source /opt/ros/humble/setup.bash
cd /workspace/pico_ws
colcon build --packages-select swift_pico
source install/setup.bash
ctest --test-dir build/swift_pico --output-on-failure
ros2 run swift_pico pico_controller_LQR
```

Launch the simulation separately and run only one controller at a time: PID and
LQR share the node name and command topics. The math test checks finite gain
generation, closed-loop discrete stability, and x/y/z control signs.
