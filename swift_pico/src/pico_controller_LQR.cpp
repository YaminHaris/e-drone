// Outer-loop position controller for the Swift Pico drone.
//
// The simulator's own attitude controller is the inner loop, so we only command
// small pitch/roll angles plus a thrust offset. The plant is modelled as a
// small-angle point mass:   input u = [pitch, roll, delta_thrust]
//                            state x = [px, py, pz, vx, vy, vz]
//
// Axes are WhyCode's camera frame (x, y, depth), NOT world axes:
//   - WhyCode y is opposite world y
//   - WhyCode depth is opposite world-up, so more thrust => smaller depth

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "error_msg/msg/error.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "swift_msgs/msg/swift_msgs.hpp"
#include "whycode_interfaces/msg/marker_array.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

// ---------------------------------------------------------------------------
// LQR math
// ---------------------------------------------------------------------------
namespace lqr {

using State = Eigen::Matrix<double, 6, 1>;
using Input = Eigen::Matrix<double, 3, 1>;
using StateMatrix = Eigen::Matrix<double, 6, 6>;
using InputMatrix = Eigen::Matrix<double, 6, 3>;
using GainMatrix = Eigen::Matrix<double, 3, 6>;

// Discrete double-integrator model per axis: position += velocity * dt,
// velocity += acceleration * dt, where acceleration comes from tilt (x, y)
// or thrust (depth).
void make_model(double dt, double mass, double gravity, StateMatrix *a, InputMatrix *b) {
  if (dt <= 0.0 || mass <= 0.0 || gravity <= 0.0) {
    throw std::invalid_argument("dt, mass and gravity must be positive");
  }
  const double half_dt2 = 0.5 * dt * dt;

  *a = StateMatrix::Identity();
  a->block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();

  *b = InputMatrix::Zero();
  (*b)(0, 0) = half_dt2 * gravity;  // pitch -> x
  (*b)(3, 0) = dt * gravity;
  (*b)(1, 1) = half_dt2 * gravity;  // roll  -> y
  (*b)(4, 1) = dt * gravity;
  (*b)(2, 2) = -half_dt2 / mass;    // extra thrust -> smaller depth
  (*b)(5, 2) = -dt / mass;
}

// K = (R + B'PB)^-1 B'PA for a given Riccati matrix P.
GainMatrix gain_from(const StateMatrix &a, const InputMatrix &b, const StateMatrix &p,
                     const Eigen::Matrix3d &r) {
  const Eigen::Matrix3d s = r + b.transpose() * p * b;
  const Eigen::LDLT<Eigen::Matrix3d> ldlt(s);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    throw std::runtime_error("R + B'PB is not positive definite");
  }
  return ldlt.solve(b.transpose() * p * a);
}

// Solve the discrete Riccati equation by plain fixed-point iteration.
// Slow-ish but it only runs once at startup and needs nothing but Eigen.
GainMatrix solve(const StateMatrix &a, const InputMatrix &b, const StateMatrix &q,
                 const Eigen::Matrix3d &r) {
  constexpr int kMaxIterations = 20000;
  constexpr double kTolerance = 1e-9;

  StateMatrix p = q;
  for (int i = 0; i < kMaxIterations; ++i) {
    const GainMatrix k = gain_from(a, b, p, r);
    const StateMatrix next = q + a.transpose() * p * a - a.transpose() * p * b * k;
    if (!next.allFinite()) throw std::runtime_error("Riccati iteration blew up");

    if ((next - p).cwiseAbs().maxCoeff() < kTolerance) {
      return gain_from(a, b, next, r);
    }
    p = next;
  }
  throw std::runtime_error("Riccati iteration did not converge");
}

// Closed loop (A - BK) is stable if every eigenvalue is inside the unit circle.
bool is_stable(const StateMatrix &a, const InputMatrix &b, const GainMatrix &k) {
  const Eigen::EigenSolver<StateMatrix> solver(a - b * k, false);
  return solver.info() == Eigen::Success &&
         solver.eigenvalues().array().abs().maxCoeff() < 1.0;
}

}  // namespace lqr

// ---------------------------------------------------------------------------
// Controller node
// ---------------------------------------------------------------------------
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMassKg = 1.5;
constexpr double kGravity = 9.81;
constexpr double kControlPeriod = 0.02;  // 50 Hz

// Hover thrust is calibrated against the sim, so it's slightly above m*g.
constexpr double kHoverThrustN = 14.896;
constexpr double kMaxThrustN = 30.0;

// The interface maps +/-4.5 deg onto the RC span. We stay at 3 deg so the outer
// loop only asks the inner attitude loop for gentle corrections.
constexpr double kRcTiltLimitRad = 4.5 * kPi / 180.0;
constexpr double kMaxTiltRad = 3.0 * kPi / 180.0;

constexpr double kRcCenter = 1500.0;
constexpr double kRcTiltSpan = 85.0;

constexpr double kMarkerTimeout = 0.20;  // seconds before we call the state stale
constexpr int kDisarmTicks = 5;          // send "disarm" this long before arming

double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

bool has_valid_marker(const whycode_interfaces::msg::MarkerArray &msg) {
  if (msg.markers.empty()) return false;
  const auto &p = msg.markers.front().position.position;
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

}  // namespace

class PicoController final : public rclcpp::Node {
 public:
  PicoController() : Node("pico_controller") {
    target_ << declare_parameter("target_x", 1.5), declare_parameter("target_y", -1.5),
        declare_parameter("target_z", 13.0);

    // Tuning knobs (all live ROS params so you can change them without rebuilding).
    const double q_pos = declare_parameter("q_xy_position", 5.0);
    const double q_vel = declare_parameter("q_xy_velocity", 1.0);
    const double r_tilt = declare_parameter("r_tilt", 100.0);
    alpha_ = declare_parameter("filter_alpha", 0.6);  // position trust in a new marker
    beta_ = declare_parameter("filter_beta", 0.2);    // velocity trust in a new marker

    setup_lqr(q_pos, q_vel, r_tilt);

    command_pub_ = create_publisher<swift_msgs::msg::SwiftMsgs>("/drone_command", 10);
    error_pub_ = create_publisher<error_msg::msg::Error>("/pos_error", 10);
    heartbeat_pub_ = create_publisher<std_msgs::msg::Header>("/controller_heartbeat", 10);
    marker_sub_ = create_subscription<whycode_interfaces::msg::MarkerArray>(
        "/whycode_node/markers", 1, std::bind(&PicoController::on_marker, this, _1));

    heartbeat_timer_ = create_wall_timer(100ms, std::bind(&PicoController::publish_heartbeat, this));
    control_timer_ = create_wall_timer(20ms, std::bind(&PicoController::control_step, this));
    publish_heartbeat();

    // Let `ros2 param set /pico_controller target_x 0.0` move the setpoint live.
    param_cb_ = add_on_set_parameters_callback(
        std::bind(&PicoController::on_parameters, this, _1));
  }

 private:
  using Clock = std::chrono::steady_clock;

  // LQR
  lqr::GainMatrix gain_ = lqr::GainMatrix::Zero();
  bool lqr_ready_ = false;
  Eigen::Vector3d target_ = Eigen::Vector3d::Zero();

  // Estimated position/velocity from WhyCode (alpha-beta filter)
  Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();
  double alpha_ = 0.6;
  double beta_ = 0.2;
  bool tracking_ = false;
  Clock::time_point last_marker_time_{};
  Clock::time_point last_warning_time_{};

  int startup_ticks_ = 0;

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  rclcpp::Publisher<swift_msgs::msg::SwiftMsgs>::SharedPtr command_pub_;
  rclcpp::Publisher<error_msg::msg::Error>::SharedPtr error_pub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr heartbeat_pub_;
  rclcpp::Subscription<whycode_interfaces::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  void setup_lqr(double q_pos, double q_vel, double r_tilt) {
    lqr::StateMatrix a;
    lqr::InputMatrix b;
    lqr::make_model(kControlPeriod, kMassKg, kGravity, &a, &b);

    lqr::StateMatrix q = lqr::StateMatrix::Zero();
    q.diagonal() << q_pos, q_pos, 20.0, q_vel, q_vel, 6.0;
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    r.diagonal() << r_tilt, r_tilt, 0.8;

    try {
      gain_ = lqr::solve(a, b, q, r);
      if (!lqr::is_stable(a, b, gain_)) throw std::runtime_error("closed loop is unstable");
      lqr_ready_ = true;
      RCLCPP_INFO(get_logger(), "LQR ready at %.0f Hz, target [%.2f, %.2f, %.2f]",
                  1.0 / kControlPeriod, target_(0), target_(1), target_(2));
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "LQR disabled: %s", e.what());
    }
  }

  // Only the target can change live. The LQR gains and filter settings are read
  // once at startup, so changing those still needs a restart.
  rcl_interfaces::msg::SetParametersResult on_parameters(
      const std::vector<rclcpp::Parameter> &params) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    Eigen::Vector3d new_target = target_;
    for (const auto &p : params) {
      const std::string &name = p.get_name();
      if (name != "target_x" && name != "target_y" && name != "target_z") continue;

      const double value = p.as_double();
      if (!std::isfinite(value)) {
        result.successful = false;
        result.reason = name + " must be finite";
        return result;
      }
      if (name == "target_x") new_target(0) = value;
      if (name == "target_y") new_target(1) = value;
      if (name == "target_z") new_target(2) = value;
    }

    // Same executor thread as the control timer, so no locking needed.
    if (new_target != target_) {
      target_ = new_target;
      RCLCPP_INFO(get_logger(), "New target [%.2f, %.2f, %.2f]",
                  target_(0), target_(1), target_(2));
    }
    return result;
  }

  void publish_heartbeat() {
    std_msgs::msg::Header msg;
    msg.stamp = get_clock()->now();
    msg.frame_id = "pico_controller_active";
    heartbeat_pub_->publish(msg);
  }

  // Marker callback: update the filtered position and velocity.
  void on_marker(const whycode_interfaces::msg::MarkerArray &msg) {
    if (!has_valid_marker(msg)) {
      RCLCPP_WARN(get_logger(), "Ignoring empty or non-finite marker message");
      return;
    }
    const auto &p = msg.markers.front().position.position;
    const Eigen::Vector3d measured(p.x, p.y, p.z);
    const auto now = Clock::now();

    if (!tracking_) {
      pos_ = measured;
      vel_.setZero();
    } else {
      const double dt = std::chrono::duration<double>(now - last_marker_time_).count();
      if (dt > kMarkerTimeout) {
        // Lost the marker for a while: restart instead of trusting old velocity.
        pos_ = measured;
        vel_.setZero();
      } else if (dt > 1e-3) {
        const Eigen::Vector3d predicted = pos_ + vel_ * dt;
        const Eigen::Vector3d residual = measured - predicted;
        pos_ = predicted + alpha_ * residual;
        vel_ += (beta_ / dt) * residual;
      } else {
        return;  // duplicate message in the same instant, nothing to learn
      }
    }
    last_marker_time_ = now;
    tracking_ = true;
  }

  // Convert desired tilt (rad) and total thrust (N) into the RC message.
  void publish_command(double pitch_rad, double roll_rad, double thrust_n) {
    pitch_rad = clamp(pitch_rad, -kMaxTiltRad, kMaxTiltRad);
    roll_rad = clamp(roll_rad, -kMaxTiltRad, kMaxTiltRad);
    thrust_n = clamp(thrust_n, 0.0, kMaxThrustN);

    // Thrust maps piecewise: below hover 1000..1500, above hover 1500..2000.
    const double thrust_range = (thrust_n >= kHoverThrustN) ? (kMaxThrustN - kHoverThrustN)
                                                            : kHoverThrustN;
    const double throttle = kRcCenter + (thrust_n - kHoverThrustN) * 500.0 / thrust_range;

    swift_msgs::msg::SwiftMsgs cmd;
    cmd.rc_pitch = std::lround(kRcCenter + pitch_rad * kRcTiltSpan / kRcTiltLimitRad);
    cmd.rc_roll = std::lround(kRcCenter + roll_rad * kRcTiltSpan / kRcTiltLimitRad);
    cmd.rc_yaw = 1500;
    cmd.rc_throttle = std::lround(clamp(throttle, 1000.0, 2000.0));
    cmd.rc_aux4 = 2000;  // armed
    command_pub_->publish(cmd);
  }

  void publish_disarm() {
    swift_msgs::msg::SwiftMsgs cmd;
    cmd.rc_roll = cmd.rc_pitch = cmd.rc_yaw = cmd.rc_throttle = cmd.rc_aux4 = 1000;
    command_pub_->publish(cmd);
  }

  void hold_neutral() { publish_command(0.0, 0.0, kHoverThrustN); }

  void control_step() {
    // Disarm for a few ticks first so the sim actually sees the arming edge.
    if (startup_ticks_ < kDisarmTicks) {
      publish_disarm();
      ++startup_ticks_;
      return;
    }

    const auto now = Clock::now();
    const double age = tracking_ ? std::chrono::duration<double>(now - last_marker_time_).count()
                                 : 1e9;

    if (!lqr_ready_ || age > kMarkerTimeout) {
      hold_neutral();
      if (now - last_warning_time_ > 1s) {
        RCLCPP_WARN(get_logger(), "Holding neutral: LQR not ready or marker data is stale");
        last_warning_time_ = now;
      }
      return;
    }

    // Markers arrive slower than the 50 Hz loop, so extrapolate the position to
    // "now" instead of feeding the controller a stair-step signal.
    lqr::State state;
    state << (pos_ + vel_ * age), vel_;

    lqr::State error = state;
    error.head<3>() -= target_;

    const lqr::Input u = -gain_ * error;
    if (!u.allFinite()) {
      RCLCPP_ERROR(get_logger(), "Rejected non-finite LQR output");
      hold_neutral();
      return;
    }

    publish_command(u(0), u(1), kHoverThrustN + u(2));

    error_msg::msg::Error err;
    err.pitch_error = static_cast<float>(error(0));
    err.roll_error = static_cast<float>(error(1));
    err.throttle_error = static_cast<float>(error(2));
    err.yaw_error = 0.0F;
    error_pub_->publish(err);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PicoController>());
  rclcpp::shutdown();
  return 0;
}