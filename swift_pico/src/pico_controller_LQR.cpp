#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

#include "error_msg/msg/error.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "swift_msgs/msg/swift_msgs.hpp"
#include "whycode_interfaces/msg/marker_array.hpp"

#include "lqr_controller_math.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

namespace {
constexpr double kMassKg = 1.5;
constexpr double kGravity = 9.81;
constexpr double kControlPeriodSeconds = 0.02;
constexpr double kHoverThrustN = 14.896;
constexpr double kMaximumThrustN = 30.0;
constexpr double kMaximumTiltRad = 4.5 * M_PI / 180.0;
// Deliberately below the 4.5 degree interface limit: the outer loop asks the
// simulator's inner attitude loop for gentle corrections, not aggressive bank.
constexpr double kOperationalMaximumTiltRad = 3.0 * M_PI / 180.0;
constexpr double kRcCenter = 1500.0;
constexpr double kRcTiltSpan = 85.0;
constexpr double kStateTimeoutSeconds = 0.20;

double clamp(const double value, const double low, const double high) {
  return std::max(low, std::min(value, high));
}

bool finite_marker(const whycode_interfaces::msg::MarkerArray &message) {
  if (message.markers.empty()) return false;
  const auto &p = message.markers.front().position.position;
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
}  // namespace

class SwiftPicoLqr final : public rclcpp::Node {
 public:
  SwiftPicoLqr() : Node("pico_controller") {
    reference_ << this->declare_parameter<double>("target_x", 1.0),
        this->declare_parameter<double>("target_y", -1.0),
        this->declare_parameter<double>("target_z", 13.0), 0.0, 0.0, 0.0;
    const double q_xy_position = this->declare_parameter<double>("q_xy_position", 10.0);
    const double q_xy_velocity = this->declare_parameter<double>("q_xy_velocity", 8.0);
    const double r_tilt = this->declare_parameter<double>("r_tilt", 10.0);

    swift_pico::lqr::StateMatrix a;
    swift_pico::lqr::InputMatrix b;
    swift_pico::lqr::make_discrete_model(kControlPeriodSeconds, kMassKg, kGravity, &a, &b);
    swift_pico::lqr::CostMatrix q = swift_pico::lqr::CostMatrix::Zero();
    q.diagonal() << q_xy_position, q_xy_position, 20.0,
                    q_xy_velocity, q_xy_velocity, 6.0;
    swift_pico::lqr::InputCostMatrix r = swift_pico::lqr::InputCostMatrix::Zero();
    r.diagonal() << r_tilt, r_tilt, 0.8;
    try {
      gain_ = swift_pico::lqr::solve_discrete_lqr(a, b, q, r);
      if (!swift_pico::lqr::is_stable(a, b, gain_)) throw std::runtime_error("closed-loop discrete model is unstable");
      lqr_ready_ = true;
      RCLCPP_INFO(this->get_logger(), "LQR ready at %.0f Hz; reference [%.2f, %.2f, %.2f]",
                  1.0 / kControlPeriodSeconds, reference_(0), reference_(1), reference_(2));
    } catch (const std::exception &error) {
      RCLCPP_ERROR(this->get_logger(), "LQR disabled: %s", error.what());
    }

    command_pub_ = this->create_publisher<swift_msgs::msg::SwiftMsgs>("/drone_command", 10);
    error_pub_ = this->create_publisher<error_msg::msg::Error>("/pos_error", 10);
    heartbeat_pub_ = this->create_publisher<std_msgs::msg::Header>("/controller_heartbeat", 10);
    marker_sub_ = this->create_subscription<whycode_interfaces::msg::MarkerArray>(
        "/whycode_node/markers", 1, std::bind(&SwiftPicoLqr::marker_callback, this, _1));
    heartbeat_timer_ = this->create_wall_timer(100ms, std::bind(&SwiftPicoLqr::publish_heartbeat, this));
    control_timer_ = this->create_wall_timer(20ms, std::bind(&SwiftPicoLqr::control_step, this));
    publish_heartbeat();
    arm();
  }

 private:
  using Clock = std::chrono::steady_clock;
  swift_pico::lqr::State state_ = swift_pico::lqr::State::Zero();
  swift_pico::lqr::State reference_ = swift_pico::lqr::State::Zero();
  swift_pico::lqr::GainMatrix gain_ = swift_pico::lqr::GainMatrix::Zero();
  bool lqr_ready_{false};
  bool have_marker_{false};
  Clock::time_point last_marker_time_{};
  Clock::time_point last_stale_warning_{};
  rclcpp::Publisher<swift_msgs::msg::SwiftMsgs>::SharedPtr command_pub_;
  rclcpp::Publisher<error_msg::msg::Error>::SharedPtr error_pub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr heartbeat_pub_;
  rclcpp::Subscription<whycode_interfaces::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  void publish_heartbeat() {
    std_msgs::msg::Header heartbeat;
    heartbeat.stamp = this->get_clock()->now();
    heartbeat.frame_id = "pico_controller_active";
    heartbeat_pub_->publish(heartbeat);
  }

  void publish_command(const double pitch_rad, const double roll_rad, const double total_thrust_n) {
    swift_msgs::msg::SwiftMsgs command;
    const double pitch = clamp(pitch_rad, -kOperationalMaximumTiltRad, kOperationalMaximumTiltRad);
    const double roll = clamp(roll_rad, -kOperationalMaximumTiltRad, kOperationalMaximumTiltRad);
    const double thrust = clamp(total_thrust_n, 0.0, kMaximumThrustN);
    command.rc_pitch = std::lround(kRcCenter + pitch * kRcTiltSpan / kMaximumTiltRad);
    command.rc_roll = std::lround(kRcCenter + roll * kRcTiltSpan / kMaximumTiltRad);
    command.rc_yaw = 1500;
    command.rc_aux4 = 2000;
    if (thrust >= kHoverThrustN) {
      command.rc_throttle = std::lround(kRcCenter +
          (thrust - kHoverThrustN) * 500.0 / (kMaximumThrustN - kHoverThrustN));
    } else {
      command.rc_throttle = std::lround(kRcCenter +
          (thrust - kHoverThrustN) * 500.0 / kHoverThrustN);
    }
    command.rc_throttle = static_cast<int64_t>(clamp(command.rc_throttle, 1000.0, 2000.0));
    command_pub_->publish(command);
  }

  void arm() {
    swift_msgs::msg::SwiftMsgs disarm;
    disarm.rc_roll = 1000;
    disarm.rc_pitch = 1000;
    disarm.rc_yaw = 1000;
    disarm.rc_throttle = 1000;
    disarm.rc_aux4 = 1000;
    command_pub_->publish(disarm);
    publish_command(0.0, 0.0, kHoverThrustN);
  }

  void marker_callback(const whycode_interfaces::msg::MarkerArray &message) {
    if (!finite_marker(message)) {
      RCLCPP_WARN(this->get_logger(), "Ignoring empty or non-finite WhyCode marker message");
      return;
    }
    const auto now = Clock::now();
    const auto &position = message.markers.front().position.position;
    const Eigen::Vector3d measured(position.x, position.y, position.z);
    if (have_marker_) {
      const double dt = std::chrono::duration<double>(now - last_marker_time_).count();
      if (dt > 0.001 && dt <= kStateTimeoutSeconds) {
        const Eigen::Vector3d raw_velocity = (measured - state_.head<3>()) / dt;
        state_.tail<3>() = 0.30 * raw_velocity + 0.70 * state_.tail<3>();
      } else {
        state_.tail<3>().setZero();
      }
    }
    state_.head<3>() = measured;
    last_marker_time_ = now;
    have_marker_ = true;
  }

  void control_step() {
    const auto now = Clock::now();
    const bool fresh_state = have_marker_ &&
        std::chrono::duration<double>(now - last_marker_time_).count() <= kStateTimeoutSeconds;
    if (!lqr_ready_ || !fresh_state || !state_.allFinite()) {
      publish_command(0.0, 0.0, kHoverThrustN);
      if (last_stale_warning_ == Clock::time_point{} ||
          std::chrono::duration<double>(now - last_stale_warning_).count() > 1.0) {
        RCLCPP_WARN(this->get_logger(), "Holding neutral command: LQR is not ready or WhyCode state is stale");
        last_stale_warning_ = now;
      }
      return;
    }
    const swift_pico::lqr::State error = state_ - reference_;
    const swift_pico::lqr::Input control = -gain_ * error;
    if (!control.allFinite()) {
      RCLCPP_ERROR(this->get_logger(), "Non-finite LQR command rejected");
      publish_command(0.0, 0.0, kHoverThrustN);
      return;
    }
    publish_command(control(0), control(1), kHoverThrustN + control(2));
    error_msg::msg::Error error_message;
    error_message.pitch_error = static_cast<float>(error(0));
    error_message.roll_error = static_cast<float>(error(1));
    error_message.throttle_error = static_cast<float>(error(2));
    error_message.yaw_error = 0.0F;
    error_pub_->publish(error_message);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SwiftPicoLqr>());
  rclcpp::shutdown();
  return 0;
}
