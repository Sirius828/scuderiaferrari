#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <memory>
#include <string>
#include <thread>

#include "geometry_msgs/msg/twist.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;

namespace
{
std::atomic_bool g_shutdown_requested{false};

void signal_handler(int)
{
  g_shutdown_requested.store(true);
}
}  // namespace

class LineFollowerControllerCpp : public rclcpp::Node
{
public:
  LineFollowerControllerCpp()
  : Node("line_follower_controller_cpp"),
    previous_control_time_(std::chrono::steady_clock::now())
  {
    declare_parameter<double>("Kp", 0.80);
    declare_parameter<double>("Kd", 0.035);
    declare_parameter<double>("linear_speed", 0.50);
    declare_parameter<double>("wheel_radius", 0.032);
    declare_parameter<double>("max_steering", 0.85);
    declare_parameter<double>("steering_sign", 1.0);
    declare_parameter<double>("control_frequency", 50.0);
    declare_parameter<std::string>("offset_topic", "/segmentation/center_offset");
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

    load_parameters();
    validate_startup_parameters();

    rclcpp::QoS sensor_qos(10);
    sensor_qos.best_effort();

    offset_subscription_ = create_subscription<std_msgs::msg::Float32>(
      offset_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::offset_callback, this, std::placeholders::_1));

    cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    const auto period = std::chrono::duration<double>(1.0 / control_frequency_);
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&LineFollowerControllerCpp::control_loop, this));

    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&LineFollowerControllerCpp::parameters_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "C++ PD line follower initialized: Kp=%.3f Kd=%.3f linear_speed=%.3fm/s "
      "wheel_radius=%.3fm wheel_rps=%.3f max_steering=%.3f steering_sign=%.0f "
      "frequency=%.1fHz offset_topic=%s cmd_vel_topic=%s",
      kp_, kd_, linear_speed_mps_, wheel_radius_, speed_to_wheel_rps(linear_speed_mps_),
      max_steering_, steering_sign_, control_frequency_, offset_topic_.c_str(),
      cmd_vel_topic_.c_str());
  }

  void publish_stop_commands(int count = 3)
  {
    geometry_msgs::msg::Twist stop_msg;
    stop_msg.linear.x = 0.0;
    stop_msg.angular.z = 0.0;

    for (int i = 0; i < count; ++i) {
      cmd_vel_publisher_->publish(stop_msg);
      rclcpp::sleep_for(50ms);
    }
  }

private:
  void load_parameters()
  {
    kp_ = get_parameter("Kp").as_double();
    kd_ = get_parameter("Kd").as_double();
    linear_speed_mps_ = get_parameter("linear_speed").as_double();
    wheel_radius_ = get_parameter("wheel_radius").as_double();
    max_steering_ = get_parameter("max_steering").as_double();
    steering_sign_ = get_parameter("steering_sign").as_double();
    control_frequency_ = get_parameter("control_frequency").as_double();
    offset_topic_ = get_parameter("offset_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
  }

  void validate_startup_parameters()
  {
    if (control_frequency_ <= 0.0) {
      throw std::runtime_error("control_frequency must be > 0");
    }
    if (wheel_radius_ < 0.0) {
      throw std::runtime_error("wheel_radius must be >= 0");
    }
    if (linear_speed_mps_ < 0.0) {
      throw std::runtime_error("linear_speed must be >= 0");
    }
    if (max_steering_ < 0.0) {
      throw std::runtime_error("max_steering must be >= 0");
    }
    if (steering_sign_ != 1.0 && steering_sign_ != -1.0) {
      throw std::runtime_error("steering_sign must be 1.0 or -1.0");
    }
    if (offset_topic_.empty()) {
      throw std::runtime_error("offset_topic must not be empty");
    }
    if (cmd_vel_topic_.empty()) {
      throw std::runtime_error("cmd_vel_topic must not be empty");
    }
  }

  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    double pending_kp = kp_;
    double pending_kd = kd_;
    double pending_linear_speed = linear_speed_mps_;
    double pending_wheel_radius = wheel_radius_;
    double pending_max_steering = max_steering_;
    double pending_steering_sign = steering_sign_;

    for (const auto & parameter : parameters) {
      const auto & name = parameter.get_name();

      if (name == "control_frequency" || name == "offset_topic" || name == "cmd_vel_topic") {
        return parameter_result(false, name + " is startup-only; restart the node to apply it");
      }

      if (name == "Kp") {
        pending_kp = parameter.as_double();
      } else if (name == "Kd") {
        pending_kd = parameter.as_double();
      } else if (name == "linear_speed") {
        pending_linear_speed = parameter.as_double();
      } else if (name == "wheel_radius") {
        pending_wheel_radius = parameter.as_double();
      } else if (name == "max_steering") {
        pending_max_steering = parameter.as_double();
      } else if (name == "steering_sign") {
        pending_steering_sign = parameter.as_double();
      }
    }

    if (pending_linear_speed < 0.0) {
      return parameter_result(false, "linear_speed must be >= 0");
    }
    if (pending_wheel_radius < 0.0) {
      return parameter_result(false, "wheel_radius must be >= 0");
    }
    if (pending_max_steering < 0.0) {
      return parameter_result(false, "max_steering must be >= 0");
    }
    if (pending_steering_sign != 1.0 && pending_steering_sign != -1.0) {
      return parameter_result(false, "steering_sign must be 1.0 or -1.0");
    }

    kp_ = pending_kp;
    kd_ = pending_kd;
    linear_speed_mps_ = pending_linear_speed;
    wheel_radius_ = pending_wheel_radius;
    max_steering_ = pending_max_steering;
    steering_sign_ = pending_steering_sign;

    RCLCPP_INFO(
      get_logger(),
      "Updated controller parameters: Kp=%.3f Kd=%.3f linear_speed=%.3fm/s "
      "wheel_radius=%.3fm wheel_rps=%.3f max_steering=%.3f steering_sign=%.0f",
      kp_, kd_, linear_speed_mps_, wheel_radius_, speed_to_wheel_rps(linear_speed_mps_),
      max_steering_, steering_sign_);

    return parameter_result(true, "");
  }

  rcl_interfaces::msg::SetParametersResult parameter_result(
    bool successful, const std::string & reason)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = successful;
    result.reason = reason;
    return result;
  }

  void offset_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    current_offset_ = static_cast<double>(msg->data);
    has_offset_ = true;
  }

  void control_loop()
  {
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = now - previous_control_time_;
    const double dt = std::max(1e-6, elapsed.count());

    const double derivative = (current_offset_ - previous_offset_) / dt;
    double steering = kp_ * current_offset_ + kd_ * derivative;
    steering *= steering_sign_;
    steering = std::clamp(steering, -max_steering_, max_steering_);

    geometry_msgs::msg::Twist cmd_msg;
    cmd_msg.linear.x = speed_to_wheel_rps(linear_speed_mps_);
    cmd_msg.angular.z = steering;
    cmd_vel_publisher_->publish(cmd_msg);

    previous_offset_ = current_offset_;
    previous_control_time_ = now;

    if (!has_offset_ && !waiting_for_offset_logged_) {
      RCLCPP_WARN(
        get_logger(),
        "No offset received yet; publishing straight command using offset=0.0");
      waiting_for_offset_logged_ = true;
    }
  }

  double speed_to_wheel_rps(double speed_mps) const
  {
    if (wheel_radius_ > 0.0) {
      constexpr double kPi = 3.14159265358979323846;
      return speed_mps / (2.0 * kPi * wheel_radius_);
    }
    return speed_mps;
  }

  double kp_{0.80};
  double kd_{0.035};
  double linear_speed_mps_{0.50};
  double wheel_radius_{0.032};
  double max_steering_{0.85};
  double steering_sign_{1.0};
  double control_frequency_{50.0};
  std::string offset_topic_{"/segmentation/center_offset"};
  std::string cmd_vel_topic_{"/cmd_vel"};

  double current_offset_{0.0};
  double previous_offset_{0.0};
  bool has_offset_{false};
  bool waiting_for_offset_logged_{false};
  std::chrono::steady_clock::time_point previous_control_time_;

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

int main(int argc, char ** argv)
{
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::shared_ptr<LineFollowerControllerCpp> node;

  try {
    node = std::make_shared<LineFollowerControllerCpp>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);

    while (rclcpp::ok() && !g_shutdown_requested.load()) {
      executor.spin_some();
      rclcpp::sleep_for(2ms);
    }
  } catch (const std::exception & exc) {
    if (node) {
      RCLCPP_ERROR(node->get_logger(), "Unexpected error: %s", exc.what());
    } else {
      fprintf(stderr, "Failed to start line_follower_controller_cpp: %s\n", exc.what());
    }
  }

  if (node && rclcpp::ok()) {
    RCLCPP_INFO(node->get_logger(), "Shutting down - publishing zero /cmd_vel commands...");
    node->publish_stop_commands(3);
  }

  node.reset();
  rclcpp::shutdown();
  return 0;
}
