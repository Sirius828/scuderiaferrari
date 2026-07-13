#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/int8.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

namespace
{
std::atomic_bool g_shutdown_requested{false};

void signal_handler(int)
{
  g_shutdown_requested.store(true);
}
}  // namespace

struct ControllerParameters
{
  double kp{0.80};
  double kd{0.035};
  double linear_speed_mps{0.60};
  double min_linear_speed_mps{0.25};
  double wheel_radius{0.032};
  double max_steering{0.85};
  double straight_max_steering{0.30};
  double curve_max_steering{0.85};
  double steering_offset_start{0.05};
  double steering_offset_full{0.60};
  double speed_offset_start{0.05};
  double speed_offset_full{0.60};
  double speed_accel_rate{0.20};
  double speed_decel_rate{2.0};
  double derivative_limit{3.0};
  double derivative_filter_alpha{0.35};
  double steering_slew_rate{4.0};
  double steering_return_slew_rate{1.5};
  double offset_timeout{0.25};
  double geometry_stall_timeout{0.20};
  double invalid_hold_speed_mps{0.25};
  double offset_y07_weight{0.20};
  double offset_y08_weight{0.30};
  double offset_y09_weight{0.50};
  double heading_feedback_gain{0.35};
  double curvature_speed_weight{0.50};
  double curve_offset_relief_start{0.15};
  double curve_offset_relief_full{0.60};
  bool enable_dynamic_speed{true};
  bool enable_dynamic_steering_limit{true};
  bool enable_curve_offset_allowance{true};
  bool enable_curve_offset_relief{true};
  bool enable_perception_stop_request{true};
  double straight_allowed_offset{0.05};
  double curve_allowed_offset{0.60};
  double min_perception_confidence{0.50};
};

class LineFollowerControllerCpp : public rclcpp::Node
{
public:
  LineFollowerControllerCpp()
  : Node("line_follower_controller_cpp"),
    last_geometry_content_change_time_(std::chrono::steady_clock::now()),
    previous_control_time_(std::chrono::steady_clock::now())
  {
    declare_parameter<double>("Kp", params_.kp);
    declare_parameter<double>("Kd", params_.kd);
    declare_parameter<double>("linear_speed", params_.linear_speed_mps);
    declare_parameter<double>("min_linear_speed", params_.min_linear_speed_mps);
    declare_parameter<double>("wheel_radius", params_.wheel_radius);
    declare_parameter<double>("max_steering", params_.max_steering);
    declare_parameter<double>("straight_max_steering", params_.straight_max_steering);
    declare_parameter<double>("curve_max_steering", params_.curve_max_steering);
    declare_parameter<double>("steering_offset_start", params_.steering_offset_start);
    declare_parameter<double>("steering_offset_full", params_.steering_offset_full);
    declare_parameter<double>("speed_offset_start", params_.speed_offset_start);
    declare_parameter<double>("speed_offset_full", params_.speed_offset_full);
    declare_parameter<double>("speed_accel_rate", params_.speed_accel_rate);
    declare_parameter<double>("speed_decel_rate", params_.speed_decel_rate);
    declare_parameter<double>("derivative_limit", params_.derivative_limit);
    declare_parameter<double>("derivative_filter_alpha", params_.derivative_filter_alpha);
    declare_parameter<double>("steering_slew_rate", params_.steering_slew_rate);
    declare_parameter<double>("steering_return_slew_rate", params_.steering_return_slew_rate);
    declare_parameter<double>("offset_timeout", params_.offset_timeout);
    declare_parameter<double>("geometry_stall_timeout", params_.geometry_stall_timeout);
    declare_parameter<double>("invalid_hold_speed", params_.invalid_hold_speed_mps);
    declare_parameter<double>("offset_y07_weight", params_.offset_y07_weight);
    declare_parameter<double>("offset_y08_weight", params_.offset_y08_weight);
    declare_parameter<double>("offset_y09_weight", params_.offset_y09_weight);
    declare_parameter<double>("heading_feedback_gain", params_.heading_feedback_gain);
    declare_parameter<double>("curvature_speed_weight", params_.curvature_speed_weight);
    declare_parameter<double>("curve_offset_relief_start", params_.curve_offset_relief_start);
    declare_parameter<double>("curve_offset_relief_full", params_.curve_offset_relief_full);
    declare_parameter<bool>("enable_dynamic_speed", params_.enable_dynamic_speed);
    declare_parameter<bool>("enable_dynamic_steering_limit", params_.enable_dynamic_steering_limit);
    declare_parameter<bool>("enable_curve_offset_allowance", params_.enable_curve_offset_allowance);
    declare_parameter<bool>("enable_curve_offset_relief", params_.enable_curve_offset_relief);
    declare_parameter<bool>("enable_perception_stop_request", params_.enable_perception_stop_request);
    declare_parameter<double>("straight_allowed_offset", params_.straight_allowed_offset);
    declare_parameter<double>("curve_allowed_offset", params_.curve_allowed_offset);
    declare_parameter<double>("min_perception_confidence", params_.min_perception_confidence);
    declare_parameter<double>("steering_sign", 1.0);
    declare_parameter<double>("control_frequency", 50.0);
    declare_parameter<bool>("autonomous_enabled_on_start", false);
    declare_parameter<std::string>("offset_y07_topic", "/segmentation/offset_y07");
    declare_parameter<std::string>("offset_y08_topic", "/segmentation/offset_y08");
    declare_parameter<std::string>("offset_y09_topic", "/segmentation/offset_y09");
    declare_parameter<std::string>("heading_error_topic", "/segmentation/heading_error");
    declare_parameter<std::string>("curvature_topic", "/segmentation/curvature");
    declare_parameter<std::string>("lane_state_topic", "/perception/lane_state");
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

    load_parameters();
    validate_startup_parameters();

    rclcpp::QoS sensor_qos(10);
    sensor_qos.best_effort();

    offset_y07_subscription_ = create_subscription<std_msgs::msg::Float32>(
      offset_y07_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::offset_y07_callback, this, std::placeholders::_1));
    offset_y08_subscription_ = create_subscription<std_msgs::msg::Float32>(
      offset_y08_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::offset_y08_callback, this, std::placeholders::_1));
    offset_y09_subscription_ = create_subscription<std_msgs::msg::Float32>(
      offset_y09_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::offset_y09_callback, this, std::placeholders::_1));
    heading_error_subscription_ = create_subscription<std_msgs::msg::Float32>(
      heading_error_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::heading_error_callback, this, std::placeholders::_1));
    curvature_subscription_ = create_subscription<std_msgs::msg::Float32>(
      curvature_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::curvature_callback, this, std::placeholders::_1));
    valid_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/segmentation/is_valid", sensor_qos,
      std::bind(&LineFollowerControllerCpp::valid_callback, this, std::placeholders::_1));
    stop_request_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/perception/stop_request", 10,
      std::bind(&LineFollowerControllerCpp::stop_request_callback, this, std::placeholders::_1));
    emergency_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/race/emergency_stop", 10,
      std::bind(&LineFollowerControllerCpp::emergency_callback, this, std::placeholders::_1));
    lane_state_subscription_ = create_subscription<std_msgs::msg::String>(
      lane_state_topic_, 10,
      std::bind(&LineFollowerControllerCpp::lane_state_callback, this, std::placeholders::_1));

    cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    chassis_enable_publisher_ = create_publisher<std_msgs::msg::Int8>("/chassis/enable", 10);
    debug_publisher_ = create_publisher<std_msgs::msg::String>("/line_follower/debug", 10);

    start_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/start",
      std::bind(
        &LineFollowerControllerCpp::start_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    stop_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/stop",
      std::bind(
        &LineFollowerControllerCpp::stop_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    set_enabled_service_ = create_service<std_srvs::srv::SetBool>(
      "/line_follower/set_enabled",
      std::bind(
        &LineFollowerControllerCpp::set_enabled_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    const auto period = std::chrono::duration<double>(1.0 / control_frequency_);
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&LineFollowerControllerCpp::control_loop, this));

    parameter_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&LineFollowerControllerCpp::parameters_callback, this, std::placeholders::_1));

    auto_start_pending_ = autonomous_enabled_on_start_;
    RCLCPP_INFO(
      get_logger(),
      "Safe C++ line follower initialized: Kp=%.3f Kd=%.3f speed=%.3fm/s min=%.3fm/s "
      "timeout=%.3fs steering_sign=%.0f frequency=%.1fHz",
      params_.kp, params_.kd, params_.linear_speed_mps, params_.min_linear_speed_mps,
      params_.offset_timeout, steering_sign_, control_frequency_);
    RCLCPP_INFO(
      get_logger(),
      "Controller starts disabled; use /line_follower/start or /line_follower/set_enabled true");
  }

  void publish_stop_commands(int count = 5)
  {
    for (int i = 0; i < count; ++i) {
      publish_stop_state();
      rclcpp::sleep_for(50ms);
    }
  }

private:
  void load_parameters()
  {
    params_.kp = get_parameter("Kp").as_double();
    params_.kd = get_parameter("Kd").as_double();
    params_.linear_speed_mps = get_parameter("linear_speed").as_double();
    params_.min_linear_speed_mps = get_parameter("min_linear_speed").as_double();
    params_.wheel_radius = get_parameter("wheel_radius").as_double();
    params_.max_steering = get_parameter("max_steering").as_double();
    params_.straight_max_steering = get_parameter("straight_max_steering").as_double();
    params_.curve_max_steering = get_parameter("curve_max_steering").as_double();
    params_.steering_offset_start = get_parameter("steering_offset_start").as_double();
    params_.steering_offset_full = get_parameter("steering_offset_full").as_double();
    params_.speed_offset_start = get_parameter("speed_offset_start").as_double();
    params_.speed_offset_full = get_parameter("speed_offset_full").as_double();
    params_.speed_accel_rate = get_parameter("speed_accel_rate").as_double();
    params_.speed_decel_rate = get_parameter("speed_decel_rate").as_double();
    params_.derivative_limit = get_parameter("derivative_limit").as_double();
    params_.derivative_filter_alpha = get_parameter("derivative_filter_alpha").as_double();
    params_.steering_slew_rate = get_parameter("steering_slew_rate").as_double();
    params_.steering_return_slew_rate = get_parameter("steering_return_slew_rate").as_double();
    params_.offset_timeout = get_parameter("offset_timeout").as_double();
    params_.geometry_stall_timeout = get_parameter("geometry_stall_timeout").as_double();
    params_.invalid_hold_speed_mps = get_parameter("invalid_hold_speed").as_double();
    params_.offset_y07_weight = get_parameter("offset_y07_weight").as_double();
    params_.offset_y08_weight = get_parameter("offset_y08_weight").as_double();
    params_.offset_y09_weight = get_parameter("offset_y09_weight").as_double();
    params_.heading_feedback_gain = get_parameter("heading_feedback_gain").as_double();
    params_.curvature_speed_weight = get_parameter("curvature_speed_weight").as_double();
    params_.curve_offset_relief_start = get_parameter("curve_offset_relief_start").as_double();
    params_.curve_offset_relief_full = get_parameter("curve_offset_relief_full").as_double();
    params_.enable_dynamic_speed = get_parameter("enable_dynamic_speed").as_bool();
    params_.enable_dynamic_steering_limit = get_parameter("enable_dynamic_steering_limit").as_bool();
    params_.enable_curve_offset_allowance = get_parameter("enable_curve_offset_allowance").as_bool();
    params_.enable_curve_offset_relief = get_parameter("enable_curve_offset_relief").as_bool();
    params_.enable_perception_stop_request = get_parameter("enable_perception_stop_request").as_bool();
    params_.straight_allowed_offset = get_parameter("straight_allowed_offset").as_double();
    params_.curve_allowed_offset = get_parameter("curve_allowed_offset").as_double();
    params_.min_perception_confidence = get_parameter("min_perception_confidence").as_double();
    steering_sign_ = get_parameter("steering_sign").as_double();
    control_frequency_ = get_parameter("control_frequency").as_double();
    autonomous_enabled_on_start_ = get_parameter("autonomous_enabled_on_start").as_bool();
    offset_y07_topic_ = get_parameter("offset_y07_topic").as_string();
    offset_y08_topic_ = get_parameter("offset_y08_topic").as_string();
    offset_y09_topic_ = get_parameter("offset_y09_topic").as_string();
    heading_error_topic_ = get_parameter("heading_error_topic").as_string();
    curvature_topic_ = get_parameter("curvature_topic").as_string();
    lane_state_topic_ = get_parameter("lane_state_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
  }

  bool validate_controller_parameters(
    const ControllerParameters & parameters, std::string * reason) const
  {
    const auto fail = [reason](const std::string & message) {
        if (reason) {
          *reason = message;
        }
        return false;
      };

    if (!std::isfinite(parameters.kp) || parameters.kp < 0.0) {
      return fail("Kp must be finite and >= 0");
    }
    if (!std::isfinite(parameters.kd) || parameters.kd < 0.0) {
      return fail("Kd must be finite and >= 0");
    }
    if (!std::isfinite(parameters.linear_speed_mps) || parameters.linear_speed_mps < 0.0) {
      return fail("linear_speed must be finite and >= 0");
    }
    if (!std::isfinite(parameters.min_linear_speed_mps) ||
      parameters.min_linear_speed_mps < 0.0 ||
      parameters.min_linear_speed_mps > parameters.linear_speed_mps)
    {
      return fail("min_linear_speed must be >= 0 and <= linear_speed");
    }
    if (!std::isfinite(parameters.wheel_radius) || parameters.wheel_radius < 0.0) {
      return fail("wheel_radius must be finite and >= 0");
    }
    if (!std::isfinite(parameters.max_steering) ||
      parameters.max_steering < 0.0 || parameters.max_steering > 1.0)
    {
      return fail("max_steering must be in [0, 1]");
    }
    if (!std::isfinite(parameters.straight_max_steering) ||
      parameters.straight_max_steering < 0.0 ||
      parameters.straight_max_steering > parameters.max_steering)
    {
      return fail("straight_max_steering must be in [0, max_steering]");
    }
    if (!std::isfinite(parameters.curve_max_steering) ||
      parameters.curve_max_steering < parameters.straight_max_steering ||
      parameters.curve_max_steering > parameters.max_steering)
    {
      return fail("curve_max_steering must be >= straight_max_steering and <= max_steering");
    }
    if (!std::isfinite(parameters.steering_offset_start) ||
      parameters.steering_offset_start < 0.0 || parameters.steering_offset_start >= 1.0)
    {
      return fail("steering_offset_start must be in [0, 1)");
    }
    if (!std::isfinite(parameters.steering_offset_full) ||
      parameters.steering_offset_full <= parameters.steering_offset_start ||
      parameters.steering_offset_full > 1.0)
    {
      return fail("steering_offset_full must be > start and <= 1");
    }
    if (!std::isfinite(parameters.speed_offset_start) ||
      parameters.speed_offset_start < 0.0 || parameters.speed_offset_start >= 1.0)
    {
      return fail("speed_offset_start must be in [0, 1)");
    }
    if (!std::isfinite(parameters.speed_offset_full) ||
      parameters.speed_offset_full <= parameters.speed_offset_start ||
      parameters.speed_offset_full > 1.0)
    {
      return fail("speed_offset_full must be > start and <= 1");
    }
    if (!std::isfinite(parameters.speed_accel_rate) || parameters.speed_accel_rate < 0.0) {
      return fail("speed_accel_rate must be >= 0");
    }
    if (!std::isfinite(parameters.speed_decel_rate) || parameters.speed_decel_rate < 0.0) {
      return fail("speed_decel_rate must be >= 0");
    }
    if (!std::isfinite(parameters.derivative_limit) || parameters.derivative_limit < 0.0) {
      return fail("derivative_limit must be >= 0");
    }
    if (!std::isfinite(parameters.derivative_filter_alpha) ||
      parameters.derivative_filter_alpha < 0.0 || parameters.derivative_filter_alpha > 1.0)
    {
      return fail("derivative_filter_alpha must be in [0, 1]");
    }
    if (!std::isfinite(parameters.steering_slew_rate) || parameters.steering_slew_rate < 0.0) {
      return fail("steering_slew_rate must be >= 0");
    }
    if (!std::isfinite(parameters.steering_return_slew_rate) ||
      parameters.steering_return_slew_rate < 0.0)
    {
      return fail("steering_return_slew_rate must be >= 0");
    }
    if (!std::isfinite(parameters.offset_timeout) || parameters.offset_timeout <= 0.0) {
      return fail("offset_timeout must be > 0");
    }
    if (!std::isfinite(parameters.geometry_stall_timeout) ||
      parameters.geometry_stall_timeout < 0.0)
    {
      return fail("geometry_stall_timeout must be >= 0");
    }
    if (!std::isfinite(parameters.invalid_hold_speed_mps) ||
      parameters.invalid_hold_speed_mps < 0.0 ||
      parameters.invalid_hold_speed_mps > parameters.linear_speed_mps)
    {
      return fail("invalid_hold_speed must be >= 0 and <= linear_speed");
    }
    if (!std::isfinite(parameters.offset_y07_weight) || parameters.offset_y07_weight < 0.0 ||
      !std::isfinite(parameters.offset_y08_weight) || parameters.offset_y08_weight < 0.0 ||
      !std::isfinite(parameters.offset_y09_weight) || parameters.offset_y09_weight < 0.0)
    {
      return fail("offset weights must be finite and >= 0");
    }
    if (parameters.offset_y07_weight + parameters.offset_y08_weight +
        parameters.offset_y09_weight <= 1e-9)
    {
      return fail("offset weights must have a positive sum");
    }
    if (!std::isfinite(parameters.heading_feedback_gain) || parameters.heading_feedback_gain < 0.0) {
      return fail("heading_feedback_gain must be >= 0");
    }
    if (!std::isfinite(parameters.curvature_speed_weight) ||
      parameters.curvature_speed_weight < 0.0 || parameters.curvature_speed_weight > 1.0)
    {
      return fail("curvature_speed_weight must be in [0, 1]");
    }
    if (!std::isfinite(parameters.curve_offset_relief_start) ||
      parameters.curve_offset_relief_start < 0.0 || parameters.curve_offset_relief_start >= 1.0)
    {
      return fail("curve_offset_relief_start must be in [0, 1)");
    }
    if (!std::isfinite(parameters.curve_offset_relief_full) ||
      parameters.curve_offset_relief_full <= parameters.curve_offset_relief_start ||
      parameters.curve_offset_relief_full > 1.0)
    {
      return fail("curve_offset_relief_full must be > start and <= 1");
    }
    if (!std::isfinite(parameters.straight_allowed_offset) ||
      parameters.straight_allowed_offset < 0.0 || parameters.straight_allowed_offset > 1.0)
    {
      return fail("straight_allowed_offset must be in [0, 1]");
    }
    if (!std::isfinite(parameters.curve_allowed_offset) ||
      parameters.curve_allowed_offset < parameters.straight_allowed_offset ||
      parameters.curve_allowed_offset > 1.0)
    {
      return fail("curve_allowed_offset must be >= straight_allowed_offset and <= 1");
    }
    if (!std::isfinite(parameters.min_perception_confidence) ||
      parameters.min_perception_confidence < 0.0 || parameters.min_perception_confidence > 1.0)
    {
      return fail("min_perception_confidence must be in [0, 1]");
    }
    return true;
  }

  void validate_startup_parameters()
  {
    std::string reason;
    if (!validate_controller_parameters(params_, &reason)) {
      throw std::runtime_error(reason);
    }
    if (!std::isfinite(steering_sign_) || (steering_sign_ != 1.0 && steering_sign_ != -1.0)) {
      throw std::runtime_error("steering_sign must be 1.0 or -1.0");
    }
    if (!std::isfinite(control_frequency_) || control_frequency_ <= 0.0) {
      throw std::runtime_error("control_frequency must be finite and > 0");
    }
    if (offset_y07_topic_.empty() || offset_y08_topic_.empty() || offset_y09_topic_.empty() ||
      heading_error_topic_.empty() || curvature_topic_.empty()) {
      throw std::runtime_error("geometry feedback topics must not be empty");
    }
    if (cmd_vel_topic_.empty()) {
      throw std::runtime_error("cmd_vel_topic must not be empty");
    }
  }

  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    ControllerParameters pending = params_;
    double pending_steering_sign = steering_sign_;

    for (const auto & parameter : parameters) {
      const auto & name = parameter.get_name();
      if (name == "control_frequency" || name == "offset_y07_topic" ||
        name == "offset_y08_topic" || name == "offset_y09_topic" ||
        name == "heading_error_topic" ||
        name == "curvature_topic" || name == "lane_state_topic" || name == "cmd_vel_topic" ||
        name == "autonomous_enabled_on_start")
      {
        return parameter_result(false, name + " is startup-only; restart the node to apply it");
      }

      if (name == "Kp") {
        pending.kp = parameter.as_double();
      } else if (name == "Kd") {
        pending.kd = parameter.as_double();
      } else if (name == "linear_speed") {
        pending.linear_speed_mps = parameter.as_double();
      } else if (name == "min_linear_speed") {
        pending.min_linear_speed_mps = parameter.as_double();
      } else if (name == "wheel_radius") {
        pending.wheel_radius = parameter.as_double();
      } else if (name == "max_steering") {
        pending.max_steering = parameter.as_double();
      } else if (name == "straight_max_steering") {
        pending.straight_max_steering = parameter.as_double();
      } else if (name == "curve_max_steering") {
        pending.curve_max_steering = parameter.as_double();
      } else if (name == "steering_offset_start") {
        pending.steering_offset_start = parameter.as_double();
      } else if (name == "steering_offset_full") {
        pending.steering_offset_full = parameter.as_double();
      } else if (name == "speed_offset_start") {
        pending.speed_offset_start = parameter.as_double();
      } else if (name == "speed_offset_full") {
        pending.speed_offset_full = parameter.as_double();
      } else if (name == "speed_accel_rate") {
        pending.speed_accel_rate = parameter.as_double();
      } else if (name == "speed_decel_rate") {
        pending.speed_decel_rate = parameter.as_double();
      } else if (name == "derivative_limit") {
        pending.derivative_limit = parameter.as_double();
      } else if (name == "derivative_filter_alpha") {
        pending.derivative_filter_alpha = parameter.as_double();
      } else if (name == "steering_slew_rate") {
        pending.steering_slew_rate = parameter.as_double();
      } else if (name == "steering_return_slew_rate") {
        pending.steering_return_slew_rate = parameter.as_double();
      } else if (name == "offset_timeout") {
        pending.offset_timeout = parameter.as_double();
      } else if (name == "geometry_stall_timeout") {
        pending.geometry_stall_timeout = parameter.as_double();
      } else if (name == "invalid_hold_speed") {
        pending.invalid_hold_speed_mps = parameter.as_double();
      } else if (name == "offset_y07_weight") {
        pending.offset_y07_weight = parameter.as_double();
      } else if (name == "offset_y08_weight") {
        pending.offset_y08_weight = parameter.as_double();
      } else if (name == "offset_y09_weight") {
        pending.offset_y09_weight = parameter.as_double();
      } else if (name == "heading_feedback_gain") {
        pending.heading_feedback_gain = parameter.as_double();
      } else if (name == "curvature_speed_weight") {
        pending.curvature_speed_weight = parameter.as_double();
      } else if (name == "curve_offset_relief_start") {
        pending.curve_offset_relief_start = parameter.as_double();
      } else if (name == "curve_offset_relief_full") {
        pending.curve_offset_relief_full = parameter.as_double();
      } else if (name == "enable_dynamic_speed") {
        pending.enable_dynamic_speed = parameter.as_bool();
      } else if (name == "enable_dynamic_steering_limit") {
        pending.enable_dynamic_steering_limit = parameter.as_bool();
      } else if (name == "enable_curve_offset_allowance") {
        pending.enable_curve_offset_allowance = parameter.as_bool();
      } else if (name == "enable_curve_offset_relief") {
        pending.enable_curve_offset_relief = parameter.as_bool();
      } else if (name == "enable_perception_stop_request") {
        pending.enable_perception_stop_request = parameter.as_bool();
      } else if (name == "straight_allowed_offset") {
        pending.straight_allowed_offset = parameter.as_double();
      } else if (name == "curve_allowed_offset") {
        pending.curve_allowed_offset = parameter.as_double();
      } else if (name == "min_perception_confidence") {
        pending.min_perception_confidence = parameter.as_double();
      } else if (name == "steering_sign") {
        pending_steering_sign = parameter.as_double();
      }
    }

    std::string reason;
    if (!validate_controller_parameters(pending, &reason)) {
      return parameter_result(false, reason);
    }
    if (!std::isfinite(pending_steering_sign) ||
      (pending_steering_sign != 1.0 && pending_steering_sign != -1.0))
    {
      return parameter_result(false, "steering_sign must be 1.0 or -1.0");
    }

    params_ = pending;
    steering_sign_ = pending_steering_sign;
    if (!params_.enable_perception_stop_request) {
      stop_request_active_ = false;
    }
    RCLCPP_INFO(
      get_logger(),
      "Updated controller parameters: Kp=%.3f Kd=%.3f speed=%.3fm/s min=%.3fm/s "
      "straight_max=%.3f curve_max=%.3f timeout=%.3fs",
      params_.kp, params_.kd, params_.linear_speed_mps, params_.min_linear_speed_mps,
      params_.straight_max_steering, params_.curve_max_steering, params_.offset_timeout);
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

  void offset_y07_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_offset_value(
      msg->data, &current_offset_y07_, &has_offset_y07_, &last_offset_y07_time_);
  }

  void offset_y08_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_offset_value(
      msg->data, &current_offset_y08_, &has_offset_y08_, &last_offset_y08_time_);
  }

  void offset_y09_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_offset_value(
      msg->data, &current_offset_y09_, &has_offset_y09_, &last_offset_y09_time_);
  }

  void update_offset_value(
    float value, double * target, bool * has_value,
    std::chrono::steady_clock::time_point * timestamp)
  {
    if (!std::isfinite(value)) {
      *has_value = false;
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double clamped = std::clamp(static_cast<double>(value), -1.0, 1.0);
    if (!*has_value || std::abs(clamped - *target) > 1e-6) {
      last_geometry_content_change_time_ = now;
    }
    *target = clamped;
    *has_value = true;
    *timestamp = now;
  }

  void valid_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    is_valid_ = msg->data;
    has_valid_message_ = true;
    last_valid_time_ = std::chrono::steady_clock::now();
  }

  void heading_error_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_geometry_value(msg->data, &current_heading_error_, &has_heading_error_, &last_heading_error_time_);
  }

  void curvature_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_geometry_value(msg->data, &current_curvature_, &has_curvature_, &last_curvature_time_);
  }

  void update_geometry_value(
    float value, double * target, bool * has_value,
    std::chrono::steady_clock::time_point * timestamp)
  {
    if (!std::isfinite(value)) {
      *has_value = false;
      return;
    }
    *target = std::clamp(static_cast<double>(value), -1.0, 1.0);
    *has_value = true;
    *timestamp = std::chrono::steady_clock::now();
  }

  void stop_request_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!params_.enable_perception_stop_request) {
      stop_request_active_ = false;
      return;
    }
    stop_request_active_ = msg->data;
    if (params_.enable_perception_stop_request && stop_request_active_) {
      lock_and_stop("perception_stop");
    }
  }

  void emergency_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    emergency_stop_active_ = msg->data;
    if (emergency_stop_active_) {
      lock_and_stop("emergency_stop");
    }
  }

  void lane_state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    double confidence = 0.0;
    bool is_valid = false;
    std::string road_state;
    if (!extract_json_number(msg->data, "confidence", &confidence) ||
      !extract_json_bool(msg->data, "is_valid", &is_valid) ||
      !extract_json_string(msg->data, "road_state", &road_state))
    {
      has_lane_state_ = false;
      return;
    }
    lane_confidence_ = std::clamp(confidence, 0.0, 1.0);
    lane_state_valid_ = is_valid;
    lane_road_state_ = road_state;
    has_lane_state_ = true;
    last_lane_state_time_ = std::chrono::steady_clock::now();
  }

  static bool extract_json_number(
    const std::string & json, const std::string & key, double * value)
  {
    const std::string needle = "\"" + key + "\":";
    const size_t start = json.find(needle);
    if (start == std::string::npos) {
      return false;
    }
    size_t value_start = start + needle.size();
    while (value_start < json.size() && std::isspace(static_cast<unsigned char>(json[value_start]))) {
      ++value_start;
    }
    try {
      size_t consumed = 0;
      const double parsed = std::stod(json.substr(value_start), &consumed);
      if (consumed == 0 || !std::isfinite(parsed)) {
        return false;
      }
      *value = parsed;
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  static bool extract_json_bool(
    const std::string & json, const std::string & key, bool * value)
  {
    const std::string needle = "\"" + key + "\":";
    const size_t start = json.find(needle);
    if (start == std::string::npos) {
      return false;
    }
    size_t value_start = start + needle.size();
    while (value_start < json.size() && std::isspace(static_cast<unsigned char>(json[value_start]))) {
      ++value_start;
    }
    if (json.compare(value_start, 4, "true") == 0) {
      *value = true;
      return true;
    }
    if (json.compare(value_start, 5, "false") == 0) {
      *value = false;
      return true;
    }
    return false;
  }

  static bool extract_json_string(
    const std::string & json, const std::string & key, std::string * value)
  {
    const std::string needle = "\"" + key + "\":\"";
    const size_t start = json.find(needle);
    if (start == std::string::npos) {
      return false;
    }
    const size_t value_start = start + needle.size();
    const size_t value_end = json.find('"', value_start);
    if (value_end == std::string::npos) {
      return false;
    }
    *value = json.substr(value_start, value_end - value_start);
    return true;
  }

  bool start_ready(std::string * reason) const
  {
    const auto now = std::chrono::steady_clock::now();
    if (!has_offset_y07_ || !has_offset_y08_ || !has_offset_y09_) {
      if (reason) {
        *reason = "one or more preview offsets have not been received";
      }
      return false;
    }
    if (offset_age_seconds(last_offset_y07_time_, has_offset_y07_, now) > params_.offset_timeout ||
        offset_age_seconds(last_offset_y08_time_, has_offset_y08_, now) > params_.offset_timeout ||
        offset_age_seconds(last_offset_y09_time_, has_offset_y09_, now) > params_.offset_timeout) {
      if (reason) {
        *reason = "one or more preview offsets are stale";
      }
      return false;
    }
    if (!has_valid_message_ || !is_valid_) {
      if (reason) {
        *reason = "perception is not valid";
      }
      return false;
    }
    if (valid_age_seconds(now) > params_.offset_timeout) {
      if (reason) {
        *reason = "validity state is stale";
      }
      return false;
    }
    if (!lane_state_ready(now)) {
      if (reason) {
        *reason = "lane state is invalid, low-confidence, or stale";
      }
      return false;
    }
    if (!geometry_ready(now)) {
      if (reason) {
        *reason = "lane geometry feedback is missing or stale";
      }
      return false;
    }
    if (params_.enable_perception_stop_request && stop_request_active_) {
      if (reason) {
        *reason = "perception stop request is active";
      }
      return false;
    }
    if (emergency_stop_active_) {
      if (reason) {
        *reason = "emergency stop is active";
      }
      return false;
    }
    return true;
  }

  bool try_start(std::string * reason)
  {
    if (auto_enabled_) {
      return true;
    }
    if (!start_ready(reason)) {
      publish_stop_state();
      return false;
    }

    const auto now = std::chrono::steady_clock::now();
    auto_enabled_ = true;
    safety_locked_ = false;
    auto_start_pending_ = false;
    stop_reason_ = "running";
    invalid_since_.reset();
    last_geometry_content_change_time_ = now;
    current_speed_mps_ = 0.0;
    current_steering_ = 0.0;
    previous_control_error_ = compute_control_error();
    filtered_derivative_ = 0.0;
    previous_control_time_ = now;
    publish_motion_command(0.0, 0.0);
    publish_chassis_enable(true);
    RCLCPP_INFO(get_logger(), "Line following started");
    return true;
  }

  void disable_control(const std::string & reason, bool lock)
  {
    auto_enabled_ = false;
    auto_start_pending_ = false;
    safety_locked_ = lock;
    stop_reason_ = reason;
    invalid_since_.reset();
    current_speed_mps_ = 0.0;
    current_steering_ = 0.0;
    filtered_derivative_ = 0.0;
    previous_control_error_ = compute_control_error();
    previous_control_time_ = std::chrono::steady_clock::now();
    publish_stop_state();
  }

  void lock_and_stop(const std::string & reason)
  {
    if (!auto_enabled_ && safety_locked_ && stop_reason_ == reason) {
      publish_stop_state();
      return;
    }
    disable_control(reason, true);
    RCLCPP_ERROR(get_logger(), "Line following locked and stopped: %s", reason.c_str());
  }

  void start_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::string reason;
    response->success = try_start(&reason);
    response->message = response->success ? "line following started" : "start rejected: " + reason;
  }

  void stop_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    disable_control("service_stop", false);
    response->success = true;
    response->message = "line following stopped; chassis disabled";
    RCLCPP_WARN(get_logger(), "Line following stopped by service");
  }

  void set_enabled_service_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    if (request->data) {
      std::string reason;
      response->success = try_start(&reason);
      response->message = response->success ?
        "autonomous line following enabled" : "enable rejected: " + reason;
      return;
    }

    disable_control("service_stop", false);
    response->success = true;
    response->message = "autonomous line following disabled; chassis disabled";
    RCLCPP_WARN(get_logger(), "Line following disabled by service");
  }

  void control_loop()
  {
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = now - previous_control_time_;
    const double dt = std::clamp(elapsed.count(), 1e-4, 0.2);

    if (auto_start_pending_) {
      std::string reason;
      if (!try_start(&reason)) {
        last_mode_ = "auto_start_wait";
      }
    }

    if (!auto_enabled_) {
      publish_stop_state();
      if (last_mode_ != "auto_start_wait") {
        last_mode_ = safety_locked_ ? "locked_stop" : "disabled";
      }
      publish_debug(now);
      previous_control_time_ = now;
      previous_control_error_ = compute_control_error();
      filtered_derivative_ = 0.0;
      return;
    }

    if (emergency_stop_active_) {
      lock_and_stop("emergency_stop");
      last_mode_ = "emergency_stop";
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }
    if (params_.enable_perception_stop_request && stop_request_active_) {
      lock_and_stop("perception_stop");
      last_mode_ = "perception_stop";
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }

    if (geometry_content_stalled(now)) {
      lock_and_stop("geometry_stall");
      last_mode_ = "geometry_stall";
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }

    if (!perception_ready(now)) {
      if (!invalid_since_) {
        invalid_since_ = now;
      }
      const double invalid_age = std::chrono::duration<double>(now - *invalid_since_).count();
      if (invalid_age > params_.offset_timeout) {
        lock_and_stop("perception_timeout");
        last_mode_ = "perception_timeout";
        publish_debug(now);
        previous_control_time_ = now;
        return;
      }

      current_speed_mps_ = apply_speed_slew(
        std::min(current_speed_mps_, params_.invalid_hold_speed_mps), dt);
      publish_chassis_enable(true);
      publish_motion_command(current_speed_mps_, current_steering_);
      last_mode_ = "invalid_hold";
      publish_debug(now);
      previous_control_time_ = now;
      filtered_derivative_ = 0.0;
      return;
    }

    invalid_since_.reset();
    const double control_error = compute_control_error();
    const double curve_risk = compute_curve_risk();
    const double target_speed = compute_target_speed(curve_risk);
    current_speed_mps_ = apply_speed_slew(target_speed, dt);
    const double dynamic_max_steering = compute_dynamic_max_steering(curve_risk);
    const double raw_derivative = (control_error - previous_control_error_) / dt;
    double derivative = raw_derivative;
    if (params_.derivative_limit > 0.0) {
      derivative = std::clamp(derivative, -params_.derivative_limit, params_.derivative_limit);
    }
    filtered_derivative_ = params_.derivative_filter_alpha * derivative +
      (1.0 - params_.derivative_filter_alpha) * filtered_derivative_;

    double desired_steering = steering_sign_ *
      (params_.kp * control_error + params_.kd * filtered_derivative_);
    desired_steering = std::clamp(
      desired_steering, -dynamic_max_steering, dynamic_max_steering);
    current_steering_ = apply_steering_slew(desired_steering, dt);
    // The dynamic limit is a target limit.  Clamping to a rapidly shrinking
    // dynamic limit here would bypass steering_slew_rate on curve exit and
    // unload the servo abruptly.  Keep only the absolute safety clamp after
    // the slew limiter so both steering attack and release remain continuous.
    current_steering_ = std::clamp(
      current_steering_, -params_.max_steering, params_.max_steering);

    publish_chassis_enable(true);
    publish_motion_command(current_speed_mps_, current_steering_);
    last_mode_ = "running";
    publish_debug(now);
    previous_control_error_ = control_error;
    previous_control_time_ = now;
  }

  bool perception_ready(const std::chrono::steady_clock::time_point & now) const
  {
    return has_offset_y07_ && has_offset_y08_ && has_offset_y09_ &&
           has_valid_message_ && is_valid_ &&
           lane_state_ready(now) &&
           offset_age_seconds(last_offset_y07_time_, has_offset_y07_, now) <= params_.offset_timeout &&
           offset_age_seconds(last_offset_y08_time_, has_offset_y08_, now) <= params_.offset_timeout &&
           offset_age_seconds(last_offset_y09_time_, has_offset_y09_, now) <= params_.offset_timeout &&
           valid_age_seconds(now) <= params_.offset_timeout &&
           geometry_ready(now);
  }

  bool geometry_content_stalled(const std::chrono::steady_clock::time_point & now) const
  {
    if (params_.geometry_stall_timeout <= 0.0 || current_speed_mps_ < 0.30) {
      return false;
    }
    return std::chrono::duration<double>(
      now - last_geometry_content_change_time_).count() > params_.geometry_stall_timeout;
  }

  bool lane_state_ready(const std::chrono::steady_clock::time_point & now) const
  {
    return has_lane_state_ && lane_state_valid_ &&
           lane_confidence_ >= params_.min_perception_confidence &&
           lane_road_state_ != "LOW_CONFIDENCE" &&
           geometry_age_seconds(last_lane_state_time_, has_lane_state_, now) <= params_.offset_timeout;
  }

  bool geometry_ready(const std::chrono::steady_clock::time_point & now) const
  {
    return has_offset_y07_ && has_offset_y08_ && has_offset_y09_ &&
           has_heading_error_ && has_curvature_ &&
           geometry_age_seconds(last_offset_y07_time_, has_offset_y07_, now) <= params_.offset_timeout &&
           geometry_age_seconds(last_offset_y08_time_, has_offset_y08_, now) <= params_.offset_timeout &&
           geometry_age_seconds(last_offset_y09_time_, has_offset_y09_, now) <= params_.offset_timeout &&
           geometry_age_seconds(last_heading_error_time_, has_heading_error_, now) <= params_.offset_timeout &&
           geometry_age_seconds(last_curvature_time_, has_curvature_, now) <= params_.offset_timeout;
  }

  double geometry_age_seconds(
    const std::chrono::steady_clock::time_point & timestamp, bool has_value,
    const std::chrono::steady_clock::time_point & now) const
  {
    if (!has_value) {
      return std::numeric_limits<double>::infinity();
    }
    return std::chrono::duration<double>(now - timestamp).count();
  }

  double offset_age_seconds(
    const std::chrono::steady_clock::time_point & timestamp, bool has_value,
    const std::chrono::steady_clock::time_point & now) const
  {
    if (!has_value) {
      return std::numeric_limits<double>::infinity();
    }
    return std::chrono::duration<double>(now - timestamp).count();
  }

  double valid_age_seconds(const std::chrono::steady_clock::time_point & now) const
  {
    if (!has_valid_message_) {
      return std::numeric_limits<double>::infinity();
    }
    return std::chrono::duration<double>(now - last_valid_time_).count();
  }

  double compute_target_speed(double risk) const
  {
    if (!params_.enable_dynamic_speed) {
      return params_.linear_speed_mps;
    }
    const double abs_offset = std::abs(risk);
    if (abs_offset <= params_.speed_offset_start) {
      return params_.linear_speed_mps;
    }
    const double ratio = std::clamp(
      (abs_offset - params_.speed_offset_start) /
      (params_.speed_offset_full - params_.speed_offset_start), 0.0, 1.0);
    const double slowdown = ratio;
    const double speed_range = params_.linear_speed_mps - params_.min_linear_speed_mps;
    return params_.linear_speed_mps - speed_range * slowdown;
  }

  double compute_dynamic_max_steering(double risk) const
  {
    if (!params_.enable_dynamic_steering_limit) {
      return params_.max_steering;
    }
    const double abs_offset = std::abs(risk);
    if (abs_offset <= params_.steering_offset_start) {
      return params_.straight_max_steering;
    }
    const double ratio = std::clamp(
      (abs_offset - params_.steering_offset_start) /
      (params_.steering_offset_full - params_.steering_offset_start), 0.0, 1.0);
    return params_.straight_max_steering +
      (params_.curve_max_steering - params_.straight_max_steering) * ratio;
  }

  double compute_weighted_offset() const
  {
    const double weight_sum = params_.offset_y07_weight +
      params_.offset_y08_weight + params_.offset_y09_weight;
    if (weight_sum <= 1e-9) {
      return 0.0;
    }
    return std::clamp(
      (params_.offset_y07_weight * current_offset_y07_ +
       params_.offset_y08_weight * current_offset_y08_ +
       params_.offset_y09_weight * current_offset_y09_) / weight_sum,
      -1.0, 1.0);
  }

  double compute_control_error() const
  {
    const double weighted_offset = compute_weighted_offset();
    // Image y grows downward.  A centerline bending toward positive x therefore
    // has a negative dx/dy heading, so subtract heading to make both feedback
    // terms request the same steering direction through a bend.
    return std::clamp(
      weighted_offset -
      params_.heading_feedback_gain * current_heading_error_, -1.0, 1.0);
  }

  double compute_curve_risk() const
  {
    const double lateral_risk = std::abs(current_offset_y09_);
    const double heading_risk = std::abs(current_heading_error_);
    const double curvature_risk = params_.curvature_speed_weight * std::abs(current_curvature_);
    const double predictive_risk = compute_predictive_offset_risk();
    return std::clamp(
      std::max({lateral_risk, heading_risk, curvature_risk, predictive_risk}), 0.0, 1.0);
  }

  double compute_allowed_offset() const
  {
    const double curve_strength = compute_curve_strength();
    if (curve_strength <= params_.curve_offset_relief_start) {
      return params_.straight_allowed_offset;
    }
    const double ratio = std::clamp(
      (curve_strength - params_.curve_offset_relief_start) /
      (params_.curve_offset_relief_full - params_.curve_offset_relief_start), 0.0, 1.0);
    return params_.straight_allowed_offset +
      (params_.curve_allowed_offset - params_.straight_allowed_offset) * ratio;
  }

  double compute_predictive_offset_risk() const
  {
    if (!params_.enable_curve_offset_allowance) {
      return compute_offset_relief() * std::abs(compute_weighted_offset());
    }
    return std::max(0.0, std::abs(compute_weighted_offset()) - compute_allowed_offset());
  }

  double compute_curve_strength() const
  {
    return std::clamp(
      std::max(std::abs(current_heading_error_), std::abs(current_curvature_)), 0.0, 1.0);
  }

  double compute_offset_relief() const
  {
    if (!params_.enable_curve_offset_relief) {
      return 1.0;
    }
    const double curve_strength = compute_curve_strength();
    if (curve_strength <= params_.curve_offset_relief_start) {
      return 1.0;
    }
    const double ratio = std::clamp(
      (curve_strength - params_.curve_offset_relief_start) /
      (params_.curve_offset_relief_full - params_.curve_offset_relief_start), 0.0, 1.0);
    return 1.0 - ratio;
  }

  double apply_speed_slew(double target_speed, double dt)
  {
    // Normal dynamic-speed targets are already bounded by min_linear_speed.
    // Safety states such as invalid_hold intentionally request a lower speed,
    // so do not raise those targets back to the normal driving floor here.
    target_speed = std::clamp(
      target_speed, 0.0, params_.linear_speed_mps);
    if (target_speed > current_speed_mps_) {
      if (params_.speed_accel_rate <= 0.0) {
        return target_speed;
      }
      return std::min(target_speed, current_speed_mps_ + params_.speed_accel_rate * dt);
    }
    if (target_speed < current_speed_mps_) {
      if (params_.speed_decel_rate <= 0.0) {
        return target_speed;
      }
      return std::max(target_speed, current_speed_mps_ - params_.speed_decel_rate * dt);
    }
    return target_speed;
  }

  double apply_steering_slew(double target_steering, double dt) const
  {
    if (params_.steering_slew_rate <= 0.0) {
      return target_steering;
    }
    double slew_rate = params_.steering_slew_rate;
    if (params_.steering_return_slew_rate > 0.0 &&
      std::abs(target_steering) < std::abs(current_steering_))
    {
      slew_rate = params_.steering_return_slew_rate;
    }
    const double max_delta = slew_rate * dt;
    const double delta = std::clamp(
      target_steering - current_steering_, -max_delta, max_delta);
    return current_steering_ + delta;
  }

  void publish_motion_command(double speed_mps, double steering)
  {
    geometry_msgs::msg::Twist command;
    command.linear.x = speed_to_wheel_rps(std::max(0.0, speed_mps));
    command.angular.z = std::clamp(steering, -1.0, 1.0);
    cmd_vel_publisher_->publish(command);
  }

  void publish_chassis_enable(bool enabled)
  {
    std_msgs::msg::Int8 message;
    message.data = enabled ? 1 : 0;
    chassis_enable_publisher_->publish(message);
  }

  void publish_stop_state()
  {
    publish_motion_command(0.0, 0.0);
    publish_chassis_enable(false);
  }

  void publish_debug(const std::chrono::steady_clock::time_point & now)
  {
    std_msgs::msg::String message;
    std::ostringstream text;
    const double control_error = compute_control_error();
    const double curve_risk = compute_curve_risk();
    const double curve_strength = compute_curve_strength();
    const double offset_relief = compute_offset_relief();
    const double allowed_offset = compute_allowed_offset();
    const double weighted_offset = compute_weighted_offset();
    const double offset_excess = std::max(0.0, std::abs(weighted_offset) - allowed_offset);
    const double dynamic_max = compute_dynamic_max_steering(curve_risk);
    text << std::fixed << std::setprecision(3)
         << "mode=" << last_mode_
         << " enabled=" << (auto_enabled_ ? "True" : "False")
         << " locked=" << (safety_locked_ ? "True" : "False")
         << " valid=" << (is_valid_ ? "True" : "False")
         << " lane_valid=" << (lane_state_valid_ ? "True" : "False")
         << " confidence=" << lane_confidence_
         << " road_state=" << lane_road_state_
         << " lane_age=" << geometry_age_seconds(last_lane_state_time_, has_lane_state_, now)
         << " offset_y07_age=" << geometry_age_seconds(last_offset_y07_time_, has_offset_y07_, now)
         << " offset_y08_age=" << geometry_age_seconds(last_offset_y08_time_, has_offset_y08_, now)
         << " offset_y09_age=" << geometry_age_seconds(last_offset_y09_time_, has_offset_y09_, now)
         << " heading_age=" << geometry_age_seconds(last_heading_error_time_, has_heading_error_, now)
         << " curvature_age=" << geometry_age_seconds(last_curvature_time_, has_curvature_, now)
         << " geometry_content_age=" << std::chrono::duration<double>(
      now - last_geometry_content_change_time_).count()
         << " geometry_stalled=" << (geometry_content_stalled(now) ? "True" : "False")
         << " perception_ready=" << (perception_ready(now) ? "True" : "False")
         << " stop_request=" << (stop_request_active_ ? "True" : "False")
         << " emergency=" << (emergency_stop_active_ ? "True" : "False")
         << " offset_y07=" << current_offset_y07_
         << " offset_y08=" << current_offset_y08_
         << " offset_y09=" << current_offset_y09_
         << " weighted_offset=" << weighted_offset
         << " heading_error=" << current_heading_error_
         << " curvature=" << current_curvature_
         << " control_error=" << control_error
         << " curve_strength=" << curve_strength
         << " offset_relief=" << offset_relief
         << " allowed_offset=" << allowed_offset
         << " offset_excess=" << offset_excess
         << " curve_risk=" << curve_risk
         << " speed_mps=" << current_speed_mps_
         << " wheel_rps=" << speed_to_wheel_rps(current_speed_mps_)
         << " steering_cmd=" << current_steering_
         << " max_steer=" << dynamic_max
         << " stop_reason=" << stop_reason_;
    message.data = text.str();
    debug_publisher_->publish(message);
  }

  double speed_to_wheel_rps(double speed_mps) const
  {
    if (params_.wheel_radius > 0.0) {
      constexpr double kPi = 3.14159265358979323846;
      return speed_mps / (2.0 * kPi * params_.wheel_radius);
    }
    return speed_mps;
  }

  ControllerParameters params_{};
  double steering_sign_{1.0};
  double control_frequency_{50.0};
  bool autonomous_enabled_on_start_{false};
  bool auto_start_pending_{false};
  std::string offset_y07_topic_{"/segmentation/offset_y07"};
  std::string offset_y08_topic_{"/segmentation/offset_y08"};
  std::string offset_y09_topic_{"/segmentation/offset_y09"};
  std::string heading_error_topic_{"/segmentation/heading_error"};
  std::string curvature_topic_{"/segmentation/curvature"};
  std::string lane_state_topic_{"/perception/lane_state"};
  std::string cmd_vel_topic_{"/cmd_vel"};

  bool auto_enabled_{false};
  bool safety_locked_{false};
  bool has_offset_y07_{false};
  bool has_offset_y08_{false};
  bool has_offset_y09_{false};
  bool has_valid_message_{false};
  bool has_heading_error_{false};
  bool has_curvature_{false};
  bool has_lane_state_{false};
  bool is_valid_{false};
  bool lane_state_valid_{false};
  bool stop_request_active_{false};
  bool emergency_stop_active_{false};
  double current_offset_y07_{0.0};
  double current_offset_y08_{0.0};
  double current_offset_y09_{0.0};
  double current_heading_error_{0.0};
  double current_curvature_{0.0};
  double lane_confidence_{0.0};
  double previous_control_error_{0.0};
  double filtered_derivative_{0.0};
  double current_speed_mps_{0.0};
  double current_steering_{0.0};
  std::optional<std::chrono::steady_clock::time_point> invalid_since_;
  std::chrono::steady_clock::time_point last_valid_time_;
  std::chrono::steady_clock::time_point last_offset_y07_time_;
  std::chrono::steady_clock::time_point last_offset_y08_time_;
  std::chrono::steady_clock::time_point last_offset_y09_time_;
  std::chrono::steady_clock::time_point last_heading_error_time_;
  std::chrono::steady_clock::time_point last_curvature_time_;
  std::chrono::steady_clock::time_point last_lane_state_time_;
  std::chrono::steady_clock::time_point last_geometry_content_change_time_;
  std::string lane_road_state_{"UNKNOWN"};
  std::chrono::steady_clock::time_point previous_control_time_;
  std::string last_mode_{"disabled"};
  std::string stop_reason_{"startup_disabled"};

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y07_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y08_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y09_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr heading_error_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr curvature_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr valid_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_request_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr lane_state_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr chassis_enable_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_enabled_service_;
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
      fprintf(stderr, "Failed to start line_follower_control_cpp: %s\n", exc.what());
    }
  }

  if (node) {
    RCLCPP_INFO(node->get_logger(), "Shutting down - publishing zero /cmd_vel and disabling chassis...");
    node->publish_stop_commands(5);
  }

  node.reset();
  rclcpp::shutdown();
  return 0;
}
