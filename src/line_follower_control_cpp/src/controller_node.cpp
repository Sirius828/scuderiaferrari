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
#include "rcl_interfaces/msg/floating_point_range.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/int64.hpp"
#include "std_msgs/msg/int8.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "line_follower_control_cpp/finish_turn_state.hpp"
#include "line_follower_control_cpp/guideboard_reverse_state.hpp"

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
  double high_error_rescue_speed_mps{0.0};
  double high_error_rescue_start{0.70};
  double inner_side_min_linear_speed_mps{0.0};
  double inner_side_offset_threshold{0.05};
  double wheel_radius{0.032};
  double max_steering{0.85};
  double straight_max_steering{0.30};
  double curve_max_steering{0.85};
  double steering_offset_start{0.05};
  double steering_offset_full{0.60};
  double curve_entry_max_steering{0.0};
  double curve_entry_error_start{0.45};
  double curve_entry_error_override{0.85};
  double curve_entry_heading_confirm{0.45};
  double curve_entry_window_time{0.0};
  double speed_offset_start{0.05};
  double speed_offset_full{0.60};
  double speed_slowdown_exponent{1.0};
  double speed_error_rate_start{0.0};
  double speed_error_rate_full{0.0};
  double speed_accel_rate{0.20};
  double speed_decel_rate{2.0};
  double branch_max_speed_mps{0.0};
  double branch_min_speed_mps{0.0};
  double derivative_limit{3.0};
  double derivative_filter_alpha{0.35};
  double derivative_brake_gain{1.0};
  double steering_slew_rate{4.0};
  double steering_return_slew_rate{1.5};
  double branch_max_steering{0.60};
  double branch_exit_hold_time{0.80};
  double branch_error_slew_rate{0.60};
  double branch_error_recovery_rate{2.00};
  double offset_timeout{0.25};
  double geometry_stall_timeout{0.20};
  bool enable_geometry_stall_auto_resume{true};
  double geometry_stall_recovery_time{0.50};
  double invalid_hold_speed_mps{0.25};
  bool enable_line_loss_command_hold{false};
  double offset_y07_weight{0.20};
  double offset_y08_weight{0.30};
  double offset_y09_weight{0.50};
  double global_offset_blend{0.25};
  double centerline_bias{0.0};
  double global_offset_deadband{0.02};
  bool enable_adaptive_offset_weights{false};
  double near_offset_y07_weight{0.35};
  double near_offset_y08_weight{0.35};
  double near_offset_y09_weight{0.30};
  double near_offset_advantage_start{0.06};
  double near_offset_advantage_full{0.30};
  double heading_feedback_gain{0.35};
  double lookahead_transition_gain{0.0};
  double curve_outer_bias{0.0};
  double curve_outer_bias_start{0.35};
  double curve_outer_bias_full{0.70};
  double curve_outer_bias_hold_time{0.0};
  bool lock_branch_outer_bias{false};
  double branch_outer_bias_release_time{0.0};
  double curvature_speed_weight{0.50};
  double far_offset_speed_weight{0.0};
  double curve_offset_relief_start{0.15};
  double curve_offset_relief_full{0.60};
  bool enable_dynamic_speed{true};
  bool enable_dynamic_steering_limit{true};
  bool enable_curve_offset_allowance{true};
  bool enable_curve_offset_relief{true};
  double straight_allowed_offset{0.05};
  double curve_allowed_offset{0.60};
  double min_perception_confidence{0.50};
  int64_t finish_turn_encoder_counts{20000};
  double finish_turn_speed_mps{0.8};
  double finish_turn_steering{-1.0};
  double finish_turn_timeout_sec{8.0};
  double finish_turn_encoder_max_age_sec{0.30};
  int64_t guideboard_reverse_encoder_counts{1250};
  double guideboard_reverse_speed_mps{0.40};
  double guideboard_reverse_steering{0.0};
  double guideboard_reverse_timeout_sec{3.0};
  double guideboard_reverse_encoder_max_age_sec{0.30};
  int64_t guideboard_reverse_encoder_jitter_counts{10};
  int64_t guideboard_reverse_encoder_max_step_counts{500};
  bool guideboard_reverse_require_single_encoder_publisher{true};
};

class LineFollowerControllerCpp : public rclcpp::Node
{
public:
  LineFollowerControllerCpp()
  : Node("line_follower_controller_cpp"),
    last_frame_signature_time_(std::chrono::steady_clock::now()),
    last_frame_signature_change_time_(std::chrono::steady_clock::now()),
    previous_control_time_(std::chrono::steady_clock::now())
  {
    declare_parameter<double>("Kp", params_.kp);
    declare_parameter<double>("Kd", params_.kd);
    declare_parameter<double>("linear_speed", params_.linear_speed_mps);
    declare_parameter<double>("min_linear_speed", params_.min_linear_speed_mps);
    declare_parameter<double>(
      "high_error_rescue_speed", params_.high_error_rescue_speed_mps);
    declare_parameter<double>(
      "high_error_rescue_start", params_.high_error_rescue_start);
    declare_parameter<double>(
      "inner_side_min_linear_speed", params_.inner_side_min_linear_speed_mps);
    declare_parameter<double>(
      "inner_side_offset_threshold", params_.inner_side_offset_threshold);
    declare_parameter<double>("wheel_radius", params_.wheel_radius);
    declare_parameter<double>("max_steering", params_.max_steering);
    declare_parameter<double>("straight_max_steering", params_.straight_max_steering);
    declare_parameter<double>("curve_max_steering", params_.curve_max_steering);
    declare_parameter<double>("steering_offset_start", params_.steering_offset_start);
    declare_parameter<double>("steering_offset_full", params_.steering_offset_full);
    declare_parameter<double>("curve_entry_max_steering", params_.curve_entry_max_steering);
    declare_parameter<double>("curve_entry_error_start", params_.curve_entry_error_start);
    declare_parameter<double>("curve_entry_error_override", params_.curve_entry_error_override);
    declare_parameter<double>(
      "curve_entry_heading_confirm", params_.curve_entry_heading_confirm);
    declare_parameter<double>("curve_entry_window_time", params_.curve_entry_window_time);
    declare_parameter<double>("speed_offset_start", params_.speed_offset_start);
    declare_parameter<double>("speed_offset_full", params_.speed_offset_full);
    declare_parameter<double>("speed_slowdown_exponent", params_.speed_slowdown_exponent);
    declare_parameter<double>("speed_error_rate_start", params_.speed_error_rate_start);
    declare_parameter<double>("speed_error_rate_full", params_.speed_error_rate_full);
    declare_parameter<double>("speed_accel_rate", params_.speed_accel_rate);
    declare_parameter<double>("speed_decel_rate", params_.speed_decel_rate);
    declare_parameter<double>("branch_max_speed", params_.branch_max_speed_mps);
    declare_parameter<double>("branch_min_speed", params_.branch_min_speed_mps);
    declare_parameter<double>("derivative_limit", params_.derivative_limit);
    declare_parameter<double>("derivative_filter_alpha", params_.derivative_filter_alpha);
    declare_parameter<double>("derivative_brake_gain", params_.derivative_brake_gain);
    declare_parameter<double>("steering_slew_rate", params_.steering_slew_rate);
    declare_parameter<double>("steering_return_slew_rate", params_.steering_return_slew_rate);
    declare_parameter<double>("branch_max_steering", params_.branch_max_steering);
    declare_parameter<double>("branch_exit_hold_time", params_.branch_exit_hold_time);
    declare_parameter<double>("branch_error_slew_rate", params_.branch_error_slew_rate);
    declare_parameter<double>("branch_error_recovery_rate", params_.branch_error_recovery_rate);
    declare_parameter<double>("offset_timeout", params_.offset_timeout);
    declare_parameter<double>("geometry_stall_timeout", params_.geometry_stall_timeout);
    declare_parameter<bool>(
      "enable_geometry_stall_auto_resume", params_.enable_geometry_stall_auto_resume);
    declare_parameter<double>(
      "geometry_stall_recovery_time", params_.geometry_stall_recovery_time);
    declare_parameter<double>("invalid_hold_speed", params_.invalid_hold_speed_mps);
    declare_parameter<bool>(
      "enable_line_loss_command_hold", params_.enable_line_loss_command_hold);
    declare_parameter<double>("offset_y07_weight", params_.offset_y07_weight);
    declare_parameter<double>("offset_y08_weight", params_.offset_y08_weight);
    declare_parameter<double>("offset_y09_weight", params_.offset_y09_weight);
    const auto numeric_descriptor = [](
      const std::string & description, double minimum, double maximum)
      {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        descriptor.description = description;
        rcl_interfaces::msg::FloatingPointRange range;
        range.from_value = minimum;
        range.to_value = maximum;
        range.step = 0.0;
        descriptor.floating_point_range.push_back(range);
        return descriptor;
      };
    declare_parameter<double>(
      "global_offset_blend", params_.global_offset_blend,
      numeric_descriptor(
        "Blend ratio between the existing weighted offset and the whole fitted-line offset",
        0.0, 1.0));
    declare_parameter<double>(
      "centerline_bias", params_.centerline_bias,
      numeric_descriptor(
        "Constant normalized center correction used to compensate a persistent side bias",
        -1.0, 1.0));
    declare_parameter<double>(
      "global_offset_deadband", params_.global_offset_deadband,
      numeric_descriptor(
        "Deadband for the difference between whole-line and weighted offsets",
        0.0, 1.0));
    declare_parameter<bool>(
      "enable_adaptive_offset_weights", params_.enable_adaptive_offset_weights);
    declare_parameter<double>("near_offset_y07_weight", params_.near_offset_y07_weight);
    declare_parameter<double>("near_offset_y08_weight", params_.near_offset_y08_weight);
    declare_parameter<double>("near_offset_y09_weight", params_.near_offset_y09_weight);
    declare_parameter<double>(
      "near_offset_advantage_start", params_.near_offset_advantage_start);
    declare_parameter<double>(
      "near_offset_advantage_full", params_.near_offset_advantage_full);
    declare_parameter<double>("heading_feedback_gain", params_.heading_feedback_gain);
    declare_parameter<double>("lookahead_transition_gain", params_.lookahead_transition_gain);
    declare_parameter<double>("curve_outer_bias", params_.curve_outer_bias);
    declare_parameter<double>("curve_outer_bias_start", params_.curve_outer_bias_start);
    declare_parameter<double>("curve_outer_bias_full", params_.curve_outer_bias_full);
    declare_parameter<double>("curve_outer_bias_hold_time", params_.curve_outer_bias_hold_time);
    declare_parameter<bool>("lock_branch_outer_bias", params_.lock_branch_outer_bias);
    declare_parameter<double>(
      "branch_outer_bias_release_time", params_.branch_outer_bias_release_time);
    declare_parameter<double>("curvature_speed_weight", params_.curvature_speed_weight);
    declare_parameter<double>("far_offset_speed_weight", params_.far_offset_speed_weight);
    declare_parameter<double>("curve_offset_relief_start", params_.curve_offset_relief_start);
    declare_parameter<double>("curve_offset_relief_full", params_.curve_offset_relief_full);
    declare_parameter<bool>("enable_dynamic_speed", params_.enable_dynamic_speed);
    declare_parameter<bool>("enable_dynamic_steering_limit", params_.enable_dynamic_steering_limit);
    declare_parameter<bool>("enable_curve_offset_allowance", params_.enable_curve_offset_allowance);
    declare_parameter<bool>("enable_curve_offset_relief", params_.enable_curve_offset_relief);
    declare_parameter<double>("straight_allowed_offset", params_.straight_allowed_offset);
    declare_parameter<double>("curve_allowed_offset", params_.curve_allowed_offset);
    declare_parameter<double>("min_perception_confidence", params_.min_perception_confidence);
    declare_parameter<int64_t>(
      "finish_turn_encoder_counts", params_.finish_turn_encoder_counts);
    declare_parameter<double>("finish_turn_speed_mps", params_.finish_turn_speed_mps);
    declare_parameter<double>("finish_turn_steering", params_.finish_turn_steering);
    declare_parameter<double>("finish_turn_timeout_sec", params_.finish_turn_timeout_sec);
    declare_parameter<double>(
      "finish_turn_encoder_max_age_sec", params_.finish_turn_encoder_max_age_sec);
    declare_parameter<int64_t>(
      "guideboard_reverse_encoder_counts", params_.guideboard_reverse_encoder_counts);
    declare_parameter<double>(
      "guideboard_reverse_speed_mps", params_.guideboard_reverse_speed_mps);
    declare_parameter<double>(
      "guideboard_reverse_steering", params_.guideboard_reverse_steering);
    declare_parameter<double>(
      "guideboard_reverse_timeout_sec", params_.guideboard_reverse_timeout_sec);
    declare_parameter<double>(
      "guideboard_reverse_encoder_max_age_sec",
      params_.guideboard_reverse_encoder_max_age_sec);
    declare_parameter<int64_t>(
      "guideboard_reverse_encoder_jitter_counts",
      params_.guideboard_reverse_encoder_jitter_counts);
    declare_parameter<int64_t>(
      "guideboard_reverse_encoder_max_step_counts",
      params_.guideboard_reverse_encoder_max_step_counts);
    declare_parameter<bool>(
      "guideboard_reverse_require_single_encoder_publisher",
      params_.guideboard_reverse_require_single_encoder_publisher);
    declare_parameter<double>("steering_sign", 1.0);
    declare_parameter<double>("control_frequency", 50.0);
    declare_parameter<bool>("autonomous_enabled_on_start", false);
    declare_parameter<std::string>("offset_y07_topic", "/segmentation/offset_y07");
    declare_parameter<std::string>("offset_y08_topic", "/segmentation/offset_y08");
    declare_parameter<std::string>("offset_y09_topic", "/segmentation/offset_y09");
    declare_parameter<std::string>("global_offset_topic", "/segmentation/global_offset");
    declare_parameter<std::string>("heading_error_topic", "/segmentation/heading_error");
    declare_parameter<std::string>("curvature_topic", "/segmentation/curvature");
    declare_parameter<std::string>("lane_state_topic", "/perception/lane_state");
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    declare_parameter<std::string>(
      "finish_turn_encoder_topic", "/chassis/encoder_count");

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
    global_offset_subscription_ = create_subscription<std_msgs::msg::Float32>(
      global_offset_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::global_offset_callback, this, std::placeholders::_1));
    heading_error_subscription_ = create_subscription<std_msgs::msg::Float32>(
      heading_error_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::heading_error_callback, this, std::placeholders::_1));
    curvature_subscription_ = create_subscription<std_msgs::msg::Float32>(
      curvature_topic_, sensor_qos,
      std::bind(&LineFollowerControllerCpp::curvature_callback, this, std::placeholders::_1));
    valid_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/segmentation/is_valid", sensor_qos,
      std::bind(&LineFollowerControllerCpp::valid_callback, this, std::placeholders::_1));
    emergency_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/race/emergency_stop", 10,
      std::bind(&LineFollowerControllerCpp::emergency_callback, this, std::placeholders::_1));
    lane_state_subscription_ = create_subscription<std_msgs::msg::String>(
      lane_state_topic_, 10,
      std::bind(&LineFollowerControllerCpp::lane_state_callback, this, std::placeholders::_1));
    frame_signature_subscription_ = create_subscription<std_msgs::msg::UInt64>(
      "/perception/frame_signature", sensor_qos,
      std::bind(
        &LineFollowerControllerCpp::frame_signature_callback, this, std::placeholders::_1));
    finish_turn_encoder_subscription_ = create_subscription<std_msgs::msg::Int64>(
      finish_turn_encoder_topic_, rclcpp::QoS(10).reliable(),
      std::bind(
        &LineFollowerControllerCpp::finish_turn_encoder_callback, this,
        std::placeholders::_1));

    cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    chassis_enable_publisher_ = create_publisher<std_msgs::msg::Int8>("/chassis/enable", 10);
    debug_publisher_ = create_publisher<std_msgs::msg::String>("/line_follower/debug", 10);
    rclcpp::QoS reverse_state_qos(1);
    reverse_state_qos.reliable().transient_local();
    guideboard_reverse_state_publisher_ = create_publisher<std_msgs::msg::String>(
      "/line_follower/guideboard_reverse_state", reverse_state_qos);

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
    obstacle_pause_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/obstacle_pause",
      std::bind(
        &LineFollowerControllerCpp::obstacle_pause_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    obstacle_resume_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/obstacle_resume",
      std::bind(
        &LineFollowerControllerCpp::obstacle_resume_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    finish_turn_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/finish_turn",
      std::bind(
        &LineFollowerControllerCpp::finish_turn_service_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    guideboard_reverse_service_ = create_service<std_srvs::srv::Trigger>(
      "/line_follower/guideboard_reverse",
      std::bind(
        &LineFollowerControllerCpp::guideboard_reverse_service_callback, this,
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
      "Controller starts disabled; use /line_follower/start or /line_follower/set_enabled true; "
      "line_loss_command_hold=%s",
      params_.enable_line_loss_command_hold ? "true" : "false");
    publish_guideboard_reverse_state(std::chrono::steady_clock::now());
  }

  void publish_stop_commands(int count = 5)
  {
    for (int i = 0; i < count; ++i) {
      publish_stop_state();
      rclcpp::sleep_for(50ms);
    }
  }

private:
  static double steady_seconds(const std::chrono::steady_clock::time_point & now)
  {
    return std::chrono::duration<double>(now.time_since_epoch()).count();
  }

  line_follower_control_cpp::FinishTurnConfig finish_turn_config() const
  {
    line_follower_control_cpp::FinishTurnConfig config;
    config.target_encoder_counts = params_.finish_turn_encoder_counts;
    config.timeout_sec = params_.finish_turn_timeout_sec;
    config.encoder_max_age_sec = params_.finish_turn_encoder_max_age_sec;
    return config;
  }

  line_follower_control_cpp::GuideboardReverseConfig guideboard_reverse_config() const
  {
    line_follower_control_cpp::GuideboardReverseConfig config;
    config.target_encoder_counts = params_.guideboard_reverse_encoder_counts;
    config.timeout_sec = params_.guideboard_reverse_timeout_sec;
    config.encoder_max_age_sec = params_.guideboard_reverse_encoder_max_age_sec;
    config.encoder_jitter_counts = params_.guideboard_reverse_encoder_jitter_counts;
    config.encoder_max_step_counts = params_.guideboard_reverse_encoder_max_step_counts;
    return config;
  }

  void load_parameters()
  {
    params_.kp = get_parameter("Kp").as_double();
    params_.kd = get_parameter("Kd").as_double();
    params_.linear_speed_mps = get_parameter("linear_speed").as_double();
    params_.min_linear_speed_mps = get_parameter("min_linear_speed").as_double();
    params_.high_error_rescue_speed_mps =
      get_parameter("high_error_rescue_speed").as_double();
    params_.high_error_rescue_start =
      get_parameter("high_error_rescue_start").as_double();
    params_.inner_side_min_linear_speed_mps =
      get_parameter("inner_side_min_linear_speed").as_double();
    params_.inner_side_offset_threshold =
      get_parameter("inner_side_offset_threshold").as_double();
    params_.wheel_radius = get_parameter("wheel_radius").as_double();
    params_.max_steering = get_parameter("max_steering").as_double();
    params_.straight_max_steering = get_parameter("straight_max_steering").as_double();
    params_.curve_max_steering = get_parameter("curve_max_steering").as_double();
    params_.steering_offset_start = get_parameter("steering_offset_start").as_double();
    params_.steering_offset_full = get_parameter("steering_offset_full").as_double();
    params_.curve_entry_max_steering = get_parameter("curve_entry_max_steering").as_double();
    params_.curve_entry_error_start = get_parameter("curve_entry_error_start").as_double();
    params_.curve_entry_error_override = get_parameter("curve_entry_error_override").as_double();
    params_.curve_entry_heading_confirm =
      get_parameter("curve_entry_heading_confirm").as_double();
    params_.curve_entry_window_time = get_parameter("curve_entry_window_time").as_double();
    params_.speed_offset_start = get_parameter("speed_offset_start").as_double();
    params_.speed_offset_full = get_parameter("speed_offset_full").as_double();
    params_.speed_slowdown_exponent = get_parameter("speed_slowdown_exponent").as_double();
    params_.speed_error_rate_start = get_parameter("speed_error_rate_start").as_double();
    params_.speed_error_rate_full = get_parameter("speed_error_rate_full").as_double();
    params_.speed_accel_rate = get_parameter("speed_accel_rate").as_double();
    params_.speed_decel_rate = get_parameter("speed_decel_rate").as_double();
    params_.branch_max_speed_mps = get_parameter("branch_max_speed").as_double();
    params_.branch_min_speed_mps = get_parameter("branch_min_speed").as_double();
    params_.derivative_limit = get_parameter("derivative_limit").as_double();
    params_.derivative_filter_alpha = get_parameter("derivative_filter_alpha").as_double();
    params_.derivative_brake_gain = get_parameter("derivative_brake_gain").as_double();
    params_.steering_slew_rate = get_parameter("steering_slew_rate").as_double();
    params_.steering_return_slew_rate = get_parameter("steering_return_slew_rate").as_double();
    params_.branch_max_steering = get_parameter("branch_max_steering").as_double();
    params_.branch_exit_hold_time = get_parameter("branch_exit_hold_time").as_double();
    params_.branch_error_slew_rate = get_parameter("branch_error_slew_rate").as_double();
    params_.branch_error_recovery_rate = get_parameter("branch_error_recovery_rate").as_double();
    params_.offset_timeout = get_parameter("offset_timeout").as_double();
    params_.geometry_stall_timeout = get_parameter("geometry_stall_timeout").as_double();
    params_.enable_geometry_stall_auto_resume =
      get_parameter("enable_geometry_stall_auto_resume").as_bool();
    params_.geometry_stall_recovery_time =
      get_parameter("geometry_stall_recovery_time").as_double();
    params_.invalid_hold_speed_mps = get_parameter("invalid_hold_speed").as_double();
    params_.enable_line_loss_command_hold =
      get_parameter("enable_line_loss_command_hold").as_bool();
    params_.offset_y07_weight = get_parameter("offset_y07_weight").as_double();
    params_.offset_y08_weight = get_parameter("offset_y08_weight").as_double();
    params_.offset_y09_weight = get_parameter("offset_y09_weight").as_double();
    params_.global_offset_blend = get_parameter("global_offset_blend").as_double();
    params_.centerline_bias = get_parameter("centerline_bias").as_double();
    params_.global_offset_deadband = get_parameter("global_offset_deadband").as_double();
    params_.enable_adaptive_offset_weights =
      get_parameter("enable_adaptive_offset_weights").as_bool();
    params_.near_offset_y07_weight = get_parameter("near_offset_y07_weight").as_double();
    params_.near_offset_y08_weight = get_parameter("near_offset_y08_weight").as_double();
    params_.near_offset_y09_weight = get_parameter("near_offset_y09_weight").as_double();
    params_.near_offset_advantage_start =
      get_parameter("near_offset_advantage_start").as_double();
    params_.near_offset_advantage_full =
      get_parameter("near_offset_advantage_full").as_double();
    params_.heading_feedback_gain = get_parameter("heading_feedback_gain").as_double();
    params_.lookahead_transition_gain = get_parameter("lookahead_transition_gain").as_double();
    params_.curve_outer_bias = get_parameter("curve_outer_bias").as_double();
    params_.curve_outer_bias_start = get_parameter("curve_outer_bias_start").as_double();
    params_.curve_outer_bias_full = get_parameter("curve_outer_bias_full").as_double();
    params_.curve_outer_bias_hold_time = get_parameter("curve_outer_bias_hold_time").as_double();
    params_.lock_branch_outer_bias = get_parameter("lock_branch_outer_bias").as_bool();
    params_.branch_outer_bias_release_time =
      get_parameter("branch_outer_bias_release_time").as_double();
    params_.curvature_speed_weight = get_parameter("curvature_speed_weight").as_double();
    params_.far_offset_speed_weight = get_parameter("far_offset_speed_weight").as_double();
    params_.curve_offset_relief_start = get_parameter("curve_offset_relief_start").as_double();
    params_.curve_offset_relief_full = get_parameter("curve_offset_relief_full").as_double();
    params_.enable_dynamic_speed = get_parameter("enable_dynamic_speed").as_bool();
    params_.enable_dynamic_steering_limit = get_parameter("enable_dynamic_steering_limit").as_bool();
    params_.enable_curve_offset_allowance = get_parameter("enable_curve_offset_allowance").as_bool();
    params_.enable_curve_offset_relief = get_parameter("enable_curve_offset_relief").as_bool();
    params_.straight_allowed_offset = get_parameter("straight_allowed_offset").as_double();
    params_.curve_allowed_offset = get_parameter("curve_allowed_offset").as_double();
    params_.min_perception_confidence = get_parameter("min_perception_confidence").as_double();
    params_.finish_turn_encoder_counts =
      get_parameter("finish_turn_encoder_counts").as_int();
    params_.finish_turn_speed_mps = get_parameter("finish_turn_speed_mps").as_double();
    params_.finish_turn_steering = get_parameter("finish_turn_steering").as_double();
    params_.finish_turn_timeout_sec = get_parameter("finish_turn_timeout_sec").as_double();
    params_.finish_turn_encoder_max_age_sec =
      get_parameter("finish_turn_encoder_max_age_sec").as_double();
    params_.guideboard_reverse_encoder_counts =
      get_parameter("guideboard_reverse_encoder_counts").as_int();
    params_.guideboard_reverse_speed_mps =
      get_parameter("guideboard_reverse_speed_mps").as_double();
    params_.guideboard_reverse_steering =
      get_parameter("guideboard_reverse_steering").as_double();
    params_.guideboard_reverse_timeout_sec =
      get_parameter("guideboard_reverse_timeout_sec").as_double();
    params_.guideboard_reverse_encoder_max_age_sec =
      get_parameter("guideboard_reverse_encoder_max_age_sec").as_double();
    params_.guideboard_reverse_encoder_jitter_counts =
      get_parameter("guideboard_reverse_encoder_jitter_counts").as_int();
    params_.guideboard_reverse_encoder_max_step_counts =
      get_parameter("guideboard_reverse_encoder_max_step_counts").as_int();
    params_.guideboard_reverse_require_single_encoder_publisher =
      get_parameter("guideboard_reverse_require_single_encoder_publisher").as_bool();
    steering_sign_ = get_parameter("steering_sign").as_double();
    control_frequency_ = get_parameter("control_frequency").as_double();
    autonomous_enabled_on_start_ = get_parameter("autonomous_enabled_on_start").as_bool();
    offset_y07_topic_ = get_parameter("offset_y07_topic").as_string();
    offset_y08_topic_ = get_parameter("offset_y08_topic").as_string();
    offset_y09_topic_ = get_parameter("offset_y09_topic").as_string();
    global_offset_topic_ = get_parameter("global_offset_topic").as_string();
    heading_error_topic_ = get_parameter("heading_error_topic").as_string();
    curvature_topic_ = get_parameter("curvature_topic").as_string();
    lane_state_topic_ = get_parameter("lane_state_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    finish_turn_encoder_topic_ = get_parameter("finish_turn_encoder_topic").as_string();
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
    if (!std::isfinite(parameters.high_error_rescue_speed_mps) ||
      parameters.high_error_rescue_speed_mps < 0.0 ||
      parameters.high_error_rescue_speed_mps > parameters.linear_speed_mps)
    {
      return fail("high_error_rescue_speed must be in [0, linear_speed]");
    }
    if (!std::isfinite(parameters.high_error_rescue_start) ||
      parameters.high_error_rescue_start < 0.0 ||
      parameters.high_error_rescue_start >= parameters.curve_entry_error_override)
    {
      return fail("high_error_rescue_start must be in [0, curve_entry_error_override)");
    }
    if (!std::isfinite(parameters.inner_side_min_linear_speed_mps) ||
      parameters.inner_side_min_linear_speed_mps < 0.0 ||
      parameters.inner_side_min_linear_speed_mps > parameters.linear_speed_mps ||
      (parameters.inner_side_min_linear_speed_mps > 0.0 &&
      parameters.inner_side_min_linear_speed_mps < parameters.min_linear_speed_mps))
    {
      return fail(
        "inner_side_min_linear_speed must be 0 or in [min_linear_speed, linear_speed]");
    }
    if (!std::isfinite(parameters.inner_side_offset_threshold) ||
      parameters.inner_side_offset_threshold < 0.0 ||
      parameters.inner_side_offset_threshold > 1.0)
    {
      return fail("inner_side_offset_threshold must be in [0, 1]");
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
    if (!std::isfinite(parameters.curve_entry_max_steering) ||
      parameters.curve_entry_max_steering < 0.0 ||
      parameters.curve_entry_max_steering > parameters.max_steering)
    {
      return fail("curve_entry_max_steering must be in [0, max_steering]");
    }
    if (!std::isfinite(parameters.curve_entry_error_start) ||
      parameters.curve_entry_error_start < 0.0 ||
      parameters.curve_entry_error_start >= 1.0)
    {
      return fail("curve_entry_error_start must be in [0, 1)");
    }
    if (!std::isfinite(parameters.curve_entry_error_override) ||
      parameters.curve_entry_error_override <= parameters.curve_entry_error_start ||
      parameters.curve_entry_error_override > 1.0)
    {
      return fail("curve_entry_error_override must be > error_start and <= 1");
    }
    if (!std::isfinite(parameters.curve_entry_heading_confirm) ||
      parameters.curve_entry_heading_confirm <= 0.0 ||
      parameters.curve_entry_heading_confirm > 1.0)
    {
      return fail("curve_entry_heading_confirm must be in (0, 1]");
    }
    if (!std::isfinite(parameters.curve_entry_window_time) ||
      parameters.curve_entry_window_time < 0.0)
    {
      return fail("curve_entry_window_time must be >= 0");
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
    if (!std::isfinite(parameters.speed_slowdown_exponent) ||
      parameters.speed_slowdown_exponent < 0.5 || parameters.speed_slowdown_exponent > 4.0)
    {
      return fail("speed_slowdown_exponent must be in [0.5, 4]");
    }
    if (!std::isfinite(parameters.speed_error_rate_start) ||
      !std::isfinite(parameters.speed_error_rate_full) ||
      parameters.speed_error_rate_start < 0.0 ||
      parameters.speed_error_rate_full < 0.0 ||
      (parameters.speed_error_rate_full > 0.0 &&
      parameters.speed_error_rate_full <= parameters.speed_error_rate_start))
    {
      return fail(
        "speed_error_rate_start/full must be >= 0, with full > start when enabled");
    }
    if (!std::isfinite(parameters.speed_accel_rate) || parameters.speed_accel_rate < 0.0) {
      return fail("speed_accel_rate must be >= 0");
    }
    if (!std::isfinite(parameters.speed_decel_rate) || parameters.speed_decel_rate < 0.0) {
      return fail("speed_decel_rate must be >= 0");
    }
    if (!std::isfinite(parameters.branch_max_speed_mps) ||
      parameters.branch_max_speed_mps < 0.0 ||
      parameters.branch_max_speed_mps > parameters.linear_speed_mps)
    {
      return fail("branch_max_speed must be 0 or in (0, linear_speed]");
    }
    if (!std::isfinite(parameters.branch_min_speed_mps) ||
      parameters.branch_min_speed_mps < 0.0 ||
      parameters.branch_min_speed_mps > parameters.linear_speed_mps ||
      (parameters.branch_min_speed_mps > 0.0 && parameters.branch_max_speed_mps > 0.0 &&
      parameters.branch_min_speed_mps > parameters.branch_max_speed_mps))
    {
      return fail("branch_min_speed must be 0 or <= branch_max_speed and linear_speed");
    }
    if (!std::isfinite(parameters.derivative_limit) || parameters.derivative_limit < 0.0) {
      return fail("derivative_limit must be >= 0");
    }
    if (!std::isfinite(parameters.lookahead_transition_gain) ||
      parameters.lookahead_transition_gain < 0.0 ||
      parameters.lookahead_transition_gain > 1.0)
    {
      return fail("lookahead_transition_gain must be in [0, 1]");
    }
    if (!std::isfinite(parameters.derivative_filter_alpha) ||
      parameters.derivative_filter_alpha < 0.0 || parameters.derivative_filter_alpha > 1.0)
    {
      return fail("derivative_filter_alpha must be in [0, 1]");
    }
    if (!std::isfinite(parameters.derivative_brake_gain) ||
      parameters.derivative_brake_gain < 1.0 || parameters.derivative_brake_gain > 3.0)
    {
      return fail("derivative_brake_gain must be in [1, 3]");
    }
    if (!std::isfinite(parameters.steering_slew_rate) || parameters.steering_slew_rate < 0.0) {
      return fail("steering_slew_rate must be >= 0");
    }
    if (!std::isfinite(parameters.steering_return_slew_rate) ||
      parameters.steering_return_slew_rate < 0.0)
    {
      return fail("steering_return_slew_rate must be >= 0");
    }
    if (!std::isfinite(parameters.branch_max_steering) ||
      parameters.branch_max_steering < 0.0 ||
      parameters.branch_max_steering > parameters.max_steering)
    {
      return fail("branch_max_steering must be in [0, max_steering]");
    }
    if (!std::isfinite(parameters.branch_exit_hold_time) ||
      parameters.branch_exit_hold_time < 0.0)
    {
      return fail("branch_exit_hold_time must be >= 0");
    }
    if (!std::isfinite(parameters.branch_error_slew_rate) ||
      parameters.branch_error_slew_rate < 0.0)
    {
      return fail("branch_error_slew_rate must be >= 0");
    }
    if (!std::isfinite(parameters.branch_error_recovery_rate) ||
      parameters.branch_error_recovery_rate < 0.0)
    {
      return fail("branch_error_recovery_rate must be >= 0");
    }
    if (!std::isfinite(parameters.offset_timeout) || parameters.offset_timeout <= 0.0) {
      return fail("offset_timeout must be > 0");
    }
    if (!std::isfinite(parameters.geometry_stall_timeout) ||
      parameters.geometry_stall_timeout < 0.0)
    {
      return fail("geometry_stall_timeout must be >= 0");
    }
    if (!std::isfinite(parameters.geometry_stall_recovery_time) ||
      parameters.geometry_stall_recovery_time < 0.0 ||
      parameters.geometry_stall_recovery_time > 10.0)
    {
      return fail("geometry_stall_recovery_time must be in [0, 10]");
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
    if (!std::isfinite(parameters.near_offset_y07_weight) ||
      parameters.near_offset_y07_weight < 0.0 ||
      !std::isfinite(parameters.near_offset_y08_weight) ||
      parameters.near_offset_y08_weight < 0.0 ||
      !std::isfinite(parameters.near_offset_y09_weight) ||
      parameters.near_offset_y09_weight < 0.0)
    {
      return fail("near offset weights must be finite and >= 0");
    }
    if (parameters.near_offset_y07_weight + parameters.near_offset_y08_weight +
        parameters.near_offset_y09_weight <= 1e-9)
    {
      return fail("near offset weights must have a positive sum");
    }
    if (!std::isfinite(parameters.global_offset_blend) ||
      parameters.global_offset_blend < 0.0 || parameters.global_offset_blend > 1.0)
    {
      return fail("global_offset_blend must be in [0, 1]");
    }
    if (!std::isfinite(parameters.centerline_bias) ||
      parameters.centerline_bias < -1.0 || parameters.centerline_bias > 1.0)
    {
      return fail("centerline_bias must be in [-1, 1]");
    }
    if (!std::isfinite(parameters.global_offset_deadband) ||
      parameters.global_offset_deadband < 0.0 || parameters.global_offset_deadband > 1.0)
    {
      return fail("global_offset_deadband must be in [0, 1]");
    }
    if (!std::isfinite(parameters.near_offset_advantage_start) ||
      !std::isfinite(parameters.near_offset_advantage_full) ||
      parameters.near_offset_advantage_start < 0.0 ||
      parameters.near_offset_advantage_full <= parameters.near_offset_advantage_start)
    {
      return fail("near offset advantage full must be > start >= 0");
    }
    if (!std::isfinite(parameters.heading_feedback_gain) || parameters.heading_feedback_gain < 0.0) {
      return fail("heading_feedback_gain must be >= 0");
    }
    if (!std::isfinite(parameters.curve_outer_bias) ||
      std::abs(parameters.curve_outer_bias) > 0.5)
    {
      return fail("curve_outer_bias must be in [-0.5, 0.5] (negative targets curve inside)");
    }
    if (!std::isfinite(parameters.curve_outer_bias_start) ||
      parameters.curve_outer_bias_start < 0.0 || parameters.curve_outer_bias_start >= 1.0)
    {
      return fail("curve_outer_bias_start must be in [0, 1)");
    }
    if (!std::isfinite(parameters.curve_outer_bias_full) ||
      parameters.curve_outer_bias_full <= parameters.curve_outer_bias_start ||
      parameters.curve_outer_bias_full > 1.0)
    {
      return fail("curve_outer_bias_full must be > start and <= 1");
    }
    if (!std::isfinite(parameters.curve_outer_bias_hold_time) ||
      parameters.curve_outer_bias_hold_time < 0.0)
    {
      return fail("curve_outer_bias_hold_time must be >= 0");
    }
    if (!std::isfinite(parameters.branch_outer_bias_release_time) ||
      parameters.branch_outer_bias_release_time < 0.0 ||
      parameters.branch_outer_bias_release_time > 5.0)
    {
      return fail("branch_outer_bias_release_time must be in [0, 5]");
    }
    if (!std::isfinite(parameters.curvature_speed_weight) ||
      parameters.curvature_speed_weight < 0.0 || parameters.curvature_speed_weight > 1.0)
    {
      return fail("curvature_speed_weight must be in [0, 1]");
    }
    if (!std::isfinite(parameters.far_offset_speed_weight) ||
      parameters.far_offset_speed_weight < 0.0 ||
      parameters.far_offset_speed_weight > 1.0)
    {
      return fail("far_offset_speed_weight must be in [0, 1]");
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
    if (parameters.finish_turn_encoder_counts <= 0) {
      return fail("finish_turn_encoder_counts must be > 0");
    }
    if (!std::isfinite(parameters.finish_turn_speed_mps) ||
      parameters.finish_turn_speed_mps <= 0.0)
    {
      return fail("finish_turn_speed_mps must be finite and > 0");
    }
    if (!std::isfinite(parameters.finish_turn_steering) ||
      parameters.finish_turn_steering < -1.0 || parameters.finish_turn_steering > 1.0)
    {
      return fail("finish_turn_steering must be in [-1, 1]");
    }
    if (!std::isfinite(parameters.finish_turn_timeout_sec) ||
      parameters.finish_turn_timeout_sec <= 0.0)
    {
      return fail("finish_turn_timeout_sec must be finite and > 0");
    }
    if (!std::isfinite(parameters.finish_turn_encoder_max_age_sec) ||
      parameters.finish_turn_encoder_max_age_sec <= 0.0)
    {
      return fail("finish_turn_encoder_max_age_sec must be finite and > 0");
    }
    if (parameters.guideboard_reverse_encoder_counts <= 0) {
      return fail("guideboard_reverse_encoder_counts must be > 0");
    }
    if (!std::isfinite(parameters.guideboard_reverse_speed_mps) ||
      parameters.guideboard_reverse_speed_mps <= 0.0)
    {
      return fail("guideboard_reverse_speed_mps must be finite and > 0");
    }
    if (!std::isfinite(parameters.guideboard_reverse_steering) ||
      parameters.guideboard_reverse_steering < -1.0 ||
      parameters.guideboard_reverse_steering > 1.0)
    {
      return fail("guideboard_reverse_steering must be in [-1, 1]");
    }
    if (!std::isfinite(parameters.guideboard_reverse_timeout_sec) ||
      parameters.guideboard_reverse_timeout_sec <= 0.0)
    {
      return fail("guideboard_reverse_timeout_sec must be finite and > 0");
    }
    if (!std::isfinite(parameters.guideboard_reverse_encoder_max_age_sec) ||
      parameters.guideboard_reverse_encoder_max_age_sec <= 0.0)
    {
      return fail("guideboard_reverse_encoder_max_age_sec must be finite and > 0");
    }
    if (parameters.guideboard_reverse_encoder_jitter_counts < 0) {
      return fail("guideboard_reverse_encoder_jitter_counts must be >= 0");
    }
    if (parameters.guideboard_reverse_encoder_max_step_counts <=
      parameters.guideboard_reverse_encoder_jitter_counts)
    {
      return fail(
        "guideboard_reverse_encoder_max_step_counts must be greater than jitter counts");
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
      heading_error_topic_.empty() || curvature_topic_.empty() || global_offset_topic_.empty()) {
      throw std::runtime_error("geometry feedback topics must not be empty");
    }
    if (cmd_vel_topic_.empty() || finish_turn_encoder_topic_.empty()) {
      throw std::runtime_error("cmd_vel_topic and finish_turn_encoder_topic must not be empty");
    }
  }

  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    ControllerParameters pending = params_;
    double pending_steering_sign = steering_sign_;

    for (const auto & parameter : parameters) {
      const auto & name = parameter.get_name();
      if (finish_turn_state_.active() &&
        name.compare(0, std::string("finish_turn_").size(), "finish_turn_") == 0)
      {
        return parameter_result(
          false, name + " cannot change while the finish turn is active");
      }
      if (guideboard_reverse_state_.active() &&
        name.compare(
          0, std::string("guideboard_reverse_").size(), "guideboard_reverse_") == 0)
      {
        return parameter_result(
          false, name + " cannot change while guideboard reverse is active");
      }
      if (name == "control_frequency" || name == "offset_y07_topic" ||
        name == "offset_y08_topic" || name == "offset_y09_topic" ||
        name == "global_offset_topic" ||
        name == "heading_error_topic" ||
        name == "curvature_topic" || name == "lane_state_topic" || name == "cmd_vel_topic" ||
        name == "finish_turn_encoder_topic" ||
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
      } else if (name == "high_error_rescue_speed") {
        pending.high_error_rescue_speed_mps = parameter.as_double();
      } else if (name == "high_error_rescue_start") {
        pending.high_error_rescue_start = parameter.as_double();
      } else if (name == "inner_side_min_linear_speed") {
        pending.inner_side_min_linear_speed_mps = parameter.as_double();
      } else if (name == "inner_side_offset_threshold") {
        pending.inner_side_offset_threshold = parameter.as_double();
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
      } else if (name == "speed_slowdown_exponent") {
        pending.speed_slowdown_exponent = parameter.as_double();
      } else if (name == "speed_error_rate_start") {
        pending.speed_error_rate_start = parameter.as_double();
      } else if (name == "speed_error_rate_full") {
        pending.speed_error_rate_full = parameter.as_double();
      } else if (name == "speed_accel_rate") {
        pending.speed_accel_rate = parameter.as_double();
      } else if (name == "speed_decel_rate") {
        pending.speed_decel_rate = parameter.as_double();
      } else if (name == "branch_max_speed") {
        pending.branch_max_speed_mps = parameter.as_double();
      } else if (name == "branch_min_speed") {
        pending.branch_min_speed_mps = parameter.as_double();
      } else if (name == "derivative_limit") {
        pending.derivative_limit = parameter.as_double();
      } else if (name == "derivative_filter_alpha") {
        pending.derivative_filter_alpha = parameter.as_double();
      } else if (name == "derivative_brake_gain") {
        pending.derivative_brake_gain = parameter.as_double();
      } else if (name == "steering_slew_rate") {
        pending.steering_slew_rate = parameter.as_double();
      } else if (name == "steering_return_slew_rate") {
        pending.steering_return_slew_rate = parameter.as_double();
      } else if (name == "curve_entry_max_steering") {
        pending.curve_entry_max_steering = parameter.as_double();
      } else if (name == "curve_entry_error_start") {
        pending.curve_entry_error_start = parameter.as_double();
      } else if (name == "curve_entry_error_override") {
        pending.curve_entry_error_override = parameter.as_double();
      } else if (name == "curve_entry_heading_confirm") {
        pending.curve_entry_heading_confirm = parameter.as_double();
      } else if (name == "curve_entry_window_time") {
        pending.curve_entry_window_time = parameter.as_double();
      } else if (name == "branch_max_steering") {
        pending.branch_max_steering = parameter.as_double();
      } else if (name == "branch_exit_hold_time") {
        pending.branch_exit_hold_time = parameter.as_double();
      } else if (name == "branch_error_slew_rate") {
        pending.branch_error_slew_rate = parameter.as_double();
      } else if (name == "branch_error_recovery_rate") {
        pending.branch_error_recovery_rate = parameter.as_double();
      } else if (name == "offset_timeout") {
        pending.offset_timeout = parameter.as_double();
      } else if (name == "geometry_stall_timeout") {
        pending.geometry_stall_timeout = parameter.as_double();
      } else if (name == "enable_geometry_stall_auto_resume") {
        pending.enable_geometry_stall_auto_resume = parameter.as_bool();
      } else if (name == "geometry_stall_recovery_time") {
        pending.geometry_stall_recovery_time = parameter.as_double();
      } else if (name == "invalid_hold_speed") {
        pending.invalid_hold_speed_mps = parameter.as_double();
      } else if (name == "enable_line_loss_command_hold") {
        pending.enable_line_loss_command_hold = parameter.as_bool();
      } else if (name == "offset_y07_weight") {
        pending.offset_y07_weight = parameter.as_double();
      } else if (name == "offset_y08_weight") {
        pending.offset_y08_weight = parameter.as_double();
      } else if (name == "offset_y09_weight") {
        pending.offset_y09_weight = parameter.as_double();
      } else if (name == "global_offset_blend") {
        pending.global_offset_blend = parameter.as_double();
      } else if (name == "centerline_bias") {
        pending.centerline_bias = parameter.as_double();
      } else if (name == "global_offset_deadband") {
        pending.global_offset_deadband = parameter.as_double();
      } else if (name == "enable_adaptive_offset_weights") {
        pending.enable_adaptive_offset_weights = parameter.as_bool();
      } else if (name == "near_offset_y07_weight") {
        pending.near_offset_y07_weight = parameter.as_double();
      } else if (name == "near_offset_y08_weight") {
        pending.near_offset_y08_weight = parameter.as_double();
      } else if (name == "near_offset_y09_weight") {
        pending.near_offset_y09_weight = parameter.as_double();
      } else if (name == "near_offset_advantage_start") {
        pending.near_offset_advantage_start = parameter.as_double();
      } else if (name == "near_offset_advantage_full") {
        pending.near_offset_advantage_full = parameter.as_double();
      } else if (name == "heading_feedback_gain") {
        pending.heading_feedback_gain = parameter.as_double();
      } else if (name == "lookahead_transition_gain") {
        pending.lookahead_transition_gain = parameter.as_double();
      } else if (name == "curve_outer_bias") {
        pending.curve_outer_bias = parameter.as_double();
      } else if (name == "curve_outer_bias_start") {
        pending.curve_outer_bias_start = parameter.as_double();
      } else if (name == "curve_outer_bias_full") {
        pending.curve_outer_bias_full = parameter.as_double();
      } else if (name == "curve_outer_bias_hold_time") {
        pending.curve_outer_bias_hold_time = parameter.as_double();
      } else if (name == "lock_branch_outer_bias") {
        pending.lock_branch_outer_bias = parameter.as_bool();
      } else if (name == "branch_outer_bias_release_time") {
        pending.branch_outer_bias_release_time = parameter.as_double();
      } else if (name == "curvature_speed_weight") {
        pending.curvature_speed_weight = parameter.as_double();
      } else if (name == "far_offset_speed_weight") {
        pending.far_offset_speed_weight = parameter.as_double();
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
      } else if (name == "straight_allowed_offset") {
        pending.straight_allowed_offset = parameter.as_double();
      } else if (name == "curve_allowed_offset") {
        pending.curve_allowed_offset = parameter.as_double();
      } else if (name == "min_perception_confidence") {
        pending.min_perception_confidence = parameter.as_double();
      } else if (name == "finish_turn_encoder_counts") {
        pending.finish_turn_encoder_counts = parameter.as_int();
      } else if (name == "finish_turn_speed_mps") {
        pending.finish_turn_speed_mps = parameter.as_double();
      } else if (name == "finish_turn_steering") {
        pending.finish_turn_steering = parameter.as_double();
      } else if (name == "finish_turn_timeout_sec") {
        pending.finish_turn_timeout_sec = parameter.as_double();
      } else if (name == "finish_turn_encoder_max_age_sec") {
        pending.finish_turn_encoder_max_age_sec = parameter.as_double();
      } else if (name == "guideboard_reverse_encoder_counts") {
        pending.guideboard_reverse_encoder_counts = parameter.as_int();
      } else if (name == "guideboard_reverse_speed_mps") {
        pending.guideboard_reverse_speed_mps = parameter.as_double();
      } else if (name == "guideboard_reverse_steering") {
        pending.guideboard_reverse_steering = parameter.as_double();
      } else if (name == "guideboard_reverse_timeout_sec") {
        pending.guideboard_reverse_timeout_sec = parameter.as_double();
      } else if (name == "guideboard_reverse_encoder_max_age_sec") {
        pending.guideboard_reverse_encoder_max_age_sec = parameter.as_double();
      } else if (name == "guideboard_reverse_encoder_jitter_counts") {
        pending.guideboard_reverse_encoder_jitter_counts = parameter.as_int();
      } else if (name == "guideboard_reverse_encoder_max_step_counts") {
        pending.guideboard_reverse_encoder_max_step_counts = parameter.as_int();
      } else if (name == "guideboard_reverse_require_single_encoder_publisher") {
        pending.guideboard_reverse_require_single_encoder_publisher = parameter.as_bool();
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
    if (!params_.enable_geometry_stall_auto_resume) {
      cancel_geometry_stall_resume();
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

  void global_offset_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    update_offset_value(
      msg->data, &current_global_offset_, &has_global_offset_, &last_global_offset_time_);
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
    if (!std::isfinite(msg->data)) {
      return;
    }
    const double heading = std::clamp(static_cast<double>(msg->data), -1.0, 1.0);
    if (std::abs(heading) >= params_.curve_outer_bias_full) {
      last_strong_curve_turn_sign_ = heading < 0.0 ? 1.0 : -1.0;
      last_strong_curve_time_ = std::chrono::steady_clock::now();
      has_strong_curve_memory_ = true;
    }
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

  void emergency_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    emergency_stop_active_ = msg->data;
    if (emergency_stop_active_) {
      finish_turn_state_.cancel("emergency_stop");
      guideboard_reverse_state_.cancel("emergency_stop");
      lock_and_stop("emergency_stop");
    }
  }

  void finish_turn_encoder_callback(const std_msgs::msg::Int64::SharedPtr msg)
  {
    const double now_sec = steady_seconds(std::chrono::steady_clock::now());
    finish_turn_state_.set_encoder_count(msg->data, now_sec);
    guideboard_reverse_state_.set_encoder_count(msg->data, now_sec);
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
    const auto now = std::chrono::steady_clock::now();
    const bool entering_branch = road_state == "BRANCH" && lane_road_state_ != "BRANCH";
    if (entering_branch && params_.lock_branch_outer_bias) {
      const double candidate_bias = compute_curve_outer_bias();
      latched_branch_outer_bias_ =
        std::abs(candidate_bias) >= 0.5 * std::abs(params_.curve_outer_bias) ?
        candidate_bias : 0.0;
    }
    lane_confidence_ = std::clamp(confidence, 0.0, 1.0);
    lane_state_valid_ = is_valid;
    lane_road_state_ = road_state;
    has_lane_state_ = true;
    last_lane_state_time_ = now;
    if (road_state == "BRANCH") {
      last_branch_seen_time_ = now;
      has_seen_branch_ = true;
    }
  }

  void frame_signature_callback(const std_msgs::msg::UInt64::SharedPtr msg)
  {
    const auto now = std::chrono::steady_clock::now();
    if (!has_frame_signature_ || msg->data != current_frame_signature_) {
      last_frame_signature_change_time_ = now;
    }
    current_frame_signature_ = msg->data;
    has_frame_signature_ = msg->data != 0;
    last_frame_signature_time_ = now;
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
    if (emergency_stop_active_) {
      if (reason) {
        *reason = "emergency stop is active";
      }
      return false;
    }
    return true;
  }

  bool try_start(std::string * reason, bool obstacle_resume = false)
  {
    if (finish_turn_state_.active()) {
      if (reason) {
        *reason = "finish turn is active";
      }
      return false;
    }
    if (finish_turn_state_.terminal()) {
      if (obstacle_resume) {
        if (reason) {
          *reason = "finish turn is latched; automatic resume is disabled";
        }
        publish_stop_state();
        return false;
      }
      // /line_follower/start is the explicit operator reset after a completed
      // or faulted finish maneuver.
      finish_turn_state_.reset();
    }
    if (guideboard_reverse_state_.active()) {
      if (reason) {
        *reason = "guideboard reverse is active";
      }
      return false;
    }
    if (guideboard_reverse_state_.terminal()) {
      if (obstacle_resume) {
        if (reason) {
          *reason = "guideboard reverse is latched; automatic resume is disabled";
        }
        publish_stop_state();
        return false;
      }
      guideboard_reverse_state_.reset();
      publish_guideboard_reverse_state(std::chrono::steady_clock::now());
    }
    if (obstacle_hold_active_ && !obstacle_resume) {
      if (reason) {
        *reason = "obstacle hold is active";
      }
      publish_stop_state();
      return false;
    }
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
    cancel_geometry_stall_resume();
    stop_reason_ = "running";
    invalid_since_.reset();
    last_frame_signature_change_time_ = now;
    const bool restore_obstacle_command =
      obstacle_resume && obstacle_resume_command_valid_;
    current_speed_mps_ = restore_obstacle_command ?
      std::clamp(obstacle_resume_speed_mps_, 0.0, params_.linear_speed_mps) : 0.0;
    current_steering_ = restore_obstacle_command ?
      std::clamp(obstacle_resume_steering_, -params_.max_steering, params_.max_steering) : 0.0;
    limited_control_error_ = compute_control_error();
    previous_control_error_ = limited_control_error_;
    branch_error_limiter_engaged_ = false;
    curve_entry_window_until_.reset();
    curve_entry_window_armed_ =
      std::abs(limited_control_error_) < params_.curve_entry_error_start &&
      std::abs(current_heading_error_) < 0.20;
    filtered_derivative_ = 0.0;
    previous_control_time_ = now;
    publish_motion_command(current_speed_mps_, current_steering_);
    publish_chassis_enable(true);
    RCLCPP_INFO(get_logger(), "Line following started");
    return true;
  }

  bool try_resume_line_loss_command_hold(std::string * reason)
  {
    if (!params_.enable_line_loss_command_hold ||
      !obstacle_resume_armed_ || !obstacle_resume_command_valid_)
    {
      if (reason) {
        *reason = "no resumable line-loss command is available";
      }
      return false;
    }
    if (emergency_stop_active_) {
      if (reason) {
        *reason = "emergency stop is active";
      }
      return false;
    }
    if (safety_locked_) {
      if (reason) {
        *reason = "controller safety lock is active";
      }
      return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const bool held_geometry_stalled =
      params_.geometry_stall_timeout > 0.0 &&
      obstacle_resume_speed_mps_ >= 0.30 &&
      has_frame_signature_ &&
      std::chrono::duration<double>(
        now - last_frame_signature_change_time_).count() > params_.geometry_stall_timeout;
    if (held_geometry_stalled) {
      if (reason) {
        *reason = "camera geometry content is stalled";
      }
      return false;
    }

    // This is the interrupted continuation of an already running
    // line-loss hold, not a new autonomous start.  Restore the exact command
    // saved before Human pause even if lane validity is still LOW_CONFIDENCE.
    auto_enabled_ = true;
    safety_locked_ = false;
    auto_start_pending_ = false;
    cancel_geometry_stall_resume();
    stop_reason_ = "running";
    invalid_since_ = now;
    current_speed_mps_ = std::clamp(
      obstacle_resume_speed_mps_, 0.0, params_.linear_speed_mps);
    current_steering_ = std::clamp(
      obstacle_resume_steering_, -params_.max_steering, params_.max_steering);
    limited_control_error_ = compute_control_error();
    previous_control_error_ = limited_control_error_;
    branch_error_limiter_engaged_ = false;
    curve_entry_window_until_.reset();
    curve_entry_window_armed_ = false;
    filtered_derivative_ = 0.0;
    previous_control_time_ = now;
    publish_motion_command(current_speed_mps_, current_steering_);
    publish_chassis_enable(true);
    RCLCPP_WARN(
      get_logger(),
      "Obstacle cleared during line loss; restored held command speed=%.3f steering=%.3f",
      current_speed_mps_, current_steering_);
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
    limited_control_error_ = compute_control_error();
    previous_control_error_ = limited_control_error_;
    branch_error_limiter_engaged_ = false;
    curve_entry_window_until_.reset();
    curve_entry_window_armed_ = false;
    previous_control_time_ = std::chrono::steady_clock::now();
    publish_stop_state();
  }

  void lock_and_stop(const std::string & reason)
  {
    // A safety lock or emergency stop always cancels automatic obstacle
    // resume.  Keep obstacle_hold_active_ unchanged so a Human that is still
    // present continues to block manual /start calls.
    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    if (reason != "geometry_stall") {
      cancel_geometry_stall_resume();
    }
    finish_turn_state_.cancel(reason);
    guideboard_reverse_state_.cancel(reason);
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
    // An explicit operator stop must cancel any pending automatic resume.
    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    cancel_geometry_stall_resume();
    finish_turn_state_.cancel("service_stop");
    guideboard_reverse_state_.cancel("service_stop");
    disable_control("service_stop", false);
    response->success = true;
    response->message = "line following stopped; chassis disabled";
    RCLCPP_WARN(get_logger(), "Line following stopped by service");
  }

  void obstacle_pause_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    // A Human hold always takes priority over an automatic stall recovery.
    cancel_geometry_stall_resume();
    if (finish_turn_state_.active()) {
      finish_turn_state_.cancel("obstacle_pause");
      obstacle_hold_active_ = true;
      obstacle_resume_armed_ = false;
      obstacle_resume_command_valid_ = false;
      disable_control("obstacle_pause", false);
      response->success = true;
      response->message = "finish turn cancelled by obstacle pause; controller remains stopped";
      RCLCPP_WARN(get_logger(), "Finish turn cancelled by obstacle pause");
      return;
    }
    if (guideboard_reverse_state_.active()) {
      guideboard_reverse_state_.cancel("obstacle_pause");
      obstacle_hold_active_ = true;
      obstacle_resume_armed_ = false;
      obstacle_resume_command_valid_ = false;
      disable_control("obstacle_pause", false);
      publish_guideboard_reverse_state(std::chrono::steady_clock::now());
      response->success = true;
      response->message =
        "guideboard reverse cancelled by obstacle pause; controller remains stopped";
      RCLCPP_WARN(get_logger(), "Guideboard reverse cancelled by obstacle pause");
      return;
    }
    if (obstacle_hold_active_) {
      publish_stop_state();
      response->success = true;
      response->message = obstacle_resume_armed_ ?
        "obstacle hold already active; resume remains armed" :
        "obstacle hold already active; controller remains stopped";
      return;
    }

    const bool was_running = auto_enabled_;
    obstacle_hold_active_ = true;
    obstacle_resume_armed_ = was_running;
    if (was_running) {
      obstacle_resume_speed_mps_ = current_speed_mps_;
      obstacle_resume_steering_ = current_steering_;
      obstacle_resume_command_valid_ = true;
      disable_control("obstacle_pause", false);
      response->message = "obstacle pause applied; resume armed";
    } else {
      // Preserve an existing safety lock and its reason.  In particular, an
      // obstacle detected while waiting to start must never convert that state
      // into an automatically resumable pause.
      obstacle_resume_command_valid_ = false;
      publish_stop_state();
      response->message = "obstacle pause applied; controller was not running";
    }
    response->success = true;
    RCLCPP_WARN(
      get_logger(), "Obstacle hold applied (resume_armed=%s)",
      obstacle_resume_armed_ ? "true" : "false");
  }

  void obstacle_resume_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!obstacle_hold_active_ && !obstacle_resume_armed_) {
      publish_stop_state();
      response->success = true;
      response->message = "no obstacle pause is active; controller remains stopped";
      return;
    }

    if (!obstacle_resume_armed_) {
      obstacle_hold_active_ = false;
      obstacle_resume_command_valid_ = false;
      publish_stop_state();
      response->success = true;
      response->message = "obstacle cleared; controller was not running before pause";
      RCLCPP_INFO(get_logger(), "Obstacle cleared without automatic resume");
      return;
    }

    std::string reason;
    bool resumed_from_line_loss_hold = false;
    if (!try_start(&reason, true)) {
      std::string held_command_reason;
      if (!try_resume_line_loss_command_hold(&held_command_reason)) {
        // Keep the hold and arm latched while a non-line-loss safety
        // condition is active.  The perception client may safely retry.
        obstacle_hold_active_ = true;
        response->success = false;
        response->message = "obstacle resume rejected: " + reason +
          "; held-command resume rejected: " + held_command_reason;
        return;
      }
      resumed_from_line_loss_hold = true;
    }

    obstacle_hold_active_ = false;
    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    response->success = true;
    response->message = resumed_from_line_loss_hold ?
      "obstacle cleared; held line-loss command restored" :
      "obstacle cleared; line following resumed";
    RCLCPP_INFO(get_logger(), "Obstacle cleared; line following resumed");
  }

  void finish_turn_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (guideboard_reverse_state_.phase() !=
      line_follower_control_cpp::GuideboardReversePhase::Idle)
    {
      publish_stop_state();
      response->success = false;
      response->message = "finish turn rejected: guideboard reverse is active or latched";
      return;
    }
    if (emergency_stop_active_) {
      publish_stop_state();
      response->success = false;
      response->message = "finish turn rejected: emergency stop is active";
      return;
    }
    if (obstacle_hold_active_) {
      publish_stop_state();
      response->success = false;
      response->message = "finish turn rejected: obstacle hold is active";
      return;
    }
    if (safety_locked_ && !finish_turn_state_.active()) {
      publish_stop_state();
      response->success = false;
      response->message = "finish turn rejected: controller safety lock is active";
      return;
    }

    const bool was_active = finish_turn_state_.active();
    std::string reason;
    response->success = finish_turn_state_.start(
      finish_turn_config(), steady_seconds(std::chrono::steady_clock::now()), &reason);
    response->message = reason;
    if (!response->success || was_active ||
      finish_turn_state_.phase() == line_follower_control_cpp::FinishTurnPhase::Complete)
    {
      return;
    }

    auto_enabled_ = false;
    auto_start_pending_ = false;
    safety_locked_ = false;
    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    cancel_geometry_stall_resume();
    invalid_since_.reset();
    current_speed_mps_ = params_.finish_turn_speed_mps;
    current_steering_ = params_.finish_turn_steering;
    stop_reason_ = "finish_turn_rotating";
    last_mode_ = "finish_turn_rotating";
    publish_chassis_enable(true);
    publish_motion_command(current_speed_mps_, current_steering_);
    RCLCPP_WARN(
      get_logger(),
      "Finish turn started: speed=%.3fm/s steering=%.3f target=%ld timeout=%.2fs",
      params_.finish_turn_speed_mps, params_.finish_turn_steering,
      static_cast<long>(params_.finish_turn_encoder_counts),
      params_.finish_turn_timeout_sec);
  }

  bool guideboard_reverse_encoder_publisher_ready(std::string * reason) const
  {
    if (!params_.guideboard_reverse_require_single_encoder_publisher) {
      return true;
    }
    const size_t publisher_count = count_publishers(finish_turn_encoder_topic_);
    if (publisher_count == 1) {
      return true;
    }
    if (reason) {
      *reason = "expected exactly one encoder publisher, found " +
        std::to_string(publisher_count);
    }
    return false;
  }

  void guideboard_reverse_service_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (emergency_stop_active_) {
      publish_stop_state();
      response->success = false;
      response->message = "guideboard reverse rejected: emergency stop is active";
      return;
    }
    if (obstacle_hold_active_) {
      publish_stop_state();
      response->success = false;
      response->message = "guideboard reverse rejected: obstacle hold is active";
      return;
    }
    if (finish_turn_state_.phase() != line_follower_control_cpp::FinishTurnPhase::Idle) {
      publish_stop_state();
      response->success = false;
      response->message = "guideboard reverse rejected: finish turn is active or latched";
      return;
    }
    if (safety_locked_ && !guideboard_reverse_state_.active()) {
      publish_stop_state();
      response->success = false;
      response->message = "guideboard reverse rejected: controller safety lock is active";
      return;
    }

    std::string reason;
    if (!guideboard_reverse_encoder_publisher_ready(&reason)) {
      guideboard_reverse_state_.latch_fault("encoder_publisher_count_invalid");
      disable_control("encoder_publisher_count_invalid", true);
      publish_guideboard_reverse_state(std::chrono::steady_clock::now());
      response->success = false;
      response->message = "guideboard reverse rejected: " + reason;
      return;
    }

    const bool was_active = guideboard_reverse_state_.active();
    response->success = guideboard_reverse_state_.start(
      guideboard_reverse_config(), steady_seconds(std::chrono::steady_clock::now()), &reason);
    response->message = reason;
    if (!response->success) {
      guideboard_reverse_state_.latch_fault(reason);
      disable_control("guideboard_reverse_start_rejected", true);
    }
    publish_guideboard_reverse_state(std::chrono::steady_clock::now());
    if (!response->success || was_active ||
      guideboard_reverse_state_.phase() ==
      line_follower_control_cpp::GuideboardReversePhase::Complete)
    {
      return;
    }

    auto_enabled_ = false;
    auto_start_pending_ = false;
    safety_locked_ = false;
    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    cancel_geometry_stall_resume();
    invalid_since_.reset();
    current_speed_mps_ = -params_.guideboard_reverse_speed_mps;
    current_steering_ = params_.guideboard_reverse_steering;
    stop_reason_ = "guideboard_reverse_reversing";
    last_mode_ = "guideboard_reverse_reversing";
    publish_chassis_enable(true);
    publish_motion_command(current_speed_mps_, current_steering_);
    RCLCPP_WARN(
      get_logger(),
      "GuideBoard reverse started: speed=-%.3fm/s steering=%.3f target=%ld timeout=%.2fs",
      params_.guideboard_reverse_speed_mps, params_.guideboard_reverse_steering,
      static_cast<long>(params_.guideboard_reverse_encoder_counts),
      params_.guideboard_reverse_timeout_sec);
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

    obstacle_resume_armed_ = false;
    obstacle_resume_command_valid_ = false;
    cancel_geometry_stall_resume();
    finish_turn_state_.cancel("service_stop");
    guideboard_reverse_state_.cancel("service_stop");
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

    if (emergency_stop_active_) {
      finish_turn_state_.cancel("emergency_stop");
      guideboard_reverse_state_.cancel("emergency_stop");
      lock_and_stop("emergency_stop");
      last_mode_ = "emergency_stop";
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }

    if (guideboard_reverse_state_.phase() !=
      line_follower_control_cpp::GuideboardReversePhase::Idle)
    {
      if (guideboard_reverse_state_.active()) {
        std::string publisher_reason;
        if (!guideboard_reverse_encoder_publisher_ready(&publisher_reason)) {
          guideboard_reverse_state_.cancel("encoder_publisher_count_invalid");
        }
      }
      const auto reverse_snapshot = guideboard_reverse_state_.update(
        guideboard_reverse_config(), steady_seconds(now));
      if (reverse_snapshot.phase ==
        line_follower_control_cpp::GuideboardReversePhase::Reversing)
      {
        current_speed_mps_ = -params_.guideboard_reverse_speed_mps;
        current_steering_ = params_.guideboard_reverse_steering;
        publish_chassis_enable(true);
        publish_motion_command(current_speed_mps_, current_steering_);
        last_mode_ = "guideboard_reverse_reversing";
        stop_reason_ = "guideboard_reverse_reversing";
      } else {
        current_speed_mps_ = 0.0;
        current_steering_ = 0.0;
        auto_enabled_ = false;
        safety_locked_ = reverse_snapshot.phase ==
          line_follower_control_cpp::GuideboardReversePhase::Fault;
        stop_reason_ = reverse_snapshot.reason;
        last_mode_ = reverse_snapshot.phase ==
          line_follower_control_cpp::GuideboardReversePhase::Complete ?
          "guideboard_reverse_complete" : "guideboard_reverse_fault";
        publish_stop_state();
      }
      publish_guideboard_reverse_state(now);
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }

    if (finish_turn_state_.active()) {
      const auto finish_snapshot = finish_turn_state_.update(
        finish_turn_config(), steady_seconds(now));
      if (finish_snapshot.phase == line_follower_control_cpp::FinishTurnPhase::Rotating) {
        current_speed_mps_ = params_.finish_turn_speed_mps;
        current_steering_ = params_.finish_turn_steering;
        publish_chassis_enable(true);
        publish_motion_command(current_speed_mps_, current_steering_);
        last_mode_ = "finish_turn_rotating";
        stop_reason_ = "finish_turn_rotating";
      } else {
        current_speed_mps_ = 0.0;
        current_steering_ = 0.0;
        auto_enabled_ = false;
        safety_locked_ =
          finish_snapshot.phase == line_follower_control_cpp::FinishTurnPhase::Fault;
        stop_reason_ = finish_snapshot.reason;
        last_mode_ = finish_snapshot.phase ==
          line_follower_control_cpp::FinishTurnPhase::Complete ?
          "finish_turn_complete" : "finish_turn_fault";
        publish_stop_state();
        if (finish_snapshot.phase == line_follower_control_cpp::FinishTurnPhase::Complete) {
          RCLCPP_WARN(
            get_logger(), "Finish turn complete: encoder delta=%ld/%ld; chassis disabled",
            static_cast<long>(finish_snapshot.encoder_delta),
            static_cast<long>(finish_snapshot.encoder_target));
        } else {
          RCLCPP_ERROR(
            get_logger(), "Finish turn fault: %s delta=%ld/%ld; chassis disabled",
            finish_snapshot.reason.c_str(), static_cast<long>(finish_snapshot.encoder_delta),
            static_cast<long>(finish_snapshot.encoder_target));
        }
      }
      publish_debug(now);
      previous_control_time_ = now;
      return;
    }

    if (auto_start_pending_) {
      std::string reason;
      if (!try_start(&reason)) {
        last_mode_ = "auto_start_wait";
      }
    }

    if (!auto_enabled_) {
      if (try_geometry_stall_auto_resume(now)) {
        last_mode_ = "geometry_recovered";
        publish_debug(now);
        previous_control_time_ = now;
        return;
      }
      publish_stop_state();
      if (last_mode_ != "auto_start_wait") {
        last_mode_ = safety_locked_ ? "locked_stop" : "disabled";
      }
      publish_debug(now);
      previous_control_time_ = now;
      limited_control_error_ = compute_control_error();
      previous_control_error_ = limited_control_error_;
      branch_error_limiter_engaged_ = false;
      filtered_derivative_ = 0.0;
      return;
    }

    if (geometry_content_stalled(now)) {
      geometry_stall_resume_armed_ = params_.enable_geometry_stall_auto_resume;
      geometry_stall_stop_time_ = now;
      geometry_stall_recovery_since_.reset();
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

      if (params_.enable_line_loss_command_hold) {
        // Keep the exact last command alive so the chassis watchdog continues
        // to receive a command while lane confidence temporarily drops.
        // Emergency stop and geometry-stall checks run before this branch and
        // still force a zero command when either safety condition is active.
        publish_chassis_enable(true);
        publish_motion_command(current_speed_mps_, current_steering_);
        last_mode_ = "line_loss_command_hold";
        publish_debug(now);
        previous_control_time_ = now;
        filtered_derivative_ = 0.0;
        return;
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
      // Do not keep a large stale steering command while perception is invalid.
      // Decelerate and unwind the servo together; if perception recovers, the
      // normal attack slew can safely build the required steering again.
      current_steering_ = apply_steering_slew(0.0, dt);
      publish_chassis_enable(true);
      publish_motion_command(current_speed_mps_, current_steering_);
      last_mode_ = "invalid_hold";
      publish_debug(now);
      previous_control_time_ = now;
      filtered_derivative_ = 0.0;
      return;
    }

    invalid_since_.reset();
    const double raw_control_error = compute_control_error();
    const double control_error = apply_branch_error_slew(raw_control_error, dt, now);
    const double raw_derivative = (control_error - previous_control_error_) / dt;
    current_error_rate_speed_risk_ = compute_error_rate_speed_risk(raw_derivative);
    const double curve_risk = compute_curve_risk();
    const double normal_target_speed = compute_effective_target_speed(curve_risk, now);
    const double target_speed = compute_high_error_rescue_target(
      normal_target_speed, raw_control_error);
    current_speed_mps_ = apply_speed_slew(target_speed, dt);
    update_curve_entry_window(control_error, now);
    const double base_dynamic_max_steering = compute_effective_max_steering(curve_risk, now);
    const double dynamic_max_steering = compute_curve_entry_max_steering(
      base_dynamic_max_steering, control_error, now);
    double derivative = raw_derivative;
    if (params_.derivative_limit > 0.0) {
      derivative = std::clamp(derivative, -params_.derivative_limit, params_.derivative_limit);
    }
    filtered_derivative_ = params_.derivative_filter_alpha * derivative +
      (1.0 - params_.derivative_filter_alpha) * filtered_derivative_;

    const double derivative_gain = compute_derivative_gain(control_error);
    double desired_steering = steering_sign_ *
      (params_.kp * control_error + derivative_gain * filtered_derivative_);
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

  void cancel_geometry_stall_resume()
  {
    geometry_stall_resume_armed_ = false;
    geometry_stall_recovery_since_.reset();
  }

  bool try_geometry_stall_auto_resume(
    const std::chrono::steady_clock::time_point & now)
  {
    if (!geometry_stall_resume_armed_) {
      return false;
    }
    if (!safety_locked_ || stop_reason_ != "geometry_stall") {
      cancel_geometry_stall_resume();
      return false;
    }
    if (obstacle_hold_active_ || emergency_stop_active_) {
      geometry_stall_recovery_since_.reset();
      return false;
    }

    // Require feedback generated after the stall, not merely still-fresh data
    // that was queued before the stop.  Continuous perception_ready() for the
    // recovery window then proves the producer is publishing again.
    const bool received_after_stall =
      has_offset_y07_ && has_lane_state_ && has_frame_signature_ &&
      last_offset_y07_time_ > geometry_stall_stop_time_ &&
      last_lane_state_time_ > geometry_stall_stop_time_ &&
      last_frame_signature_time_ > geometry_stall_stop_time_ &&
      last_frame_signature_change_time_ > geometry_stall_stop_time_;
    if (!received_after_stall || !perception_ready(now)) {
      geometry_stall_recovery_since_.reset();
      return false;
    }

    if (!geometry_stall_recovery_since_) {
      geometry_stall_recovery_since_ = now;
      RCLCPP_WARN(
        get_logger(),
        "Geometry feedback recovered; waiting %.2fs before automatic resume",
        params_.geometry_stall_recovery_time);
      return false;
    }
    const double healthy_age = std::chrono::duration<double>(
      now - *geometry_stall_recovery_since_).count();
    if (healthy_age < params_.geometry_stall_recovery_time) {
      return false;
    }

    std::string reason;
    if (!try_start(&reason)) {
      geometry_stall_recovery_since_.reset();
      return false;
    }
    RCLCPP_WARN(
      get_logger(), "Line following automatically resumed after geometry stall");
    return true;
  }

  bool geometry_content_stalled(const std::chrono::steady_clock::time_point & now) const
  {
    if (params_.geometry_stall_timeout <= 0.0 || current_speed_mps_ < 0.30 ||
      !has_frame_signature_)
    {
      return false;
    }
    return std::chrono::duration<double>(
      now - last_frame_signature_change_time_).count() > params_.geometry_stall_timeout;
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

  double compute_target_speed(double risk, double minimum_speed) const
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
    const double slowdown = std::pow(ratio, params_.speed_slowdown_exponent);
    const double speed_range = params_.linear_speed_mps - minimum_speed;
    return params_.linear_speed_mps - speed_range * slowdown;
  }

  double compute_target_speed(double risk) const
  {
    return compute_target_speed(risk, params_.min_linear_speed_mps);
  }

  double compute_effective_minimum_speed() const
  {
    if (params_.inner_side_min_linear_speed_mps > 0.0 &&
      current_offset_y09_ <= -params_.inner_side_offset_threshold)
    {
      return std::max(
        params_.min_linear_speed_mps, params_.inner_side_min_linear_speed_mps);
    }
    return params_.min_linear_speed_mps;
  }

  double compute_effective_target_speed(
    double risk, const std::chrono::steady_clock::time_point & now) const
  {
    const bool in_branch = branch_guard_active(now);
    const double minimum_speed = in_branch && params_.branch_min_speed_mps > 0.0 ?
      params_.branch_min_speed_mps : compute_effective_minimum_speed();
    double target_speed = compute_target_speed(risk, minimum_speed);
    if (in_branch && params_.branch_max_speed_mps > 0.0) {
      target_speed = std::min(target_speed, params_.branch_max_speed_mps);
    }
    return target_speed;
  }

  double compute_high_error_rescue_target(double normal_target, double raw_error) const
  {
    const double abs_error = std::abs(raw_error);
    if (params_.high_error_rescue_speed_mps <= 0.0 ||
      abs_error <= params_.high_error_rescue_start) {
      return normal_target;
    }

    const double rescue_target = std::min(normal_target, params_.high_error_rescue_speed_mps);
    const double rescue_ratio = std::clamp(
      (abs_error - params_.high_error_rescue_start) /
      (params_.curve_entry_error_override - params_.high_error_rescue_start),
      0.0, 1.0);
    return normal_target + rescue_ratio * (rescue_target - normal_target);
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

  bool curve_entry_window_active(
    const std::chrono::steady_clock::time_point & now) const
  {
    return params_.curve_entry_window_time > 0.0 &&
           curve_entry_window_until_.has_value() && now < *curve_entry_window_until_;
  }

  void update_curve_entry_window(
    double control_error, const std::chrono::steady_clock::time_point & now)
  {
    if (params_.curve_entry_window_time <= 0.0) {
      curve_entry_window_until_.reset();
      curve_entry_window_armed_ = false;
      return;
    }

    const double abs_error = std::abs(control_error);
    const double abs_heading = std::abs(current_heading_error_);
    if (!curve_entry_window_armed_) {
      // Rearm only after a genuine straight section.  This prevents an
      // inside-edge correction in the same bend from being mistaken for a
      // second corner entry and limited again.
      const double rearm_error = std::min(0.25, 0.75 * params_.curve_entry_error_start);
      if (!curve_entry_window_active(now) && abs_error < rearm_error && abs_heading < 0.20) {
        curve_entry_window_armed_ = true;
      }
      return;
    }

    if (abs_error < params_.curve_entry_error_start ||
      abs_error >= params_.curve_entry_error_override ||
      abs_heading >= params_.curve_entry_heading_confirm)
    {
      return;
    }
    // With a confirmed image-space bend, heading and steering/error have
    // opposite signs.  Near-zero heading is also an unconfirmed entry.
    if (abs_heading > 0.05 && control_error * current_heading_error_ >= 0.0) {
      return;
    }

    curve_entry_window_until_ = now + std::chrono::duration_cast<
      std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(params_.curve_entry_window_time));
    curve_entry_window_armed_ = false;
  }

  bool curve_entry_guard_active(
    double control_error, const std::chrono::steady_clock::time_point & now) const
  {
    if (params_.curve_entry_max_steering <= 0.0) {
      return false;
    }
    if (params_.curve_entry_window_time > 0.0) {
      // A severe error always regains full steering authority, even during
      // the short entry-shaping window.
      return curve_entry_window_active(now) &&
             std::abs(control_error) < params_.curve_entry_error_override;
    }
    const double abs_error = std::abs(control_error);
    if (abs_error < params_.curve_entry_error_start ||
      abs_error >= params_.curve_entry_error_override)
    {
      return false;
    }
    const double abs_heading = std::abs(current_heading_error_);
    if (abs_heading >= params_.curve_entry_heading_confirm) {
      return false;
    }
    // With a confirmed image-space bend, heading and steering/error have
    // opposite signs.  Near-zero heading is also unconfirmed and guarded.
    return abs_heading <= 0.05 || control_error * current_heading_error_ < 0.0;
  }

  double compute_curve_entry_max_steering(
    double base_limit, double control_error,
    const std::chrono::steady_clock::time_point & now) const
  {
    if (!curve_entry_guard_active(control_error, now)) {
      return base_limit;
    }
    if (params_.curve_entry_window_time > 0.0 && curve_entry_window_until_) {
      const double remaining = std::chrono::duration<double>(
        *curve_entry_window_until_ - now).count();
      const double progress = std::clamp(
        1.0 - remaining / params_.curve_entry_window_time, 0.0, 1.0);
      // Start with a small first steering command, then continuously restore
      // full authority as the chassis settles into the bend.  This avoids a
      // second steering step when a fixed entry cap expires.
      const double progressive_limit = params_.curve_entry_max_steering +
        progress * (base_limit - params_.curve_entry_max_steering);
      return std::min(base_limit, progressive_limit);
    }
    return std::min(base_limit, params_.curve_entry_max_steering);
  }

  bool branch_guard_active(const std::chrono::steady_clock::time_point & now) const
  {
    if (lane_road_state_ == "BRANCH") {
      return true;
    }
    if (!has_seen_branch_ || params_.branch_exit_hold_time <= 0.0) {
      return false;
    }
    return std::chrono::duration<double>(now - last_branch_seen_time_).count() <=
           params_.branch_exit_hold_time;
  }

  double branch_outer_bias_blend(const std::chrono::steady_clock::time_point & now) const
  {
    if (!params_.lock_branch_outer_bias || !has_seen_branch_ ||
      std::abs(latched_branch_outer_bias_) <= 1e-6)
    {
      return 0.0;
    }
    if (lane_road_state_ == "BRANCH") {
      return 1.0;
    }
    if (params_.branch_outer_bias_release_time <= 0.0) {
      return branch_guard_active(now) ? 1.0 : 0.0;
    }
    const double age = std::chrono::duration<double>(now - last_branch_seen_time_).count();
    return std::clamp(1.0 - age / params_.branch_outer_bias_release_time, 0.0, 1.0);
  }

  double compute_effective_max_steering(
    double risk, const std::chrono::steady_clock::time_point & now) const
  {
    double limit = compute_dynamic_max_steering(risk);
    if (branch_guard_active(now) && params_.branch_max_steering > 0.0) {
      limit = std::min(limit, params_.branch_max_steering);
    }
    return limit;
  }

  double apply_branch_error_slew(
    double target_error, double dt, const std::chrono::steady_clock::time_point & now)
  {
    double rate = params_.branch_error_slew_rate;
    if (branch_guard_active(now)) {
      branch_error_limiter_engaged_ = true;
    } else if (branch_error_limiter_engaged_) {
      rate = params_.branch_error_recovery_rate;
    } else {
      limited_control_error_ = target_error;
      return limited_control_error_;
    }

    if (rate <= 0.0) {
      limited_control_error_ = target_error;
      branch_error_limiter_engaged_ = false;
      return limited_control_error_;
    }

    // The branch limiter must never hide a genuine near-edge error from the
    // curve-entry rescue override.  Keep smoothing ordinary template jumps,
    // but immediately expose a severe growing (or sign-reversing) raw error so
    // full steering authority remains available.
    if (std::abs(target_error) >= params_.curve_entry_error_override &&
      (target_error * limited_control_error_ <= 0.0 ||
      std::abs(target_error) > std::abs(limited_control_error_)))
    {
      limited_control_error_ = target_error;
      return limited_control_error_;
    }

    const double max_delta = rate * dt;
    const double delta = target_error - limited_control_error_;
    if (std::abs(delta) <= max_delta) {
      limited_control_error_ = target_error;
      if (!branch_guard_active(now)) {
        branch_error_limiter_engaged_ = false;
      }
    } else {
      limited_control_error_ += std::copysign(max_delta, delta);
    }
    return limited_control_error_;
  }

  double compute_near_offset_blend() const
  {
    if (!params_.enable_adaptive_offset_weights) {
      return 0.0;
    }
    const double near_advantage =
      std::abs(current_offset_y09_) - std::abs(current_offset_y07_);
    return std::clamp(
      (near_advantage - params_.near_offset_advantage_start) /
      (params_.near_offset_advantage_full - params_.near_offset_advantage_start),
      0.0, 1.0);
  }

  double compute_weighted_offset() const
  {
    const double near_blend = compute_near_offset_blend();
    const double y07_weight = params_.offset_y07_weight +
      near_blend * (params_.near_offset_y07_weight - params_.offset_y07_weight);
    const double y08_weight = params_.offset_y08_weight +
      near_blend * (params_.near_offset_y08_weight - params_.offset_y08_weight);
    const double y09_weight = params_.offset_y09_weight +
      near_blend * (params_.near_offset_y09_weight - params_.offset_y09_weight);
    const double weight_sum = y07_weight + y08_weight + y09_weight;
    if (weight_sum <= 1e-9) {
      return 0.0;
    }
    return std::clamp(
      (y07_weight * current_offset_y07_ +
       y08_weight * current_offset_y08_ +
       y09_weight * current_offset_y09_) / weight_sum,
      -1.0, 1.0);
  }

  double compute_global_position_error() const
  {
    double position_error = compute_weighted_offset();
    if (has_global_offset_ &&
      offset_age_seconds(
        last_global_offset_time_, has_global_offset_, std::chrono::steady_clock::now()) <=
      params_.offset_timeout)
    {
      double correction = current_global_offset_ - position_error;
      if (std::abs(correction) <= params_.global_offset_deadband) {
        correction = 0.0;
      } else {
        correction = std::copysign(
          std::abs(correction) - params_.global_offset_deadband, correction);
      }
      position_error += params_.global_offset_blend * correction;
    }
    return std::clamp(position_error + params_.centerline_bias, -1.0, 1.0);
  }

  double compute_control_error() const
  {
    const double position_error = compute_global_position_error();
    const double lookahead_transition = params_.lookahead_transition_gain *
      (1.0 - compute_near_offset_blend()) *
      (current_offset_y07_ - current_offset_y09_);
    // Image y grows downward.  A centerline bending toward positive x therefore
    // has a negative dx/dy heading, so subtract heading to make both feedback
    // terms request the same steering direction through a bend.
    return std::clamp(
      position_error -
      params_.heading_feedback_gain * current_heading_error_ -
      compute_curve_outer_bias() + lookahead_transition, -1.0, 1.0);
  }

  double compute_derivative_gain(double control_error) const
  {
    // When the filtered derivative opposes the proportional term, the error
    // magnitude is already falling.  Extra gain here brakes steering before
    // the error crosses zero without amplifying the initial turn-in command.
    if (control_error * filtered_derivative_ < 0.0) {
      return params_.kd * params_.derivative_brake_gain;
    }
    return params_.kd;
  }

  double compute_curve_outer_bias() const
  {
    if (std::abs(params_.curve_outer_bias) <= 1e-9) {
      return 0.0;
    }
    const auto now = std::chrono::steady_clock::now();
    const double abs_heading = std::abs(current_heading_error_);
    double instant_bias = 0.0;
    double current_turn_sign = 0.0;
    if (abs_heading > params_.curve_outer_bias_start) {
      const double ratio = std::clamp(
        (abs_heading - params_.curve_outer_bias_start) /
        (params_.curve_outer_bias_full - params_.curve_outer_bias_start), 0.0, 1.0);
      // Heading and steering have opposite signs in a normal image-space bend.
      // This signed bias is subtracted from the control error: positive targets
      // the outside of the bend, while negative targets the inside.
      current_turn_sign = current_heading_error_ < 0.0 ? 1.0 : -1.0;
      instant_bias = params_.curve_outer_bias * ratio * current_turn_sign;
    }

    double normal_bias = instant_bias;
    if (has_strong_curve_memory_ && params_.curve_outer_bias_hold_time > 0.0 &&
      (current_turn_sign == 0.0 || current_turn_sign == last_strong_curve_turn_sign_))
    {
      const double memory_age = std::chrono::duration<double>(
        now - last_strong_curve_time_).count();
      if (memory_age < params_.curve_outer_bias_hold_time) {
        const double held_ratio = std::clamp(
          1.0 - memory_age / params_.curve_outer_bias_hold_time, 0.0, 1.0);
        const double held_bias =
          params_.curve_outer_bias * held_ratio * last_strong_curve_turn_sign_;
        if (std::abs(held_bias) > std::abs(normal_bias)) {
          normal_bias = held_bias;
        }
      }
    }

    const double branch_blend = branch_outer_bias_blend(now);
    return normal_bias + branch_blend * (latched_branch_outer_bias_ - normal_bias);
  }

  double compute_curve_risk() const
  {
    const double lateral_risk = std::abs(current_offset_y09_);
    const double heading_risk = std::abs(current_heading_error_);
    const double curvature_risk = params_.curvature_speed_weight * std::abs(current_curvature_);
    // The far sample moves before the near sample on corner entry.  Keep this
    // speed-only term separate from steering weights so it can trigger early
    // braking without making turn-in steering more aggressive.
    const double far_offset_risk =
      params_.far_offset_speed_weight * std::abs(current_offset_y07_);
    const double predictive_risk = compute_predictive_offset_risk();
    return std::clamp(
      std::max({lateral_risk, heading_risk, curvature_risk, far_offset_risk, predictive_risk,
        current_error_rate_speed_risk_}), 0.0, 1.0);
  }

  double compute_error_rate_speed_risk(double error_rate) const
  {
    if (params_.speed_error_rate_full <= params_.speed_error_rate_start) {
      return 0.0;
    }
    return std::clamp(
      (std::abs(error_rate) - params_.speed_error_rate_start) /
      (params_.speed_error_rate_full - params_.speed_error_rate_start), 0.0, 1.0);
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
    command.linear.x = speed_to_wheel_rps(speed_mps);
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

  void publish_guideboard_reverse_state(
    const std::chrono::steady_clock::time_point & now)
  {
    const auto reverse = guideboard_reverse_state_.snapshot(steady_seconds(now));
    std_msgs::msg::String message;
    std::ostringstream text;
    text << std::fixed << std::setprecision(3)
         << "{\"phase\":\"" <<
      line_follower_control_cpp::GuideboardReverseState::phase_name(reverse.phase) << "\""
         << ",\"encoder_start_count\":" << reverse.encoder_start_count
         << ",\"encoder_count\":" << reverse.encoder_count
         << ",\"encoder_direction\":" << reverse.encoder_direction
         << ",\"encoder_delta\":" << reverse.encoder_delta
         << ",\"encoder_target\":" << reverse.encoder_target
         << ",\"elapsed_sec\":" << reverse.elapsed_sec
         << ",\"encoder_age_sec\":" << reverse.encoder_age_sec
         << ",\"reason\":\"" << reverse.reason << "\"}";
    message.data = text.str();
    guideboard_reverse_state_publisher_->publish(message);
  }

  void publish_debug(const std::chrono::steady_clock::time_point & now)
  {
    std_msgs::msg::String message;
    std::ostringstream text;
    const auto finish_turn = finish_turn_state_.snapshot(steady_seconds(now));
    const auto guideboard_reverse =
      guideboard_reverse_state_.snapshot(steady_seconds(now));
    const double raw_control_error = compute_control_error();
    const double control_error = auto_enabled_ ? limited_control_error_ : raw_control_error;
    const double curve_outer_bias = compute_curve_outer_bias();
    const double curve_risk = compute_curve_risk();
    const double far_offset_speed_risk =
      params_.far_offset_speed_weight * std::abs(current_offset_y07_);
    const double curve_strength = compute_curve_strength();
    const double offset_relief = compute_offset_relief();
    const double allowed_offset = compute_allowed_offset();
    const double weighted_offset = compute_weighted_offset();
    const double global_position_error = compute_global_position_error();
    const double near_offset_blend = compute_near_offset_blend();
    const double offset_excess = std::max(0.0, std::abs(weighted_offset) - allowed_offset);
    const bool branch_guard = branch_guard_active(now);
    const double speed_minimum = branch_guard && params_.branch_min_speed_mps > 0.0 ?
      params_.branch_min_speed_mps : compute_effective_minimum_speed();
    const double dynamic_target_speed = compute_target_speed(curve_risk, speed_minimum);
    const double normal_target_speed = compute_effective_target_speed(curve_risk, now);
    const double target_speed = compute_high_error_rescue_target(
      normal_target_speed, raw_control_error);
    const bool branch_speed_limited = normal_target_speed + 1e-6 < dynamic_target_speed;
    const double base_dynamic_max = compute_effective_max_steering(curve_risk, now);
    const bool curve_entry_guard = curve_entry_guard_active(control_error, now);
    const double dynamic_max = compute_curve_entry_max_steering(
      base_dynamic_max, control_error, now);
    text << std::fixed << std::setprecision(3)
         << "mode=" << last_mode_
         << " enabled=" << (auto_enabled_ ? "True" : "False")
         << " locked=" << (safety_locked_ ? "True" : "False")
         << " obstacle_hold=" << (obstacle_hold_active_ ? "True" : "False")
         << " obstacle_resume_armed=" << (obstacle_resume_armed_ ? "True" : "False")
         << " obstacle_command_saved=" <<
      (obstacle_resume_command_valid_ ? "True" : "False")
         << " obstacle_saved_speed=" << obstacle_resume_speed_mps_
         << " obstacle_saved_steering=" << obstacle_resume_steering_
         << " valid=" << (is_valid_ ? "True" : "False")
         << " lane_valid=" << (lane_state_valid_ ? "True" : "False")
         << " confidence=" << lane_confidence_
         << " road_state=" << lane_road_state_
         << " branch_guard=" << (branch_guard ? "True" : "False")
         << " lane_age=" << geometry_age_seconds(last_lane_state_time_, has_lane_state_, now)
         << " offset_y07_age=" << geometry_age_seconds(last_offset_y07_time_, has_offset_y07_, now)
         << " offset_y08_age=" << geometry_age_seconds(last_offset_y08_time_, has_offset_y08_, now)
         << " offset_y09_age=" << geometry_age_seconds(last_offset_y09_time_, has_offset_y09_, now)
         << " heading_age=" << geometry_age_seconds(last_heading_error_time_, has_heading_error_, now)
         << " curvature_age=" << geometry_age_seconds(last_curvature_time_, has_curvature_, now)
         << " frame_signature=" << current_frame_signature_
         << " frame_signature_age=" << geometry_age_seconds(
      last_frame_signature_time_, has_frame_signature_, now)
         << " frame_content_age=" << geometry_age_seconds(
      last_frame_signature_change_time_, has_frame_signature_, now)
         << " geometry_stalled=" << (geometry_content_stalled(now) ? "True" : "False")
         << " geometry_resume_armed=" <<
      (geometry_stall_resume_armed_ ? "True" : "False")
         << " geometry_recovery_age=" <<
      (geometry_stall_recovery_since_ ?
      std::chrono::duration<double>(now - *geometry_stall_recovery_since_).count() : 0.0)
         << " perception_ready=" << (perception_ready(now) ? "True" : "False")
         << " line_loss_command_hold=" <<
      (params_.enable_line_loss_command_hold ? "True" : "False")
         << " emergency=" << (emergency_stop_active_ ? "True" : "False")
         << " offset_y07=" << current_offset_y07_
         << " offset_y08=" << current_offset_y08_
         << " offset_y09=" << current_offset_y09_
         << " weighted_offset=" << weighted_offset
         << " global_offset=" << current_global_offset_
         << " global_offset_age=" << offset_age_seconds(
      last_global_offset_time_, has_global_offset_, now)
         << " global_position_error=" << global_position_error
         << " near_offset_blend=" << near_offset_blend
         << " heading_error=" << current_heading_error_
         << " curvature=" << current_curvature_
         << " raw_control_error=" << raw_control_error
         << " control_error=" << control_error
         << " branch_error_limiter=" << (branch_error_limiter_engaged_ ? "True" : "False")
         << " curve_outer_bias=" << curve_outer_bias
         << " branch_outer_bias=" << latched_branch_outer_bias_
         << " branch_outer_bias_locked=" <<
      (branch_outer_bias_blend(now) > 1e-6 ? "True" : "False")
         << " curve_strength=" << curve_strength
         << " offset_relief=" << offset_relief
         << " allowed_offset=" << allowed_offset
         << " offset_excess=" << offset_excess
         << " curve_risk=" << curve_risk
         << " far_offset_speed_risk=" << far_offset_speed_risk
         << " error_rate_risk=" << current_error_rate_speed_risk_
         << " curve_entry_guard=" << (curve_entry_guard ? "True" : "False")
         << " curve_entry_armed=" << (curve_entry_window_armed_ ? "True" : "False")
         << " filtered_derivative=" << filtered_derivative_
         << " proportional_term=" << params_.kp * control_error
         << " derivative_braking=" <<
      (control_error * filtered_derivative_ < 0.0 ? "True" : "False")
         << " effective_kd=" << compute_derivative_gain(control_error)
         << " derivative_term=" << compute_derivative_gain(control_error) * filtered_derivative_
         << " base_max_steer=" << base_dynamic_max
         << " target_speed=" << target_speed
         << " high_error_rescue=" <<
      (target_speed + 1e-6 < normal_target_speed ? "True" : "False")
         << " branch_speed_limited=" << (branch_speed_limited ? "True" : "False")
         << " speed_mps=" << current_speed_mps_
         << " wheel_rps=" << speed_to_wheel_rps(current_speed_mps_)
         << " steering_cmd=" << current_steering_
         << " max_steer=" << dynamic_max
         << " finish_turn_state=" <<
      line_follower_control_cpp::FinishTurnState::phase_name(finish_turn.phase)
         << " finish_turn_start_count=" << finish_turn.encoder_start_count
         << " finish_turn_encoder_count=" << finish_turn.encoder_count
         << " finish_turn_encoder_delta=" << finish_turn.encoder_delta
         << " finish_turn_encoder_target=" << finish_turn.encoder_target
         << " finish_turn_elapsed=" << finish_turn.elapsed_sec
         << " finish_turn_encoder_age=" << finish_turn.encoder_age_sec
         << " finish_turn_reason=" << finish_turn.reason
         << " guideboard_reverse_state=" <<
      line_follower_control_cpp::GuideboardReverseState::phase_name(guideboard_reverse.phase)
         << " guideboard_reverse_start_count=" << guideboard_reverse.encoder_start_count
         << " guideboard_reverse_encoder_count=" << guideboard_reverse.encoder_count
         << " guideboard_reverse_direction=" << guideboard_reverse.encoder_direction
         << " guideboard_reverse_encoder_delta=" << guideboard_reverse.encoder_delta
         << " guideboard_reverse_encoder_target=" << guideboard_reverse.encoder_target
         << " guideboard_reverse_elapsed=" << guideboard_reverse.elapsed_sec
         << " guideboard_reverse_encoder_age=" << guideboard_reverse.encoder_age_sec
         << " guideboard_reverse_reason=" << guideboard_reverse.reason
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
  std::string global_offset_topic_{"/segmentation/global_offset"};
  std::string heading_error_topic_{"/segmentation/heading_error"};
  std::string curvature_topic_{"/segmentation/curvature"};
  std::string lane_state_topic_{"/perception/lane_state"};
  std::string cmd_vel_topic_{"/cmd_vel"};
  std::string finish_turn_encoder_topic_{"/chassis/encoder_count"};

  line_follower_control_cpp::FinishTurnState finish_turn_state_;
  line_follower_control_cpp::GuideboardReverseState guideboard_reverse_state_;

  bool auto_enabled_{false};
  bool safety_locked_{false};
  bool obstacle_hold_active_{false};
  bool obstacle_resume_armed_{false};
  bool obstacle_resume_command_valid_{false};
  double obstacle_resume_speed_mps_{0.0};
  double obstacle_resume_steering_{0.0};
  bool geometry_stall_resume_armed_{false};
  bool has_frame_signature_{false};
  bool has_offset_y07_{false};
  bool has_offset_y08_{false};
  bool has_offset_y09_{false};
  bool has_global_offset_{false};
  bool has_valid_message_{false};
  bool has_heading_error_{false};
  bool has_curvature_{false};
  bool has_lane_state_{false};
  bool is_valid_{false};
  bool lane_state_valid_{false};
  bool emergency_stop_active_{false};
  double current_offset_y07_{0.0};
  double current_offset_y08_{0.0};
  double current_offset_y09_{0.0};
  double current_global_offset_{0.0};
  double current_heading_error_{0.0};
  double current_curvature_{0.0};
  double current_error_rate_speed_risk_{0.0};
  double lane_confidence_{0.0};
  double previous_control_error_{0.0};
  double limited_control_error_{0.0};
  double filtered_derivative_{0.0};
  double current_speed_mps_{0.0};
  double current_steering_{0.0};
  uint64_t current_frame_signature_{0};
  std::optional<std::chrono::steady_clock::time_point> invalid_since_;
  std::optional<std::chrono::steady_clock::time_point> geometry_stall_recovery_since_;
  std::optional<std::chrono::steady_clock::time_point> curve_entry_window_until_;
  std::chrono::steady_clock::time_point last_valid_time_;
  std::chrono::steady_clock::time_point last_offset_y07_time_;
  std::chrono::steady_clock::time_point last_offset_y08_time_;
  std::chrono::steady_clock::time_point last_offset_y09_time_;
  std::chrono::steady_clock::time_point last_global_offset_time_;
  std::chrono::steady_clock::time_point last_heading_error_time_;
  std::chrono::steady_clock::time_point last_curvature_time_;
  std::chrono::steady_clock::time_point last_lane_state_time_;
  std::chrono::steady_clock::time_point last_branch_seen_time_;
  std::chrono::steady_clock::time_point last_frame_signature_time_;
  std::chrono::steady_clock::time_point last_frame_signature_change_time_;
  std::chrono::steady_clock::time_point geometry_stall_stop_time_;
  std::string lane_road_state_{"UNKNOWN"};
  bool has_seen_branch_{false};
  bool branch_error_limiter_engaged_{false};
  bool curve_entry_window_armed_{false};
  bool has_strong_curve_memory_{false};
  double last_strong_curve_turn_sign_{0.0};
  double latched_branch_outer_bias_{0.0};
  std::chrono::steady_clock::time_point last_strong_curve_time_;
  std::chrono::steady_clock::time_point previous_control_time_;
  std::string last_mode_{"disabled"};
  std::string stop_reason_{"startup_disabled"};

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y07_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y08_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_y09_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr global_offset_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr heading_error_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr curvature_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr valid_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr lane_state_subscription_;
  rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr frame_signature_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr finish_turn_encoder_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr chassis_enable_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guideboard_reverse_state_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr obstacle_pause_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr obstacle_resume_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr finish_turn_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr guideboard_reverse_service_;
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
    RCLCPP_INFO(
      node->get_logger(),
      "Shutting down - publishing zero /cmd_vel and disabling chassis...");
    node->publish_stop_commands(5);
  }

  node.reset();
  rclcpp::shutdown();
  return 0;
}
