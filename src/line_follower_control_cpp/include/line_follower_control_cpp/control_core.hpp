#pragma once

#include <optional>
#include <string>

namespace line_follower_control_cpp
{

enum class CurveState
{
  Straight,
  Curve,
};

const char * curve_state_name(CurveState state);

struct ControlConfig
{
  // The only track-behaviour tuning parameters.
  double kp{0.80};
  double kd{0.05};
  double curve_feedforward_gain{0.25};
  double straight_speed_mps{0.60};
  double curve_speed_mps{0.60};
  double curve_enter_threshold{0.20};
  double curve_exit_threshold{0.10};

  // Chassis calibration/safety limits.  These are not track-tuning knobs.
  double max_steering{0.80};
  double steering_sign{1.0};
};

struct LaneSample
{
  bool valid{true};
  double offset_y07{0.0};
  double offset_y09{0.0};
  double heading_error{0.0};
  double timestamp_seconds{0.0};
};

struct ControlSnapshot
{
  bool has_sample{false};
  double near_error{0.0};
  double error_rate{0.0};
  double filtered_derivative{0.0};
  double curve_signal{0.0};
  double curve_score{0.0};
  CurveState curve_state{CurveState::Straight};
  double p_term{0.0};
  double d_term{0.0};
  double ff_term{0.0};
  double unclamped_steer{0.0};
  double target_steering{0.0};
  double steering_cmd{0.0};
  double target_speed_mps{0.0};
  double speed_mps{0.0};
  bool saturated{false};
};

class ControlCore
{
public:
  explicit ControlCore(const ControlConfig & config = ControlConfig{});

  static bool validate_config(const ControlConfig & config, std::string * reason = nullptr);
  bool set_config(const ControlConfig & config, std::string * reason = nullptr);

  // Updates feedback and curve state exactly once per new perception frame.
  // Returns false for a non-finite or non-monotonic sample.
  bool update_lane_sample(const LaneSample & sample);

  // Advances output slew limiters at the controller timer rate.  Overrides are
  // used only by safety states such as invalid-hold.
  const ControlSnapshot & advance(
    double dt_seconds,
    std::optional<double> speed_override_mps = std::nullopt,
    std::optional<double> steering_override = std::nullopt);

  void reset();
  const ControlSnapshot & snapshot() const {return snapshot_;}
  const ControlConfig & config() const {return config_;}

private:
  void invalidate_sample();
  void update_curve_state(double timestamp_seconds);
  static double slew(double current, double target, double rate, double dt_seconds);

  ControlConfig config_{};
  ControlSnapshot snapshot_{};
  bool has_previous_sample_{false};
  double previous_near_error_{0.0};
  double previous_sample_time_seconds_{0.0};
  std::optional<double> curve_entry_since_seconds_;
  std::optional<double> curve_exit_since_seconds_;
};

}  // namespace line_follower_control_cpp
