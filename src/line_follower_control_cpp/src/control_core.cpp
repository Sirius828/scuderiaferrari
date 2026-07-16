#include "line_follower_control_cpp/control_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace line_follower_control_cpp
{
namespace
{
constexpr double kDerivativeFilterAlpha = 0.30;
constexpr double kDerivativeLimit = 1.30;
constexpr double kCurveEnterHoldSeconds = 0.04;
constexpr double kCurveExitHoldSeconds = 0.12;
constexpr double kSteeringAttackRate = 4.0;
constexpr double kSteeringReturnRate = 2.5;
constexpr double kSpeedAccelRate = 0.85;
constexpr double kSpeedDecelRate = 5.0;

bool finite(double value)
{
  return std::isfinite(value);
}
}  // namespace

const char * curve_state_name(CurveState state)
{
  return state == CurveState::Curve ? "CURVE" : "STRAIGHT";
}

ControlCore::ControlCore(const ControlConfig & config)
{
  std::string reason;
  if (!validate_config(config, &reason)) {
    throw std::invalid_argument(reason);
  }
  config_ = config;
  reset();
}

bool ControlCore::validate_config(const ControlConfig & config, std::string * reason)
{
  const auto fail = [reason](const std::string & message) {
      if (reason != nullptr) {
        *reason = message;
      }
      return false;
    };

  if (!finite(config.kp) || config.kp < 0.0 || config.kp > 3.0) {
    return fail("Kp must be in [0, 3]");
  }
  if (!finite(config.kd) || config.kd < 0.0 || config.kd > 1.0) {
    return fail("Kd must be in [0, 1]");
  }
  if (!finite(config.curve_feedforward_gain) ||
    config.curve_feedforward_gain < 0.0 || config.curve_feedforward_gain > 1.0)
  {
    return fail("curve_feedforward_gain must be in [0, 1]");
  }
  if (!finite(config.straight_speed_mps) || config.straight_speed_mps < 0.0) {
    return fail("straight_speed must be finite and >= 0");
  }
  if (!finite(config.curve_speed_mps) || config.curve_speed_mps < 0.0 ||
    config.curve_speed_mps > config.straight_speed_mps)
  {
    return fail("curve_speed must be in [0, straight_speed]");
  }
  if (!finite(config.curve_enter_threshold) || config.curve_enter_threshold <= 0.0 ||
    config.curve_enter_threshold > 1.0)
  {
    return fail("curve_enter_threshold must be in (0, 1]");
  }
  if (!finite(config.curve_exit_threshold) || config.curve_exit_threshold < 0.0 ||
    config.curve_exit_threshold >= config.curve_enter_threshold)
  {
    return fail("curve_exit_threshold must be in [0, curve_enter_threshold)");
  }
  if (!finite(config.max_steering) || config.max_steering <= 0.0 ||
    config.max_steering > 1.0)
  {
    return fail("max_steering must be in (0, 1]");
  }
  if (!finite(config.steering_sign) ||
    (config.steering_sign != 1.0 && config.steering_sign != -1.0))
  {
    return fail("steering_sign must be 1.0 or -1.0");
  }
  return true;
}

bool ControlCore::set_config(const ControlConfig & config, std::string * reason)
{
  if (!validate_config(config, reason)) {
    return false;
  }
  config_ = config;
  snapshot_.target_speed_mps = snapshot_.curve_state == CurveState::Curve ?
    config_.curve_speed_mps : config_.straight_speed_mps;
  return true;
}

bool ControlCore::update_lane_sample(const LaneSample & sample)
{
  if (!sample.valid) {
    invalidate_sample();
    return true;
  }
  if (!finite(sample.offset_y07) || !finite(sample.offset_y09) ||
    !finite(sample.heading_error) || !finite(sample.timestamp_seconds))
  {
    return false;
  }
  if (has_previous_sample_ && sample.timestamp_seconds <= previous_sample_time_seconds_) {
    return false;
  }

  const double near_error = std::clamp(sample.offset_y09, -1.0, 1.0);
  double derivative = 0.0;
  if (has_previous_sample_) {
    const double dt = sample.timestamp_seconds - previous_sample_time_seconds_;
    if (dt >= 1e-4 && dt <= 0.2) {
      derivative = std::clamp(
        (near_error - previous_near_error_) / dt, -kDerivativeLimit, kDerivativeLimit);
      snapshot_.filtered_derivative =
        kDerivativeFilterAlpha * derivative +
        (1.0 - kDerivativeFilterAlpha) * snapshot_.filtered_derivative;
    } else {
      snapshot_.filtered_derivative = 0.0;
    }
  } else {
    snapshot_.filtered_derivative = 0.0;
  }

  snapshot_.has_sample = true;
  snapshot_.near_error = near_error;
  snapshot_.error_rate = derivative;
  snapshot_.curve_signal = std::clamp(
    (std::clamp(sample.offset_y07, -1.0, 1.0) - near_error) +
    std::clamp(sample.heading_error, -1.0, 1.0), -1.0, 1.0);
  snapshot_.curve_score = std::abs(snapshot_.curve_signal);
  update_curve_state(sample.timestamp_seconds);

  snapshot_.p_term = config_.kp * snapshot_.near_error;
  snapshot_.d_term = config_.kd * snapshot_.filtered_derivative;
  snapshot_.ff_term = config_.curve_feedforward_gain * snapshot_.curve_signal;
  snapshot_.unclamped_steer = config_.steering_sign *
    (snapshot_.p_term + snapshot_.d_term + snapshot_.ff_term);
  snapshot_.target_steering = std::clamp(
    snapshot_.unclamped_steer, -config_.max_steering, config_.max_steering);
  snapshot_.saturated =
    std::abs(snapshot_.unclamped_steer) > config_.max_steering + 1e-9;
  snapshot_.target_speed_mps = snapshot_.curve_state == CurveState::Curve ?
    config_.curve_speed_mps : config_.straight_speed_mps;

  has_previous_sample_ = true;
  previous_near_error_ = near_error;
  previous_sample_time_seconds_ = sample.timestamp_seconds;
  return true;
}

void ControlCore::invalidate_sample()
{
  snapshot_.has_sample = false;
  snapshot_.near_error = 0.0;
  snapshot_.error_rate = 0.0;
  snapshot_.filtered_derivative = 0.0;
  snapshot_.curve_signal = 0.0;
  snapshot_.curve_score = 0.0;
  snapshot_.p_term = 0.0;
  snapshot_.d_term = 0.0;
  snapshot_.ff_term = 0.0;
  snapshot_.unclamped_steer = 0.0;
  snapshot_.target_steering = 0.0;
  snapshot_.target_speed_mps = 0.0;
  snapshot_.saturated = false;
  has_previous_sample_ = false;
  curve_entry_since_seconds_.reset();
  curve_exit_since_seconds_.reset();
}

void ControlCore::update_curve_state(double timestamp_seconds)
{
  if (snapshot_.curve_state == CurveState::Straight) {
    curve_exit_since_seconds_.reset();
    if (snapshot_.curve_score >= config_.curve_enter_threshold) {
      if (!curve_entry_since_seconds_) {
        curve_entry_since_seconds_ = timestamp_seconds;
      } else if (timestamp_seconds - *curve_entry_since_seconds_ >= kCurveEnterHoldSeconds) {
        snapshot_.curve_state = CurveState::Curve;
        curve_entry_since_seconds_.reset();
      }
    } else {
      curve_entry_since_seconds_.reset();
    }
    return;
  }

  curve_entry_since_seconds_.reset();
  if (snapshot_.curve_score <= config_.curve_exit_threshold) {
    if (!curve_exit_since_seconds_) {
      curve_exit_since_seconds_ = timestamp_seconds;
    } else if (timestamp_seconds - *curve_exit_since_seconds_ >= kCurveExitHoldSeconds) {
      snapshot_.curve_state = CurveState::Straight;
      curve_exit_since_seconds_.reset();
    }
  } else {
    curve_exit_since_seconds_.reset();
  }
}

double ControlCore::slew(double current, double target, double rate, double dt_seconds)
{
  const double max_delta = std::max(0.0, rate) * std::max(0.0, dt_seconds);
  return current + std::clamp(target - current, -max_delta, max_delta);
}

const ControlSnapshot & ControlCore::advance(
  double dt_seconds, std::optional<double> speed_override_mps,
  std::optional<double> steering_override)
{
  const double dt = std::clamp(dt_seconds, 0.0, 0.2);
  const double speed_target = std::clamp(
    speed_override_mps.value_or(snapshot_.target_speed_mps), 0.0,
    config_.straight_speed_mps);
  const double steering_target = std::clamp(
    steering_override.value_or(snapshot_.target_steering),
    -config_.max_steering, config_.max_steering);

  const double speed_rate = speed_target >= snapshot_.speed_mps ?
    kSpeedAccelRate : kSpeedDecelRate;
  snapshot_.speed_mps = slew(snapshot_.speed_mps, speed_target, speed_rate, dt);

  const bool returning =
    (std::abs(snapshot_.steering_cmd) > 1e-9 &&
    snapshot_.steering_cmd * steering_target <= 0.0) ||
    std::abs(steering_target) < std::abs(snapshot_.steering_cmd);
  const double steering_rate = returning ? kSteeringReturnRate : kSteeringAttackRate;
  snapshot_.steering_cmd = slew(
    snapshot_.steering_cmd, steering_target, steering_rate, dt);
  snapshot_.steering_cmd = std::clamp(
    snapshot_.steering_cmd, -config_.max_steering, config_.max_steering);
  return snapshot_;
}

void ControlCore::reset()
{
  snapshot_ = ControlSnapshot{};
  snapshot_.target_speed_mps = config_.straight_speed_mps;
  has_previous_sample_ = false;
  previous_near_error_ = 0.0;
  previous_sample_time_seconds_ = 0.0;
  curve_entry_since_seconds_.reset();
  curve_exit_since_seconds_.reset();
}

}  // namespace line_follower_control_cpp
