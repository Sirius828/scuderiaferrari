#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace line_follower_control_cpp
{

enum class GuideboardReversePhase
{
  Idle,
  Reversing,
  Complete,
  Fault,
};

struct GuideboardReverseConfig
{
  int64_t target_encoder_counts{1250};
  double timeout_sec{3.0};
  double encoder_max_age_sec{0.30};
  int64_t encoder_jitter_counts{10};
  int64_t encoder_max_step_counts{500};
};

struct GuideboardReverseSnapshot
{
  GuideboardReversePhase phase{GuideboardReversePhase::Idle};
  int64_t encoder_start_count{0};
  int64_t encoder_count{0};
  int encoder_direction{0};
  int64_t encoder_delta{0};
  int64_t encoder_target{0};
  double elapsed_sec{0.0};
  double encoder_age_sec{-1.0};
  std::string reason{"idle"};
};

class GuideboardReverseState
{
public:
  void set_encoder_count(int64_t count, double now_sec)
  {
    if (phase_ == GuideboardReversePhase::Reversing && has_encoder_count_) {
      const int64_t step = count - encoder_count_;
      if (std::llabs(step) > config_.encoder_max_step_counts) {
        fault("encoder_step_too_large");
      } else if (phase_ == GuideboardReversePhase::Reversing) {
        update_progress(count);
      }
    }
    encoder_count_ = count;
    encoder_time_sec_ = now_sec;
    has_encoder_count_ = true;
  }

  bool start(const GuideboardReverseConfig & config, double now_sec, std::string * reason)
  {
    if (phase_ == GuideboardReversePhase::Reversing) {
      set_reason(reason, "guideboard reverse already active");
      return true;
    }
    if (phase_ == GuideboardReversePhase::Complete) {
      set_reason(reason, "guideboard reverse already complete");
      return true;
    }
    if (phase_ == GuideboardReversePhase::Fault) {
      set_reason(
        reason, "guideboard reverse fault is latched; call /line_follower/start to reset");
      return false;
    }
    if (config.target_encoder_counts <= 0 || config.timeout_sec <= 0.0 ||
      config.encoder_max_age_sec <= 0.0 || config.encoder_jitter_counts < 0 ||
      config.encoder_max_step_counts <= config.encoder_jitter_counts)
    {
      set_reason(reason, "invalid guideboard reverse configuration");
      return false;
    }
    if (!has_encoder_count_) {
      set_reason(reason, "encoder count has not been received");
      return false;
    }
    if (std::max(0.0, now_sec - encoder_time_sec_) > config.encoder_max_age_sec) {
      set_reason(reason, "encoder count is stale");
      return false;
    }

    config_ = config;
    phase_ = GuideboardReversePhase::Reversing;
    encoder_start_count_ = encoder_count_;
    encoder_direction_ = 0;
    encoder_delta_ = 0;
    encoder_target_ = config.target_encoder_counts;
    start_time_sec_ = now_sec;
    elapsed_sec_ = 0.0;
    reason_ = "reversing";
    set_reason(reason, "guideboard reverse accepted");
    return true;
  }

  GuideboardReverseSnapshot update(const GuideboardReverseConfig & config, double now_sec)
  {
    if (phase_ != GuideboardReversePhase::Reversing) {
      return snapshot(now_sec);
    }

    elapsed_sec_ = std::max(0.0, now_sec - start_time_sec_);
    if (!has_encoder_count_ ||
      std::max(0.0, now_sec - encoder_time_sec_) > config.encoder_max_age_sec)
    {
      fault("encoder_stale");
      return snapshot(now_sec);
    }
    if (encoder_delta_ >= encoder_target_) {
      phase_ = GuideboardReversePhase::Complete;
      reason_ = "encoder_target_reached";
      return snapshot(now_sec);
    }
    if (elapsed_sec_ >= config.timeout_sec) {
      fault("guideboard_reverse_timeout");
    }
    return snapshot(now_sec);
  }

  void cancel(const std::string & reason)
  {
    if (phase_ == GuideboardReversePhase::Reversing) {
      fault(reason);
    }
  }

  void latch_fault(const std::string & reason)
  {
    if (phase_ != GuideboardReversePhase::Complete) {
      fault(reason);
    }
  }

  void reset()
  {
    phase_ = GuideboardReversePhase::Idle;
    encoder_start_count_ = encoder_count_;
    encoder_direction_ = 0;
    encoder_delta_ = 0;
    encoder_target_ = 0;
    elapsed_sec_ = 0.0;
    reason_ = "idle";
  }

  bool active() const {return phase_ == GuideboardReversePhase::Reversing;}
  bool terminal() const
  {
    return phase_ == GuideboardReversePhase::Complete ||
           phase_ == GuideboardReversePhase::Fault;
  }
  GuideboardReversePhase phase() const {return phase_;}

  GuideboardReverseSnapshot snapshot(double now_sec) const
  {
    GuideboardReverseSnapshot value;
    value.phase = phase_;
    value.encoder_start_count = encoder_start_count_;
    value.encoder_count = encoder_count_;
    value.encoder_direction = encoder_direction_;
    value.encoder_delta = encoder_delta_;
    value.encoder_target = encoder_target_;
    value.elapsed_sec = elapsed_sec_;
    value.encoder_age_sec = has_encoder_count_ ?
      std::max(0.0, now_sec - encoder_time_sec_) : -1.0;
    value.reason = reason_;
    return value;
  }

  static const char * phase_name(GuideboardReversePhase phase)
  {
    switch (phase) {
      case GuideboardReversePhase::Idle:
        return "IDLE";
      case GuideboardReversePhase::Reversing:
        return "REVERSING";
      case GuideboardReversePhase::Complete:
        return "COMPLETE";
      case GuideboardReversePhase::Fault:
        return "FAULT";
    }
    return "UNKNOWN";
  }

private:
  static void set_reason(std::string * target, const std::string & value)
  {
    if (target) {
      *target = value;
    }
  }

  void update_progress(int64_t count)
  {
    const int64_t from_start = count - encoder_start_count_;
    if (encoder_direction_ == 0) {
      if (std::llabs(from_start) <= config_.encoder_jitter_counts) {
        return;
      }
      encoder_direction_ = from_start > 0 ? 1 : -1;
    }

    const int64_t directed_position = from_start * encoder_direction_;
    if (directed_position + config_.encoder_jitter_counts < encoder_delta_) {
      fault("encoder_direction_reversed");
      return;
    }
    encoder_delta_ = std::max(encoder_delta_, directed_position);
  }

  void fault(const std::string & reason)
  {
    phase_ = GuideboardReversePhase::Fault;
    reason_ = reason;
  }

  GuideboardReverseConfig config_{};
  GuideboardReversePhase phase_{GuideboardReversePhase::Idle};
  bool has_encoder_count_{false};
  int64_t encoder_start_count_{0};
  int64_t encoder_count_{0};
  int encoder_direction_{0};
  int64_t encoder_delta_{0};
  int64_t encoder_target_{0};
  double encoder_time_sec_{0.0};
  double start_time_sec_{0.0};
  double elapsed_sec_{0.0};
  std::string reason_{"idle"};
};

}  // namespace line_follower_control_cpp
