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
  Settling,
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
  double settle_min_sec{0.25};
  int64_t settle_stable_samples{3};
};

struct GuideboardReverseSnapshot
{
  GuideboardReversePhase phase{GuideboardReversePhase::Idle};
  int64_t encoder_start_count{0};
  int64_t encoder_count{0};
  int encoder_direction{0};
  int64_t encoder_delta{0};
  int64_t encoder_target{0};
  double settle_elapsed_sec{0.0};
  int64_t settle_stable_samples{0};
  int64_t settle_stable_samples_target{0};
  double elapsed_sec{0.0};
  double encoder_age_sec{-1.0};
  std::string reason{"idle"};
};

class GuideboardReverseState
{
public:
  void set_encoder_count(int64_t count, double now_sec)
  {
    if (active() && has_encoder_count_) {
      const int64_t step = count - encoder_count_;
      if (std::llabs(step) > config_.encoder_max_step_counts) {
        fault("encoder_step_too_large");
      } else if (phase_ == GuideboardReversePhase::Settling) {
        if (std::llabs(step) <= config_.encoder_jitter_counts) {
          settle_stable_samples_ = std::min(
            settle_stable_samples_ + 1, config_.settle_stable_samples);
        } else {
          settle_stable_samples_ = 0;
        }
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
    if (active()) {
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
      config.encoder_max_step_counts <= config.encoder_jitter_counts ||
      !std::isfinite(config.settle_min_sec) || config.settle_min_sec < 0.0 ||
      config.settle_stable_samples <= 0)
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
    phase_ = GuideboardReversePhase::Settling;
    encoder_start_count_ = encoder_count_;
    encoder_direction_ = 0;
    encoder_delta_ = 0;
    encoder_target_ = config.target_encoder_counts;
    settle_start_time_sec_ = now_sec;
    reverse_start_time_sec_ = 0.0;
    settle_elapsed_sec_ = 0.0;
    settle_stable_samples_ = 0;
    elapsed_sec_ = 0.0;
    reason_ = "settling";
    set_reason(reason, "guideboard reverse settling");
    return true;
  }

  GuideboardReverseSnapshot update(const GuideboardReverseConfig & config, double now_sec)
  {
    if (!active()) {
      return snapshot(now_sec);
    }

    if (!has_encoder_count_ ||
      std::max(0.0, now_sec - encoder_time_sec_) > config.encoder_max_age_sec)
    {
      fault("encoder_stale");
      return snapshot(now_sec);
    }
    if (phase_ == GuideboardReversePhase::Settling) {
      settle_elapsed_sec_ = std::max(0.0, now_sec - settle_start_time_sec_);
      if (settle_elapsed_sec_ >= config.settle_min_sec &&
        settle_stable_samples_ >= config.settle_stable_samples)
      {
        phase_ = GuideboardReversePhase::Reversing;
        encoder_start_count_ = encoder_count_;
        encoder_direction_ = 0;
        encoder_delta_ = 0;
        reverse_start_time_sec_ = now_sec;
        elapsed_sec_ = 0.0;
        reason_ = "reversing";
      }
      return snapshot(now_sec);
    }

    elapsed_sec_ = std::max(0.0, now_sec - reverse_start_time_sec_);
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
    if (active()) {
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
    settle_elapsed_sec_ = 0.0;
    settle_stable_samples_ = 0;
    elapsed_sec_ = 0.0;
    reason_ = "idle";
  }

  bool active() const
  {
    return phase_ == GuideboardReversePhase::Settling ||
           phase_ == GuideboardReversePhase::Reversing;
  }
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
    value.settle_elapsed_sec = settle_elapsed_sec_;
    value.settle_stable_samples = settle_stable_samples_;
    value.settle_stable_samples_target = config_.settle_stable_samples;
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
      case GuideboardReversePhase::Settling:
        return "SETTLING";
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
  double settle_start_time_sec_{0.0};
  double reverse_start_time_sec_{0.0};
  double settle_elapsed_sec_{0.0};
  int64_t settle_stable_samples_{0};
  double elapsed_sec_{0.0};
  std::string reason_{"idle"};
};

}  // namespace line_follower_control_cpp
