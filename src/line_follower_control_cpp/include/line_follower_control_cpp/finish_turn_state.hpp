#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace line_follower_control_cpp
{

enum class FinishTurnPhase
{
  Idle,
  Rotating,
  Complete,
  Fault,
};

struct FinishTurnConfig
{
  int64_t target_encoder_counts{20000};
  double timeout_sec{8.0};
  double encoder_max_age_sec{0.30};
};

struct FinishTurnSnapshot
{
  FinishTurnPhase phase{FinishTurnPhase::Idle};
  int64_t encoder_start_count{0};
  int64_t encoder_count{0};
  int64_t encoder_delta{0};
  int64_t encoder_target{0};
  double elapsed_sec{0.0};
  double encoder_age_sec{0.0};
  std::string reason{"idle"};
};

class FinishTurnState
{
public:
  void set_encoder_count(int64_t count, double now_sec)
  {
    if (phase_ == FinishTurnPhase::Rotating && has_encoder_count_ && count < encoder_count_) {
      encoder_decreased_ = true;
    }
    encoder_count_ = count;
    encoder_time_sec_ = now_sec;
    has_encoder_count_ = true;
  }

  bool start(const FinishTurnConfig & config, double now_sec, std::string * reason)
  {
    if (phase_ == FinishTurnPhase::Rotating) {
      set_reason(reason, "finish turn already active");
      return true;
    }
    if (phase_ == FinishTurnPhase::Complete) {
      set_reason(reason, "finish turn already complete");
      return true;
    }
    if (phase_ == FinishTurnPhase::Fault) {
      set_reason(reason, "finish turn fault is latched; call /line_follower/start to reset");
      return false;
    }
    if (config.target_encoder_counts <= 0 || config.timeout_sec <= 0.0 ||
      config.encoder_max_age_sec <= 0.0)
    {
      set_reason(reason, "invalid finish turn configuration");
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

    phase_ = FinishTurnPhase::Rotating;
    encoder_start_count_ = encoder_count_;
    encoder_delta_ = 0;
    encoder_target_ = config.target_encoder_counts;
    start_time_sec_ = now_sec;
    elapsed_sec_ = 0.0;
    encoder_decreased_ = false;
    reason_ = "rotating";
    set_reason(reason, "finish turn accepted");
    return true;
  }

  FinishTurnSnapshot update(const FinishTurnConfig & config, double now_sec)
  {
    if (phase_ != FinishTurnPhase::Rotating) {
      return snapshot(now_sec);
    }

    elapsed_sec_ = std::max(0.0, now_sec - start_time_sec_);
    if (!has_encoder_count_ ||
      std::max(0.0, now_sec - encoder_time_sec_) > config.encoder_max_age_sec)
    {
      fault("encoder_stale");
      return snapshot(now_sec);
    }
    if (encoder_decreased_ || encoder_count_ < encoder_start_count_) {
      fault("encoder_count_decreased");
      return snapshot(now_sec);
    }

    encoder_delta_ = encoder_count_ - encoder_start_count_;
    if (encoder_delta_ >= encoder_target_) {
      phase_ = FinishTurnPhase::Complete;
      reason_ = "encoder_target_reached";
      return snapshot(now_sec);
    }
    if (elapsed_sec_ >= config.timeout_sec) {
      fault("finish_turn_timeout");
    }
    return snapshot(now_sec);
  }

  void cancel(const std::string & reason)
  {
    if (phase_ == FinishTurnPhase::Rotating) {
      phase_ = FinishTurnPhase::Fault;
      reason_ = reason;
    }
  }

  void reset()
  {
    phase_ = FinishTurnPhase::Idle;
    encoder_start_count_ = encoder_count_;
    encoder_delta_ = 0;
    encoder_target_ = 0;
    elapsed_sec_ = 0.0;
    encoder_decreased_ = false;
    reason_ = "idle";
  }

  bool active() const {return phase_ == FinishTurnPhase::Rotating;}
  bool terminal() const
  {
    return phase_ == FinishTurnPhase::Complete || phase_ == FinishTurnPhase::Fault;
  }
  FinishTurnPhase phase() const {return phase_;}
  FinishTurnSnapshot snapshot(double now_sec) const
  {
    FinishTurnSnapshot value;
    value.phase = phase_;
    value.encoder_start_count = encoder_start_count_;
    value.encoder_count = encoder_count_;
    value.encoder_delta = encoder_delta_;
    value.encoder_target = encoder_target_;
    value.elapsed_sec = elapsed_sec_;
    value.encoder_age_sec = has_encoder_count_ ?
      std::max(0.0, now_sec - encoder_time_sec_) : -1.0;
    value.reason = reason_;
    return value;
  }

  static const char * phase_name(FinishTurnPhase phase)
  {
    switch (phase) {
      case FinishTurnPhase::Idle:
        return "IDLE";
      case FinishTurnPhase::Rotating:
        return "ROTATING";
      case FinishTurnPhase::Complete:
        return "COMPLETE";
      case FinishTurnPhase::Fault:
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

  void fault(const std::string & reason)
  {
    phase_ = FinishTurnPhase::Fault;
    reason_ = reason;
  }

  FinishTurnPhase phase_{FinishTurnPhase::Idle};
  bool has_encoder_count_{false};
  bool encoder_decreased_{false};
  int64_t encoder_start_count_{0};
  int64_t encoder_count_{0};
  int64_t encoder_delta_{0};
  int64_t encoder_target_{0};
  double encoder_time_sec_{0.0};
  double start_time_sec_{0.0};
  double elapsed_sec_{0.0};
  std::string reason_{"idle"};
};

}  // namespace line_follower_control_cpp
