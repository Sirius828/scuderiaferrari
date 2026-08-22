#include <gtest/gtest.h>

#include <string>

#include "line_follower_control_cpp/guideboard_reverse_state.hpp"

namespace
{
using line_follower_control_cpp::GuideboardReverseConfig;
using line_follower_control_cpp::GuideboardReversePhase;
using line_follower_control_cpp::GuideboardReverseState;

GuideboardReverseConfig config()
{
  GuideboardReverseConfig value;
  value.target_encoder_counts = 1250;
  value.timeout_sec = 3.0;
  value.encoder_max_age_sec = 0.30;
  value.encoder_jitter_counts = 10;
  value.encoder_max_step_counts = 500;
  value.settle_min_sec = 0.20;
  value.settle_stable_samples = 3;
  return value;
}

void finish_settling(
  GuideboardReverseState & state, const GuideboardReverseConfig & cfg,
  int64_t encoder_count, double start_sec)
{
  state.set_encoder_count(encoder_count, start_sec + 0.05);
  state.set_encoder_count(encoder_count, start_sec + 0.10);
  state.set_encoder_count(encoder_count, start_sec + 0.21);
  const auto settled = state.update(cfg, start_sec + 0.21);
  ASSERT_EQ(settled.phase, GuideboardReversePhase::Reversing);
  EXPECT_EQ(settled.encoder_start_count, encoder_count);
  EXPECT_DOUBLE_EQ(settled.elapsed_sec, 0.0);
}

TEST(GuideboardReverseState, CompletesWithIncreasingEncoder)
{
  GuideboardReverseState state;
  state.set_encoder_count(10000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  EXPECT_EQ(state.phase(), GuideboardReversePhase::Settling);
  finish_settling(state, config(), 10000, 1.0);
  state.set_encoder_count(10400, 1.3);
  state.set_encoder_count(10800, 1.4);
  state.set_encoder_count(11250, 1.5);
  const auto result = state.update(config(), 1.5);
  EXPECT_EQ(result.phase, GuideboardReversePhase::Complete);
  EXPECT_EQ(result.encoder_direction, 1);
  EXPECT_EQ(result.encoder_delta, 1250);
}

TEST(GuideboardReverseState, CompletesWithDecreasingEncoder)
{
  GuideboardReverseState state;
  state.set_encoder_count(10000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  finish_settling(state, config(), 10000, 1.0);
  state.set_encoder_count(9600, 1.3);
  state.set_encoder_count(9200, 1.4);
  state.set_encoder_count(8750, 1.5);
  const auto result = state.update(config(), 1.5);
  EXPECT_EQ(result.phase, GuideboardReversePhase::Complete);
  EXPECT_EQ(result.encoder_direction, -1);
  EXPECT_EQ(result.encoder_delta, 1250);
}

TEST(GuideboardReverseState, SettlesForwardResidualBeforeMeasuringReverse)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  state.set_encoder_count(1100, 1.05);
  state.set_encoder_count(1200, 1.10);
  state.set_encoder_count(1205, 1.15);
  state.set_encoder_count(1208, 1.20);
  state.set_encoder_count(1207, 1.25);
  const auto settled = state.update(config(), 1.25);
  ASSERT_EQ(settled.phase, GuideboardReversePhase::Reversing);
  EXPECT_EQ(settled.encoder_start_count, 1207);
  EXPECT_EQ(settled.encoder_direction, 0);

  state.set_encoder_count(1100, 1.30);
  state.set_encoder_count(900, 1.35);
  const auto reversing = state.snapshot(1.35);
  EXPECT_EQ(reversing.phase, GuideboardReversePhase::Reversing);
  EXPECT_EQ(reversing.encoder_direction, -1);
  EXPECT_EQ(reversing.encoder_delta, 307);
}

TEST(GuideboardReverseState, WaitsForMinimumTimeAndConsecutiveStableSamples)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  state.set_encoder_count(1000, 1.05);
  state.set_encoder_count(1000, 1.10);
  state.set_encoder_count(1000, 1.15);
  EXPECT_EQ(state.update(config(), 1.15).phase, GuideboardReversePhase::Settling);
  state.set_encoder_count(1020, 1.18);
  state.set_encoder_count(1020, 1.20);
  state.set_encoder_count(1020, 1.22);
  EXPECT_EQ(state.update(config(), 1.22).phase, GuideboardReversePhase::Settling);
  state.set_encoder_count(1020, 1.24);
  const auto settled = state.update(config(), 1.24);
  EXPECT_EQ(settled.phase, GuideboardReversePhase::Reversing);
  EXPECT_EQ(settled.encoder_start_count, 1020);
}

TEST(GuideboardReverseState, IgnoresJitterButRejectsDirectionReversalAfterSettling)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  finish_settling(state, config(), 1000, 1.0);
  state.set_encoder_count(1100, 1.3);
  state.set_encoder_count(1092, 1.35);
  EXPECT_EQ(state.snapshot(1.35).phase, GuideboardReversePhase::Reversing);
  state.set_encoder_count(1080, 1.4);
  EXPECT_EQ(state.snapshot(1.4).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(state.snapshot(1.4).reason, "encoder_direction_reversed");
}

TEST(GuideboardReverseState, RejectsLargeStep)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  state.set_encoder_count(1600, 1.1);
  EXPECT_EQ(state.snapshot(1.1).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(state.snapshot(1.1).reason, "encoder_step_too_large");
}

TEST(GuideboardReverseState, FaultsOnStaleEncoderAndTimeout)
{
  GuideboardReverseState stale;
  stale.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(stale.start(config(), 1.0, &reason));
  EXPECT_EQ(stale.update(config(), 1.31).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(stale.snapshot(1.31).reason, "encoder_stale");

  GuideboardReverseConfig timeout_config = config();
  timeout_config.encoder_max_age_sec = 10.0;
  GuideboardReverseState timeout;
  timeout.set_encoder_count(1000, 1.0);
  ASSERT_TRUE(timeout.start(timeout_config, 1.0, &reason));
  timeout.set_encoder_count(1100, 2.0);
  timeout.set_encoder_count(1200, 3.0);
  timeout.set_encoder_count(1300, 4.0);
  EXPECT_EQ(timeout.update(timeout_config, 4.0).phase, GuideboardReversePhase::Settling);
  timeout.set_encoder_count(1300, 4.1);
  timeout.set_encoder_count(1300, 4.2);
  timeout.set_encoder_count(1300, 4.3);
  ASSERT_EQ(timeout.update(timeout_config, 4.3).phase, GuideboardReversePhase::Reversing);
  EXPECT_EQ(timeout.update(timeout_config, 7.31).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(timeout.snapshot(7.31).reason, "guideboard_reverse_timeout");
}

TEST(GuideboardReverseState, RejectsMissingOrStaleInitialEncoder)
{
  GuideboardReverseState state;
  std::string reason;
  EXPECT_FALSE(state.start(config(), 1.0, &reason));
  EXPECT_EQ(reason, "encoder count has not been received");
  state.set_encoder_count(1000, 1.0);
  EXPECT_FALSE(state.start(config(), 1.31, &reason));
  EXPECT_EQ(reason, "encoder count is stale");
}

TEST(GuideboardReverseState, RejectsInvalidSettlingConfiguration)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  auto invalid = config();
  invalid.settle_stable_samples = 0;
  EXPECT_FALSE(state.start(invalid, 1.0, &reason));
  EXPECT_EQ(reason, "invalid guideboard reverse configuration");
}
}  // namespace
