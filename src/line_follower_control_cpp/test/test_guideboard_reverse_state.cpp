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
  return value;
}

TEST(GuideboardReverseState, CompletesWithIncreasingEncoder)
{
  GuideboardReverseState state;
  state.set_encoder_count(10000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  state.set_encoder_count(10400, 1.1);
  state.set_encoder_count(10800, 1.2);
  state.set_encoder_count(11250, 1.3);
  const auto result = state.update(config(), 1.3);
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
  state.set_encoder_count(9600, 1.1);
  state.set_encoder_count(9200, 1.2);
  state.set_encoder_count(8750, 1.3);
  const auto result = state.update(config(), 1.3);
  EXPECT_EQ(result.phase, GuideboardReversePhase::Complete);
  EXPECT_EQ(result.encoder_direction, -1);
  EXPECT_EQ(result.encoder_delta, 1250);
}

TEST(GuideboardReverseState, IgnoresJitterButRejectsDirectionReversal)
{
  GuideboardReverseState state;
  state.set_encoder_count(1000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  state.set_encoder_count(1008, 1.05);
  EXPECT_EQ(state.snapshot(1.05).encoder_direction, 0);
  state.set_encoder_count(1100, 1.1);
  state.set_encoder_count(1092, 1.15);
  EXPECT_EQ(state.snapshot(1.15).phase, GuideboardReversePhase::Reversing);
  state.set_encoder_count(1080, 1.2);
  EXPECT_EQ(state.snapshot(1.2).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(state.snapshot(1.2).reason, "encoder_direction_reversed");
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
  EXPECT_EQ(timeout.update(timeout_config, 4.0).phase, GuideboardReversePhase::Fault);
  EXPECT_EQ(timeout.snapshot(4.0).reason, "guideboard_reverse_timeout");
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
}  // namespace
