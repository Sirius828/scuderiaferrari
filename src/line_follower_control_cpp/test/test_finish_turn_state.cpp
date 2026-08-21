#include <gtest/gtest.h>

#include <string>

#include "line_follower_control_cpp/finish_turn_state.hpp"

namespace line_follower_control_cpp
{
namespace
{

FinishTurnConfig config()
{
  FinishTurnConfig value;
  value.target_encoder_counts = 20000;
  value.timeout_sec = 8.0;
  value.encoder_max_age_sec = 0.30;
  return value;
}

TEST(FinishTurnStateTest, RequiresFreshEncoderBeforeStart)
{
  FinishTurnState state;
  std::string reason;
  EXPECT_FALSE(state.start(config(), 1.0, &reason));
  EXPECT_EQ(reason, "encoder count has not been received");

  state.set_encoder_count(100, 1.0);
  EXPECT_FALSE(state.start(config(), 1.31, &reason));
  EXPECT_EQ(reason, "encoder count is stale");
}

TEST(FinishTurnStateTest, UsesEncoderDeltaFromArbitraryBaseline)
{
  FinishTurnState state;
  state.set_encoder_count(50000, 1.0);
  std::string reason;
  ASSERT_TRUE(state.start(config(), 1.0, &reason));
  EXPECT_EQ(state.phase(), FinishTurnPhase::Rotating);

  state.set_encoder_count(69999, 2.0);
  auto running = state.update(config(), 2.0);
  EXPECT_EQ(running.phase, FinishTurnPhase::Rotating);
  EXPECT_EQ(running.encoder_delta, 19999);

  state.set_encoder_count(70000, 2.1);
  auto complete = state.update(config(), 2.1);
  EXPECT_EQ(complete.phase, FinishTurnPhase::Complete);
  EXPECT_EQ(complete.encoder_delta, 20000);
  EXPECT_EQ(complete.reason, "encoder_target_reached");
}

TEST(FinishTurnStateTest, StaleEncoderFaultsAndStops)
{
  FinishTurnState state;
  state.set_encoder_count(1000, 1.0);
  ASSERT_TRUE(state.start(config(), 1.0, nullptr));

  auto snapshot = state.update(config(), 1.31);
  EXPECT_EQ(snapshot.phase, FinishTurnPhase::Fault);
  EXPECT_EQ(snapshot.reason, "encoder_stale");
}

TEST(FinishTurnStateTest, EncoderDecreaseFaultsImmediately)
{
  FinishTurnState state;
  state.set_encoder_count(1000, 1.0);
  ASSERT_TRUE(state.start(config(), 1.0, nullptr));
  state.set_encoder_count(1100, 1.1);
  ASSERT_EQ(state.update(config(), 1.1).phase, FinishTurnPhase::Rotating);

  state.set_encoder_count(1099, 1.2);
  auto snapshot = state.update(config(), 1.2);
  EXPECT_EQ(snapshot.phase, FinishTurnPhase::Fault);
  EXPECT_EQ(snapshot.reason, "encoder_count_decreased");
}

TEST(FinishTurnStateTest, OverallTimeoutFaultsEvenWithFreshEncoder)
{
  FinishTurnState state;
  state.set_encoder_count(1000, 1.0);
  ASSERT_TRUE(state.start(config(), 1.0, nullptr));
  state.set_encoder_count(1500, 9.0);

  auto snapshot = state.update(config(), 9.0);
  EXPECT_EQ(snapshot.phase, FinishTurnPhase::Fault);
  EXPECT_EQ(snapshot.reason, "finish_turn_timeout");
}

TEST(FinishTurnStateTest, CancelLatchesUntilExplicitReset)
{
  FinishTurnState state;
  state.set_encoder_count(1000, 1.0);
  ASSERT_TRUE(state.start(config(), 1.0, nullptr));
  state.cancel("service_stop");
  EXPECT_EQ(state.phase(), FinishTurnPhase::Fault);

  std::string reason;
  EXPECT_FALSE(state.start(config(), 1.1, &reason));
  state.reset();
  state.set_encoder_count(1000, 1.1);
  EXPECT_TRUE(state.start(config(), 1.1, &reason));
}

}  // namespace
}  // namespace line_follower_control_cpp
