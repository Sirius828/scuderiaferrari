#include <gtest/gtest.h>

#include <stdexcept>

#include "line_follower_control_cpp/control_core.hpp"

namespace line_follower_control_cpp
{
namespace
{

LaneSample sample(double time, double y07, double y09, double heading)
{
  LaneSample value;
  value.timestamp_seconds = time;
  value.offset_y07 = y07;
  value.offset_y09 = y09;
  value.heading_error = heading;
  return value;
}

TEST(ControlCoreTest, PureStraightUsesNearErrorForPd)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, 0.2, 0.2, 0.0)));
  EXPECT_NEAR(core.snapshot().p_term, 0.16, 1e-9);
  EXPECT_DOUBLE_EQ(core.snapshot().ff_term, 0.0);
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Straight);
}

TEST(ControlCoreTest, CurveFeedforwardRemainsWithZeroNearError)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, 0.20, 0.0, 0.40)));
  EXPECT_NEAR(core.snapshot().curve_signal, 0.60, 1e-9);
  EXPECT_NEAR(core.snapshot().ff_term, 0.15, 1e-9);
  EXPECT_DOUBLE_EQ(core.snapshot().p_term, 0.0);
}

TEST(ControlCoreTest, HeadingCancelsStraightLinePreviewSlope)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, -0.10, 0.10, 0.20)));
  EXPECT_NEAR(core.snapshot().curve_signal, 0.0, 1e-9);
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Straight);
}

TEST(ControlCoreTest, DerivativeOnlyChangesForNewSample)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(core.update_lane_sample(sample(1.02, 0.1, 0.1, 0.0)));
  const double derivative = core.snapshot().filtered_derivative;
  EXPECT_GT(derivative, 0.0);
  EXPECT_FALSE(core.update_lane_sample(sample(1.02, 0.3, 0.3, 0.0)));
  core.advance(0.02);
  EXPECT_DOUBLE_EQ(core.snapshot().filtered_derivative, derivative);
}

TEST(ControlCoreTest, CurveStateUsesEntryAndExitHysteresis)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.00, 0.3, 0.0, 0.0)));
  ASSERT_TRUE(core.update_lane_sample(sample(1.03, 0.3, 0.0, 0.0)));
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Straight);
  ASSERT_TRUE(core.update_lane_sample(sample(1.05, 0.3, 0.0, 0.0)));
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Curve);

  ASSERT_TRUE(core.update_lane_sample(sample(1.08, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(core.update_lane_sample(sample(1.15, 0.0, 0.0, 0.0)));
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Curve);
  ASSERT_TRUE(core.update_lane_sample(sample(1.21, 0.0, 0.0, 0.0)));
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Straight);
}

TEST(ControlCoreTest, SteeringIsLimitedAndSlewed)
{
  ControlConfig config;
  config.kp = 3.0;
  config.kd = 0.0;
  config.curve_feedforward_gain = 0.0;
  config.max_steering = 0.8;
  ControlCore core(config);
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, 1.0, 1.0, 0.0)));
  EXPECT_TRUE(core.snapshot().saturated);
  EXPECT_DOUBLE_EQ(core.snapshot().target_steering, 0.8);
  core.advance(0.02);
  EXPECT_NEAR(core.snapshot().steering_cmd, 0.08, 1e-9);

  core.advance(0.18);
  ASSERT_NEAR(core.snapshot().steering_cmd, 0.8, 1e-9);
  ASSERT_TRUE(core.update_lane_sample(sample(1.02, 0.0, 0.0, 0.0)));
  core.advance(0.02);
  EXPECT_NEAR(core.snapshot().steering_cmd, 0.75, 1e-9);
}

TEST(ControlCoreTest, RejectsInvalidConfiguration)
{
  ControlConfig config;
  config.curve_exit_threshold = config.curve_enter_threshold;
  std::string reason;
  EXPECT_FALSE(ControlCore::validate_config(config, &reason));
  EXPECT_FALSE(reason.empty());
  EXPECT_THROW(ControlCore invalid_core(config), std::invalid_argument);
}

TEST(ControlCoreTest, InvalidLaneTargetsAStopAndClearsFeedback)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.0, 0.2, 0.2, 0.0)));
  core.advance(0.2);
  ASSERT_GT(core.snapshot().speed_mps, 0.0);

  LaneSample invalid;
  invalid.valid = false;
  ASSERT_TRUE(core.update_lane_sample(invalid));
  EXPECT_FALSE(core.snapshot().has_sample);
  EXPECT_DOUBLE_EQ(core.snapshot().target_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(core.snapshot().target_steering, 0.0);
  core.advance(0.2);
  EXPECT_DOUBLE_EQ(core.snapshot().speed_mps, 0.0);
}

TEST(ControlCoreTest, InvalidFrameDoesNotCauseAFalseCurveExit)
{
  ControlCore core;
  ASSERT_TRUE(core.update_lane_sample(sample(1.00, 0.3, 0.0, 0.0)));
  ASSERT_TRUE(core.update_lane_sample(sample(1.05, 0.3, 0.0, 0.0)));
  ASSERT_EQ(core.snapshot().curve_state, CurveState::Curve);

  LaneSample invalid;
  invalid.valid = false;
  ASSERT_TRUE(core.update_lane_sample(invalid));
  EXPECT_EQ(core.snapshot().curve_state, CurveState::Curve);
}

}  // namespace
}  // namespace line_follower_control_cpp
