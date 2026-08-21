#include <gtest/gtest.h>

#include "track_perception_cpp/lane_decision.hpp"

namespace track_perception_cpp
{
namespace
{

constexpr int kWidth = 160;
constexpr int kHeight = 120;

LaneDecisionConfig makeConfig(int required_occurrences = 2)
{
  LaneDecisionConfig config;
  config.enable_segment_branch_logic = false;
  config.enable_encoder_branch_hold = false;
  config.enable_guideboard_branch_selection = false;
  config.enable_human_obstacle_stop = false;
  config.enable_car_obstacle_avoidance = false;
  config.enable_finish_stop = true;
  config.finish_stop_min_confidence = 0.45f;
  config.finish_stop_arm_y_ratio = 0.70f;
  config.finish_stop_lost_frames = 3;
  config.finish_stop_required_occurrences = required_occurrences;
  config.fit_min_points = 3;
  return config;
}

cv::Mat roadMask()
{
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  mask(cv::Rect(40, 0, 80, kHeight)).setTo(1);
  return mask;
}

Detection stopDetection(float confidence, float bottom_y)
{
  Detection stop;
  stop.class_name = "Stop";
  stop.confidence = confidence;
  stop.bbox = cv::Rect2f(65.0f, bottom_y - 20.0f, 30.0f, 20.0f);
  stop.center = cv::Point2f(80.0f, bottom_y - 10.0f);
  return stop;
}

LaneState completeStopPass(LaneDecision * decision)
{
  (void)decision->decide(roadMask(), {stopDetection(0.95f, 50.0f)});
  (void)decision->decide(roadMask(), {stopDetection(0.95f, 100.0f)});
  LaneState state;
  for (int i = 0; i < 3; ++i) {
    state = decision->decide(roadMask(), {});
  }
  return state;
}

TEST(LaneDecisionFinishStopTest, LowConfidenceAndUnarmedStopDoNotCount)
{
  LaneDecision decision;
  decision.configure(makeConfig());

  (void)decision.decide(roadMask(), {stopDetection(0.20f, 100.0f)});
  EXPECT_EQ(decision.debugInfo().finish_stop_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().finish_stop_occurrence_count, 0);

  (void)decision.decide(roadMask(), {stopDetection(0.95f, 50.0f)});
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().finish_stop_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().finish_stop_occurrence_count, 0);
}

TEST(LaneDecisionFinishStopTest, TwoCompletePassesTriggerOnceOnSecondPass)
{
  LaneDecision decision;
  decision.configure(makeConfig(2));

  auto first = completeStopPass(&decision);
  EXPECT_NE(first.task_state, "FINISH_STOP");
  EXPECT_EQ(decision.debugInfo().finish_stop_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().finish_stop_occurrence_count, 1);
  EXPECT_EQ(decision.debugInfo().finish_stop_required_occurrences, 2);

  auto second = completeStopPass(&decision);
  EXPECT_EQ(second.task_state, "FINISH_STOP");
  EXPECT_TRUE(decision.debugInfo().finish_stop_active);
  EXPECT_EQ(decision.debugInfo().finish_stop_state, "FINISH_STOP");
  EXPECT_EQ(decision.debugInfo().finish_stop_occurrence_count, 2);

  for (int i = 0; i < 5; ++i) {
    (void)decision.decide(roadMask(), {});
  }
  EXPECT_EQ(decision.debugInfo().finish_stop_occurrence_count, 2);
}

TEST(LaneDecisionFinishStopTest, RequiredOccurrenceParameterSupportsOneAndThree)
{
  LaneDecision one;
  one.configure(makeConfig(1));
  EXPECT_EQ(completeStopPass(&one).task_state, "FINISH_STOP");

  LaneDecision three;
  three.configure(makeConfig(3));
  EXPECT_NE(completeStopPass(&three).task_state, "FINISH_STOP");
  EXPECT_NE(completeStopPass(&three).task_state, "FINISH_STOP");
  EXPECT_EQ(three.debugInfo().finish_stop_occurrence_count, 2);
  EXPECT_EQ(completeStopPass(&three).task_state, "FINISH_STOP");
}

}  // namespace
}  // namespace track_perception_cpp
