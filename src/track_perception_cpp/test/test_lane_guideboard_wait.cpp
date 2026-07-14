#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "track_perception_cpp/lane_decision.hpp"

namespace track_perception_cpp {
namespace {

LaneDecisionConfig makeConfig() {
  LaneDecisionConfig config;
  config.band_count = 4;
  config.band_y_min_ratio = 0.0f;
  config.band_y_max_ratio = 1.0f;
  config.band_height_ratio = 0.2f;
  config.min_segment_width_px = 5;
  config.min_segment_gap_px = 10;
  config.min_pixels_per_band = 5;
  config.branch_detect_min_bands = 1;
  config.branch_confirm_frames = 1;
  config.branch_detect_far_band_ratio = 1.0f;
  config.enable_encoder_branch_hold = true;
  config.enable_guideboard_branch_selection = true;
  config.guideboard_detect_y0_ratio = 0.0f;
  config.guideboard_detect_y1_ratio = 1.0f;
  config.guideboard_require_hint = true;
  config.guideboard_unknown_branch = "left";
  config.guideboard_hint_wait_timeout_sec = 0.02;
  config.encoder_hold_counts = 9000;
  config.encoder_hold_right_counts = 20000;
  config.fit_min_points = 1;
  return config;
}

cv::Mat branchMask() {
  cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
  mask(cv::Rect(5, 0, 25, 100)).setTo(1);
  mask(cv::Rect(70, 0, 25, 100)).setTo(1);
  return mask;
}

std::vector<Detection> guideboardDetection() {
  Detection sign;
  sign.class_name = "GuideBoard";
  sign.confidence = 0.95f;
  sign.bbox = cv::Rect2f(40.0f, 35.0f, 20.0f, 20.0f);
  sign.center = cv::Point2f(50.0f, 45.0f);
  return {sign};
}

TEST(LaneGuideboardWaitTest, WaitsForHintThenFallsBackStraight) {
  LaneDecision decision;
  decision.configure(makeConfig());
  decision.setGuideboardBranchHint("", false);

  auto first = decision.decide(branchMask(), guideboardDetection());
  EXPECT_TRUE(decision.debugInfo().guideboard_waiting_for_hint);
  EXPECT_TRUE(first.branch_side.empty());

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  auto timed_out = decision.decide(branchMask(), guideboardDetection());
  EXPECT_FALSE(decision.debugInfo().guideboard_waiting_for_hint);
  EXPECT_EQ(timed_out.branch_side, "left");
}

TEST(LaneGuideboardWaitTest, StableHintLocksImmediately) {
  LaneDecision decision;
  decision.configure(makeConfig());
  decision.setGuideboardBranchHint("right", true);

  auto state = decision.decide(branchMask(), guideboardDetection());
  EXPECT_FALSE(decision.debugInfo().guideboard_waiting_for_hint);
  EXPECT_TRUE(decision.debugInfo().guideboard_hint_valid);
  EXPECT_EQ(state.branch_side, "right");
  EXPECT_EQ(decision.debugInfo().encoder_hold_target, 20000);
}

TEST(LaneGuideboardWaitTest, UsesLateHintAfterGuideboardLeavesRoi) {
  LaneDecision decision;
  decision.configure(makeConfig());
  decision.setGuideboardBranchHint("", false);

  auto first = decision.decide(branchMask(), guideboardDetection());
  ASSERT_TRUE(decision.debugInfo().guideboard_waiting_for_hint);
  ASSERT_TRUE(first.branch_side.empty());

  decision.setGuideboardBranchHint("right", true);
  auto state = decision.decide(branchMask(), {});
  EXPECT_FALSE(decision.debugInfo().guideboard_seen);
  EXPECT_TRUE(decision.debugInfo().guideboard_hint_valid);
  EXPECT_EQ(state.branch_side, "right");
}

}  // namespace
}  // namespace track_perception_cpp
