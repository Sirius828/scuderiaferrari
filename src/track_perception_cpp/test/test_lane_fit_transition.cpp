#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "track_perception_cpp/lane_decision.hpp"

namespace track_perception_cpp {
namespace {

LaneDecisionConfig transitionConfig(const std::string& side = "left") {
  LaneDecisionConfig config;
  config.band_count = 10;
  config.band_y_min_ratio = 0.0f;
  config.band_y_max_ratio = 1.0f;
  config.band_height_ratio = 0.08f;
  config.min_segment_width_px = 5;
  config.min_segment_gap_px = 10;
  config.min_pixels_per_band = 5;
  config.branch_detect_min_bands = 1;
  config.branch_confirm_frames = 1;
  config.branch_detect_far_band_ratio = 1.0f;
  config.outer_side = side;
  config.enable_guideboard_branch_selection = false;
  config.enable_encoder_branch_hold = true;
  config.encoder_hold_counts = 10;
  config.encoder_hold_right_counts = 10;
  config.fit_min_points = 4;
  config.fit_order = 2;
  config.branch_fit_order = 2;
  config.enable_fit_point_jump_filter = false;
  config.enable_fit_point_trend_filter = false;
  config.enable_left_boundary_template_line = false;
  config.car_fit_hold_timeout_sec = 0.30;
  return config;
}

cv::Mat straightMask() {
  cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
  mask(cv::Rect(40, 0, 20, 100)).setTo(1);
  return mask;
}

cv::Mat branchMask() {
  cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
  mask(cv::Rect(5, 0, 20, 100)).setTo(1);
  mask(cv::Rect(75, 0, 20, 100)).setTo(1);
  return mask;
}

cv::Mat failedMask() {
  return cv::Mat::zeros(100, 100, CV_8UC1);
}

TEST(LaneFitTransitionTest, NormalToLeftBranchIsContinuousAndConverges) {
  LaneDecision decision;
  decision.configure(transitionConfig("left"));
  decision.setEncoderCount(0, nowSeconds());

  const LaneState normal = decision.decide(straightMask(), {});
  ASSERT_TRUE(normal.is_valid);
  ASSERT_EQ(decision.debugInfo().fit_source, "NORMAL");

  const LaneState transition = decision.decide(branchMask(), {});
  EXPECT_EQ(decision.debugInfo().fit_source, "BRANCH_LEFT");
  EXPECT_TRUE(decision.debugInfo().fit_transition_active);
  EXPECT_DOUBLE_EQ(decision.debugInfo().fit_transition_ratio, 0.0);
  EXPECT_NEAR(transition.offset_y09, normal.offset_y09, 1e-4);
  EXPECT_FALSE(decision.debugInfo().candidate_fit_coeffs.empty());

  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  const LaneState settled = decision.decide(branchMask(), {});
  EXPECT_FALSE(decision.debugInfo().fit_transition_active);
  EXPECT_DOUBLE_EQ(decision.debugInfo().fit_transition_ratio, 1.0);
  EXPECT_LT(settled.offset_y09, -0.4f);
}

TEST(LaneFitTransitionTest, RightBranchUsesIndependentSourceName) {
  LaneDecision decision;
  decision.configure(transitionConfig("right"));
  decision.setEncoderCount(0, nowSeconds());

  ASSERT_TRUE(decision.decide(straightMask(), {}).is_valid);
  const LaneState transition = decision.decide(branchMask(), {});
  EXPECT_TRUE(transition.is_valid);
  EXPECT_EQ(decision.debugInfo().fit_source, "BRANCH_RIGHT");
  EXPECT_TRUE(decision.debugInfo().fit_transition_active);
}

TEST(LaneFitTransitionTest, ReleaseDuringTransitionRestartsFromPublishedCurve) {
  LaneDecision decision;
  decision.configure(transitionConfig("left"));
  decision.setEncoderCount(0, nowSeconds());

  const LaneState normal = decision.decide(straightMask(), {});
  ASSERT_TRUE(decision.decide(branchMask(), {}).is_valid);
  ASSERT_TRUE(decision.debugInfo().fit_transition_active);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const LaneState mid_transition = decision.decide(branchMask(), {});
  ASSERT_TRUE(decision.debugInfo().fit_transition_active);
  ASSERT_LT(mid_transition.offset_y09, normal.offset_y09 - 0.05f);

  decision.setEncoderCount(20, nowSeconds());
  const LaneState released = decision.decide(straightMask(), {});
  EXPECT_EQ(decision.debugInfo().fit_source, "NORMAL");
  EXPECT_TRUE(decision.debugInfo().fit_transition_active);
  EXPECT_DOUBLE_EQ(decision.debugInfo().fit_transition_ratio, 0.0);
  EXPECT_NEAR(released.offset_y09, mid_transition.offset_y09, 1e-4);

  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  const LaneState settled = decision.decide(straightMask(), {});
  EXPECT_FALSE(decision.debugInfo().fit_transition_active);
  EXPECT_NEAR(settled.offset_y09, normal.offset_y09, 1e-3);
}

TEST(LaneFitTransitionTest, FailedFitHoldsAndPausesTransition) {
  LaneDecision decision;
  decision.configure(transitionConfig("left"));
  decision.setEncoderCount(0, nowSeconds());

  ASSERT_TRUE(decision.decide(straightMask(), {}).is_valid);
  ASSERT_TRUE(decision.decide(branchMask(), {}).is_valid);
  ASSERT_TRUE(decision.debugInfo().fit_transition_active);

  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  const LaneState held = decision.decide(failedMask(), {});
  EXPECT_TRUE(held.is_valid);
  EXPECT_TRUE(decision.debugInfo().fit_hold_active);
  EXPECT_TRUE(decision.debugInfo().fit_transition_active);
  EXPECT_DOUBLE_EQ(decision.debugInfo().fit_transition_ratio, 0.0);

  const LaneState recovered = decision.decide(branchMask(), {});
  EXPECT_TRUE(recovered.is_valid);
  EXPECT_TRUE(decision.debugInfo().fit_transition_active);
  EXPECT_LT(decision.debugInfo().fit_transition_ratio, 0.2);
}

}  // namespace
}  // namespace track_perception_cpp
