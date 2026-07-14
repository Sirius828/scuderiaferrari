#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "track_perception_cpp/lane_decision.hpp"

namespace track_perception_cpp {
namespace {

LaneDecisionConfig makeConfig() {
  LaneDecisionConfig config;
  config.enable_segment_branch_logic = true;
  config.band_count = 12;
  config.band_y_min_ratio = 0.0f;
  config.band_y_max_ratio = 1.0f;
  config.band_height_ratio = 0.08f;
  config.min_segment_width_px = 10;
  config.min_segment_gap_px = 20;
  config.min_pixels_per_band = 10;
  config.branch_detect_min_bands = 100;
  config.branch_confirm_frames = 1;
  config.enable_encoder_branch_hold = false;
  config.enable_guideboard_branch_selection = false;
  config.fit_min_points = 4;
  config.fit_order = 2;
  config.branch_fit_order = 2;
  config.enable_fit_point_jump_filter = false;
  config.enable_fit_point_trend_filter = false;
  config.enable_obstacle_avoidance = false;
  config.obstacle_min_bottom_y_ratio = 0.35f;
  config.obstacle_stop_labels = {"Human"};
  config.enable_obstacle_stop = true;
  config.obstacle_stop_bottom_y_ratio = 0.75f;
  config.obstacle_stop_confirm_frames = 1;
  config.car_boundary_smoothing_alpha = 0.5f;
  config.car_boundary_lost_frames = 3;
  config.car_fit_hold_timeout_sec = 0.20;
  config.offset_smoothing_alpha = 1.0f;
  config.max_offset_jump = 0.0f;
  return config;
}

cv::Mat makeMask(const std::vector<int>& centers) {
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  const int step = kHeight / 12;
  const int band_height = static_cast<int>(kHeight * 0.08f);
  for (size_t i = 0; i < centers.size() && i < 12; ++i) {
    const int x0 = centers[i] - 10;
    mask(cv::Rect(x0, static_cast<int>(i) * step, 20, band_height)).setTo(1);
  }
  return mask;
}

cv::Mat fullMask() {
  // The first eight points are on the Car side; the last four are the safe
  // near-side points that must remain in the fit.
  return makeMask({446, 443, 479, 450, 460, 440, 455, 470, 348, 348, 328, 318});
}

cv::Mat sparseMask() {
  return makeMask({320, 330, 340});
}

Detection car(float left_x) {
  Detection det;
  det.class_name = "Car";
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(left_x, 200.0f, 80.0f, 120.0f);
  det.center = cv::Point2f(left_x + 40.0f, 260.0f);
  return det;
}

Detection human() {
  Detection det;
  det.class_name = "Human";
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(250.0f, 250.0f, 50.0f, 160.0f);
  det.center = cv::Point2f(275.0f, 330.0f);
  return det;
}

Detection detection(const std::string& label, float y, float height) {
  Detection det;
  det.class_name = label;
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(100.0f, y, 40.0f, height);
  det.center = cv::Point2f(120.0f, y + height * 0.5f);
  return det;
}

TEST(LaneDecisionCarBoundaryTest, NoCarKeepsSharpTurnPointsAndQuadraticFit) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(fullMask(), {});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_EQ(debug.raw_points.size(), debug.fit_points.size());
  EXPECT_EQ(debug.car_filtered_point_count, 0);
  EXPECT_FALSE(debug.car_boundary_active);
  ASSERT_EQ(debug.fit_coeffs.size(), 3u);
  EXPECT_EQ(debug.fit_order, 2);
}

TEST(LaneDecisionCarBoundaryTest, RemovesCarSideFarPointsAndKeepsNearLeftPoints) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(fullMask(), {car(360.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(debug.car_boundary_active);
  EXPECT_FLOAT_EQ(debug.car_left_x, 360.0f);
  EXPECT_EQ(debug.car_filtered_point_count, 8);
  ASSERT_EQ(debug.fit_coeffs.size(), 3u);
  EXPECT_EQ(debug.fit_order, 2);
  ASSERT_EQ(debug.fit_points.size(), 4u);
  for (const auto& point : debug.fit_points) {
    EXPECT_LT(point.x, 360.0f);
    EXPECT_GT(point.y, 320.0f);
  }
  for (const auto& point : debug.removed_fit_points) {
    EXPECT_GT(point.x, 360.0f);
    EXPECT_LE(point.y, 320.0f);
  }
}

TEST(LaneDecisionCarBoundaryTest, AppliesLeftEdgeRuleEvenWhenCarIsInImageLeftHalf) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(fullMask(), {car(200.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(state.is_valid);
  EXPECT_TRUE(debug.car_boundary_active);
  EXPECT_FLOAT_EQ(debug.car_left_x, 200.0f);
  EXPECT_EQ(debug.car_filtered_point_count, 8);
  ASSERT_EQ(debug.fit_points.size(), 4u);
  for (const auto& point : debug.fit_points) {
    EXPECT_GT(point.y, 320.0f);
  }
  for (const auto& point : debug.removed_fit_points) {
    EXPECT_GT(point.x, 200.0f);
    EXPECT_LE(point.y, 320.0f);
  }
}

TEST(LaneDecisionCarBoundaryTest, SmoothsBoundaryAndKeepsItAcrossTwoLostFrames) {
  LaneDecision decision;
  decision.configure(makeConfig());

  decision.decide(fullMask(), {car(360.0f)});
  decision.decide(fullMask(), {car(380.0f)});
  EXPECT_NEAR(decision.debugInfo().car_left_x, 370.0f, 1e-4f);

  decision.decide(fullMask(), {});
  EXPECT_TRUE(decision.debugInfo().car_boundary_active);
  EXPECT_EQ(decision.debugInfo().car_boundary_lost_count, 1);
  decision.decide(fullMask(), {});
  EXPECT_TRUE(decision.debugInfo().car_boundary_active);
  EXPECT_EQ(decision.debugInfo().car_boundary_lost_count, 2);

  decision.decide(fullMask(), {});
  EXPECT_FALSE(decision.debugInfo().car_boundary_active);
  EXPECT_EQ(decision.debugInfo().car_boundary_lost_count, 3);
  EXPECT_EQ(decision.debugInfo().car_filtered_point_count, 0);
  EXPECT_EQ(decision.debugInfo().fit_points.size(), decision.debugInfo().raw_points.size());
}

TEST(LaneDecisionCarBoundaryTest, HoldsLastQuadraticFitThenBecomesLowConfidence) {
  LaneDecisionConfig config = makeConfig();
  config.car_fit_hold_timeout_sec = 0.01;
  LaneDecision decision;
  decision.configure(config);

  ASSERT_TRUE(decision.decide(fullMask(), {}).is_valid);
  const auto previous_coeffs = decision.debugInfo().fit_coeffs;

  const LaneState held = decision.decide(sparseMask(), {});
  EXPECT_TRUE(held.is_valid);
  EXPECT_TRUE(decision.debugInfo().fit_hold_active);
  EXPECT_EQ(decision.debugInfo().fit_coeffs, previous_coeffs);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const LaneState expired = decision.decide(sparseMask(), {});
  EXPECT_FALSE(expired.is_valid);
  EXPECT_EQ(expired.road_state, "LOW_CONFIDENCE");
  EXPECT_FALSE(decision.debugInfo().fit_hold_active);
  EXPECT_TRUE(decision.debugInfo().fit_coeffs.empty());
}

TEST(LaneDecisionCarBoundaryTest, HumanStillTriggersObstacleStopWhileCarDoesNot) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState car_state = decision.decide(fullMask(), {car(360.0f)});
  EXPECT_EQ(car_state.task_state, "CLEAR");

  const LaneState human_state = decision.decide(fullMask(), {human()});
  EXPECT_EQ(human_state.task_state, "OBSTACLE_STOP");
}

TEST(LaneDecisionCarBoundaryTest, FinishAndLowConfidenceProtectionRemainActive) {
  LaneDecision finish_decision;
  LaneDecisionConfig finish_config = makeConfig();
  finish_config.finish_stop_lost_frames = 1;
  finish_decision.configure(finish_config);
  ASSERT_EQ(finish_decision.decide(fullMask(), {detection("Stop", 300.0f, 100.0f)}).task_state,
            "CLEAR");
  EXPECT_EQ(finish_decision.decide(fullMask(), {}).task_state, "FINISH_STOP");

  LaneDecision low_confidence_decision;
  low_confidence_decision.configure(makeConfig());
  const LaneState low_confidence = low_confidence_decision.decide(sparseMask(), {});
  EXPECT_FALSE(low_confidence.is_valid);
  EXPECT_EQ(low_confidence.road_state, "LOW_CONFIDENCE");
}

}  // namespace
}  // namespace track_perception_cpp
