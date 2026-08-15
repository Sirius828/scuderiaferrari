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
  config.enable_human_obstacle_stop = true;
  config.human_left_expand_px = 25.0f;
  config.human_left_expand_width_ratio = 0.50f;
  config.human_line_margin_px = 5.0f;
  config.human_stop_effective_area_ratio = 0.020f;
  config.human_stop_confirm_frames = 1;
  config.human_clear_confirm_frames = 2;
  config.enable_human_right_edge_pass = true;
  config.human_right_edge_pass_margin_px = 45.0f;
  config.car_boundary_smoothing_alpha = 0.5f;
  config.car_boundary_lost_frames = 3;
  config.car_fit_hold_timeout_sec = 0.20;
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

cv::Mat carPushMask() {
  // Far points are outside the original Car box and should be deleted.  The
  // middle points are inside the expanded box and should be pushed left.  The
  // last points are safely left of the expanded box and should remain.
  return makeMask({450, 450, 450, 450, 400, 400, 400, 300, 300, 300, 300, 300});
}

cv::Mat sparseMask() {
  return makeMask({320, 330, 340});
}

cv::Mat straightMask() {
  return makeMask({320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320});
}

cv::Mat rightFitMask() {
  return makeMask({620, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620});
}

Detection car(float left_x) {
  Detection det;
  det.class_name = "Car";
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(left_x, 200.0f, 80.0f, 120.0f);
  det.center = cv::Point2f(left_x + 40.0f, 260.0f);
  return det;
}

Detection human(float left_x, float y, float width, float height) {
  Detection det;
  det.class_name = "Human";
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(left_x, y, width, height);
  det.center = cv::Point2f(left_x + width * 0.5f, y + height * 0.5f);
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

TEST(LaneDecisionCarBoundaryTest, KalmanFiltersBandCentersBeforeFit) {
  LaneDecisionConfig config = makeConfig();
  config.enable_centerline_kalman = true;
  config.centerline_kalman_idle_process_noise = 0.0f;
  config.centerline_kalman_motion_process_noise = 0.0f;
  config.centerline_kalman_turn_process_noise = 0.0f;
  config.centerline_kalman_measurement_noise = 0.01f;
  config.centerline_kalman_reset_innovation = 0.50f;
  config.centerline_kalman_state_timeout_sec = 1.0;

  LaneDecision decision;
  decision.configure(config);
  ASSERT_TRUE(decision.decide(makeMask(std::vector<int>(12, 320)), {}).is_valid);
  const auto first_fit_points = decision.debugInfo().fit_points;

  ASSERT_TRUE(decision.decide(makeMask(std::vector<int>(12, 360)), {}).is_valid);
  const auto& debug = decision.debugInfo();
  ASSERT_EQ(debug.fit_points.size(), first_fit_points.size());
  ASSERT_EQ(debug.fit_points.size(), debug.raw_points.size());
  EXPECT_TRUE(debug.centerline_kalman_enabled);
  EXPECT_EQ(debug.centerline_kalman_point_count,
            static_cast<int>(debug.fit_points.size()));
  for (size_t i = 0; i < debug.fit_points.size(); ++i) {
    EXPECT_GT(debug.fit_points[i].x, first_fit_points[i].x);
    EXPECT_LT(debug.fit_points[i].x, debug.raw_points[i].x);
  }
}

TEST(LaneDecisionCarBoundaryTest, OffsetOutputUsesCurrentFittedCurveWithoutSecondFilter) {
  LaneDecisionConfig config = makeConfig();
  config.enable_centerline_kalman = false;

  LaneDecision decision;
  decision.configure(config);
  const LaneState state = decision.decide(makeMask(std::vector<int>(12, 400)), {});

  ASSERT_TRUE(state.is_valid);
  EXPECT_GT(state.offset_y09, 0.20f);
  EXPECT_FLOAT_EQ(state.offset_y09, decision.debugInfo().raw_offset_y09);
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

TEST(LaneDecisionCarBoundaryTest, PushesExpandedCarPointsAndDeletesOutsideFarPoints) {
  LaneDecisionConfig config = makeConfig();
  config.enable_car_point_push_avoidance = true;
  config.car_push_expand_left_px = 40.0f;
  config.car_push_expand_bottom_px = 20.0f;
  config.car_push_expand_right_px = 0.0f;
  config.car_push_expand_top_px = 0.0f;
  config.car_push_clearance_px = 3.0f;

  LaneDecision decision;
  decision.configure(config);
  const LaneState state = decision.decide(carPushMask(), {car(360.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(debug.car_push_active);
  EXPECT_EQ(debug.car_pushed_point_count, 2);
  EXPECT_EQ(debug.car_deleted_point_count, 5);
  EXPECT_EQ(debug.car_filtered_point_count, 5);
  EXPECT_NEAR(debug.car_push_target_x, 317.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.x, 320.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.y, 200.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.width, 120.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.height, 140.0f, 1e-4f);
  ASSERT_EQ(debug.fit_coeffs.size(), 3u);
  EXPECT_EQ(debug.fit_order, 2);

  for (const auto& point : debug.pushed_fit_points) {
    EXPECT_FLOAT_EQ(point.x, 317.0f);
    EXPECT_GE(point.y, 200.0f);
    EXPECT_LE(point.y, 340.0f);
  }
  for (const auto& point : debug.removed_fit_points) {
    EXPECT_GT(point.x, 360.0f);
  }
  bool has_safe_left_point = false;
  for (const auto& point : debug.fit_points) {
    if (point.x < 320.0f) {
      has_safe_left_point = true;
    }
  }
  EXPECT_TRUE(has_safe_left_point);
}

TEST(LaneDecisionCarBoundaryTest, PushModeIsSwitchableAndDoesNotStackLegacyDeletion) {
  LaneDecisionConfig legacy_config = makeConfig();
  legacy_config.enable_car_point_push_avoidance = false;

  LaneDecision legacy;
  legacy.configure(legacy_config);
  legacy.decide(carPushMask(), {car(360.0f)});
  EXPECT_FALSE(legacy.debugInfo().car_push_active);
  EXPECT_TRUE(legacy.debugInfo().pushed_fit_points.empty());
  EXPECT_EQ(legacy.debugInfo().car_filtered_point_count, 7);

  LaneDecisionConfig push_config = legacy_config;
  push_config.enable_car_point_push_avoidance = true;
  LaneDecision pushed;
  pushed.configure(push_config);
  pushed.decide(carPushMask(), {car(360.0f)});
  EXPECT_TRUE(pushed.debugInfo().car_push_active);
  EXPECT_EQ(pushed.debugInfo().car_pushed_point_count, 2);
  EXPECT_EQ(pushed.debugInfo().car_deleted_point_count, 5);
  EXPECT_EQ(pushed.debugInfo().fit_points.size(), 7u);
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

TEST(LaneDecisionCarBoundaryTest, HumanOnRightPassesRegardlessOfAreaAndDoesNotChangeFit) {
  LaneDecision baseline;
  baseline.configure(makeConfig());
  const LaneState baseline_state = baseline.decide(straightMask(), {});
  const auto baseline_points = baseline.debugInfo().fit_points;
  const auto baseline_coeffs = baseline.debugInfo().fit_coeffs;

  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(straightMask(), {human(380.0f, 200.0f, 40.0f, 300.0f)});
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_passable);
  EXPECT_FALSE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
  EXPECT_EQ(decision.debugInfo().fit_points, baseline_points);
  EXPECT_EQ(decision.debugInfo().fit_coeffs, baseline_coeffs);
  EXPECT_EQ(state.offset_y07, baseline_state.offset_y07);
  EXPECT_EQ(state.offset_y08, baseline_state.offset_y08);
  EXPECT_EQ(state.offset_y09, baseline_state.offset_y09);
}

TEST(LaneDecisionCarBoundaryTest, HumanAtRightEdgePassesEvenWhenFitLineLeavesNoRightSpace) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      rightFitMask(), {human(590.0f, 180.0f, 40.0f, 300.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.humans.size(), 1u);
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(debug.humans.front().right_edge_passable);
  EXPECT_TRUE(debug.humans.front().passable);
  EXPECT_FALSE(debug.humans.front().stop_candidate);
  EXPECT_TRUE(debug.human_right_edge_passable);
  EXPECT_EQ(debug.human_right_edge_pass_count, 1);
  EXPECT_NEAR(debug.human_right_edge_limit_x, 595.0f, 1e-4f);
  EXPECT_FALSE(debug.human_left_seen_latched);
  EXPECT_EQ(debug.human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, RightEdgeBoundaryIsInclusiveAndOutsideStillUsesFitLine) {
  LaneDecision edge_decision;
  edge_decision.configure(makeConfig());
  const LaneState edge_state = edge_decision.decide(
      rightFitMask(), {human(555.0f, 250.0f, 40.0f, 50.0f)});
  ASSERT_EQ(edge_decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(edge_state.task_state, "CLEAR");
  EXPECT_TRUE(edge_decision.debugInfo().humans.front().right_edge_passable);
  EXPECT_TRUE(edge_decision.debugInfo().human_passable);

  LaneDecision outside_decision;
  outside_decision.configure(makeConfig());
  const LaneState outside_state = outside_decision.decide(
      rightFitMask(), {human(554.0f, 250.0f, 40.0f, 50.0f)});
  ASSERT_EQ(outside_decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(outside_state.task_state, "CLEAR");
  EXPECT_FALSE(outside_decision.debugInfo().humans.front().right_edge_passable);
  EXPECT_FALSE(outside_decision.debugInfo().humans.front().passable);
  EXPECT_TRUE(outside_decision.debugInfo().human_left_seen_latched);
}

TEST(LaneDecisionCarBoundaryTest, RightEdgePassesWhenFitIsInvalid) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      sparseMask(), {human(590.0f, 250.0f, 40.0f, 50.0f)});
  ASSERT_EQ(decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().humans.front().fit_available);
  EXPECT_TRUE(decision.debugInfo().humans.front().right_edge_passable);
  EXPECT_TRUE(decision.debugInfo().humans.front().passable);
  EXPECT_FALSE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, LeftHumanLatchesWithoutStoppingBelowAreaThreshold) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(straightMask(), {human(250.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_passable);
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "LEFT_LATCHED_WAIT_AREA");
  EXPECT_FALSE(decision.debugInfo().human_stop_candidate);

  const LaneState lost = decision.decide(straightMask(), {});
  EXPECT_EQ(lost.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_FALSE(decision.debugInfo().human_passable);
  EXPECT_EQ(decision.debugInfo().human_state, "LEFT_LATCHED_WAIT_AREA");
}

TEST(LaneDecisionCarBoundaryTest, HumanNeverEntersLegacyBandExclusionPath) {
  LaneDecisionConfig config = makeConfig();
  config.enable_obstacle_avoidance = true;
  config.obstacle_labels = {"Human"};

  LaneDecision baseline;
  baseline.configure(makeConfig());
  baseline.decide(straightMask(), {});

  LaneDecision decision;
  decision.configure(config);
  const LaneState state = decision.decide(
      straightMask(), {human(250.0f, 250.0f, 40.0f, 50.0f)});

  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().fit_points, baseline.debugInfo().fit_points);
  EXPECT_TRUE(decision.debugInfo().obstacle_zones.empty());
}

TEST(LaneDecisionCarBoundaryTest, LargeLeftHumanStopsAndLostDetectionDoesNotReleaseStop) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState stopped = decision.decide(
      straightMask(), {human(250.0f, 180.0f, 40.0f, 120.0f)});
  EXPECT_EQ(stopped.task_state, "OBSTACLE_STOP");
  EXPECT_TRUE(decision.debugInfo().human_stop_candidate);
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);

  const LaneState lost = decision.decide(straightMask(), {});
  EXPECT_EQ(lost.task_state, "OBSTACLE_STOP");
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "OBSTACLE_STOP");
}

TEST(LaneDecisionCarBoundaryTest, RightHumanClearsLeftMemoryOnlyAfterConfirmation) {
  LaneDecision decision;
  decision.configure(makeConfig());

  ASSERT_EQ(decision.decide(straightMask(), {human(250.0f, 250.0f, 40.0f, 50.0f)}).task_state,
            "CLEAR");

  const LaneState first_right = decision.decide(
      straightMask(), {human(380.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(first_right.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_TRUE(decision.debugInfo().human_right_clear_confirming);
  EXPECT_EQ(decision.debugInfo().human_right_clear_confirm_count, 1);

  const LaneState second_right = decision.decide(
      straightMask(), {human(380.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(second_right.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_left_seen_latched);
  EXPECT_TRUE(decision.debugInfo().human_passable);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, RightEdgeHumanClearsLeftMemoryImmediately) {
  LaneDecision decision;
  LaneDecisionConfig config = makeConfig();
  config.human_clear_confirm_frames = 3;
  decision.configure(config);

  decision.decide(rightFitMask(), {human(250.0f, 250.0f, 40.0f, 50.0f)});
  ASSERT_TRUE(decision.debugInfo().human_left_seen_latched);

  const LaneState edge_state = decision.decide(
      rightFitMask(), {human(590.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(edge_state.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_right_edge_passable);
  EXPECT_FALSE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_right_clear_confirm_count, 0);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, RightEdgeHumanReleasesActiveStop) {
  LaneDecision decision;
  LaneDecisionConfig config = makeConfig();
  config.human_stop_effective_area_ratio = 0.020f;
  config.human_stop_confirm_frames = 1;
  decision.configure(config);

  const LaneState stopped = decision.decide(
      rightFitMask(), {human(250.0f, 180.0f, 40.0f, 120.0f)});
  ASSERT_EQ(stopped.task_state, "OBSTACLE_STOP");

  const LaneState cleared = decision.decide(
      rightFitMask(), {human(590.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(cleared.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, NonpassableHumanBlocksRightEdgeClear) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      rightFitMask(), {human(590.0f, 250.0f, 40.0f, 50.0f),
                       human(250.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_right_edge_passable);
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_FALSE(decision.debugInfo().human_passable);
  EXPECT_EQ(decision.debugInfo().human_state, "LEFT_LATCHED_WAIT_AREA");
}

TEST(LaneDecisionCarBoundaryTest, DisablingRightEdgePassRestoresFitLineJudgement) {
  LaneDecisionConfig config = makeConfig();
  config.enable_human_right_edge_pass = false;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = decision.decide(
      rightFitMask(), {human(590.0f, 250.0f, 40.0f, 50.0f)});
  ASSERT_EQ(decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().humans.front().right_edge_passable);
  EXPECT_FALSE(decision.debugInfo().humans.front().passable);
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
}

TEST(LaneDecisionCarBoundaryTest, LostFrameResetsRightConfirmationAndDoesNotClearMemory) {
  LaneDecision decision;
  decision.configure(makeConfig());

  decision.decide(straightMask(), {human(250.0f, 250.0f, 40.0f, 50.0f)});
  decision.decide(straightMask(), {human(380.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(decision.debugInfo().human_right_clear_confirm_count, 1);

  decision.decide(straightMask(), {});
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_right_clear_confirm_count, 0);

  decision.decide(straightMask(), {human(380.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_right_clear_confirm_count, 1);
}

TEST(LaneDecisionCarBoundaryTest, HumanCannotBeClearedByRightDetectionWithoutValidFit) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      sparseMask(), {human(380.0f, 250.0f, 40.0f, 50.0f)});
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_passable);
  EXPECT_TRUE(decision.debugInfo().human_left_seen_latched);
  EXPECT_EQ(decision.debugInfo().human_state, "NO_FIT_LATCHED");
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
