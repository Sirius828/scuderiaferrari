#include <algorithm>
#include <chrono>
#include <cmath>
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
  config.human_horizontal_expand_px = 25.0f;
  config.human_horizontal_expand_width_ratio = 0.50f;
  config.human_line_sample_count = 5;
  config.human_stop_raw_area_ratio = 0.004f;
  config.human_stop_confirm_frames = 2;
  config.human_clear_confirm_frames = 2;
  config.enable_car_obstacle_avoidance = true;
  config.car_side_mask_strip_width_px = 40;
  config.car_side_mask_strip_height_px = 60;
  config.car_side_mask_gap_px = 3;
  config.car_side_mask_y_start_ratio = 0.60f;
  config.car_side_mask_min_road_ratio = 0.02f;
  config.car_side_mask_min_ratio_diff = 0.05f;
  config.car_side_confirm_frames = 2;
  config.car_bbox_smoothing_alpha = 0.5f;
  config.car_detection_lost_frames = 3;
  config.car_expand_toward_lane_width_ratio = 1.25f;
  config.car_expand_bottom_height_ratio = 0.167f;
  config.car_expand_top_height_ratio = 0.0f;
  config.car_shift_clearance_width_ratio = 0.04f;
  config.car_safe_fit_hold_timeout_sec = 0.20;
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

cv::Mat straightMask() {
  return makeMask({320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320});
}

cv::Mat rightFitMask() {
  return makeMask({620, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620});
}

cv::Mat curvedFitMask() {
  return makeMask({200, 215, 235, 260, 290, 325, 365, 410, 460, 515, 570, 620});
}

Detection car(float left_x) {
  Detection det;
  det.class_name = "Car";
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(left_x, 200.0f, 80.0f, 120.0f);
  det.center = cv::Point2f(left_x + 40.0f, 260.0f);
  return det;
}

cv::Mat withCarSideMaskEvidence(const cv::Mat& base, float car_left,
                                const std::string& car_side,
                                bool thin = false) {
  cv::Mat mask = base.clone();
  constexpr int kCarWidth = 80;
  constexpr int kGap = 3;
  constexpr int kStripWidth = 40;
  constexpr int kY0 = 248;
  constexpr int kY1 = 340;
  int strip_x = car_side == "LEFT"
                    ? static_cast<int>(std::ceil(car_left + kCarWidth)) + kGap
                    : static_cast<int>(std::floor(car_left)) - kGap - kStripWidth;
  strip_x = std::clamp(strip_x, 0, mask.cols - kStripWidth);
  if (thin) {
    mask(cv::Rect(strip_x, kY0, 3, kY1 - kY0)).setTo(1);
  } else {
    mask(cv::Rect(strip_x, kY0, kStripWidth, kY1 - kY0)).setTo(1);
  }
  return mask;
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
  EXPECT_EQ(debug.car_deleted_point_count, 0);
  EXPECT_EQ(debug.car_shifted_point_count, 0);
  EXPECT_FALSE(debug.car_avoidance_active);
  EXPECT_EQ(debug.car_side, "UNKNOWN");
  ASSERT_EQ(debug.fit_coeffs.size(), 3u);
  EXPECT_EQ(debug.fit_order, 2);
}

TEST(LaneDecisionCarBoundaryTest, CarSideMaskStripHeightExtendsBelowCarBox) {
  LaneDecisionConfig config = makeConfig();
  config.car_side_mask_strip_height_px = 200;
  LaneDecision decision;
  decision.configure(config);

  const cv::Mat mask = withCarSideMaskEvidence(straightMask(), 200.0f, "LEFT");
  ASSERT_FALSE(decision.decide(mask, {car(200.0f)}).is_valid);
  (void)decision.decide(mask, {car(200.0f)});

  const auto& debug = decision.debugInfo();
  EXPECT_EQ(debug.car_left_mask_roi.width, config.car_side_mask_strip_width_px);
  EXPECT_EQ(debug.car_right_mask_roi.width, config.car_side_mask_strip_width_px);
  EXPECT_EQ(debug.car_left_mask_roi.y, 272);
  EXPECT_EQ(debug.car_right_mask_roi.y, 272);
  EXPECT_EQ(debug.car_left_mask_roi.height, 200);
  EXPECT_EQ(debug.car_right_mask_roi.height, 200);
  EXPECT_GT(debug.car_left_mask_roi.y + debug.car_left_mask_roi.height, 320);
  EXPECT_GT(debug.car_right_mask_roi.y + debug.car_right_mask_roi.height, 320);
  EXPECT_EQ(debug.car_side, "LEFT");
  EXPECT_TRUE(debug.car_avoidance_active);
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

TEST(LaneDecisionCarBoundaryTest, LeftCarGloballyShiftsScreenedPointsRight) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask({100, 100, 100, 100, 100, 330, 330, 330, 330, 330, 330, 330}),
      200.0f, "LEFT");

  const LaneState first = decision.decide(mask, {car(200.0f)});
  EXPECT_FALSE(first.is_valid);
  EXPECT_EQ(first.road_state, "LOW_CONFIDENCE");
  EXPECT_EQ(decision.debugInfo().car_side_candidate, "LEFT");
  EXPECT_EQ(decision.debugInfo().car_side_confirm_count, 1);
  EXPECT_FALSE(decision.debugInfo().car_avoidance_active);
  EXPECT_TRUE(decision.debugInfo().car_fit_rejected);

  const LaneState second = decision.decide(mask, {car(200.0f)});
  const auto& debug = decision.debugInfo();
  EXPECT_EQ(debug.car_side, "LEFT");
  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_GT(debug.car_right_mask_ratio, debug.car_left_mask_ratio);
  EXPECT_EQ(debug.car_side_confirm_count, 2);
  EXPECT_NEAR(debug.car_expanded_bbox.x, 200.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.width, 180.0f, 1e-4f);
  EXPECT_TRUE(debug.car_initial_fit_success);
  EXPECT_GT(debug.car_global_shift_x, 0.0f);
  EXPECT_EQ(debug.car_shifted_point_count,
            static_cast<int>(debug.fit_points.size()));
  EXPECT_FALSE(debug.car_shift_out_of_bounds);
  EXPECT_GT(debug.car_deleted_point_count, 0);
  for (const auto& point : debug.removed_fit_points) {
    EXPECT_LT(point.x, 280.0f);
    EXPECT_LE(point.y, 340.0f);
  }
  EXPECT_TRUE(second.is_valid);
  EXPECT_FALSE(debug.car_fit_collision);
}

TEST(LaneDecisionCarBoundaryTest, RightCarGloballyShiftsScreenedPointsLeft) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask({500, 500, 500, 500, 500, 330, 330, 330, 330, 330, 330, 330}),
      360.0f, "RIGHT");

  EXPECT_FALSE(decision.decide(mask, {car(360.0f)}).is_valid);
  const LaneState second = decision.decide(mask, {car(360.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_EQ(debug.car_side, "RIGHT");
  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_GT(debug.car_left_mask_ratio, debug.car_right_mask_ratio);
  EXPECT_NEAR(debug.car_expanded_bbox.x, 260.0f, 1e-4f);
  EXPECT_NEAR(debug.car_expanded_bbox.width, 180.0f, 1e-4f);
  EXPECT_TRUE(debug.car_initial_fit_success);
  EXPECT_LT(debug.car_global_shift_x, 0.0f);
  EXPECT_EQ(debug.car_shifted_point_count,
            static_cast<int>(debug.fit_points.size()));
  EXPECT_FALSE(debug.car_shift_out_of_bounds);
  EXPECT_GT(debug.car_deleted_point_count, 0);
  for (const auto& point : debug.removed_fit_points) {
    EXPECT_GT(point.x, 360.0f);
    EXPECT_LE(point.y, 340.0f);
  }
  EXPECT_TRUE(second.is_valid);
  EXPECT_FALSE(debug.car_fit_collision);
}

TEST(LaneDecisionCarBoundaryTest, LeftImageEdgeCarUsesAvailableRightMask) {
  const LaneDecisionConfig config = makeConfig();
  LaneDecision decision;
  decision.configure(config);
  const cv::Mat mask = withCarSideMaskEvidence(straightMask(), 0.0f, "LEFT");

  EXPECT_FALSE(decision.decide(mask, {car(0.0f)}).is_valid);
  const LaneState second = decision.decide(mask, {car(0.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_EQ(debug.car_side, "LEFT");
  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_TRUE(debug.car_left_mask_roi.empty());
  EXPECT_FALSE(debug.car_right_mask_roi.empty());
  EXPECT_GT(debug.car_right_mask_ratio, config.car_side_mask_min_road_ratio);
  EXPECT_TRUE(second.is_valid);
}

TEST(LaneDecisionCarBoundaryTest, RightImageEdgeCarUsesAvailableLeftMask) {
  const LaneDecisionConfig config = makeConfig();
  LaneDecision decision;
  decision.configure(config);
  const cv::Mat mask = withCarSideMaskEvidence(straightMask(), 560.0f, "RIGHT");

  EXPECT_FALSE(decision.decide(mask, {car(560.0f)}).is_valid);
  const LaneState second = decision.decide(mask, {car(560.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_EQ(debug.car_side, "RIGHT");
  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_FALSE(debug.car_left_mask_roi.empty());
  EXPECT_TRUE(debug.car_right_mask_roi.empty());
  EXPECT_GT(debug.car_left_mask_ratio, config.car_side_mask_min_road_ratio);
  EXPECT_TRUE(second.is_valid);
}

TEST(LaneDecisionCarBoundaryTest, UnknownMaskSideImmediatelyRejectsFit) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = makeMask(std::vector<int>(12, 100));

  const LaneState state = decision.decide(mask, {car(360.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_FALSE(state.is_valid);
  EXPECT_EQ(state.road_state, "LOW_CONFIDENCE");
  EXPECT_EQ(debug.car_side, "UNKNOWN");
  EXPECT_EQ(debug.car_side_candidate, "UNKNOWN");
  EXPECT_FALSE(debug.car_avoidance_active);
  EXPECT_TRUE(debug.car_fit_rejected);
  EXPECT_FALSE(debug.fit_hold_active);
}

TEST(LaneDecisionCarBoundaryTest, CurvedFitDoesNotAffectMaskSideDecision) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = withCarSideMaskEvidence(curvedFitMask(), 200.0f, "LEFT");

  decision.decide(mask, {car(200.0f)});
  decision.decide(mask, {car(200.0f)});

  EXPECT_EQ(decision.debugInfo().car_side, "LEFT");
  EXPECT_TRUE(decision.debugInfo().car_avoidance_active);
  EXPECT_GT(decision.debugInfo().car_right_mask_ratio,
            decision.debugInfo().car_left_mask_ratio);
}

TEST(LaneDecisionCarBoundaryTest, DoesNotCreateVirtualPointsWhenExpandedBoxHasNoFitPoint) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask(std::vector<int>(12, 100)), 360.0f, "RIGHT", true);

  decision.decide(mask, {car(360.0f)});
  const LaneState state = decision.decide(mask, {car(360.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(state.is_valid);
  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_TRUE(debug.shifted_fit_points.empty());
  EXPECT_TRUE(debug.removed_fit_points.empty());
  EXPECT_EQ(debug.fit_points.size(), debug.raw_points.size());
}

TEST(LaneDecisionCarBoundaryTest, CarGlobalShiftRunsAfterKalmanAndRefitsSafely) {
  LaneDecisionConfig config = makeConfig();
  config.enable_centerline_kalman = true;
  config.centerline_kalman_idle_process_noise = 0.0f;
  config.centerline_kalman_motion_process_noise = 0.0f;
  config.centerline_kalman_turn_process_noise = 0.0f;
  config.centerline_kalman_measurement_noise = 0.01f;
  config.centerline_kalman_reset_innovation = 0.50f;
  LaneDecision decision;
  decision.configure(config);

  ASSERT_TRUE(decision.decide(makeMask(std::vector<int>(12, 320)), {}).is_valid);
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask(std::vector<int>(12, 320)), 360.0f, "RIGHT");
  EXPECT_FALSE(decision.decide(mask, {car(360.0f)}).is_valid);
  const LaneState shifted = decision.decide(mask, {car(360.0f)});

  const auto& debug = decision.debugInfo();
  ASSERT_TRUE(debug.car_avoidance_active);
  ASSERT_FALSE(debug.shifted_fit_points.empty());
  EXPECT_TRUE(debug.centerline_kalman_enabled);
  EXPECT_TRUE(debug.car_initial_fit_success);
  EXPECT_LT(debug.car_global_shift_x, 0.0f);
  EXPECT_TRUE(shifted.is_valid);
  EXPECT_FALSE(debug.car_fit_collision);
  EXPECT_FALSE(debug.car_fit_rejected);
  EXPECT_EQ(debug.shifted_fit_points.size(), debug.fit_points.size());
}

TEST(LaneDecisionCarBoundaryTest, GlobalShiftRejectsInsteadOfClippingAtImageEdge) {
  LaneDecisionConfig config = makeConfig();
  config.car_shift_clearance_width_ratio = 5.0f;
  LaneDecision decision;
  decision.configure(config);
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask(std::vector<int>(12, 330)), 200.0f, "LEFT");

  EXPECT_FALSE(decision.decide(mask, {car(200.0f)}).is_valid);
  const LaneState state = decision.decide(mask, {car(200.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_TRUE(debug.car_avoidance_active);
  EXPECT_TRUE(debug.car_initial_fit_success);
  EXPECT_GT(debug.car_global_shift_x, 0.0f);
  EXPECT_TRUE(debug.car_shift_out_of_bounds);
  EXPECT_TRUE(debug.shifted_fit_points.empty());
  EXPECT_FALSE(state.is_valid);
  EXPECT_EQ(state.road_state, "LOW_CONFIDENCE");
  EXPECT_TRUE(debug.car_fit_rejected);
}

TEST(LaneDecisionCarBoundaryTest, KeepsConfirmedCarAcrossTwoLostFramesThenClears) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const cv::Mat mask = withCarSideMaskEvidence(
      makeMask(std::vector<int>(12, 200)), 360.0f, "RIGHT");

  decision.decide(mask, {car(360.0f)});
  decision.decide(mask, {car(360.0f)});
  ASSERT_TRUE(decision.debugInfo().car_avoidance_active);

  decision.decide(mask, {});
  EXPECT_TRUE(decision.debugInfo().car_avoidance_active);
  EXPECT_EQ(decision.debugInfo().car_detection_lost_count, 1);
  decision.decide(mask, {});
  EXPECT_TRUE(decision.debugInfo().car_avoidance_active);
  EXPECT_EQ(decision.debugInfo().car_detection_lost_count, 2);

  const LaneState cleared = decision.decide(mask, {});
  EXPECT_FALSE(decision.debugInfo().car_avoidance_active);
  EXPECT_EQ(decision.debugInfo().car_detection_lost_count, 0);
  EXPECT_EQ(decision.debugInfo().car_side, "UNKNOWN");
  EXPECT_TRUE(cleared.is_valid);
}

TEST(LaneDecisionCarBoundaryTest, SafeCachedFitCanHoldButCollidingCacheCannot) {
  LaneDecisionConfig config = makeConfig();
  config.car_safe_fit_hold_timeout_sec = 0.20;

  LaneDecision safe;
  safe.configure(config);
  ASSERT_TRUE(safe.decide(makeMask(std::vector<int>(12, 100)), {}).is_valid);
  const cv::Mat right_evidence = withCarSideMaskEvidence(
      sparseMask(), 360.0f, "RIGHT", true);
  EXPECT_FALSE(safe.decide(right_evidence, {car(360.0f)}).is_valid);
  const LaneState safe_held = safe.decide(right_evidence, {car(360.0f)});
  EXPECT_TRUE(safe_held.is_valid);
  EXPECT_TRUE(safe.debugInfo().fit_hold_active);
  EXPECT_FALSE(safe.debugInfo().car_fit_collision);

  LaneDecision colliding;
  colliding.configure(config);
  ASSERT_TRUE(colliding.decide(makeMask(std::vector<int>(12, 320)), {}).is_valid);
  EXPECT_FALSE(colliding.decide(right_evidence, {car(360.0f)}).is_valid);
  const LaneState rejected = colliding.decide(right_evidence, {car(360.0f)});
  EXPECT_FALSE(rejected.is_valid);
  EXPECT_FALSE(colliding.debugInfo().fit_hold_active);
  EXPECT_TRUE(colliding.debugInfo().car_fit_rejected);
}

TEST(LaneDecisionCarBoundaryTest, HoldsLastQuadraticFitThenBecomesLowConfidence) {
  LaneDecisionConfig config = makeConfig();
  config.car_safe_fit_hold_timeout_sec = 0.01;
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

TEST(LaneDecisionCarBoundaryTest, NoHumanKeepsHumanStateClear) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(straightMask(), {});

  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().human_state, "NONE");
  EXPECT_EQ(decision.debugInfo().human_valid_count, 0);
  EXPECT_FALSE(decision.debugInfo().human_stop_active);
}

TEST(LaneDecisionCarBoundaryTest, HumansOnEitherSidePassAndDoNotChangeFit) {
  LaneDecision baseline;
  baseline.configure(makeConfig());
  const LaneState baseline_state = baseline.decide(straightMask(), {});
  const auto baseline_points = baseline.debugInfo().fit_points;
  const auto baseline_coeffs = baseline.debugInfo().fit_coeffs;

  LaneDecision left_decision;
  left_decision.configure(makeConfig());
  const LaneState left_state = left_decision.decide(
      straightMask(), {human(200.0f, 200.0f, 40.0f, 100.0f)});
  ASSERT_EQ(left_decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(left_state.task_state, "CLEAR");
  EXPECT_TRUE(left_decision.debugInfo().human_passable);
  EXPECT_FALSE(left_decision.debugInfo().humans.front().line_intersects);
  EXPECT_EQ(left_decision.debugInfo().human_state, "PASSABLE");

  LaneDecision right_decision;
  right_decision.configure(makeConfig());
  const LaneState right_state = right_decision.decide(
      straightMask(), {human(380.0f, 200.0f, 40.0f, 100.0f)});
  ASSERT_EQ(right_decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(right_state.task_state, "CLEAR");
  EXPECT_TRUE(right_decision.debugInfo().human_passable);
  EXPECT_FALSE(right_decision.debugInfo().humans.front().line_intersects);
  EXPECT_EQ(right_decision.debugInfo().fit_points, baseline_points);
  EXPECT_EQ(right_decision.debugInfo().fit_coeffs, baseline_coeffs);
  EXPECT_EQ(right_state.offset_y07, baseline_state.offset_y07);
  EXPECT_EQ(right_state.offset_y08, baseline_state.offset_y08);
  EXPECT_EQ(right_state.offset_y09, baseline_state.offset_y09);
}

TEST(LaneDecisionCarBoundaryTest, ExpandedBoundaryContactCountsAsIntersection) {
  LaneDecisionConfig config = makeConfig();
  config.human_horizontal_expand_px = 25.0f;
  config.human_horizontal_expand_width_ratio = 0.0f;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = decision.decide(
      straightMask(), {human(344.5f, 200.0f, 20.0f, 50.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.humans.size(), 1u);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(debug.humans.front().line_intersects);
  EXPECT_FALSE(debug.humans.front().passable);
  EXPECT_EQ(debug.human_state, "OVERLAP_WAIT_NEAR");
}

TEST(LaneDecisionCarBoundaryTest, FiveSamplesCatchCurveIntersectionNearBoxTop) {
  LaneDecision decision;
  decision.configure(makeConfig());

  decision.decide(curvedFitMask(), {human(170.0f, 0.0f, 10.0f, 160.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.humans.size(), 1u);
  const auto& human_debug = debug.humans.front();
  ASSERT_EQ(human_debug.fit_sample_points.size(), 5u);
  EXPECT_TRUE(human_debug.line_intersects);
  int intersecting_samples = 0;
  for (const auto& sample : human_debug.fit_sample_points) {
    if (sample.x >= human_debug.expanded_bbox.x &&
        sample.x <= human_debug.expanded_bbox.x + human_debug.expanded_bbox.width) {
      ++intersecting_samples;
    }
  }
  EXPECT_GT(intersecting_samples, 0);
  EXPECT_LT(intersecting_samples, 5);
}

TEST(LaneDecisionCarBoundaryTest, FarOverlapKeepsControllerRunning) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      straightMask(), {human(300.0f, 220.0f, 40.0f, 20.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.humans.size(), 1u);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_TRUE(debug.human_line_intersects);
  EXPECT_FALSE(debug.human_stop_candidate);
  EXPECT_FALSE(debug.human_stop_active);
  EXPECT_EQ(debug.human_stop_confirm_count, 0);
  EXPECT_EQ(debug.human_state, "OVERLAP_WAIT_NEAR");
}

TEST(LaneDecisionCarBoundaryTest, NearOverlapStopsAfterConfirmation) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const Detection near_human = human(300.0f, 200.0f, 40.0f, 40.0f);

  const LaneState first = decision.decide(straightMask(), {near_human});
  EXPECT_EQ(first.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().human_stop_candidate);
  EXPECT_EQ(decision.debugInfo().human_stop_confirm_count, 1);
  EXPECT_EQ(decision.debugInfo().human_state, "OVERLAP_WAIT_NEAR");

  const LaneState second = decision.decide(straightMask(), {near_human});
  EXPECT_EQ(second.task_state, "OBSTACLE_STOP");
  EXPECT_TRUE(decision.debugInfo().human_stop_active);
  EXPECT_EQ(decision.debugInfo().human_stop_confirm_count, 2);
  EXPECT_EQ(decision.debugInfo().human_count_at_stop, 1);
  EXPECT_EQ(decision.debugInfo().human_state, "OBSTACLE_STOP");
}

TEST(LaneDecisionCarBoundaryTest, RawAreaDoesNotDependOnExpansionSize) {
  LaneDecisionConfig compact_config = makeConfig();
  compact_config.human_horizontal_expand_px = 25.0f;
  LaneDecision compact;
  compact.configure(compact_config);
  compact.decide(straightMask(), {human(300.0f, 220.0f, 40.0f, 20.0f)});

  LaneDecisionConfig wide_config = makeConfig();
  wide_config.human_horizontal_expand_px = 100.0f;
  LaneDecision wide;
  wide.configure(wide_config);
  wide.decide(straightMask(), {human(300.0f, 220.0f, 40.0f, 20.0f)});

  ASSERT_EQ(compact.debugInfo().humans.size(), 1u);
  ASSERT_EQ(wide.debugInfo().humans.size(), 1u);
  EXPECT_FLOAT_EQ(compact.debugInfo().humans.front().raw_area_ratio,
                  wide.debugInfo().humans.front().raw_area_ratio);
  EXPECT_LT(compact.debugInfo().humans.front().expanded_bbox.width,
            wide.debugInfo().humans.front().expanded_bbox.width);
  EXPECT_FALSE(compact.debugInfo().human_stop_candidate);
  EXPECT_FALSE(wide.debugInfo().human_stop_candidate);
}

TEST(LaneDecisionCarBoundaryTest, InvalidFitStopsOnlyWhenHumanIsNear) {
  LaneDecision far_decision;
  far_decision.configure(makeConfig());
  const LaneState far_state = far_decision.decide(
      sparseMask(), {human(300.0f, 220.0f, 40.0f, 20.0f)});
  ASSERT_EQ(far_decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(far_state.task_state, "CLEAR");
  EXPECT_FALSE(far_decision.debugInfo().humans.front().fit_available);
  EXPECT_FALSE(far_decision.debugInfo().human_stop_candidate);
  EXPECT_EQ(far_decision.debugInfo().human_state, "NO_FIT_WAIT_NEAR");

  LaneDecision near_decision;
  near_decision.configure(makeConfig());
  const Detection near_human = human(300.0f, 200.0f, 40.0f, 40.0f);
  EXPECT_EQ(near_decision.decide(sparseMask(), {near_human}).task_state, "CLEAR");
  EXPECT_TRUE(near_decision.debugInfo().human_stop_candidate);
  EXPECT_EQ(near_decision.decide(sparseMask(), {near_human}).task_state,
            "OBSTACLE_STOP");
}

TEST(LaneDecisionCarBoundaryTest, StoppedHumanLossDoesNotReleaseAndSafeFramesDo) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const Detection near_overlap = human(300.0f, 200.0f, 40.0f, 40.0f);
  const Detection safe_left = human(200.0f, 200.0f, 40.0f, 40.0f);

  decision.decide(straightMask(), {near_overlap});
  ASSERT_EQ(decision.decide(straightMask(), {near_overlap}).task_state,
            "OBSTACLE_STOP");

  const LaneState lost = decision.decide(straightMask(), {});
  EXPECT_EQ(lost.task_state, "OBSTACLE_STOP");
  EXPECT_EQ(decision.debugInfo().human_clear_confirm_count, 0);

  const LaneState first_safe = decision.decide(straightMask(), {safe_left});
  EXPECT_EQ(first_safe.task_state, "OBSTACLE_STOP");
  EXPECT_TRUE(decision.debugInfo().human_clear_confirming);
  EXPECT_EQ(decision.debugInfo().human_clear_confirm_count, 1);

  const LaneState second_safe = decision.decide(straightMask(), {safe_left});
  EXPECT_EQ(second_safe.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_stop_active);
  EXPECT_TRUE(decision.debugInfo().human_passable);
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
}

TEST(LaneDecisionCarBoundaryTest, MissingHumanInMultipleTargetStopCannotClear) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const Detection near_overlap = human(300.0f, 200.0f, 40.0f, 40.0f);
  const Detection safe_left = human(200.0f, 200.0f, 40.0f, 40.0f);
  const Detection safe_right = human(400.0f, 200.0f, 40.0f, 40.0f);

  decision.decide(straightMask(), {near_overlap, safe_left});
  ASSERT_EQ(decision.decide(straightMask(), {near_overlap, safe_left}).task_state,
            "OBSTACLE_STOP");
  ASSERT_EQ(decision.debugInfo().human_count_at_stop, 2);

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(decision.decide(straightMask(), {safe_left}).task_state,
              "OBSTACLE_STOP");
    EXPECT_EQ(decision.debugInfo().human_clear_confirm_count, 0);
  }

  EXPECT_EQ(decision.decide(straightMask(), {safe_left, safe_right}).task_state,
            "OBSTACLE_STOP");
  EXPECT_EQ(decision.debugInfo().human_clear_confirm_count, 1);
  EXPECT_EQ(decision.decide(straightMask(), {safe_left, safe_right}).task_state,
            "CLEAR");
}

TEST(LaneDecisionCarBoundaryTest, RightEdgeHasNoSpecialBypass) {
  LaneDecision decision;
  decision.configure(makeConfig());
  const Detection edge_human = human(590.0f, 180.0f, 40.0f, 40.0f);

  const LaneState first = decision.decide(rightFitMask(), {edge_human});
  ASSERT_EQ(decision.debugInfo().humans.size(), 1u);
  EXPECT_EQ(first.task_state, "CLEAR");
  EXPECT_TRUE(decision.debugInfo().humans.front().line_intersects);
  EXPECT_TRUE(decision.debugInfo().human_stop_candidate);

  const LaneState second = decision.decide(rightFitMask(), {edge_human});
  EXPECT_EQ(second.task_state, "OBSTACLE_STOP");
}

TEST(LaneDecisionCarBoundaryTest, HumanNeverEntersLegacyBandExclusionPath) {
  LaneDecisionConfig config = makeConfig();
  config.enable_obstacle_avoidance = true;
  config.obstacle_labels = {"Human"};

  LaneDecision baseline;
  baseline.configure(makeConfig());
  const LaneState baseline_state = baseline.decide(straightMask(), {});

  LaneDecision decision;
  decision.configure(config);
  const LaneState state = decision.decide(
      straightMask(), {human(300.0f, 220.0f, 40.0f, 20.0f)});

  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().fit_points, baseline.debugInfo().fit_points);
  EXPECT_EQ(decision.debugInfo().fit_coeffs, baseline.debugInfo().fit_coeffs);
  EXPECT_EQ(state.offset_y07, baseline_state.offset_y07);
  EXPECT_EQ(state.offset_y08, baseline_state.offset_y08);
  EXPECT_EQ(state.offset_y09, baseline_state.offset_y09);
  EXPECT_TRUE(decision.debugInfo().obstacle_zones.empty());
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
