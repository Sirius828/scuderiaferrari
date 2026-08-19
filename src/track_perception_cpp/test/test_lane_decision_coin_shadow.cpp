#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>

#include "track_perception_cpp/lane_decision.hpp"

namespace track_perception_cpp {
namespace {

constexpr int kWidth = 320;
constexpr int kHeight = 240;

LaneDecisionConfig makeConfig() {
  LaneDecisionConfig config;
  config.enable_segment_branch_logic = true;
  config.band_count = 12;
  config.band_y_min_ratio = 0.0f;
  config.band_y_max_ratio = 1.0f;
  config.band_height_ratio = 0.08f;
  config.min_segment_width_px = 5;
  config.min_segment_gap_px = 15;
  config.min_pixels_per_band = 5;
  config.branch_detect_min_bands = 100;
  config.enable_encoder_branch_hold = false;
  config.enable_guideboard_branch_selection = false;
  config.fit_min_points = 4;
  config.fit_order = 2;
  config.enable_fit_point_jump_filter = false;
  config.enable_fit_point_trend_filter = false;
  config.enable_centerline_kalman = false;
  config.enable_car_obstacle_avoidance = false;
  config.enable_human_obstacle_stop = false;
  config.enable_finish_stop = false;

  config.enable_coin_shadow_evaluation = true;
  config.coin_min_confidence = 0.5f;
  config.coin_evaluate_min_y_ratio = 0.0f;
  config.coin_near_committed_y_ratio = 1.0f;
  config.coin_car_half_width_area_scale = 0.0f;
  config.coin_car_half_width_min_px = 10.0f;
  config.coin_car_half_width_max_px = 10.0f;
  config.coin_hit_margin_px = 0.0f;
  config.coin_reachable_extra_area_scale = 0.0f;
  config.coin_reachable_extra_min_px = 30.0f;
  config.coin_reachable_extra_max_px = 30.0f;
  config.coin_obstacle_expand_px = 0.0f;
  config.coin_obstacle_lookahead_ratio = 0.0f;
  config.coin_side_clear_frames = 3;
  config.obstacle_min_confidence = 0.5f;
  return config;
}

cv::Mat roadMask() {
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  mask(cv::Rect(80, 0, 160, kHeight)).setTo(1);
  return mask;
}

cv::Mat curvedRoadMask() {
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  for (int y = 0; y < kHeight; ++y) {
    const double center_x = 100.0 + 0.0025 * y * y;
    const int x0 = std::max(0, static_cast<int>(std::round(center_x - 35.0)));
    const int x1 = std::min(kWidth, static_cast<int>(std::round(center_x + 35.0)));
    if (x1 > x0) {
      mask.row(y).colRange(x0, x1).setTo(1);
    }
  }
  return mask;
}

double evaluatePolynomial(const std::vector<double>& coeffs, double y) {
  double x = 0.0;
  for (double coefficient : coeffs) {
    x = x * y + coefficient;
  }
  return x;
}

double evaluatePolynomialSlope(const std::vector<double>& coeffs, double y) {
  double slope = 0.0;
  const int degree = static_cast<int>(coeffs.size()) - 1;
  for (size_t index = 0; index + 1 < coeffs.size(); ++index) {
    slope = slope * y + coeffs[index] *
                            (degree - static_cast<int>(index));
  }
  return slope;
}

Detection detection(const std::string& class_name, float center_x,
                    float bottom_y, float width = 10.0f,
                    float height = 10.0f) {
  Detection result;
  result.class_name = class_name;
  result.confidence = 0.95f;
  result.bbox = cv::Rect2f(center_x - width * 0.5f, bottom_y - height,
                           width, height);
  result.center = cv::Point2f(center_x, bottom_y - height * 0.5f);
  return result;
}

TEST(LaneDecisionCoinShadowTest, ClassifiesOnRouteReachableAndTooFar) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      roadMask(), {detection("Gold", 160.0f, 100.0f),
                   detection("Gold", 190.0f, 100.0f),
                   detection("Gold", 230.0f, 100.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_TRUE(state.is_valid);
  ASSERT_EQ(debug.coins.size(), 3u);
  EXPECT_EQ(debug.coins[0].classification, "ON_ROUTE");
  EXPECT_EQ(debug.coins[1].classification, "REACHABLE");
  EXPECT_EQ(debug.coins[2].classification, "TOO_FAR");
  EXPECT_EQ(debug.coin_on_route_count, 1);
  EXPECT_EQ(debug.coin_reachable_count, 1);
  EXPECT_EQ(debug.coin_too_far_count, 1);
  EXPECT_EQ(debug.coin_blocked_count, 0);
  EXPECT_NEAR(debug.coins[0].hit_radius, 15.0, 1e-3);
}

TEST(LaneDecisionCoinShadowTest, ObstacleInApproachCorridorVetoesCoin) {
  LaneDecision decision;
  decision.configure(makeConfig());

  Detection human = detection("Human", 185.0f, 170.0f, 30.0f, 50.0f);
  const LaneState state = decision.decide(
      roadMask(), {detection("Gold", 190.0f, 100.0f), human});
  const auto& debug = decision.debugInfo();

  ASSERT_TRUE(state.is_valid);
  ASSERT_EQ(debug.coins.size(), 1u);
  EXPECT_EQ(debug.coins[0].classification, "BLOCKED");
  EXPECT_TRUE(debug.coins[0].obstacle_blocked);
  EXPECT_EQ(debug.coins[0].blocked_by, "Human");
  EXPECT_EQ(debug.coin_blocked_count, 1);
}

TEST(LaneDecisionCoinShadowTest, GoldNeverChangesFittedGeometry) {
  LaneDecision baseline;
  baseline.configure(makeConfig());
  const LaneState baseline_state = baseline.decide(roadMask(), {});
  const auto baseline_coeffs = baseline.debugInfo().fit_coeffs;

  LaneDecision with_coin;
  with_coin.configure(makeConfig());
  const LaneState coin_state = with_coin.decide(
      roadMask(), {detection("Gold", 190.0f, 100.0f)});
  const auto& coin_debug = with_coin.debugInfo();

  ASSERT_TRUE(baseline_state.is_valid);
  ASSERT_TRUE(coin_state.is_valid);
  EXPECT_FLOAT_EQ(coin_state.offset_y07, baseline_state.offset_y07);
  EXPECT_FLOAT_EQ(coin_state.offset_y08, baseline_state.offset_y08);
  EXPECT_FLOAT_EQ(coin_state.offset_y09, baseline_state.offset_y09);
  EXPECT_FLOAT_EQ(coin_state.heading_error, baseline_state.heading_error);
  EXPECT_FLOAT_EQ(coin_state.curvature, baseline_state.curvature);
  ASSERT_EQ(coin_debug.fit_coeffs.size(), baseline_coeffs.size());
  for (size_t i = 0; i < baseline_coeffs.size(); ++i) {
    EXPECT_DOUBLE_EQ(coin_debug.fit_coeffs[i], baseline_coeffs[i]);
  }
  EXPECT_TRUE(coin_debug.coin_route_candidate_valid)
      << coin_debug.coin_route_reject_reason;
  EXPECT_FALSE(coin_debug.coin_route_control_active);
  EXPECT_EQ(coin_debug.coin_route_target_count, 1);
}

TEST(LaneDecisionCoinShadowTest, ReportsNoFitWithoutChangingInvalidState) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(
      cv::Mat::zeros(kHeight, kWidth, CV_8UC1),
      {detection("Gold", 160.0f, 100.0f)});
  const auto& debug = decision.debugInfo();

  EXPECT_FALSE(state.is_valid);
  ASSERT_EQ(debug.coins.size(), 1u);
  EXPECT_EQ(debug.coins[0].classification, "NO_FIT");
  EXPECT_EQ(debug.coin_no_fit_count, 1);
}

TEST(LaneDecisionCoinShadowTest, FarWaitsAndNearCommitsWithoutDistanceEvaluation) {
  auto config = makeConfig();
  config.coin_evaluate_min_y_ratio = 0.50f;
  config.coin_near_committed_y_ratio = 0.86f;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = decision.decide(
      roadMask(), {detection("Gold", 160.0f, 80.0f),
                   detection("Gold", 260.0f, 220.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_TRUE(state.is_valid);
  ASSERT_EQ(debug.coins.size(), 2u);
  EXPECT_EQ(debug.coins[0].classification, "WAIT_FAR");
  EXPECT_EQ(debug.coins[1].classification, "NEAR_COMMITTED");
  EXPECT_FLOAT_EQ(debug.coins[0].normal_distance, 0.0f);
  EXPECT_FLOAT_EQ(debug.coins[1].normal_distance, 0.0f);
  EXPECT_EQ(debug.coin_wait_far_count, 1);
  EXPECT_EQ(debug.coin_near_committed_count, 1);
  EXPECT_EQ(debug.coin_too_far_count, 0);
}

TEST(LaneDecisionCoinShadowTest, EqualAreaUsesEqualThresholdAtDifferentHeights) {
  auto config = makeConfig();
  config.coin_car_half_width_area_scale = 0.5f;
  config.coin_car_half_width_min_px = 0.0f;
  config.coin_car_half_width_max_px = 100.0f;
  config.coin_reachable_extra_area_scale = 1.0f;
  config.coin_reachable_extra_min_px = 0.0f;
  config.coin_reachable_extra_max_px = 100.0f;
  LaneDecision decision;
  decision.configure(config);

  (void)decision.decide(
      roadMask(), {detection("Gold", 180.0f, 120.0f),
                   detection("Gold", 180.0f, 210.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.coins.size(), 2u);
  EXPECT_FLOAT_EQ(debug.coins[0].sqrt_bbox_area,
                  debug.coins[1].sqrt_bbox_area);
  EXPECT_NEAR(debug.coins[0].hit_radius, debug.coins[1].hit_radius, 1e-2f);
  EXPECT_FLOAT_EQ(debug.coins[0].reachable_extra,
                  debug.coins[1].reachable_extra);
}

TEST(LaneDecisionCoinShadowTest, LargerCoinGetsLargerThresholdAtSameHeight) {
  auto config = makeConfig();
  config.coin_car_half_width_area_scale = 0.5f;
  config.coin_car_half_width_min_px = 0.0f;
  config.coin_car_half_width_max_px = 100.0f;
  config.coin_reachable_extra_area_scale = 1.0f;
  config.coin_reachable_extra_min_px = 0.0f;
  config.coin_reachable_extra_max_px = 100.0f;
  LaneDecision decision;
  decision.configure(config);

  (void)decision.decide(
      roadMask(), {detection("Gold", 180.0f, 160.0f, 10.0f, 10.0f),
                   detection("Gold", 220.0f, 160.0f, 40.0f, 40.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.coins.size(), 2u);
  EXPECT_NEAR(debug.coins[0].sqrt_bbox_area, 10.0f, 1e-3f);
  EXPECT_NEAR(debug.coins[1].sqrt_bbox_area, 40.0f, 1e-3f);
  EXPECT_LT(debug.coins[0].hit_radius, debug.coins[1].hit_radius);
  EXPECT_LT(debug.coins[0].reachable_extra, debug.coins[1].reachable_extra);
  EXPECT_EQ(debug.coins[0].classification, "TOO_FAR");
  EXPECT_EQ(debug.coins[1].classification, "REACHABLE");
}

TEST(LaneDecisionCoinShadowTest, ClosestPointLiesOnCurvedFit) {
  auto config = makeConfig();
  config.coin_reachable_extra_min_px = 200.0f;
  config.coin_reachable_extra_max_px = 200.0f;
  LaneDecision decision;
  decision.configure(config);

  const Detection coin = detection("Gold", 210.0f, 110.0f);
  (void)decision.decide(curvedRoadMask(), {coin});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.coins.size(), 1u);
  ASSERT_GE(debug.fit_coeffs.size(), 2u);
  const auto& closest = debug.coins[0].local_path_point;
  EXPECT_NEAR(closest.x,
              evaluatePolynomial(debug.fit_coeffs, closest.y), 0.1);
  EXPECT_NEAR(debug.coins[0].normal_distance,
              std::hypot(coin.bbox.x + coin.bbox.width * 0.5f - closest.x,
                         coin.bbox.y + coin.bbox.height - closest.y),
              0.1);
  const double slope = evaluatePolynomialSlope(debug.fit_coeffs, closest.y);
  const double tangent_dot =
      (coin.bbox.x + coin.bbox.width * 0.5f - closest.x) * slope +
      (coin.bbox.y + coin.bbox.height - closest.y);
  EXPECT_NEAR(tangent_dot, 0.0, 0.5);
}

TEST(LaneDecisionCoinShadowTest, ExplicitControlFlagAppliesValidCoinRoute) {
  auto config = makeConfig();
  config.enable_coin_route_control = true;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = decision.decide(
      roadMask(), {detection("Gold", 190.0f, 100.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_TRUE(debug.coin_route_candidate_valid)
      << debug.coin_route_reject_reason;
  EXPECT_TRUE(debug.coin_route_control_active);
  EXPECT_GT(state.offset_y07, 0.0f);
}

TEST(LaneDecisionCoinShadowTest, SelectsOnlyOneReachableSideAndHoldsItByFrames) {
  LaneDecision decision;
  decision.configure(makeConfig());

  (void)decision.decide(
      roadMask(), {detection("Gold", 129.0f, 120.0f),
                   detection("Gold", 190.0f, 120.0f)});
  const auto& initial_debug = decision.debugInfo();
  ASSERT_EQ(initial_debug.coins.size(), 2u);
  ASSERT_EQ(initial_debug.coins[0].classification, "REACHABLE");
  ASSERT_EQ(initial_debug.coins[1].classification, "REACHABLE");
  EXPECT_EQ(initial_debug.coin_selected_side, "LEFT");
  EXPECT_TRUE(initial_debug.coins[0].selected_for_route);
  EXPECT_FALSE(initial_debug.coins[1].selected_for_route);
  EXPECT_EQ(initial_debug.coin_route_target_count, 1);
  EXPECT_TRUE(initial_debug.coins[0].route_targeted);
  EXPECT_FALSE(initial_debug.coins[1].route_targeted);

  (void)decision.decide(roadMask(), {detection("Gold", 190.0f, 120.0f)});
  EXPECT_EQ(decision.debugInfo().coin_selected_side, "LEFT");
  EXPECT_FALSE(decision.debugInfo().coins[0].selected_for_route);
  (void)decision.decide(roadMask(), {detection("Gold", 190.0f, 120.0f)});
  EXPECT_EQ(decision.debugInfo().coin_selected_side, "LEFT");
  (void)decision.decide(roadMask(), {detection("Gold", 190.0f, 120.0f)});
  EXPECT_EQ(decision.debugInfo().coin_selected_side, "RIGHT");
  EXPECT_TRUE(decision.debugInfo().coins[0].selected_for_route);
}

TEST(LaneDecisionCoinShadowTest, ChainsReachableCoinsOnlyOnSelectedSide) {
  LaneDecision decision;
  decision.configure(makeConfig());

  (void)decision.decide(
      roadMask(), {detection("Gold", 130.0f, 110.0f),
                   detection("Gold", 128.0f, 170.0f),
                   detection("Gold", 190.0f, 130.0f)});
  const auto& debug = decision.debugInfo();

  ASSERT_EQ(debug.coins.size(), 3u);
  EXPECT_EQ(debug.coin_selected_side, "LEFT");
  EXPECT_EQ(debug.coin_route_target_count, 2);
  EXPECT_TRUE(debug.coins[0].route_targeted);
  EXPECT_TRUE(debug.coins[1].route_targeted);
  EXPECT_FALSE(debug.coins[2].route_targeted);
  EXPECT_TRUE(debug.coin_route_candidate_valid)
      << debug.coin_route_reject_reason;
}

}  // namespace
}  // namespace track_perception_cpp
