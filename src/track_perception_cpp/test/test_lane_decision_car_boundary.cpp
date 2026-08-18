#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

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
  config.branch_confirm_frames = 1;
  config.enable_encoder_branch_hold = false;
  config.enable_guideboard_branch_selection = false;
  config.fit_min_points = 4;
  config.fit_order = 2;
  config.branch_fit_order = 2;
  config.enable_fit_point_jump_filter = false;
  config.enable_fit_point_trend_filter = false;
  config.enable_centerline_kalman = false;
  config.enable_human_obstacle_stop = true;
  config.enable_finish_stop = true;

  config.enable_car_obstacle_avoidance = true;
  config.obstacle_min_confidence = 0.45f;
  config.car_avoidance_min_height_ratio = 0.08f;
  config.car_avoidance_min_height_px = 12;
  config.car_side_confirm_frames = 2;
  config.car_side_fit_downward_extension_px = 0;
  config.car_encoder_detour_counts = 9000;
  config.car_encoder_return_counts = 1000;
  config.car_rearm_clear_frames = 3;
  config.car_template_min_points = 6;
  config.car_template_fit_order = 2;
  config.car_template_weight = 1.0f;
  config.car_right_template_offsets = "0,0,0,0,0,0,0,0,0,0,0,0";
  config.car_left_template_offsets = "0,0,0,0,0,0,0,0,0,0,0,0";
  config.car_encoder_fault_hold_sec = 3.0;
  return config;
}

double wallNow() {
  using Clock = std::chrono::system_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

cv::Mat roadMask(int x0 = 80, int x1 = 240) {
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  mask(cv::Rect(x0, 0, x1 - x0, kHeight)).setTo(1);
  return mask;
}

cv::Mat fullRoadMask() {
  cv::Mat mask = cv::Mat::ones(kHeight, kWidth, CV_8UC1);
  return mask;
}

cv::Mat branchMask() {
  cv::Mat mask = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
  mask(cv::Rect(20, 0, 80, kHeight)).setTo(1);
  mask(cv::Rect(180, 0, 80, kHeight)).setTo(1);
  return mask;
}

Detection carOnSide(const std::string& side) {
  Detection det;
  det.class_name = "Car";
  det.confidence = 0.95f;
  det.bbox = side == "LEFT" ? cv::Rect2f(20.0f, 80.0f, 50.0f, 80.0f)
                             : cv::Rect2f(250.0f, 80.0f, 50.0f, 80.0f);
  det.center = cv::Point2f(det.bbox.x + det.bbox.width * 0.5f,
                          det.bbox.y + det.bbox.height * 0.5f);
  return det;
}

void setEncoder(LaneDecision* decision, int64_t count,
                double age_sec = 0.0) {
  decision->setEncoderCount(count, wallNow() - age_sec);
}

LaneState confirmAndTrigger(LaneDecision* decision, const std::string& side,
                            const cv::Mat& mask, int64_t encoder = 100) {
  const Detection det = carOnSide(side);
  setEncoder(decision, encoder);
  (void)decision->decide(mask, {det});
  setEncoder(decision, encoder);
  return decision->decide(mask, {det});
}

double meanPointX(const std::vector<cv::Point3f>& points) {
  double sum = 0.0;
  for (const auto& point : points) {
    sum += point.x;
  }
  return points.empty() ? 0.0 : sum / points.size();
}

TEST(LaneDecisionCarTemplateTest, NoCarLeavesOrdinaryQuadraticFitUnchanged) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = decision.decide(roadMask(), {});
  const auto& debug = decision.debugInfo();
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(debug.car_template_state, "IDLE");
  EXPECT_FALSE(debug.car_template_active);
  ASSERT_EQ(debug.fit_coeffs.size(), 3u);
  EXPECT_NEAR(meanPointX(debug.fit_points), 159.5, 1.0);
}

TEST(LaneDecisionCarTemplateTest, RightCarUsesAllLeftRoadBoundaries) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = confirmAndTrigger(&decision, "RIGHT", roadMask());
  const auto& debug = decision.debugInfo();
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(debug.car_template_state, "DETOUR");
  EXPECT_EQ(debug.car_template_side, "RIGHT");
  EXPECT_GE(debug.car_template_point_count, 6);
  EXPECT_EQ(debug.fit_order, 2);
  for (const auto& point : debug.fit_points) {
    EXPECT_NEAR(point.x, 80.0, 1.0);
  }
}

TEST(LaneDecisionCarTemplateTest, FarCarUsesHeightRatioEvenWhenBottomIsNearHorizon) {
  LaneDecision decision;
  decision.configure(makeConfig());
  Detection far_car = carOnSide("RIGHT");
  far_car.bbox.y = 0.0f;
  far_car.bbox.height = 24.0f;
  far_car.center = cv::Point2f(
      far_car.bbox.x + far_car.bbox.width * 0.5f,
      far_car.bbox.y + far_car.bbox.height * 0.5f);

  setEncoder(&decision, 100);
  (void)decision.decide(roadMask(), {far_car});
  setEncoder(&decision, 100);
  const LaneState state = decision.decide(roadMask(), {far_car});
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  EXPECT_EQ(decision.debugInfo().car_template_side, "RIGHT");
}

TEST(LaneDecisionCarTemplateTest, SameHeightFitPointRightOfCarMeansCarIsLeft) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = confirmAndTrigger(
      &decision, "LEFT", fullRoadMask());
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_template_side, "LEFT");
  EXPECT_EQ(decision.debugInfo().car_side_fit_relation, "RIGHT_OF_CAR");
  EXPECT_NEAR(decision.debugInfo().car_side_fit_y, 150.0, 1.0);
}

TEST(LaneDecisionCarTemplateTest, SameHeightFitPointLeftOfCarMeansCarIsRight) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = confirmAndTrigger(
      &decision, "RIGHT", fullRoadMask());
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_template_side, "RIGHT");
  EXPECT_EQ(decision.debugInfo().car_side_fit_relation, "LEFT_OF_CAR");
}

TEST(LaneDecisionCarTemplateTest, CarSideUsesLowestFitPointInDownwardExtension) {
  auto config = makeConfig();
  config.car_side_fit_downward_extension_px = 100;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = confirmAndTrigger(&decision, "LEFT", roadMask());
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_side_fit_relation, "RIGHT_OF_CAR");
  EXPECT_GT(decision.debugInfo().car_side_fit_y, 160.0f);
}

TEST(LaneDecisionCarTemplateTest, LeftCarUsesLeftBoundaryOffsets) {
  LaneDecision decision;
  decision.configure(makeConfig());

  const LaneState state = confirmAndTrigger(&decision, "LEFT", roadMask());
  const auto& debug = decision.debugInfo();
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(debug.car_template_state, "DETOUR");
  EXPECT_EQ(debug.car_template_side, "LEFT");
  for (const auto& point : debug.fit_points) {
    EXPECT_NEAR(point.x, 80.0, 1.0);
  }
}

TEST(LaneDecisionCarTemplateTest, LeftCarUsesDirectPerBandEmpiricalOffsets) {
  auto config = makeConfig();
  config.car_left_template_offsets = "100,100,100,100,100,100,100,100,100,100,100,100";
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = confirmAndTrigger(&decision, "LEFT", roadMask());
  const auto& debug = decision.debugInfo();
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(debug.car_template_state, "DETOUR");
  EXPECT_EQ(debug.car_template_side, "LEFT");
  ASSERT_EQ(debug.fit_points.size(), 12u);
  for (const auto& point : debug.fit_points) {
    EXPECT_NEAR(point.x, 180.0, 1.0);
  }
}

TEST(LaneDecisionCarTemplateTest, PerBandOffsetsMoveTemplateTowardRoadInterior) {
  auto config = makeConfig();
  config.car_right_template_offsets = "5,5,5,0,0,0,0,0,0,0,0,0";
  LaneDecision decision;
  decision.configure(config);

  (void)confirmAndTrigger(&decision, "RIGHT", roadMask());
  const auto& debug = decision.debugInfo();
  ASSERT_GE(debug.fit_points.size(), 6u);
  int offset_points = 0;
  int default_points = 0;
  for (const auto& point : debug.fit_points) {
    if (std::abs(point.x - 85.0f) < 1.0f) {
      ++offset_points;
    } else if (std::abs(point.x - 80.0f) < 1.0f) {
      ++default_points;
    }
  }
  EXPECT_EQ(offset_points, 3);
  EXPECT_GT(default_points, 0);
}

TEST(LaneDecisionCarTemplateTest, DoesNotStartWithoutEnoughTemplatePoints) {
  auto config = makeConfig();
  config.car_template_min_points = 20;
  LaneDecision decision;
  decision.configure(config);

  const LaneState state = confirmAndTrigger(&decision, "RIGHT", roadMask());
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_template_state, "IDLE");
  EXPECT_FALSE(decision.debugInfo().car_template_cache_active);
  EXPECT_NEAR(meanPointX(decision.debugInfo().fit_points), 159.5, 1.0);
}

TEST(LaneDecisionCarTemplateTest, DoesNotStartWithStaleEncoderFeedback) {
  auto config = makeConfig();
  config.encoder_feedback_timeout_sec = 0.01;
  LaneDecision decision;
  decision.configure(config);

  setEncoder(&decision, 100);
  (void)decision.decide(roadMask(), {carOnSide("RIGHT")});
  setEncoder(&decision, 100, 1.0);
  const LaneState state = decision.decide(roadMask(), {carOnSide("RIGHT")});
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_side, "RIGHT");
  EXPECT_EQ(decision.debugInfo().car_template_state, "IDLE");
  EXPECT_FALSE(decision.debugInfo().car_template_cache_active);
}

TEST(LaneDecisionCarTemplateTest, ActiveDetourKeepsCachedTemplateWhenBandsDisappear) {
  LaneDecision decision;
  decision.configure(makeConfig());
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask());

  setEncoder(&decision, 200);
  const LaneState state = decision.decide(
      cv::Mat::zeros(kHeight, kWidth, CV_8UC1), {});
  const auto& debug = decision.debugInfo();
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(debug.car_template_state, "DETOUR");
  EXPECT_TRUE(debug.car_template_cache_active);
  EXPECT_NEAR(meanPointX(debug.fit_points), 80.0, 1.0);
}

TEST(LaneDecisionCarTemplateTest, EncoderTransitionsAtExactBoundariesAndBlendsReturn) {
  LaneDecision decision;
  decision.configure(makeConfig());
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask(), 100);

  setEncoder(&decision, 9099);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");

  setEncoder(&decision, 9100);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "RETURN");
  EXPECT_NEAR(decision.debugInfo().car_encoder_return_progress, 0.0, 1e-6);

  setEncoder(&decision, 9600);
  const LaneState halfway = decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "RETURN");
  EXPECT_NEAR(decision.debugInfo().car_encoder_return_progress, 0.5, 1e-6);
  EXPECT_NEAR(halfway.offset_y09, (80.0 + 159.5) * 0.5 / 160.0 - 1.0,
              0.02);

  setEncoder(&decision, 10100);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "WAIT_CLEAR");
  EXPECT_FALSE(decision.debugInfo().car_template_active);
  EXPECT_NEAR(meanPointX(decision.debugInfo().fit_points), 159.5, 1.0);
}

TEST(LaneDecisionCarTemplateTest, WaitClearRequiresThreeConsecutiveNoCarFrames) {
  LaneDecision decision;
  decision.configure(makeConfig());
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask(), 100);
  setEncoder(&decision, 10100);
  (void)decision.decide(roadMask(), {carOnSide("RIGHT")});
  ASSERT_EQ(decision.debugInfo().car_template_state, "WAIT_CLEAR");

  (void)decision.decide(roadMask(), {carOnSide("RIGHT")});
  EXPECT_EQ(decision.debugInfo().car_rearm_clear_count, 0);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_rearm_clear_count, 1);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_rearm_clear_count, 2);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "IDLE");
}

TEST(LaneDecisionCarTemplateTest, SideAndTemplateStayLatchedAfterDetectionLoss) {
  LaneDecision decision;
  decision.configure(makeConfig());
  (void)confirmAndTrigger(&decision, "LEFT", roadMask());

  setEncoder(&decision, 200);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  EXPECT_EQ(decision.debugInfo().car_template_side, "LEFT");
  EXPECT_NEAR(meanPointX(decision.debugInfo().fit_points), 80.0, 1.0);
}

TEST(LaneDecisionCarTemplateTest, OppositeFitPointEvidenceDoesNotChangeLatchedSide) {
  LaneDecision decision;
  decision.configure(makeConfig());
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask());

  setEncoder(&decision, 200);
  (void)decision.decide(roadMask(), {carOnSide("LEFT")});
  EXPECT_EQ(decision.debugInfo().car_side_candidate, "LEFT");
  EXPECT_EQ(decision.debugInfo().car_template_side, "RIGHT");
  EXPECT_NEAR(meanPointX(decision.debugInfo().fit_points), 80.0, 1.0);
}

TEST(LaneDecisionCarTemplateTest, ShortEncoderFaultHoldsTemplate) {
  auto config = makeConfig();
  config.encoder_feedback_timeout_sec = 0.01;
  config.car_encoder_fault_hold_sec = 3.0;
  LaneDecision decision;
  decision.configure(config);
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask());

  setEncoder(&decision, 100, 1.0);
  const LaneState state = decision.decide(roadMask(), {});
  EXPECT_TRUE(state.is_valid);
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  EXPECT_NEAR(meanPointX(decision.debugInfo().fit_points), 80.0, 1.0);
}

TEST(LaneDecisionCarTemplateTest, EncoderRollbackUsesFaultExitInsteadOfAbsoluteDelta) {
  auto config = makeConfig();
  config.car_encoder_fault_hold_sec = 0.0;
  LaneDecision decision;
  decision.configure(config);
  (void)confirmAndTrigger(&decision, "RIGHT", roadMask(), 100);

  setEncoder(&decision, 99);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  EXPECT_EQ(decision.debugInfo().car_encoder_delta, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(550));
  setEncoder(&decision, 99);
  (void)decision.decide(roadMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "WAIT_CLEAR");
}

TEST(LaneDecisionCarTemplateTest, BranchEncoderContinuesWhileCarOverridesOutput) {
  auto config = makeConfig();
  config.branch_detect_min_bands = 1;
  config.enable_encoder_branch_hold = true;
  config.encoder_hold_counts = 5;
  config.encoder_hold_right_counts = 5;
  LaneDecision decision;
  decision.configure(config);

  setEncoder(&decision, 100);
  (void)decision.decide(branchMask(), {carOnSide("RIGHT")});
  setEncoder(&decision, 100);
  (void)decision.decide(branchMask(), {carOnSide("RIGHT")});
  ASSERT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  ASSERT_TRUE(decision.debugInfo().encoder_hold);

  setEncoder(&decision, 106);
  (void)decision.decide(branchMask(), {});
  EXPECT_EQ(decision.debugInfo().car_template_state, "DETOUR");
  EXPECT_FALSE(decision.debugInfo().encoder_hold);
  EXPECT_TRUE(decision.debugInfo().car_template_active);
}

TEST(LaneDecisionCarTemplateTest, HumanAndFinishProtectionRemainIndependent) {
  auto config = makeConfig();
  config.human_stop_confirm_frames = 1;
  config.human_stop_raw_area_ratio = 0.001f;
  LaneDecision decision;
  decision.configure(config);

  Detection human;
  human.class_name = "Human";
  human.confidence = 0.95f;
  human.bbox = cv::Rect2f(140.0f, 80.0f, 40.0f, 100.0f);
  human.center = cv::Point2f(160.0f, 130.0f);
  const LaneState state = decision.decide(roadMask(), {human});
  EXPECT_EQ(state.task_state, "OBSTACLE_STOP");
  EXPECT_EQ(decision.debugInfo().car_template_state, "IDLE");

  const LaneState cleared_state = decision.decide(roadMask(), {});
  EXPECT_EQ(cleared_state.task_state, "CLEAR");
  EXPECT_FALSE(decision.debugInfo().human_stop_active);
  EXPECT_EQ(decision.debugInfo().human_state, "NONE");
}

TEST(LaneDecisionCarTemplateTest, OnlyNearestHumanControlsAvoidance) {
  auto config = makeConfig();
  config.human_stop_confirm_frames = 1;
  config.human_stop_raw_area_ratio = 0.001f;
  LaneDecision decision;
  decision.configure(config);

  Detection far_human;
  far_human.class_name = "Human";
  far_human.confidence = 0.95f;
  far_human.bbox = cv::Rect2f(140.0f, 20.0f, 40.0f, 80.0f);
  far_human.center = cv::Point2f(160.0f, 60.0f);

  Detection near_human;
  near_human.class_name = "Human";
  near_human.confidence = 0.95f;
  near_human.bbox = cv::Rect2f(240.0f, 120.0f, 20.0f, 80.0f);
  near_human.center = cv::Point2f(250.0f, 160.0f);

  const LaneState state = decision.decide(roadMask(), {far_human, near_human});
  ASSERT_EQ(decision.debugInfo().humans.size(), 2u);
  EXPECT_EQ(decision.debugInfo().human_candidate_count, 2);
  EXPECT_EQ(decision.debugInfo().human_valid_count, 1);
  EXPECT_EQ(state.task_state, "CLEAR");
  EXPECT_EQ(decision.debugInfo().human_state, "PASSABLE");
  EXPECT_FALSE(decision.debugInfo().humans[0].active);
  EXPECT_TRUE(decision.debugInfo().humans[1].active);
}

}  // namespace
}  // namespace track_perception_cpp
