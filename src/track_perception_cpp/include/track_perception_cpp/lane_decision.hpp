#pragma once

#include <array>
#include <optional>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "track_perception_cpp/types.hpp"

namespace track_perception_cpp {

struct LaneDecisionConfig {
  bool enable_segment_branch_logic{true};
  int band_count{13};
  float band_y_min_ratio{0.65f};
  float band_y_max_ratio{1.0f};
  float band_height_ratio{0.02f};
  int min_segment_width_px{25};
  int min_segment_gap_px{40};
  int min_pixels_per_band{80};

  int branch_detect_min_bands{2};
  int branch_confirm_frames{2};
  float branch_detect_far_band_ratio{0.7f};
  std::string outer_side{"left"};
  bool enable_guideboard_branch_selection{true};
  float guideboard_detect_y0_ratio{0.2f};
  float guideboard_detect_y1_ratio{0.7f};
  bool guideboard_require_hint{false};
  std::string guideboard_unknown_branch{"left"};
  double guideboard_hint_wait_timeout_sec{0.20};
  bool enable_encoder_branch_hold{true};
  int64_t encoder_hold_counts{9000};
  int64_t encoder_hold_right_counts{20000};
  double encoder_feedback_timeout_sec{0.30};

  int fit_min_points{5};
  int fit_order{2};
  int branch_fit_order{2};
  bool enable_fit_point_jump_filter{false};
  float max_fit_point_dx_ratio{0.22f};
  float max_fit_point_dx_px{140.0f};
  bool enable_fit_point_trend_filter{false};
  float fit_point_trend_residual_ratio{0.12f};
  float fit_point_trend_residual_px{80.0f};
  float fit_point_trend_slope_delta{0.65f};
  int fit_point_trend_min_points{6};
  float fit_point_trend_min_keep_ratio{0.75f};
  float offset_y07_ratio{0.70f};
  float offset_y08_ratio{0.80f};
  float offset_y09_ratio{0.90f};
  float heading_y_ratio{0.75f};
  // Heading is estimated from independent local line fits around the
  // near/mid/far image positions.  Weights are normalized at runtime.
  float heading_near_ratio{0.81f};
  float heading_mid_ratio{0.68f};
  float heading_far_ratio{0.55f};
  float heading_near_weight{0.10f};
  float heading_mid_weight{0.50f};
  float heading_far_weight{0.40f};
  float heading_local_window_half_ratio{0.06f};
  int heading_local_min_points{3};
  bool enable_centerline_kalman{false};
  float centerline_kalman_measurement_noise{0.0036f};
  float centerline_kalman_idle_process_noise{0.0001f};
  float centerline_kalman_motion_process_noise{0.0008f};
  float centerline_kalman_turn_process_noise{0.0008f};
  float centerline_kalman_initial_variance{0.01f};
  float centerline_kalman_encoder_reference_counts{50.0f};
  float centerline_kalman_reset_innovation{0.25f};
  double centerline_kalman_state_timeout_sec{0.30};
  double centerline_kalman_steering_timeout_sec{0.25};

  bool enable_left_boundary_template_line{false};
  std::string left_boundary_template_side{"left"};
  std::string left_boundary_template_offsets;
  int left_boundary_template_min_points{6};
  float left_boundary_template_weight{1.0f};
  std::string right_boundary_template_offsets;
  int right_boundary_template_min_points{6};
  float right_boundary_template_weight{1.0f};

  bool enable_obstacle_avoidance{false};
  std::unordered_set<std::string> obstacle_labels{"Car"};
  float obstacle_min_confidence{0.45f};
  float obstacle_x_margin_px{25.0f};
  float obstacle_y_margin_px{20.0f};
  float obstacle_min_bottom_y_ratio{0.30f};
  bool enable_human_obstacle_stop{true};
  float human_horizontal_expand_px{25.0f};
  float human_horizontal_expand_width_ratio{0.50f};
  int human_line_sample_count{5};
  float human_stop_raw_area_ratio{0.004f};
  int human_stop_confirm_frames{2};
  int human_clear_confirm_frames{2};

  bool enable_car_right_boundary_filter{true};
  float car_boundary_x_margin_px{0.0f};
  float car_boundary_y_margin_px{0.0f};
  float car_boundary_smoothing_alpha{0.5f};
  int car_boundary_lost_frames{3};
  double car_fit_hold_timeout_sec{0.20};
  bool enable_car_point_push_avoidance{false};
  float car_push_expand_left_px{40.0f};
  float car_push_expand_bottom_px{20.0f};
  float car_push_expand_right_px{0.0f};
  float car_push_expand_top_px{0.0f};
  float car_push_clearance_px{3.0f};

  bool enable_finish_stop{true};
  float finish_stop_min_confidence{0.45f};
  float finish_stop_arm_y_ratio{0.70f};
  int finish_stop_lost_frames{3};

};

struct LaneSegmentDebug {
  int x0{0};
  int x1{0};
  int center_x{0};
  int pixel_count{0};
  bool selected{false};
  bool virtual_segment{false};
  bool obstacle_cut{false};
};

struct LaneBandDebug {
  int y0{0};
  int y1{0};
  std::vector<LaneSegmentDebug> segments;
  int selected_center_x{-1};
};

struct LaneObstacleDebug {
  cv::Rect2f rect;
  std::string label;
};

struct LaneHumanDebug {
  cv::Rect2f raw_bbox;
  cv::Rect2f expanded_bbox;
  std::vector<cv::Point2f> fit_sample_points;
  float raw_area_ratio{0.0f};
  bool fit_available{false};
  bool line_intersects{false};
  bool passable{false};
  bool stop_candidate{false};
};

struct LaneDebugInfo {
  std::vector<LaneBandDebug> bands;
  std::vector<LaneObstacleDebug> obstacle_zones;
  std::vector<cv::Point3f> raw_points;
  std::vector<cv::Point3f> fit_points;
  std::vector<cv::Point3f> removed_fit_points;
  std::vector<cv::Point3f> pushed_fit_points;
  std::vector<double> fit_coeffs;
  std::vector<LaneHumanDebug> humans;
  bool human_passable{false};
  bool human_line_intersects{false};
  bool human_stop_candidate{false};
  bool human_stop_active{false};
  bool human_clear_confirming{false};
  float human_raw_area_ratio{0.0f};
  std::string human_state{"NONE"};
  int human_stop_confirm_count{0};
  int human_clear_confirm_count{0};
  int human_count_at_stop{0};
  int human_valid_count{0};
  bool car_boundary_active{false};
  float car_left_x{-1.0f};
  int car_filtered_point_count{0};
  int car_boundary_lost_count{0};
  bool car_push_active{false};
  float car_push_target_x{-1.0f};
  float car_push_expand_bottom_y{-1.0f};
  int car_pushed_point_count{0};
  int car_deleted_point_count{0};
  cv::Rect2f car_bbox;
  cv::Rect2f car_expanded_bbox;
  bool fit_hold_active{false};
  double fit_hold_age{0.0};
  int fit_order{0};
  bool branch_detected{false};
  int branch_score{0};
  bool guideboard_seen{false};
  int guideboard_count{0};
  int guideboard_roi_count{0};
  float guideboard_best_confidence{0.0f};
  cv::Point2f guideboard_best_center;
  bool guideboard_hint_valid{false};
  bool guideboard_waiting_for_hint{false};
  double guideboard_hint_wait_elapsed{0.0};
  bool guideboard_seen_latched{false};
  bool branch_event_armed{true};
  bool branch_event_rearmed{false};
  bool branch_lock_event{false};
  bool branch_lock_guideboard{false};
  uint64_t branch_event_id{0};
  std::string branch_decision_source;
  bool encoder_hold{false};
  std::string encoder_hold_side;
  int64_t encoder_count{0};
  int64_t encoder_hold_delta{0};
  int64_t encoder_hold_target{0};
  bool encoder_feedback_valid{false};
  double encoder_feedback_age{0.0};
  bool centerline_kalman_enabled{false};
  int centerline_kalman_point_count{0};
  int64_t centerline_kalman_encoder_delta{0};
  double centerline_kalman_motion_ratio{0.0};
  double centerline_kalman_process_noise{0.0};
  double centerline_kalman_steering{0.0};
  int centerline_kalman_reset_count{0};
  bool left_boundary_template_active{false};
  int left_boundary_template_points{0};
  std::string left_boundary_template_reason;
  std::string boundary_template_side;
  int raw_point_count{0};
  int fit_point_count{0};
  int segment_count{0};
  float offset_y07{0.0f};
  float offset_y08{0.0f};
  float offset_y09{0.0f};
  float raw_offset_y07{0.0f};
  float raw_offset_y08{0.0f};
  float raw_offset_y09{0.0f};
  int fit_y_min{0};
  int fit_y_max{0};
  int fit_y_span{0};
  int image_width{0};
};

class LaneDecision {
 public:
  void configure(const LaneDecisionConfig& config);
  void setGuideboardBranchHint(const std::string& branch, bool valid,
                               const std::string& decision_source = "guideboard_hint");
  void setEncoderCount(int64_t count, double timestamp);
  void setSteeringCommand(double steering_ratio, double timestamp);
  void setHeadingWeights(double near_weight, double mid_weight, double far_weight);
  LaneState decide(const cv::Mat& seg_map, const std::vector<Detection>& detections);
  const LaneDebugInfo& debugInfo() const { return debug_info_; }
  bool branchEventArmed() const { return branch_event_armed_; }
  uint64_t branchEventId() const { return branch_event_id_; }

 private:
  struct Segment {
    int x0{0};
    int x1{0};
    double width{0.0};
    double center_x{0.0};
    int pixel_count{0};
    bool obstacle_cut{false};
    bool virtual_segment{false};
  };

  struct Band {
    int index{0};
    int y0{0};
    int y1{0};
    double y_center{0.0};
    std::vector<Segment> segments;
    std::optional<Segment> selected_segment;
  };

  struct ObstacleZone {
    cv::Rect2f rect;
    std::string label;
  };

  struct CenterlineKalmanState {
    bool initialized{false};
    double x_normalized{0.0};
    double variance{0.0};
    double last_update_time{0.0};
  };

  enum class RoadClass {
    Normal,
    Branch,
    LowConfidence,
  };

  std::vector<Band> buildBands(const cv::Mat& road_mask, const std::vector<Detection>& detections);
  std::vector<ObstacleZone> getActiveObstacleZones(const std::vector<Detection>& detections,
                                                   int image_width, int image_height) const;
  std::vector<Segment> extractSegmentsInBand(const cv::Mat& band_mask) const;
  std::vector<Segment> applyObstacleExclusionToSegments(const std::vector<Segment>& segments,
                                                        int band_y0, int band_y1,
                                                        const std::vector<ObstacleZone>& zones) const;
  std::pair<bool, int> detectBranchFromBands(const std::vector<Band>& bands) const;
  std::optional<Segment> chooseTargetSegment(const Band& band, const std::string& side) const;
  RoadClass classifyRoadGeometry(const std::vector<Band>& bands,
                                 const std::vector<cv::Point3f>& raw_points) const;
  std::string roadClassName(RoadClass road_class) const;
  double calculateLaneConfidence(const std::vector<cv::Point3f>& fit_points,
                                 const std::vector<Band>& bands) const;
  bool isBoundaryTemplateReady(const std::string& side) const;
  int boundaryTemplateMinPoints(const std::string& side) const;
  bool shouldUseBoundaryTemplate(const std::string& side, bool template_trigger);
  std::vector<cv::Point3f> collectBoundaryTemplatePoints(std::vector<Band>& bands,
                                                         int image_width,
                                                         const std::string& side,
                                                         int min_band_index = 0);
  std::vector<cv::Point3f> collectCenterlinePoints(std::vector<Band>& bands, bool branch_locked,
                                                   const std::string& side,
                                                   std::optional<double> last_center_x,
                                                   int image_width);
  std::vector<cv::Point3f> filterCenterlinePoints(const std::vector<cv::Point3f>& points,
                                                  int image_width,
                                                  std::optional<double> last_center_x) const;
  std::vector<cv::Point3f> filterCenterlinePointsKalman(
      const std::vector<cv::Point3f>& points, const std::vector<Band>& bands,
      int image_width, int image_height, double timestamp,
      const std::string& target_side, bool template_active);
  void resetCenterlineKalman();
  void updateCarBoundaryState(const std::vector<Detection>& detections, int image_width,
                              int image_height);
  std::vector<cv::Point3f> filterCarRightBoundaryPoints(
      const std::vector<cv::Point3f>& points,
      std::vector<cv::Point3f>* removed_points) const;
  std::vector<cv::Point3f> applyCarPointPushAvoidance(
      const std::vector<cv::Point3f>& points, int image_width, int image_height,
      std::vector<cv::Point3f>* pushed_points,
      std::vector<cv::Point3f>* removed_points);
  bool fitCenterlineAndComputeGeometry(const std::vector<cv::Point3f>& points, int h,
                                       int fit_order, std::vector<double>* coeffs,
                                       double* heading_error, double* curvature) const;
  double offsetAtY(const std::vector<double>& coeffs, double y, int image_width) const;
  double fallbackCenterOffset(const cv::Mat& seg_map) const;
  bool checkGuideboardInFarRoi(const std::vector<Detection>& detections, int h) const;
  void updateFinishStopState(const std::vector<Detection>& detections, int image_height);
  void updateHumanStopState(const std::vector<Detection>& detections, int image_width,
                            int image_height, const std::vector<double>& fit_coeffs,
                            bool fit_valid);
  std::string taskState() const;
  void populateDebugInfo(const std::vector<Band>& bands, const std::vector<ObstacleZone>& zones,
                         const std::vector<cv::Point3f>& raw_points,
                         const std::vector<cv::Point3f>& fit_points,
                         const std::vector<cv::Point3f>& removed_fit_points,
                         const std::vector<cv::Point3f>& pushed_fit_points,
                         int image_width, int image_height,
                         const std::vector<double>& fit_coeffs);

  LaneDecisionConfig cfg_;
  LaneDebugInfo debug_info_;

  std::array<double, 3> last_offsets_{{0.0, 0.0, 0.0}};
  std::vector<CenterlineKalmanState> centerline_kalman_states_;
  bool centerline_kalman_context_valid_{false};
  int centerline_kalman_image_width_{0};
  int centerline_kalman_image_height_{0};
  bool centerline_kalman_branch_locked_{false};
  bool centerline_kalman_template_active_{false};
  std::string centerline_kalman_target_side_;
  bool centerline_kalman_encoder_baseline_valid_{false};
  int64_t centerline_kalman_last_encoder_count_{0};
  double latest_steering_command_{0.0};
  double last_steering_command_time_{0.0};
  bool has_steering_command_{false};
  bool branch_locked_{false};
  std::string locked_branch_side_{"left"};
  std::string guideboard_branch_hint_{"left"};
  std::string guideboard_branch_hint_source_{"guideboard_hint"};
  bool guideboard_branch_hint_valid_{false};
  double guideboard_hint_wait_start_sec_{0.0};
  double lock_start_time_{0.0};
  int branch_confirm_count_{0};
  bool guideboard_seen_latched_{false};
  bool branch_event_armed_{true};
  bool branch_event_rearmed_{false};
  int branch_clear_count_{0};
  uint64_t branch_event_id_{1};
  bool has_encoder_count_{false};
  int64_t latest_encoder_count_{0};
  double last_encoder_update_sec_{0.0};
  bool encoder_hold_baseline_valid_{false};
  int64_t encoder_hold_start_count_{0};
  bool encoder_hold_active_{false};
  int64_t encoder_hold_delta_{0};
  int64_t encoder_hold_target_{0};
  std::vector<double> left_boundary_template_offsets_;
  std::vector<double> right_boundary_template_offsets_;
  bool finish_stop_active_{false};
  bool human_stop_active_{false};
  std::string finish_stop_state_{"CLEAR"};
  std::string human_state_{"NONE"};
  int finish_stop_lost_count_{0};
  int human_stop_confirm_count_{0};
  int human_clear_confirm_count_{0};
  int human_count_at_stop_{0};

  bool car_boundary_active_{false};
  double car_left_x_{-1.0};
  double car_right_x_{0.0};
  double car_top_y_{0.0};
  double car_bottom_y_{0.0};
  int car_boundary_lost_count_{0};
  bool fit_hold_active_{false};
  double fit_hold_age_{0.0};
  bool has_last_valid_fit_{false};
  std::vector<double> last_valid_fit_coeffs_;
  std::array<double, 3> last_valid_offsets_{{0.0, 0.0, 0.0}};
  std::array<double, 3> last_valid_raw_offsets_{{0.0, 0.0, 0.0}};
  double last_valid_fit_heading_{0.0};
  double last_valid_fit_curvature_{0.0};
  double last_valid_fit_confidence_{0.0};
  double last_valid_fit_time_{0.0};
  std::string last_valid_road_state_{"NORMAL"};
};

}  // namespace track_perception_cpp
