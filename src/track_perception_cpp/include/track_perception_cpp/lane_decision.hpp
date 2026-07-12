#pragma once

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
  std::string guideboard_branch{"right"};
  float guideboard_detect_y0_ratio{0.2f};
  float guideboard_detect_y1_ratio{0.7f};
  bool enable_encoder_branch_hold{true};
  int64_t encoder_hold_counts{5000};
  double encoder_feedback_timeout_sec{0.30};

  int fit_min_points{5};
  int fit_order{2};
  int branch_fit_order{2};
  bool enable_fit_point_jump_filter{true};
  float max_fit_point_dx_ratio{0.22f};
  float max_fit_point_dx_px{140.0f};
  bool enable_fit_point_trend_filter{true};
  float fit_point_trend_residual_ratio{0.12f};
  float fit_point_trend_residual_px{80.0f};
  float fit_point_trend_slope_delta{0.65f};
  int fit_point_trend_min_points{6};
  float fit_point_trend_min_keep_ratio{0.75f};
  float lookahead_y_ratio{0.75f};
  bool use_heading_term{true};
  float heading_weight{0.10f};
  float near_offset_weight{0.90f};
  float max_offset_jump{2.0f};
  float offset_smoothing_alpha{0.35f};

  bool enable_left_boundary_template_line{false};
  std::string left_boundary_template_side{"left"};
  std::string left_boundary_template_offsets;
  int left_boundary_template_min_points{6};
  float left_boundary_template_weight{1.0f};
  std::string right_boundary_template_offsets;
  int right_boundary_template_min_points{6};
  float right_boundary_template_weight{1.0f};

  bool enable_obstacle_avoidance{false};
  std::unordered_set<std::string> obstacle_labels{"Human", "Car"};
  std::unordered_set<std::string> obstacle_stop_labels{"Human"};
  float obstacle_min_confidence{0.45f};
  float obstacle_x_margin_px{25.0f};
  float obstacle_y_margin_px{20.0f};
  float obstacle_min_bottom_y_ratio{0.30f};
  bool enable_obstacle_stop{true};
  float obstacle_stop_bottom_y_ratio{0.82f};
  int obstacle_stop_confirm_frames{2};
  int obstacle_stop_lost_frames{3};

  bool enable_label_fit_points{true};
  std::unordered_set<std::string> fit_point_labels{"Go"};
  float fit_point_min_confidence{0.45f};
  float fit_point_y0_ratio{0.45f};
  float fit_point_y1_ratio{1.0f};
  float fit_point_weight{1.0f};

  bool enable_start_boost_trigger{true};
  std::unordered_set<std::string> start_boost_labels{"Go", "Gate"};
  float start_boost_min_confidence{0.45f};
  float start_boost_y0_ratio{0.0f};
  float start_boost_y1_ratio{1.0f};
  int start_boost_lost_frames{3};

  bool enable_traffic_light_stop{true};
  float traffic_light_min_confidence{0.45f};
  float zebra_min_confidence{0.45f};
  float zebra_stop_y_ratio{0.70f};
  int green_light_confirm_frames{1};
  int red_light_confirm_frames{1};

  bool enable_finish_stop{true};
  float finish_stop_min_confidence{0.45f};
  float finish_stop_arm_y_ratio{0.70f};
  int finish_stop_lost_frames{3};

  bool enable_branch_event_log{false};
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

struct LaneDebugInfo {
  std::vector<LaneBandDebug> bands;
  std::vector<LaneObstacleDebug> obstacle_zones;
  std::vector<cv::Point3f> raw_points;
  std::vector<cv::Point3f> fit_points;
  std::vector<double> fit_coeffs;
  bool branch_detected{false};
  int branch_score{0};
  bool guideboard_seen{false};
  int guideboard_count{0};
  int guideboard_roi_count{0};
  float guideboard_best_confidence{0.0f};
  cv::Point2f guideboard_best_center;
  bool encoder_hold{false};
  std::string encoder_hold_side;
  int64_t encoder_count{0};
  int64_t encoder_hold_delta{0};
  int64_t encoder_hold_target{0};
  bool encoder_feedback_valid{false};
  double encoder_feedback_age{0.0};
  bool left_boundary_template_active{false};
  int left_boundary_template_points{0};
  std::string left_boundary_template_reason;
  std::string boundary_template_side;
  int raw_point_count{0};
  int fit_point_count{0};
  int segment_count{0};
  float bottom_offset{0.0f};
  float raw_control_offset{0.0f};
  float lookahead_x{0.0f};
  float lookahead_y{0.0f};
  int image_width{0};
};

class LaneDecision {
 public:
  void configure(const LaneDecisionConfig& config);
  void setGuideboardBranchHint(const std::string& branch, bool valid);
  void setEncoderCount(int64_t count, double timestamp);
  LaneState decide(const cv::Mat& seg_map, const std::vector<Detection>& detections);
  const LaneDebugInfo& debugInfo() const { return debug_info_; }

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
  void appendDetectionFitPoints(std::vector<cv::Point3f>& points,
                                const std::vector<Detection>& detections,
                                int image_height) const;
  bool fitCenterlineAndComputeOffset(const std::vector<cv::Point3f>& points, int h, int w,
                                     int fit_order, double* final_offset,
                                     std::vector<double>* coeffs, double* lateral_offset,
                                     double* heading_error, double* curvature) const;
  double smoothOffset(double raw_offset);
  double fallbackCenterOffset(const cv::Mat& seg_map) const;
  bool checkGuideboardInFarRoi(const std::vector<Detection>& detections, int h) const;
  void updateTrafficLightStopState(const std::vector<Detection>& detections, int image_height);
  void updateFinishStopState(const std::vector<Detection>& detections, int image_height);
  void updateObstacleStopState(const std::vector<Detection>& detections, int image_height);
  void updateStartBoostState(const std::vector<Detection>& detections, int image_height);
  std::string taskState() const;
  void populateDebugInfo(const std::vector<Band>& bands, const std::vector<ObstacleZone>& zones,
                         const std::vector<cv::Point3f>& raw_points,
                         const std::vector<cv::Point3f>& fit_points,
                         const std::vector<double>& fit_coeffs);

  LaneDecisionConfig cfg_;
  LaneDebugInfo debug_info_;

  double last_offset_{0.0};
  bool branch_locked_{false};
  std::string locked_branch_side_{"left"};
  std::string guideboard_branch_hint_{"left"};
  bool guideboard_branch_hint_valid_{false};
  double lock_start_time_{0.0};
  int branch_confirm_count_{0};
  bool has_encoder_count_{false};
  int64_t latest_encoder_count_{0};
  double last_encoder_update_sec_{0.0};
  bool encoder_hold_baseline_valid_{false};
  int64_t encoder_hold_start_count_{0};
  bool encoder_hold_active_{false};
  int64_t encoder_hold_delta_{0};
  std::vector<double> left_boundary_template_offsets_;
  std::vector<double> right_boundary_template_offsets_;
  bool traffic_stop_active_{false};
  bool finish_stop_active_{false};
  bool obstacle_stop_active_{false};
  bool start_boost_active_{false};
  bool start_boost_used_{false};
  bool stop_request_active_{false};
  std::string traffic_light_state_{"CLEAR"};
  std::string finish_stop_state_{"CLEAR"};
  std::string obstacle_stop_state_{"CLEAR"};
  int finish_stop_lost_count_{0};
  int obstacle_stop_confirm_count_{0};
  int obstacle_stop_lost_count_{0};
  int start_boost_lost_count_{0};
  int red_light_confirm_count_{0};
  int green_light_confirm_count_{0};
};

}  // namespace track_perception_cpp
