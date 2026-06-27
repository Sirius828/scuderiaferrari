#pragma once

#include <optional>
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
  float branch_detect_far_band_ratio{0.7f};
  std::string outer_side{"left"};
  bool enable_guideboard_branch_selection{true};
  std::string guideboard_branch{"right"};
  float guideboard_detect_y0_ratio{0.2f};
  float guideboard_detect_y1_ratio{0.7f};
  bool enable_continuity_branch_selection{false};
  float branch_continuity_max_dx_ratio{0.35f};
  float branch_continuity_near_band_ratio{0.5f};
  bool enable_locked_path_continuity{true};
  float locked_path_continuity_after_time{0.5f};
  float locked_path_continuity_max_dx_ratio{0.28f};
  float branch_lock_time{2.0f};
  float min_branch_lock_time{0.8f};
  int exit_single_path_confirm_frames{5};
  float exit_single_path_min_ratio{0.8f};

  bool enable_merge_wide_segment_logic{true};
  float merge_wide_segment_ratio{1.35f};
  int merge_wide_min_bands{2};
  int merge_wide_confirm_frames{2};
  int merge_wide_release_frames{4};
  float merge_wide_lane_width_alpha{0.2f};

  int fit_min_points{5};
  int fit_order{2};
  int branch_fit_order{2};
  bool enable_fit_point_jump_filter{true};
  float max_fit_point_dx_ratio{0.22f};
  float max_fit_point_dx_px{140.0f};
  bool enable_branch_bottom_anchor{true};
  float branch_bottom_anchor_x_ratio{0.5f};
  float branch_bottom_anchor_y_ratio{0.98f};
  float branch_bottom_anchor_weight{0.6f};
  float lookahead_y_ratio{0.75f};
  bool use_heading_term{true};
  float heading_weight{0.10f};
  float near_offset_weight{0.90f};
  float max_offset_jump{2.0f};
  float offset_smoothing_alpha{0.35f};

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
  std::vector<cv::Point3f> fit_points;
  std::vector<double> fit_coeffs;
  bool branch_detected{false};
  int branch_score{0};
  bool guideboard_seen{false};
  int guideboard_count{0};
  int guideboard_roi_count{0};
  float guideboard_best_confidence{0.0f};
  cv::Point2f guideboard_best_center;
  bool branch_locked{false};
  std::string locked_branch_side;
  int raw_point_count{0};
  int fit_point_count{0};
  int segment_count{0};
};

class LaneDecision {
 public:
  void configure(const LaneDecisionConfig& config);
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

  std::vector<Band> buildBands(const cv::Mat& road_mask, const std::vector<Detection>& detections);
  std::vector<ObstacleZone> getActiveObstacleZones(const std::vector<Detection>& detections,
                                                   int image_width, int image_height) const;
  std::vector<Segment> extractSegmentsInBand(const cv::Mat& band_mask) const;
  std::vector<Segment> applyObstacleExclusionToSegments(const std::vector<Segment>& segments,
                                                        int band_y0, int band_y1,
                                                        const std::vector<ObstacleZone>& zones) const;
  std::pair<bool, int> detectBranchFromBands(const std::vector<Band>& bands) const;
  std::optional<std::string> chooseBranchSideByContinuity(const std::vector<Band>& bands,
                                                          int image_width, double last_center_x) const;
  std::optional<Segment> chooseTargetSegment(const Band& band, const std::string& side) const;
  std::optional<Segment> chooseLockedSegmentByContinuity(const Band& band, int image_width,
                                                         double last_center_x) const;
  bool shouldUseLockedPathContinuity(double now) const;
  std::string currentRoadState() const;
  double calculateLaneConfidence(const std::vector<cv::Point3f>& fit_points,
                                 const std::vector<Band>& bands) const;
  void updateMergeWideState(const std::vector<Band>& bands, double last_center_x);
  std::optional<Segment> chooseMergeWideSegment(const Band& band, std::optional<double> last_center_x);
  std::vector<cv::Point3f> collectCenterlinePoints(std::vector<Band>& bands, bool branch_locked,
                                                   const std::string& side,
                                                   std::optional<double> last_center_x,
                                                   int image_width, double now);
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
                         const std::vector<cv::Point3f>& fit_points,
                         const std::vector<double>& fit_coeffs);
  std::optional<double> getBandLaneWidth(int band_index) const;
  void updateBandLaneWidth(int band_index, double width);
  bool isMergeWideSegment(const Band& band) const;

  LaneDecisionConfig cfg_;
  LaneDebugInfo debug_info_;

  double last_offset_{0.0};
  bool branch_locked_{false};
  std::string locked_branch_side_{"left"};
  double lock_start_time_{0.0};
  int exit_confirm_count_{0};
  bool merge_wide_locked_{false};
  std::string merge_wide_side_;
  int merge_wide_confirm_count_{0};
  int merge_wide_release_count_{0};
  std::vector<std::optional<double>> band_lane_widths_;
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
