#pragma once

#include <string>
#include <vector>

#include "track_perception_cpp/types.hpp"

namespace track_perception_cpp {

struct LaneSegmentDebug {
  int x0{0};
  int x1{0};
  int center_x{0};
  int pixel_count{0};
};

struct LaneBandDebug {
  int y0{0};
  int y1{0};
  std::vector<LaneSegmentDebug> segments;
  int selected_center_x{-1};
};

struct LaneDebugInfo {
  std::vector<LaneBandDebug> bands;
  std::vector<cv::Point2f> fit_points;
};

class LaneDecision {
 public:
  void configure(int band_count, float band_y_min_ratio, float band_y_max_ratio,
                 float band_height_ratio, int min_segment_width_px,
                 int min_pixels_per_band, float smoothing_alpha,
                 float max_offset_jump, const std::string& outer_side,
                 bool enable_traffic_light_stop, float traffic_light_min_confidence,
                 float zebra_min_confidence, float zebra_stop_y_ratio,
                 bool enable_finish_stop, float finish_stop_min_confidence,
                 float finish_stop_arm_y_ratio);
  LaneState decide(const cv::Mat& seg_map, const std::vector<Detection>& detections);
  const LaneDebugInfo& debugInfo() const { return debug_info_; }

 private:
  int band_count_{13};
  float band_y_min_ratio_{0.65f};
  float band_y_max_ratio_{1.0f};
  float band_height_ratio_{0.02f};
  int min_segment_width_px_{25};
  int min_pixels_per_band_{80};
  float smoothing_alpha_{0.35f};
  float max_offset_jump_{2.0f};
  std::string outer_side_{"left"};
  bool enable_traffic_light_stop_{true};
  float traffic_light_min_confidence_{0.45f};
  float zebra_min_confidence_{0.45f};
  float zebra_stop_y_ratio_{0.70f};
  bool enable_finish_stop_{true};
  float finish_stop_min_confidence_{0.45f};
  float finish_stop_arm_y_ratio_{0.70f};
  float last_offset_{0.0f};
  bool has_last_offset_{false};
  LaneDebugInfo debug_info_;
};

}  // namespace track_perception_cpp
