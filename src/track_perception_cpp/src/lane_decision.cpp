#include "track_perception_cpp/lane_decision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace track_perception_cpp {

double nowSeconds() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string laneStateToJson(const LaneState& state) {
  std::ostringstream ss;
  ss << "{"
     << "\"control_offset\":" << state.control_offset << ","
     << "\"lateral_offset\":" << state.lateral_offset << ","
     << "\"heading_error\":" << state.heading_error << ","
     << "\"curvature\":" << state.curvature << ","
     << "\"confidence\":" << state.confidence << ","
     << "\"is_valid\":" << (state.is_valid ? "true" : "false") << ","
     << "\"road_state\":\"" << state.road_state << "\","
     << "\"branch_side\":\"" << state.branch_side << "\","
     << "\"task_state\":\"" << state.task_state << "\","
     << "\"task_bias\":" << state.task_bias << ","
     << "\"timestamp\":" << state.timestamp
     << "}";
  return ss.str();
}

void LaneDecision::configure(int band_count, float band_y_min_ratio, float band_y_max_ratio,
                             float band_height_ratio, int min_segment_width_px,
                             int min_pixels_per_band, float smoothing_alpha,
                             float max_offset_jump, const std::string& outer_side,
                             bool enable_traffic_light_stop, float traffic_light_min_confidence,
                             float zebra_min_confidence, float zebra_stop_y_ratio,
                             bool enable_finish_stop, float finish_stop_min_confidence,
                             float finish_stop_arm_y_ratio) {
  band_count_ = std::max(1, band_count);
  band_y_min_ratio_ = band_y_min_ratio;
  band_y_max_ratio_ = band_y_max_ratio;
  band_height_ratio_ = band_height_ratio;
  min_segment_width_px_ = min_segment_width_px;
  min_pixels_per_band_ = min_pixels_per_band;
  smoothing_alpha_ = smoothing_alpha;
  max_offset_jump_ = max_offset_jump;
  outer_side_ = outer_side;
  enable_traffic_light_stop_ = enable_traffic_light_stop;
  traffic_light_min_confidence_ = traffic_light_min_confidence;
  zebra_min_confidence_ = zebra_min_confidence;
  zebra_stop_y_ratio_ = zebra_stop_y_ratio;
  enable_finish_stop_ = enable_finish_stop;
  finish_stop_min_confidence_ = finish_stop_min_confidence;
  finish_stop_arm_y_ratio_ = finish_stop_arm_y_ratio;
}

LaneState LaneDecision::decide(const cv::Mat& seg_map, const std::vector<Detection>& detections) {
  LaneState state;
  state.timestamp = nowSeconds();
  state.branch_side = outer_side_;
  debug_info_ = LaneDebugInfo{};

  if (seg_map.empty()) {
    return state;
  }

  std::vector<cv::Point2f> centers;
  int h = seg_map.rows;
  int w = seg_map.cols;
  int y_min = std::clamp(static_cast<int>(h * band_y_min_ratio_), 0, h - 1);
  int y_max = std::clamp(static_cast<int>(h * band_y_max_ratio_), y_min + 1, h);
  int band_h = std::max(1, static_cast<int>(h * band_height_ratio_));

  for (int i = 0; i < band_count_; ++i) {
    float t = band_count_ == 1 ? 0.0f : static_cast<float>(i) / (band_count_ - 1);
    int y = static_cast<int>(y_min + t * (y_max - y_min - 1));
    int y0 = std::clamp(y - band_h / 2, 0, h - 1);
    int y1 = std::clamp(y0 + band_h, y0 + 1, h);

    const int min_x = 0;
    const int max_x = w;
    int best_start = -1;
    int best_end = -1;
    int cur_start = -1;
    int cur_count = 0;
    LaneBandDebug band_debug;
    band_debug.y0 = y0;
    band_debug.y1 = y1;

    std::vector<int> col_counts(w, 0);
    for (int yy = y0; yy < y1; ++yy) {
      const uint8_t* row = seg_map.ptr<uint8_t>(yy);
      for (int x = min_x; x < max_x; ++x) {
        if (row[x] == 1) {
          col_counts[x]++;
        }
      }
    }

    for (int x = min_x; x <= max_x; ++x) {
      bool active = x < max_x && col_counts[x] > 0;
      if (active) {
        if (cur_start < 0) {
          cur_start = x;
          cur_count = 0;
        }
        cur_count += col_counts[x];
      } else if (cur_start >= 0) {
        int cur_end = x;
        int width = cur_end - cur_start;
        if (width >= min_segment_width_px_ && cur_count >= min_pixels_per_band_) {
          LaneSegmentDebug seg_debug;
          seg_debug.x0 = cur_start;
          seg_debug.x1 = cur_end;
          seg_debug.center_x = (cur_start + cur_end) / 2;
          seg_debug.pixel_count = cur_count;
          band_debug.segments.push_back(seg_debug);
          if (best_start < 0 || width > best_end - best_start) {
            best_start = cur_start;
            best_end = cur_end;
          }
        }
        cur_start = -1;
      }
    }

    if (best_start >= 0) {
      float center_x = (best_start + best_end) * 0.5f;
      float center_y = static_cast<float>((y0 + y1) * 0.5f);
      centers.emplace_back(center_x, center_y);
      band_debug.selected_center_x = static_cast<int>(center_x);
    }
    debug_info_.bands.push_back(std::move(band_debug));
  }
  debug_info_.fit_points = centers;

  if (centers.size() < 3) {
    state.is_valid = false;
    state.road_state = "LOW_CONFIDENCE";
    state.confidence = static_cast<float>(centers.size()) / std::max(1, band_count_);
    return state;
  }

  cv::Point2f near = centers.back();
  float raw_offset = (near.x - (w * 0.5f)) / (w * 0.5f);
  float offset = raw_offset;
  if (has_last_offset_) {
    float diff = raw_offset - last_offset_;
    if (max_offset_jump_ > 0.0f && std::abs(diff) > max_offset_jump_) {
      raw_offset = last_offset_ + std::copysign(max_offset_jump_, diff);
    }
    offset = last_offset_ * (1.0f - smoothing_alpha_) + raw_offset * smoothing_alpha_;
  }
  last_offset_ = offset;
  has_last_offset_ = true;

  state.control_offset = offset;
  state.lateral_offset = offset;
  state.is_valid = true;
  state.confidence = std::min(1.0f, static_cast<float>(centers.size()) / std::max(1, band_count_));
  state.road_state = "NORMAL";

  if (centers.size() >= 2) {
    cv::Point2f far = centers.front();
    state.heading_error = (near.x - far.x) / std::max(1.0f, static_cast<float>(w));
  }

  bool red_seen = false;
  bool zebra_reached = false;
  bool finish_reached = false;
  for (const auto& det : detections) {
    float bottom_ratio = (det.bbox.y + det.bbox.height) / std::max(1.0f, static_cast<float>(h));
    if (enable_traffic_light_stop_ && det.class_name == "red_light" &&
        det.confidence >= traffic_light_min_confidence_) {
      red_seen = true;
    }
    if (enable_traffic_light_stop_ && det.class_name == "Zebra" &&
        det.confidence >= zebra_min_confidence_ && bottom_ratio >= zebra_stop_y_ratio_) {
      zebra_reached = true;
    }
    if (enable_finish_stop_ && det.class_name == "Stop" &&
        det.confidence >= finish_stop_min_confidence_ && bottom_ratio >= finish_stop_arm_y_ratio_) {
      finish_reached = true;
    }
  }

  if (finish_reached) {
    state.task_state = "FINISH_STOP";
  } else if (red_seen && zebra_reached) {
    state.task_state = "TRAFFIC_STOP";
  }
  return state;
}

}  // namespace track_perception_cpp
