#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <deque>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/opencv.hpp>

#include "ppocr_direction_system.h"
#include "track_perception_cpp/lane_decision.hpp"
#include "track_perception_cpp/shm_reader.hpp"
#include "track_perception_cpp/yolo_detector.hpp"
#include "track_perception_cpp/yolo_seg.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace track_perception_cpp {

namespace {

constexpr double kGuideboardOcrMinIntervalSec = 0.08;
constexpr int kOcrVoteRequired = 2;
constexpr int kMinOcrCropSizePx = 20;

std::vector<int> parseCoreIds(const std::string& text) {
  std::vector<int> ids;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      ids.push_back(std::stoi(item));
    } catch (...) {
    }
  }
  if (ids.empty()) {
    ids = {1, 2};
  }
  return ids;
}

std::unordered_set<std::string> parseLabelSet(const std::string& text) {
  std::unordered_set<std::string> labels;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), item.end());
    if (!item.empty()) {
      labels.insert(item);
    }
  }
  return labels;
}

std::string joinLabels(const std::vector<std::string>& labels) {
  std::ostringstream ss;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << labels[i];
  }
  return ss.str();
}

std::string jsonEscape(const std::string& text) {
  std::ostringstream ss;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') {
      ss << '\\' << ch;
    } else if (ch == '\n') {
      ss << "\\n";
    } else {
      ss << ch;
    }
  }
  return ss.str();
}

double normalizeCenterX(int center_x, int image_width) {
  if (center_x < 0 || image_width <= 1) {
    return 0.0;
  }
  return (static_cast<double>(center_x) - 0.5 * static_cast<double>(image_width)) /
         (0.5 * static_cast<double>(image_width));
}

int selectedCenterAtRatio(const LaneDebugInfo& debug_info, double ratio) {
  std::vector<const LaneBandDebug*> selected;
  selected.reserve(debug_info.bands.size());
  for (const auto& band : debug_info.bands) {
    if (band.selected_center_x >= 0) {
      selected.push_back(&band);
    }
  }
  if (selected.empty()) {
    return -1;
  }
  ratio = std::clamp(ratio, 0.0, 1.0);
  size_t index = static_cast<size_t>(std::round(ratio * static_cast<double>(selected.size() - 1)));
  return selected[index]->selected_center_x;
}

int inferDebugImageWidth(const LaneDebugInfo& debug_info, int fallback_width) {
  if (debug_info.image_width > 1) {
    return debug_info.image_width;
  }
  int max_x = 0;
  for (const auto& band : debug_info.bands) {
    for (const auto& seg : band.segments) {
      max_x = std::max(max_x, seg.x1);
    }
  }
  return max_x > 1 ? max_x : fallback_width;
}

std::string laneDebugToJson(const LaneDebugInfo& debug_info, int fallback_width) {
  const int image_width = inferDebugImageWidth(debug_info, fallback_width);
  const int top_x = selectedCenterAtRatio(debug_info, 0.0);
  const int mid_x = selectedCenterAtRatio(debug_info, 0.5);
  const int bottom_x = selectedCenterAtRatio(debug_info, 1.0);
  const double top_n = normalizeCenterX(top_x, image_width);
  const double mid_n = normalizeCenterX(mid_x, image_width);
  const double bottom_n = normalizeCenterX(bottom_x, image_width);

  std::ostringstream ss;
  ss << "{"
     << "\"top_x\":" << top_x << ","
     << "\"mid_x\":" << mid_x << ","
     << "\"bottom_x\":" << bottom_x << ","
     << "\"top_norm\":" << top_n << ","
     << "\"mid_norm\":" << mid_n << ","
     << "\"bottom_norm\":" << bottom_n << ","
     << "\"center_slope_norm\":" << (top_n - bottom_n) << ","
     << "\"near_slope_norm\":" << (mid_n - bottom_n) << ","
     << "\"image_width\":" << image_width << ","
     << "\"offset_y07\":" << debug_info.offset_y07 << ","
     << "\"offset_y08\":" << debug_info.offset_y08 << ","
     << "\"offset_y09\":" << debug_info.offset_y09 << ","
     << "\"raw_offset_y07\":" << debug_info.raw_offset_y07 << ","
     << "\"raw_offset_y08\":" << debug_info.raw_offset_y08 << ","
     << "\"raw_offset_y09\":" << debug_info.raw_offset_y09 << ","
     << "\"fit_y_min\":" << debug_info.fit_y_min << ","
     << "\"fit_y_max\":" << debug_info.fit_y_max << ","
     << "\"fit_y_span\":" << debug_info.fit_y_span << ","
     << "\"raw_points\":" << debug_info.raw_point_count << ","
     << "\"fit_points\":" << debug_info.fit_point_count << ","
     << "\"segments\":" << debug_info.segment_count << ","
     << "\"branch_detected\":" << (debug_info.branch_detected ? "true" : "false") << ","
     << "\"branch_score\":" << debug_info.branch_score << ","
     << "\"encoder_hold\":" << (debug_info.encoder_hold ? "true" : "false") << ","
     << "\"encoder_hold_side\":\"" << jsonEscape(debug_info.encoder_hold_side) << "\","
     << "\"encoder_count\":" << debug_info.encoder_count << ","
     << "\"encoder_hold_delta\":" << debug_info.encoder_hold_delta << ","
     << "\"encoder_hold_target\":" << debug_info.encoder_hold_target << ","
     << "\"encoder_feedback_valid\":" << (debug_info.encoder_feedback_valid ? "true" : "false") << ","
     << "\"encoder_feedback_age\":" << debug_info.encoder_feedback_age << ","
     << "\"lb_template\":" << (debug_info.left_boundary_template_active ? "true" : "false") << ","
     << "\"template_side\":\"" << jsonEscape(debug_info.boundary_template_side) << "\","
     << "\"lb_points\":" << debug_info.left_boundary_template_points << ","
     << "\"lb_reason\":\"" << jsonEscape(debug_info.left_boundary_template_reason) << "\","
     << "\"bands\":[";
  for (size_t i = 0; i < debug_info.bands.size(); ++i) {
    const auto& band = debug_info.bands[i];
    if (i > 0) {
      ss << ",";
    }
    ss << "{\"y0\":" << band.y0
       << ",\"y1\":" << band.y1
       << ",\"selected_center_x\":" << band.selected_center_x
       << ",\"segments\":[";
    for (size_t j = 0; j < band.segments.size(); ++j) {
      const auto& segment = band.segments[j];
      if (j > 0) {
        ss << ",";
      }
      ss << "{\"x0\":" << segment.x0
         << ",\"x1\":" << segment.x1
         << ",\"center_x\":" << segment.center_x
         << ",\"pixel_count\":" << segment.pixel_count
         << ",\"selected\":" << (segment.selected ? "true" : "false")
         << ",\"virtual\":" << (segment.virtual_segment ? "true" : "false")
         << ",\"obstacle_cut\":" << (segment.obstacle_cut ? "true" : "false")
         << "}";
    }
    ss << "]}";
  }
  ss << "],\"raw_points\":[";
  for (size_t i = 0; i < debug_info.raw_points.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    const auto& point = debug_info.raw_points[i];
    ss << "[" << point.x << "," << point.y << "," << point.z << "]";
  }
  ss << "],\"fit_points\":[";
  for (size_t i = 0; i < debug_info.fit_points.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    const auto& point = debug_info.fit_points[i];
    ss << "[" << point.x << "," << point.y << "," << point.z << "]";
  }
  ss << "],\"fit_coeffs\":[";
  for (size_t i = 0; i < debug_info.fit_coeffs.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << debug_info.fit_coeffs[i];
  }
  ss << "]"
     << "}";
  return ss.str();
}

}  // namespace

class FusedPerceptionNode : public rclcpp::Node {
 public:
  FusedPerceptionNode() : Node("fused_perception_node") {
    declareParameters();
    loadParameters();
    initialize();

    timer_ = create_wall_timer(1ms, std::bind(&FusedPerceptionNode::tick, this));
  }

  ~FusedPerceptionNode() override {
    waitForGuideboardOcr();
  }

 private:
  struct OcrTaskResult {
    int ret{-1};
    PPOCRDirectionResult result{};
    int crop_width{0};
    int crop_height{0};
  };

  void declareParameters() {
    declare_parameter<std::string>("shm_name", "shm_ar_video");
    declare_parameter<bool>("enable_flip", true);
    declare_parameter<int>("flip_code", 0);
    declare_parameter<std::string>("input_format", "RGB");
    declare_parameter<std::string>("model_input_format", "RGB");
    declare_parameter<bool>("enable_perf_stats", true);
    declare_parameter<double>("perf_interval", 2.0);
    declare_parameter<bool>("show_window", false);
    declare_parameter<double>("blend_alpha", 1.0);
    declare_parameter<bool>("show_branch_debug", true);
    declare_parameter<bool>("enable_debug_screenshots", false);
    declare_parameter<double>("debug_screenshot_interval_sec", 0.0);
    declare_parameter<bool>("debug_screenshot_branch_only", false);
    declare_parameter<std::string>("debug_screenshot_dir", "/tmp/fused_perception_screenshots");

    declare_parameter<std::string>("det_model_path", "model/yolov8_n_det_split_int8_v2.rknn");
    declare_parameter<std::string>("label_list_path", "model/label_list.txt");
    declare_parameter<int>("det_core_id", 0);
    declare_parameter<int>("det_input_width", 384);
    declare_parameter<int>("det_input_height", 288);
    declare_parameter<double>("det_conf_threshold", 0.5);
    declare_parameter<double>("det_nms_threshold", 0.45);
    declare_parameter<bool>("det_raw_output", false);
    declare_parameter<bool>("publish_detections", true);
    declare_parameter<int>("publish_rate", 0);
    declare_parameter<bool>("use_fast_postprocess", false);

    declare_parameter<std::string>("seg_model_dir", "model");
    declare_parameter<std::string>("seg_model_filename", "yolov8_seg_n_384x160_split_int8.rknn");
    declare_parameter<std::string>("seg_core_ids", "1,2");
    declare_parameter<int>("seg_input_width", 384);
    declare_parameter<int>("seg_input_height", 160);
    declare_parameter<double>("seg_crop_y0_ratio", 0.0);
    declare_parameter<double>("seg_crop_y1_ratio", 1.0);
    declare_parameter<int>("seg_pad_value", 114);
    declare_parameter<double>("seg_conf_threshold", 0.45);
    declare_parameter<double>("seg_nms_threshold", 0.45);
    declare_parameter<double>("seg_nms_contain_threshold", 0.85);
    declare_parameter<double>("seg_mask_threshold", 0.45);
    declare_parameter<int>("seg_max_detections", 30);
    declare_parameter<bool>("seg_raw_output", false);

    declare_parameter<int>("band_count", 13);
    declare_parameter<double>("band_y_min_ratio", 0.65);
    declare_parameter<double>("band_y_max_ratio", 1.0);
    declare_parameter<double>("band_height_ratio", 0.02);
    declare_parameter<int>("min_segment_width_px", 25);
    declare_parameter<int>("min_segment_gap_px", 40);
    declare_parameter<int>("min_pixels_per_band", 80);
    declare_parameter<int>("branch_detect_min_bands", 2);
    declare_parameter<int>("branch_confirm_frames", 2);
    declare_parameter<double>("branch_detect_far_band_ratio", 0.7);
    declare_parameter<bool>("enable_encoder_branch_hold", true);
    declare_parameter<std::string>("encoder_count_topic", "/chassis/encoder_count");
    declare_parameter<int64_t>("encoder_hold_counts", 5000);
    declare_parameter<double>("encoder_feedback_timeout_sec", 0.30);
    declare_parameter<bool>("enable_guideboard_branch_selection", true);
    declare_parameter<std::string>("guideboard_branch", "right");
    declare_parameter<double>("guideboard_detect_y0_ratio", 0.2);
    declare_parameter<double>("guideboard_detect_y1_ratio", 0.7);
    declare_parameter<bool>("enable_guideboard_ocr", true);
    declare_parameter<std::string>("ocr_det_model_path", "model/ppocrv4_det.rknn");
    declare_parameter<std::string>("ocr_rec_model_path", "model/ppocrv4_rec.rknn");
    declare_parameter<double>("ocr_min_score", 0.75);
    declare_parameter<double>("ocr_crop_padding_ratio", 0.25);
    declare_parameter<int>("ocr_vote_window", 3);
    declare_parameter<bool>("enable_segment_branch_logic", true);
    declare_parameter<int>("fit_min_points", 5);
    declare_parameter<int>("fit_order", 2);
    declare_parameter<int>("branch_fit_order", 2);
    declare_parameter<bool>("enable_fit_point_jump_filter", true);
    declare_parameter<double>("max_fit_point_dx_ratio", 0.22);
    declare_parameter<double>("max_fit_point_dx_px", 140.0);
    declare_parameter<bool>("enable_fit_point_trend_filter", true);
    declare_parameter<double>("fit_point_trend_residual_ratio", 0.12);
    declare_parameter<double>("fit_point_trend_residual_px", 80.0);
    declare_parameter<double>("fit_point_trend_slope_delta", 0.65);
    declare_parameter<int>("fit_point_trend_min_points", 6);
    declare_parameter<double>("fit_point_trend_min_keep_ratio", 0.75);
    declare_parameter<bool>("enable_obstacle_avoidance", false);
    declare_parameter<std::string>("obstacle_labels", "Human,Car");
    declare_parameter<std::string>("obstacle_stop_labels", "Human");
    declare_parameter<double>("obstacle_min_confidence", 0.45);
    declare_parameter<double>("obstacle_x_margin_px", 25.0);
    declare_parameter<double>("obstacle_y_margin_px", 20.0);
    declare_parameter<double>("obstacle_max_age", 0.3);
    declare_parameter<double>("obstacle_min_bottom_y_ratio", 0.30);
    declare_parameter<bool>("enable_obstacle_stop", true);
    declare_parameter<double>("obstacle_stop_bottom_y_ratio", 0.82);
    declare_parameter<int>("obstacle_stop_confirm_frames", 2);
    declare_parameter<int>("obstacle_stop_lost_frames", 3);
    declare_parameter<bool>("enable_label_fit_points", true);
    declare_parameter<std::string>("fit_point_labels", "Go");
    declare_parameter<double>("fit_point_min_confidence", 0.45);
    declare_parameter<double>("fit_point_y0_ratio", 0.45);
    declare_parameter<double>("fit_point_y1_ratio", 1.0);
    declare_parameter<double>("fit_point_weight", 1.0);
    declare_parameter<bool>("enable_start_boost_trigger", true);
    declare_parameter<std::string>("start_boost_labels", "Go,Gate");
    declare_parameter<double>("start_boost_min_confidence", 0.45);
    declare_parameter<double>("start_boost_y0_ratio", 0.0);
    declare_parameter<double>("start_boost_y1_ratio", 1.0);
    declare_parameter<int>("start_boost_lost_frames", 3);
    declare_parameter<bool>("enable_traffic_light_stop", true);
    declare_parameter<double>("traffic_light_min_confidence", 0.45);
    declare_parameter<double>("zebra_min_confidence", 0.45);
    declare_parameter<double>("zebra_stop_y_ratio", 0.70);
    declare_parameter<double>("traffic_light_max_age", 0.5);
    declare_parameter<int>("green_light_confirm_frames", 1);
    declare_parameter<int>("red_light_confirm_frames", 1);
    declare_parameter<bool>("enable_finish_stop", true);
    declare_parameter<double>("finish_stop_min_confidence", 0.45);
    declare_parameter<double>("finish_stop_arm_y_ratio", 0.70);
    declare_parameter<int>("finish_stop_lost_frames", 3);
    declare_parameter<double>("finish_stop_max_age", 0.5);
    declare_parameter<double>("offset_y07_ratio", 0.70);
    declare_parameter<double>("offset_y08_ratio", 0.80);
    declare_parameter<double>("offset_y09_ratio", 0.90);
    declare_parameter<double>("max_offset_jump", 2.0);
    declare_parameter<double>("offset_smoothing_alpha", 0.35);
    declare_parameter<bool>("enable_left_boundary_template_line", false);
    declare_parameter<std::string>("left_boundary_template_side", "left");
    declare_parameter<std::string>("left_boundary_template_offsets", "");
    declare_parameter<int>("left_boundary_template_min_points", 6);
    declare_parameter<double>("left_boundary_template_weight", 1.0);
    declare_parameter<std::string>("right_boundary_template_offsets", "");
    declare_parameter<int>("right_boundary_template_min_points", 6);
    declare_parameter<double>("right_boundary_template_weight", 1.0);
    declare_parameter<std::string>("outer_side", "left");
    declare_parameter<bool>("enable_status_log", false);
    declare_parameter<bool>("enable_branch_event_log", false);
    declare_parameter<bool>("publish_lane_state", true);
  }

  void loadParameters() {
    shm_name_ = get_parameter("shm_name").as_string();
    enable_flip_ = get_parameter("enable_flip").as_bool();
    flip_code_ = static_cast<int>(get_parameter("flip_code").as_int());
    input_format_ = get_parameter("input_format").as_string();
    enable_perf_stats_ = get_parameter("enable_perf_stats").as_bool();
    perf_interval_ = get_parameter("perf_interval").as_double();
    show_window_ = get_parameter("show_window").as_bool();
    blend_alpha_ = static_cast<float>(get_parameter("blend_alpha").as_double());
    show_branch_debug_ = get_parameter("show_branch_debug").as_bool();
    enable_debug_screenshots_ = get_parameter("enable_debug_screenshots").as_bool();
    debug_screenshot_interval_sec_ = get_parameter("debug_screenshot_interval_sec").as_double();
    debug_screenshot_branch_only_ = get_parameter("debug_screenshot_branch_only").as_bool();
    debug_screenshot_dir_ = get_parameter("debug_screenshot_dir").as_string();
    publish_detections_ = get_parameter("publish_detections").as_bool();
    publish_lane_state_ = get_parameter("publish_lane_state").as_bool();
    enable_status_log_ = get_parameter("enable_status_log").as_bool();
    enable_branch_event_log_ = get_parameter("enable_branch_event_log").as_bool();
    enable_guideboard_ocr_ = get_parameter("enable_guideboard_ocr").as_bool();
    ocr_det_model_path_ = resolveOwnPackagePath(get_parameter("ocr_det_model_path").as_string());
    ocr_rec_model_path_ = resolveOwnPackagePath(get_parameter("ocr_rec_model_path").as_string());
    ocr_min_score_ = static_cast<float>(get_parameter("ocr_min_score").as_double());
    ocr_crop_padding_ratio_ = static_cast<float>(get_parameter("ocr_crop_padding_ratio").as_double());
    ocr_vote_window_ = static_cast<int>(get_parameter("ocr_vote_window").as_int());

    det_model_path_ = resolveTrackPerceptionPath(get_parameter("det_model_path").as_string());
    label_list_path_ = resolveTrackPerceptionPath(get_parameter("label_list_path").as_string());
    det_core_id_ = static_cast<int>(get_parameter("det_core_id").as_int());
    det_input_width_ = static_cast<int>(get_parameter("det_input_width").as_int());
    det_input_height_ = static_cast<int>(get_parameter("det_input_height").as_int());
    det_conf_threshold_ = static_cast<float>(get_parameter("det_conf_threshold").as_double());
    det_nms_threshold_ = static_cast<float>(get_parameter("det_nms_threshold").as_double());
    det_raw_output_ = get_parameter("det_raw_output").as_bool();

    std::string seg_model_dir = get_parameter("seg_model_dir").as_string();
    std::string seg_model_filename = get_parameter("seg_model_filename").as_string();
    fs::path seg_path = fs::path(seg_model_dir) / seg_model_filename;
    seg_model_path_ = resolveTrackPerceptionPath(seg_path.string());
    seg_core_ids_ = parseCoreIds(get_parameter("seg_core_ids").as_string());
    seg_input_width_ = static_cast<int>(get_parameter("seg_input_width").as_int());
    seg_input_height_ = static_cast<int>(get_parameter("seg_input_height").as_int());
    seg_crop_y0_ratio_ = static_cast<float>(get_parameter("seg_crop_y0_ratio").as_double());
    seg_crop_y1_ratio_ = static_cast<float>(get_parameter("seg_crop_y1_ratio").as_double());
    seg_pad_value_ = static_cast<int>(get_parameter("seg_pad_value").as_int());
    seg_conf_threshold_ = static_cast<float>(get_parameter("seg_conf_threshold").as_double());
    seg_nms_threshold_ = static_cast<float>(get_parameter("seg_nms_threshold").as_double());
    seg_nms_contain_threshold_ =
        static_cast<float>(get_parameter("seg_nms_contain_threshold").as_double());
    seg_mask_threshold_ = static_cast<float>(get_parameter("seg_mask_threshold").as_double());
    seg_max_detections_ = static_cast<int>(get_parameter("seg_max_detections").as_int());
    seg_raw_output_ = get_parameter("seg_raw_output").as_bool();

    LaneDecisionConfig lane_cfg;
    lane_cfg.enable_segment_branch_logic = get_parameter("enable_segment_branch_logic").as_bool();
    lane_cfg.band_count = static_cast<int>(get_parameter("band_count").as_int());
    lane_cfg.band_y_min_ratio = static_cast<float>(get_parameter("band_y_min_ratio").as_double());
    lane_cfg.band_y_max_ratio = static_cast<float>(get_parameter("band_y_max_ratio").as_double());
    lane_cfg.band_height_ratio = static_cast<float>(get_parameter("band_height_ratio").as_double());
    lane_cfg.min_segment_width_px = static_cast<int>(get_parameter("min_segment_width_px").as_int());
    lane_cfg.min_segment_gap_px = static_cast<int>(get_parameter("min_segment_gap_px").as_int());
    lane_cfg.min_pixels_per_band = static_cast<int>(get_parameter("min_pixels_per_band").as_int());
    lane_cfg.branch_detect_min_bands = static_cast<int>(get_parameter("branch_detect_min_bands").as_int());
    lane_cfg.branch_confirm_frames = static_cast<int>(get_parameter("branch_confirm_frames").as_int());
    lane_cfg.branch_detect_far_band_ratio = static_cast<float>(get_parameter("branch_detect_far_band_ratio").as_double());
    lane_cfg.enable_encoder_branch_hold = get_parameter("enable_encoder_branch_hold").as_bool();
    lane_cfg.encoder_hold_counts = get_parameter("encoder_hold_counts").as_int();
    lane_cfg.encoder_feedback_timeout_sec = get_parameter("encoder_feedback_timeout_sec").as_double();
    encoder_count_topic_ = get_parameter("encoder_count_topic").as_string();
    lane_cfg.outer_side = get_parameter("outer_side").as_string();
    lane_cfg.enable_guideboard_branch_selection = get_parameter("enable_guideboard_branch_selection").as_bool();
    lane_cfg.guideboard_branch = get_parameter("guideboard_branch").as_string();
    lane_cfg.guideboard_detect_y0_ratio = static_cast<float>(get_parameter("guideboard_detect_y0_ratio").as_double());
    lane_cfg.guideboard_detect_y1_ratio = static_cast<float>(get_parameter("guideboard_detect_y1_ratio").as_double());
    lane_guideboard_y0_ratio_ = lane_cfg.guideboard_detect_y0_ratio;
    lane_guideboard_y1_ratio_ = lane_cfg.guideboard_detect_y1_ratio;
    lane_cfg.fit_min_points = static_cast<int>(get_parameter("fit_min_points").as_int());
    lane_cfg.fit_order = static_cast<int>(get_parameter("fit_order").as_int());
    lane_cfg.branch_fit_order = static_cast<int>(get_parameter("branch_fit_order").as_int());
    lane_cfg.enable_fit_point_jump_filter = get_parameter("enable_fit_point_jump_filter").as_bool();
    lane_cfg.max_fit_point_dx_ratio = static_cast<float>(get_parameter("max_fit_point_dx_ratio").as_double());
    lane_cfg.max_fit_point_dx_px = static_cast<float>(get_parameter("max_fit_point_dx_px").as_double());
    lane_cfg.enable_fit_point_trend_filter = get_parameter("enable_fit_point_trend_filter").as_bool();
    lane_cfg.fit_point_trend_residual_ratio = static_cast<float>(get_parameter("fit_point_trend_residual_ratio").as_double());
    lane_cfg.fit_point_trend_residual_px = static_cast<float>(get_parameter("fit_point_trend_residual_px").as_double());
    lane_cfg.fit_point_trend_slope_delta = static_cast<float>(get_parameter("fit_point_trend_slope_delta").as_double());
    lane_cfg.fit_point_trend_min_points = static_cast<int>(get_parameter("fit_point_trend_min_points").as_int());
    lane_cfg.fit_point_trend_min_keep_ratio = static_cast<float>(get_parameter("fit_point_trend_min_keep_ratio").as_double());
    lane_cfg.offset_y07_ratio = static_cast<float>(get_parameter("offset_y07_ratio").as_double());
    lane_cfg.offset_y08_ratio = static_cast<float>(get_parameter("offset_y08_ratio").as_double());
    lane_cfg.offset_y09_ratio = static_cast<float>(get_parameter("offset_y09_ratio").as_double());
    lane_cfg.max_offset_jump = static_cast<float>(get_parameter("max_offset_jump").as_double());
    lane_cfg.offset_smoothing_alpha = static_cast<float>(get_parameter("offset_smoothing_alpha").as_double());
    lane_cfg.enable_left_boundary_template_line =
        get_parameter("enable_left_boundary_template_line").as_bool();
    lane_cfg.left_boundary_template_side = get_parameter("left_boundary_template_side").as_string();
    lane_cfg.left_boundary_template_offsets = get_parameter("left_boundary_template_offsets").as_string();
    lane_cfg.left_boundary_template_min_points =
        static_cast<int>(get_parameter("left_boundary_template_min_points").as_int());
    lane_cfg.left_boundary_template_weight =
        static_cast<float>(get_parameter("left_boundary_template_weight").as_double());
    lane_cfg.right_boundary_template_offsets = get_parameter("right_boundary_template_offsets").as_string();
    lane_cfg.right_boundary_template_min_points =
        static_cast<int>(get_parameter("right_boundary_template_min_points").as_int());
    lane_cfg.right_boundary_template_weight =
        static_cast<float>(get_parameter("right_boundary_template_weight").as_double());
    lane_cfg.enable_obstacle_avoidance = get_parameter("enable_obstacle_avoidance").as_bool();
    lane_cfg.obstacle_labels = parseLabelSet(get_parameter("obstacle_labels").as_string());
    lane_cfg.obstacle_stop_labels = parseLabelSet(get_parameter("obstacle_stop_labels").as_string());
    lane_cfg.obstacle_min_confidence = static_cast<float>(get_parameter("obstacle_min_confidence").as_double());
    lane_cfg.obstacle_x_margin_px = static_cast<float>(get_parameter("obstacle_x_margin_px").as_double());
    lane_cfg.obstacle_y_margin_px = static_cast<float>(get_parameter("obstacle_y_margin_px").as_double());
    lane_cfg.obstacle_min_bottom_y_ratio = static_cast<float>(get_parameter("obstacle_min_bottom_y_ratio").as_double());
    lane_cfg.enable_obstacle_stop = get_parameter("enable_obstacle_stop").as_bool();
    lane_cfg.obstacle_stop_bottom_y_ratio = static_cast<float>(get_parameter("obstacle_stop_bottom_y_ratio").as_double());
    lane_cfg.obstacle_stop_confirm_frames = static_cast<int>(get_parameter("obstacle_stop_confirm_frames").as_int());
    lane_cfg.obstacle_stop_lost_frames = static_cast<int>(get_parameter("obstacle_stop_lost_frames").as_int());
    lane_cfg.enable_label_fit_points = get_parameter("enable_label_fit_points").as_bool();
    lane_cfg.fit_point_labels = parseLabelSet(get_parameter("fit_point_labels").as_string());
    lane_cfg.fit_point_min_confidence = static_cast<float>(get_parameter("fit_point_min_confidence").as_double());
    lane_cfg.fit_point_y0_ratio = static_cast<float>(get_parameter("fit_point_y0_ratio").as_double());
    lane_cfg.fit_point_y1_ratio = static_cast<float>(get_parameter("fit_point_y1_ratio").as_double());
    lane_cfg.fit_point_weight = static_cast<float>(get_parameter("fit_point_weight").as_double());
    lane_cfg.enable_start_boost_trigger = get_parameter("enable_start_boost_trigger").as_bool();
    lane_cfg.start_boost_labels = parseLabelSet(get_parameter("start_boost_labels").as_string());
    lane_cfg.start_boost_min_confidence = static_cast<float>(get_parameter("start_boost_min_confidence").as_double());
    lane_cfg.start_boost_y0_ratio = static_cast<float>(get_parameter("start_boost_y0_ratio").as_double());
    lane_cfg.start_boost_y1_ratio = static_cast<float>(get_parameter("start_boost_y1_ratio").as_double());
    lane_cfg.start_boost_lost_frames = static_cast<int>(get_parameter("start_boost_lost_frames").as_int());
    lane_cfg.enable_traffic_light_stop = get_parameter("enable_traffic_light_stop").as_bool();
    lane_cfg.traffic_light_min_confidence = static_cast<float>(get_parameter("traffic_light_min_confidence").as_double());
    lane_cfg.zebra_min_confidence = static_cast<float>(get_parameter("zebra_min_confidence").as_double());
    lane_cfg.zebra_stop_y_ratio = static_cast<float>(get_parameter("zebra_stop_y_ratio").as_double());
    lane_cfg.green_light_confirm_frames = static_cast<int>(get_parameter("green_light_confirm_frames").as_int());
    lane_cfg.red_light_confirm_frames = static_cast<int>(get_parameter("red_light_confirm_frames").as_int());
    lane_cfg.enable_finish_stop = get_parameter("enable_finish_stop").as_bool();
    lane_cfg.finish_stop_min_confidence = static_cast<float>(get_parameter("finish_stop_min_confidence").as_double());
    lane_cfg.finish_stop_arm_y_ratio = static_cast<float>(get_parameter("finish_stop_arm_y_ratio").as_double());
    lane_cfg.finish_stop_lost_frames = static_cast<int>(get_parameter("finish_stop_lost_frames").as_int());
    lane_cfg.enable_branch_event_log = get_parameter("enable_branch_event_log").as_bool();
    lane_decision_.configure(lane_cfg);
  }

  std::string resolveTrackPerceptionPath(const std::string& path) const {
    fs::path p(path);
    if (p.is_absolute()) {
      return p.string();
    }
    std::string share_dir = ament_index_cpp::get_package_share_directory("track_perception");
    return (fs::path(share_dir) / p).string();
  }

  std::string resolveOwnPackagePath(const std::string& path) const {
    fs::path p(path);
    if (p.is_absolute()) {
      return p.string();
    }
    std::string share_dir = ament_index_cpp::get_package_share_directory("track_perception_cpp");
    return (fs::path(share_dir) / p).string();
  }

  const Detection* selectGuideboardForOcr(const std::vector<Detection>& detections, int image_height) const {
    int y0 = static_cast<int>(image_height * lane_guideboard_y0_ratio_);
    int y1 = static_cast<int>(image_height * lane_guideboard_y1_ratio_);
    const Detection* best = nullptr;
    for (const auto& det : detections) {
      if (det.class_name != "GuideBoard" || det.center.y < y0 || det.center.y > y1 ||
          det.bbox.width < kMinOcrCropSizePx || det.bbox.height < kMinOcrCropSizePx) {
        continue;
      }
      if (best == nullptr || det.confidence > best->confidence) {
        best = &det;
      }
    }
    return best;
  }

  cv::Rect makePaddedCropRect(const cv::Rect2f& bbox, const cv::Size& image_size) const {
    float pad = std::max(bbox.width, bbox.height) * std::max(0.0f, ocr_crop_padding_ratio_);
    int x0 = static_cast<int>(std::floor(bbox.x - pad));
    int y0 = static_cast<int>(std::floor(bbox.y - pad));
    int x1 = static_cast<int>(std::ceil(bbox.x + bbox.width + pad));
    int y1 = static_cast<int>(std::ceil(bbox.y + bbox.height + pad));
    x0 = std::clamp(x0, 0, std::max(0, image_size.width - 1));
    y0 = std::clamp(y0, 0, std::max(0, image_size.height - 1));
    x1 = std::clamp(x1, x0 + 1, image_size.width);
    y1 = std::clamp(y1, y0 + 1, image_size.height);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
  }

  bool isOcrResultReliable(const PPOCRDirectionResult& result) const {
    return result.status == PPOCR_STATUS_OK && result.raw_direction != -1 &&
           !result.text_touch_edge && result.ocr_score >= ocr_min_score_;
  }

  void addOcrVote(int raw_direction) {
    int window = std::clamp(ocr_vote_window_, kOcrVoteRequired, 9);
    guideboard_ocr_votes_.push_back(raw_direction);
    while (static_cast<int>(guideboard_ocr_votes_.size()) > window) {
      guideboard_ocr_votes_.pop_front();
    }

    int straight_votes = 0;
    int right_votes = 0;
    for (int vote : guideboard_ocr_votes_) {
      if (vote == 0) {
        ++straight_votes;
      } else if (vote == 1) {
        ++right_votes;
      }
    }
    if (straight_votes >= kOcrVoteRequired) {
      stable_guideboard_branch_ = "left";
      stable_guideboard_raw_direction_ = 0;
    } else if (right_votes >= kOcrVoteRequired) {
      stable_guideboard_branch_ = "right";
      stable_guideboard_raw_direction_ = 1;
    }
  }

  void consumeGuideboardOcrResult() {
    if (!guideboard_ocr_future_.valid() ||
        guideboard_ocr_future_.wait_for(0ms) != std::future_status::ready) {
      return;
    }

    OcrTaskResult task;
    try {
      task = guideboard_ocr_future_.get();
    } catch (const std::exception& e) {
      task.result.status = PPOCR_STATUS_INFERENCE_FAILED;
      task.result.error = e.what();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "GuideBoard OCR async exception: %s crop=%dx%d",
                           task.result.error.c_str(), ocr_task_crop_width_,
                           ocr_task_crop_height_);
    } catch (...) {
      task.result.status = PPOCR_STATUS_INFERENCE_FAILED;
      task.result.error = "unknown OCR async exception";
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "GuideBoard OCR async exception: %s crop=%dx%d",
                           task.result.error.c_str(), ocr_task_crop_width_,
                           ocr_task_crop_height_);
    }

    const PPOCRDirectionResult& result = task.result;
    last_ocr_text_ = result.text;
    last_ocr_status_ = result.status;
    last_ocr_raw_direction_ = result.raw_direction;
    last_ocr_score_ = result.ocr_score;
    last_ocr_time_ms_ = result.time_ms;
    last_ocr_touch_edge_ = result.text_touch_edge;

    if (task.ret == 0 && isOcrResultReliable(result)) {
      addOcrVote(result.raw_direction);
    }
    if (result.time_ms > 0.0) {
      sum_ocr_ms_ += result.time_ms;
      ++ocr_run_count_;
    }

    if (enable_branch_event_log_) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 500,
          "GuideBoard OCR ret=%d text=%s raw=%d status=%d score=%.2f edge=%d time=%.1fms "
          "stable=%s votes=%zu crop=%dx%d skipped=%d boxes=%s error=%s",
          task.ret, last_ocr_text_.c_str(), last_ocr_raw_direction_, last_ocr_status_,
          last_ocr_score_, last_ocr_touch_edge_, last_ocr_time_ms_,
          stable_guideboard_branch_.c_str(), guideboard_ocr_votes_.size(), task.crop_width,
          task.crop_height, result.skipped_box_count, result.skipped_box_summary.c_str(),
          result.error.c_str());
    }
  }

  void waitForGuideboardOcr() {
    if (!guideboard_ocr_future_.valid()) {
      return;
    }
    try {
      guideboard_ocr_future_.wait();
      guideboard_ocr_future_.get();
    } catch (...) {
    }
  }

  void updateGuideboardOcr(const cv::Mat& frame_rgb, const std::vector<Detection>& detections) {
    if (!enable_guideboard_ocr_ || !guideboard_ocr_ready_ || frame_rgb.empty()) {
      lane_decision_.setGuideboardBranchHint("left", enable_guideboard_ocr_);
      return;
    }

    consumeGuideboardOcrResult();

    const Detection* guideboard = selectGuideboardForOcr(detections, frame_rgb.rows);
    if (guideboard == nullptr) {
      lane_decision_.setGuideboardBranchHint(stable_guideboard_branch_, true);
      return;
    }

    double now = nowSeconds();
    if (guideboard_ocr_future_.valid()) {
      lane_decision_.setGuideboardBranchHint(stable_guideboard_branch_, true);
      return;
    }
    if (now - last_guideboard_ocr_sec_ < kGuideboardOcrMinIntervalSec) {
      lane_decision_.setGuideboardBranchHint(stable_guideboard_branch_, true);
      return;
    }
    last_guideboard_ocr_sec_ = now;

    cv::Rect crop_rect = makePaddedCropRect(guideboard->bbox, frame_rgb.size());
    if (crop_rect.width < kMinOcrCropSizePx || crop_rect.height < kMinOcrCropSizePx) {
      lane_decision_.setGuideboardBranchHint(stable_guideboard_branch_, true);
      return;
    }

    cv::Mat crop = frame_rgb(crop_rect).clone();
    ocr_task_crop_width_ = crop.cols;
    ocr_task_crop_height_ = crop.rows;
    guideboard_ocr_future_ = std::async(std::launch::async, [this, crop = std::move(crop)]() {
      OcrTaskResult task;
      task.crop_width = crop.cols;
      task.crop_height = crop.rows;
      try {
        task.ret = guideboard_ocr_.run_mat(crop, &task.result);
      } catch (const cv::Exception& e) {
        task.result.status = PPOCR_STATUS_INFERENCE_FAILED;
        task.result.error = e.what();
      } catch (const std::exception& e) {
        task.result.status = PPOCR_STATUS_INFERENCE_FAILED;
        task.result.error = e.what();
      } catch (...) {
        task.result.status = PPOCR_STATUS_INFERENCE_FAILED;
        task.result.error = "unknown OCR exception";
      }
      return task;
    });
    lane_decision_.setGuideboardBranchHint(stable_guideboard_branch_, true);
  }

  void initialize() {
    detection_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/detection/results", 10);
    label_pub_ = create_publisher<std_msgs::msg::String>("/detection/labels", 10);
    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    offset_y07_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/offset_y07", sensor_qos);
    offset_y08_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/offset_y08", sensor_qos);
    offset_y09_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/offset_y09", sensor_qos);
    heading_error_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/heading_error", sensor_qos);
    curvature_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/curvature", sensor_qos);
    is_valid_pub_ = create_publisher<std_msgs::msg::Bool>("/segmentation/is_valid", sensor_qos);
    stop_request_pub_ = create_publisher<std_msgs::msg::Bool>("/perception/stop_request", 10);
    lane_state_pub_ = create_publisher<std_msgs::msg::String>("/perception/lane_state", 10);
    lane_debug_pub_ = create_publisher<std_msgs::msg::String>("/perception/lane_debug", 10);
    encoder_count_sub_ = create_subscription<std_msgs::msg::Int64>(
      encoder_count_topic_, rclcpp::QoS(10).reliable(),
      [this](const std_msgs::msg::Int64::SharedPtr msg) {
        lane_decision_.setEncoderCount(msg->data, nowSeconds());
      });

    shm_reader_ = std::make_unique<ShmReader>(shm_name_);
    if (!shm_reader_->connect()) {
      RCLCPP_WARN(get_logger(), "SHM not connected yet: /dev/shm/%s", shm_name_.c_str());
    }

    if (!detector_.init(det_model_path_, label_list_path_, det_core_id_, det_input_width_,
                        det_input_height_, det_conf_threshold_, det_nms_threshold_,
                        det_raw_output_)) {
      throw std::runtime_error("failed to initialize detector model");
    }
    if (!segmenter_.init(seg_model_path_, seg_core_ids_, seg_input_width_, seg_input_height_,
                         seg_crop_y0_ratio_, seg_crop_y1_ratio_, seg_pad_value_,
                         seg_conf_threshold_, seg_nms_threshold_, seg_mask_threshold_,
                         seg_nms_contain_threshold_, seg_max_detections_, seg_raw_output_)) {
      throw std::runtime_error("failed to initialize segmentation model");
    }
    if (enable_guideboard_ocr_) {
      int ret = guideboard_ocr_.init(ocr_det_model_path_.c_str(), ocr_rec_model_path_.c_str());
      if (ret != 0) {
        throw std::runtime_error("failed to initialize guideboard OCR model");
      }
      guideboard_ocr_ready_ = true;
      RCLCPP_INFO(get_logger(), "guideboard OCR ready: det=%s rec=%s",
                  ocr_det_model_path_.c_str(), ocr_rec_model_path_.c_str());
    }

    std_msgs::msg::String labels_msg;
    labels_msg.data = joinLabels(detector_.labels());
    label_pub_->publish(labels_msg);

    perf_start_sec_ = nowSeconds();
    RCLCPP_INFO(get_logger(),
                "fused perception ready: det=%s seg=%s shm=/dev/shm/%s show_window=%d blend_alpha=%.2f",
                det_model_path_.c_str(), seg_model_path_.c_str(), shm_name_.c_str(),
                show_window_, blend_alpha_);
  }

  void tick() {
    if (busy_.exchange(true)) {
      return;
    }

    auto clear_busy = std::shared_ptr<void>(nullptr, [this](void*) { busy_.store(false); });

    Frame frame;
    if (!shm_reader_->readLatest(frame)) {
      return;
    }

    if (last_fid_ != 0 && frame.fid > last_fid_) {
      upstream_frames_ += static_cast<uint64_t>(frame.fid - last_fid_);
    }
    last_fid_ = frame.fid;

    auto frame_rgb = frame.image;
    if (enable_flip_) {
      cv::flip(frame_rgb, frame_rgb, flip_code_);
    }
    if (input_format_ == "BGR" || input_format_ == "bgr") {
      cv::cvtColor(frame_rgb, frame_rgb, cv::COLOR_BGR2RGB);
    }

    PerfStats stats;
    std::vector<Detection> detections;
    cv::Mat seg_map;

    auto det_future = std::async(std::launch::async, [&]() {
      std::vector<Detection> local_detections;
      double rknn_ms = 0.0;
      double post_ms = 0.0;
      bool ok = detector_.infer(frame_rgb, local_detections, &rknn_ms, &post_ms);
      return std::make_tuple(ok, std::move(local_detections), rknn_ms, post_ms);
    });

    auto seg_future = std::async(std::launch::async, [&]() {
      cv::Mat local_seg_map;
      std::vector<YoloSegInstance> local_instances;
      double rknn_ms = 0.0;
      double post_ms = 0.0;
      bool ok = segmenter_.infer(frame_rgb, local_seg_map, &rknn_ms, &post_ms);
      local_instances = segmenter_.lastInstances();
      return std::make_tuple(ok, local_seg_map, std::move(local_instances), rknn_ms, post_ms);
    });

    auto det_result = det_future.get();
    auto seg_result = seg_future.get();
    bool det_ok = std::get<0>(det_result);
    bool seg_ok = std::get<0>(seg_result);
    detections = std::move(std::get<1>(det_result));
    seg_map = std::get<1>(seg_result);
    std::vector<YoloSegInstance> seg_instances = std::move(std::get<2>(seg_result));
    stats.det_rknn_ms = std::get<2>(det_result);
    stats.det_post_ms = std::get<3>(det_result);
    stats.seg_rknn_ms = std::get<3>(seg_result);
    stats.seg_post_ms = std::get<4>(seg_result);

    updateGuideboardOcr(frame_rgb, detections);

    auto t_decision0 = std::chrono::steady_clock::now();
    LaneState lane_state = lane_decision_.decide(seg_map, detections);
    auto t_decision1 = std::chrono::steady_clock::now();
    stats.decision_ms = std::chrono::duration<double, std::milli>(t_decision1 - t_decision0).count();
    const LaneDebugInfo& lane_debug = lane_decision_.debugInfo();
    logDecisionStatus(lane_state, lane_debug);

    refreshDebugParameters();
    if (show_window_ || enable_debug_screenshots_) {
      showDebugWindow(frame_rgb, seg_map, detections, seg_instances, lane_state, lane_debug);
    }

    auto t_pub0 = std::chrono::steady_clock::now();
    publishAll(detections, lane_state, lane_debug);
    auto t_pub1 = std::chrono::steady_clock::now();
    stats.publish_ms = std::chrono::duration<double, std::milli>(t_pub1 - t_pub0).count();

    ++processed_frames_;
    sum_det_rknn_ms_ += stats.det_rknn_ms;
    sum_det_post_ms_ += stats.det_post_ms;
    sum_seg_rknn_ms_ += stats.seg_rknn_ms;
    sum_seg_post_ms_ += stats.seg_post_ms;
    sum_decision_ms_ += stats.decision_ms;
    sum_publish_ms_ += stats.publish_ms;
    last_det_count_ = detections.size();
    const YoloSegStats& seg_stats = segmenter_.lastStats();
    last_seg_model_instances_ = seg_stats.kept_instances;
    last_seg_model_conf_mean_ = seg_stats.score_mean;
    last_seg_model_conf_min_ = seg_stats.score_min;
    last_seg_model_conf_max_ = seg_stats.score_max;
    sum_seg_model_score_ += static_cast<double>(seg_stats.score_mean) * seg_stats.kept_instances;
    sum_seg_model_instances_ += static_cast<uint64_t>(seg_stats.kept_instances);

    if (!det_ok || !seg_ok) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "inference failure: det_ok=%d seg_ok=%d", det_ok, seg_ok);
    }

    logPerfIfNeeded();
  }

  void publishAll(const std::vector<Detection>& detections, const LaneState& lane_state,
                  const LaneDebugInfo& lane_debug) {
    std_msgs::msg::Float32MultiArray det_msg;
    det_msg.data.reserve(detections.size() * 8);
    for (const auto& det : detections) {
      det_msg.data.push_back(static_cast<float>(det.class_id));
      det_msg.data.push_back(det.confidence);
      det_msg.data.push_back(det.bbox.x);
      det_msg.data.push_back(det.bbox.y);
      det_msg.data.push_back(det.bbox.x + det.bbox.width);
      det_msg.data.push_back(det.bbox.y + det.bbox.height);
      det_msg.data.push_back(det.center.x);
      det_msg.data.push_back(det.center.y);
    }
    if (publish_detections_) {
      detection_pub_->publish(det_msg);
    }

    std_msgs::msg::String labels_msg;
    std::vector<std::string> current_labels;
    current_labels.reserve(detections.size());
    for (const auto& det : detections) {
      current_labels.push_back(det.class_name);
    }
    labels_msg.data = joinLabels(current_labels);
    label_pub_->publish(labels_msg);

    std_msgs::msg::Float32 offset_y07_msg;
    offset_y07_msg.data = lane_state.offset_y07;
    offset_y07_pub_->publish(offset_y07_msg);

    std_msgs::msg::Float32 offset_y08_msg;
    offset_y08_msg.data = lane_state.offset_y08;
    offset_y08_pub_->publish(offset_y08_msg);

    std_msgs::msg::Float32 offset_y09_msg;
    offset_y09_msg.data = lane_state.offset_y09;
    offset_y09_pub_->publish(offset_y09_msg);

    std_msgs::msg::Float32 heading_msg;
    heading_msg.data = lane_state.heading_error;
    heading_error_pub_->publish(heading_msg);

    std_msgs::msg::Float32 curvature_msg;
    curvature_msg.data = lane_state.curvature;
    curvature_pub_->publish(curvature_msg);

    std_msgs::msg::Bool valid_msg;
    valid_msg.data = lane_state.is_valid;
    is_valid_pub_->publish(valid_msg);

    std_msgs::msg::Bool stop_msg;
    stop_msg.data = lane_state.task_state != "CLEAR";
    stop_request_pub_->publish(stop_msg);

    std_msgs::msg::String lane_msg;
    lane_msg.data = laneStateToJson(lane_state);
    if (publish_lane_state_) {
      lane_state_pub_->publish(lane_msg);
    }

    std_msgs::msg::String lane_debug_msg;
    lane_debug_msg.data = laneDebugToJson(lane_debug, seg_input_width_);
    lane_debug_pub_->publish(lane_debug_msg);
  }

  void showDebugWindow(const cv::Mat& frame_rgb, const cv::Mat& seg_map,
                       const std::vector<Detection>& detections,
                       const std::vector<YoloSegInstance>& seg_instances,
                       const LaneState& lane_state,
                       const LaneDebugInfo& debug_info) {
    if (frame_rgb.empty()) {
      return;
    }

    cv::Mat vis;
    cv::cvtColor(frame_rgb, vis, cv::COLOR_RGB2BGR);
    if (!seg_map.empty() && seg_map.size() == vis.size()) {
      float alpha = std::clamp(blend_alpha_, 0.0f, 1.0f);
      cv::Mat mask_u8;
      seg_map.convertTo(mask_u8, CV_8UC1, 255.0);

      cv::Mat overlay = vis.clone();
      overlay.setTo(cv::Scalar(255, 80, 20), mask_u8);
      if (alpha >= 0.999f) {
        vis = cv::Mat::zeros(vis.size(), vis.type());
        vis.setTo(cv::Scalar(255, 80, 20), mask_u8);
      } else if (alpha > 0.0f) {
        cv::addWeighted(vis, 1.0f - alpha, overlay, alpha, 0.0, vis);
      }
    }

    for (const auto& det : detections) {
      cv::Scalar color = detectionColor(det.class_name);
      cv::rectangle(vis, det.bbox, color, 2);
      cv::circle(vis, det.center, 3, color, -1);
      std::ostringstream label;
      label << det.class_name << " " << static_cast<int>(det.confidence * 100.0f) << "%";
      cv::putText(vis, label.str(), cv::Point(static_cast<int>(det.bbox.x), static_cast<int>(det.bbox.y) - 4),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
    }

    for (const auto& instance : seg_instances) {
      cv::Scalar color(255, 255, 0);
      cv::rectangle(vis, instance.bbox, color, 2);
      std::ostringstream label;
      label << "seg " << static_cast<int>(std::round(instance.score * 100.0f)) << "%";
      int x = std::clamp(static_cast<int>(std::round(instance.bbox.x)), 0, std::max(0, vis.cols - 1));
      int y = std::clamp(static_cast<int>(std::round(instance.bbox.y)) - 5, 12, std::max(12, vis.rows - 1));
      cv::putText(vis, label.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  color, 2, cv::LINE_AA);
    }

    if (show_branch_debug_) {
      drawLaneDebug(vis, debug_info);
    }

    std::ostringstream status;
    status << lane_state.road_state << " offsets=" << lane_state.offset_y07 << ","
           << lane_state.offset_y08 << "," << lane_state.offset_y09
           << " valid=" << (lane_state.is_valid ? 1 : 0) << " task=" << lane_state.task_state
           << " enc_hold=" << (debug_info.encoder_hold ? 1 : 0)
           << " enc_delta=" << debug_info.encoder_hold_delta
           << " lb_tpl=" << (debug_info.left_boundary_template_active ? 1 : 0)
           << " tpl=" << debug_info.boundary_template_side
           << " alpha=" << std::clamp(blend_alpha_, 0.0f, 1.0f);
    cv::putText(vis, status.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    if (show_window_) {
      cv::imshow("fused_perception", vis);
      int key = cv::waitKey(1) & 0xff;
      if (key == 's' || key == 'S') {
        saveDebugScreenshot(vis, "key");
      }
    }
    if (enable_debug_screenshots_) {
      double now = nowSeconds();
      bool branch_ok = !debug_screenshot_branch_only_ || debug_info.encoder_hold;
      double interval = std::max(0.0, debug_screenshot_interval_sec_);
      if (branch_ok && interval > 0.0 && now - last_debug_screenshot_sec_ >= interval) {
        saveDebugScreenshot(vis, debug_info.encoder_hold ? "branch" : "auto");
        last_debug_screenshot_sec_ = now;
      }
    }
  }

  void saveDebugScreenshot(const cv::Mat& vis, const std::string& reason) {
    if (vis.empty()) {
      return;
    }
    try {
      fs::create_directories(debug_screenshot_dir_);
      auto now = std::chrono::system_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
      std::string filename = "fused_perception_" + reason + "_" + std::to_string(ms) + "_" +
                             std::to_string(debug_screenshot_count_++) + ".jpg";
      fs::path path = fs::path(debug_screenshot_dir_) / filename;
      fs::path latest = fs::path(debug_screenshot_dir_) / "latest.jpg";
      cv::imwrite(path.string(), vis);
      cv::imwrite(latest.string(), vis);
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                           "saved debug screenshot: %s", path.string().c_str());
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "failed to save debug screenshot: %s", e.what());
    }
  }

  static cv::Scalar detectionColor(const std::string& class_name) {
    static const std::unordered_map<std::string, cv::Scalar> colors = {
        {"Human", cv::Scalar(0, 255, 0)},
        {"Car", cv::Scalar(0, 0, 255)},
        {"Stop", cv::Scalar(0, 165, 255)},
        {"Gold", cv::Scalar(0, 255, 255)},
        {"Go", cv::Scalar(255, 255, 0)},
        {"Gate", cv::Scalar(255, 0, 255)},
        {"GuideBoard", cv::Scalar(255, 255, 255)},
        {"red_light", cv::Scalar(0, 0, 255)},
        {"yellow_light", cv::Scalar(0, 255, 255)},
        {"green_light", cv::Scalar(0, 255, 0)},
        {"Zebra", cv::Scalar(255, 255, 255)},
    };
    auto it = colors.find(class_name);
    return it == colors.end() ? cv::Scalar(255, 255, 255) : it->second;
  }

  void drawLaneDebug(cv::Mat& vis, const LaneDebugInfo& debug_info) const {
    bool template_active = debug_info.left_boundary_template_active;
    for (const auto& band : debug_info.bands) {
      int y0 = std::clamp(band.y0, 0, std::max(0, vis.rows - 1));
      int y1 = std::clamp(band.y1, y0 + 1, vis.rows);
      cv::rectangle(vis, cv::Point(0, y0), cv::Point(vis.cols - 1, y1 - 1),
                    cv::Scalar(255, 200, 100), 1);

      for (const auto& seg : band.segments) {
        int x0 = std::clamp(seg.x0, 0, std::max(0, vis.cols - 1));
        int x1 = std::clamp(seg.x1, 0, std::max(0, vis.cols - 1));
        int cy = (y0 + y1) / 2;
        int center_x = std::clamp(static_cast<int>(std::round(seg.center_x)), 0, std::max(0, vis.cols - 1));
        if (template_active) {
          cv::circle(vis, cv::Point(center_x, cy), 1, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
          continue;
        }
        if (seg.virtual_segment) {
          cv::circle(vis, cv::Point(center_x, cy), 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          continue;
        }
        cv::line(vis, cv::Point(x0, y0), cv::Point(x0, y1 - 1), cv::Scalar(0, 255, 0), 1);
        cv::line(vis, cv::Point(x1, y0), cv::Point(x1, y1 - 1), cv::Scalar(0, 255, 0), 1);
        cv::circle(vis, cv::Point(center_x, cy), 3, cv::Scalar(0, 255, 255), -1);
      }

      if (band.selected_center_x >= 0) {
        int cy = (y0 + y1) / 2;
        cv::circle(vis, cv::Point(std::clamp(band.selected_center_x, 0, std::max(0, vis.cols - 1)), cy),
                   template_active ? 4 : 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
      }
    }

    const auto& points = debug_info.fit_points;
    for (const auto& point : points) {
      cv::circle(vis, cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))),
                 template_active ? 3 : 1, cv::Scalar(0, 255, 0), -1);
    }

    if (!debug_info.fit_coeffs.empty()) {
      int y0 = debug_info.bands.empty() ? 0 : debug_info.bands.front().y0;
      int y1 = debug_info.bands.empty() ? vis.rows - 1 : debug_info.bands.back().y1;
      if (!points.empty()) {
        auto [min_it, max_it] = std::minmax_element(
            points.begin(), points.end(),
            [](const cv::Point3f& a, const cv::Point3f& b) { return a.y < b.y; });
        y0 = std::clamp(static_cast<int>(std::round(min_it->y)), 0, std::max(0, vis.rows - 1));
        y1 = std::clamp(static_cast<int>(std::round(max_it->y)), y0 + 1, vis.rows - 1);
      }
      cv::Point prev;
      bool has_prev = false;
      for (int i = 0; i < 100; ++i) {
        double t = i / 99.0;
        double y = y0 + (y1 - y0) * t;
        double x = 0.0;
        for (double c : debug_info.fit_coeffs) {
          x = x * y + c;
        }
        cv::Point cur(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
        if (has_prev && prev.x >= 0 && prev.x < vis.cols && prev.y >= 0 && prev.y < vis.rows &&
            cur.x >= 0 && cur.x < vis.cols && cur.y >= 0 && cur.y < vis.rows) {
          cv::line(vis, prev, cur, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }
        prev = cur;
        has_prev = true;
      }
    } else {
      for (size_t i = 1; i < points.size(); ++i) {
        cv::Point p0(static_cast<int>(std::round(points[i - 1].x)),
                     static_cast<int>(std::round(points[i - 1].y)));
        cv::Point p1(static_cast<int>(std::round(points[i].x)),
                     static_cast<int>(std::round(points[i].y)));
        if (p0.x >= 0 && p0.x < vis.cols && p0.y >= 0 && p0.y < vis.rows &&
            p1.x >= 0 && p1.x < vis.cols && p1.y >= 0 && p1.y < vis.rows) {
          cv::line(vis, p0, p1, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }
      }
    }
  }

  void refreshDebugParameters() {
    show_window_ = get_parameter("show_window").as_bool();
    blend_alpha_ = std::clamp(static_cast<float>(get_parameter("blend_alpha").as_double()), 0.0f, 1.0f);
    show_branch_debug_ = get_parameter("show_branch_debug").as_bool();
    enable_debug_screenshots_ = get_parameter("enable_debug_screenshots").as_bool();
    debug_screenshot_interval_sec_ = get_parameter("debug_screenshot_interval_sec").as_double();
    debug_screenshot_branch_only_ = get_parameter("debug_screenshot_branch_only").as_bool();
    debug_screenshot_dir_ = get_parameter("debug_screenshot_dir").as_string();
    enable_status_log_ = get_parameter("enable_status_log").as_bool();
    enable_branch_event_log_ = get_parameter("enable_branch_event_log").as_bool();
    enable_perf_stats_ = get_parameter("enable_perf_stats").as_bool();
  }

  void logDecisionStatus(const LaneState& lane_state, const LaneDebugInfo& debug_info) {
    if (enable_branch_event_log_) {
      if (debug_info.guideboard_seen && !last_guideboard_seen_) {
        RCLCPP_INFO(get_logger(),
                    "GuideBoard usable: roi_count=%d total=%d best_conf=%.2f center=(%.1f,%.1f); "
                    "branch_detected=%d score=%d encoder_hold=%d side=%s",
                    debug_info.guideboard_roi_count, debug_info.guideboard_count,
                    debug_info.guideboard_best_confidence, debug_info.guideboard_best_center.x,
                    debug_info.guideboard_best_center.y, debug_info.branch_detected,
                    debug_info.branch_score, debug_info.encoder_hold,
                    debug_info.encoder_hold_side.c_str());
      } else if (!debug_info.guideboard_seen && debug_info.guideboard_count > 0 &&
                 !last_guideboard_seen_) {
        RCLCPP_INFO(get_logger(),
                    "GuideBoard detected but not usable: roi_count=%d total=%d best_conf=%.2f "
                    "center=(%.1f,%.1f)",
                    debug_info.guideboard_roi_count, debug_info.guideboard_count,
                    debug_info.guideboard_best_confidence, debug_info.guideboard_best_center.x,
                    debug_info.guideboard_best_center.y);
      }
      if (debug_info.branch_detected != last_branch_detected_ ||
          debug_info.branch_score != last_branch_score_) {
        RCLCPP_INFO(get_logger(),
                    "branch_detected=%d score=%d encoder_hold=%d encoder_delta=%ld/%ld "
                    "lb_tpl=%d tpl_side=%s lb_pts=%d lb_reason=%s "
                    "segments=%d raw_points=%d fit_points=%d",
                    debug_info.branch_detected, debug_info.branch_score,
                    debug_info.encoder_hold, static_cast<long>(debug_info.encoder_hold_delta),
                    static_cast<long>(debug_info.encoder_hold_target),
                    debug_info.left_boundary_template_active, debug_info.boundary_template_side.c_str(),
                    debug_info.left_boundary_template_points, debug_info.left_boundary_template_reason.c_str(),
                    debug_info.segment_count,
                    debug_info.raw_point_count, debug_info.fit_point_count);
      }
      if (lane_state.road_state != last_road_state_ ||
          lane_state.branch_side != last_branch_side_) {
        RCLCPP_INFO(get_logger(), "road_state=%s branch_side=%s offset=%.3f valid=%d guideboard=%d",
                    lane_state.road_state.c_str(), lane_state.branch_side.c_str(),
                    lane_state.offset_y09, lane_state.is_valid, debug_info.guideboard_seen);
      }
      if (lane_state.task_state != last_task_state_) {
        RCLCPP_INFO(get_logger(), "task_state=%s stop_request=%d",
                    lane_state.task_state.c_str(), lane_state.task_state != "CLEAR");
      }

      last_guideboard_seen_ = debug_info.guideboard_seen;
      last_branch_detected_ = debug_info.branch_detected;
      last_branch_score_ = debug_info.branch_score;
      last_road_state_ = lane_state.road_state;
      last_branch_side_ = lane_state.branch_side;
      last_task_state_ = lane_state.task_state;
    }

    if (!enable_status_log_) {
      return;
    }
    double now = nowSeconds();
    if (now - last_status_log_sec_ < 0.5) {
      return;
    }
    const YoloSegStats& seg_stats = segmenter_.lastStats();
    RCLCPP_INFO(get_logger(),
                "status road=%s branch=%s offsets=%.3f,%.3f,%.3f heading=%.3f conf=%.2f valid=%d "
                "seg_conf=%.3f[%.3f,%.3f]/%d "
                "branch_detected=%d score=%d guideboard_roi=%d/%d guideboard_best=%.2f@(%.0f,%.0f) "
                "encoder_hold=%d encoder_delta=%ld/%ld encoder_valid=%d age=%.2f "
                "lb_tpl=%d tpl_side=%s lb_pts=%d lb_reason=%s obstacles=%zu segments=%d points=%d/%d task=%s",
                lane_state.road_state.c_str(), lane_state.branch_side.c_str(),
                lane_state.offset_y07, lane_state.offset_y08, lane_state.offset_y09,
                lane_state.heading_error,
                lane_state.confidence, lane_state.is_valid, seg_stats.score_mean,
                seg_stats.score_min, seg_stats.score_max, seg_stats.kept_instances,
                debug_info.branch_detected,
                debug_info.branch_score, debug_info.guideboard_roi_count, debug_info.guideboard_count,
                debug_info.guideboard_best_confidence, debug_info.guideboard_best_center.x,
                debug_info.guideboard_best_center.y, debug_info.encoder_hold,
                static_cast<long>(debug_info.encoder_hold_delta),
                static_cast<long>(debug_info.encoder_hold_target),
                debug_info.encoder_feedback_valid, debug_info.encoder_feedback_age,
                debug_info.left_boundary_template_active, debug_info.boundary_template_side.c_str(),
                debug_info.left_boundary_template_points, debug_info.left_boundary_template_reason.c_str(),
                debug_info.obstacle_zones.size(), debug_info.segment_count,
                debug_info.raw_point_count, debug_info.fit_point_count,
                lane_state.task_state.c_str());
    last_status_log_sec_ = now;
  }

  void logPerfIfNeeded() {
    if (!enable_perf_stats_) {
      return;
    }
    double now = nowSeconds();
    double dt = now - perf_start_sec_;
    if (dt < perf_interval_) {
      return;
    }

    uint64_t frames = std::max<uint64_t>(1, processed_frames_);
    double processed_fps = processed_frames_ / std::max(0.001, dt);
    double upstream_fps = upstream_frames_ / std::max(0.001, dt);
    double seg_model_conf_avg = sum_seg_model_instances_ > 0
                                    ? sum_seg_model_score_ / static_cast<double>(sum_seg_model_instances_)
                                    : 0.0;
    RCLCPP_INFO(get_logger(),
                "perf %.1fs upstream=%.1f fps processed=%.1f fps det=%.2f/%.2f ms "
                "seg=%.2f/%.2f ms ocr=%.2f ms/%lu decision=%.2f ms publish=%.2f ms detections=%zu "
                "seg_model_conf_avg=%.3f seg_model_instances=%lu last_seg_conf=%.3f[%.3f,%.3f]/%d",
                dt, upstream_fps, processed_fps, sum_det_rknn_ms_ / frames,
                sum_det_post_ms_ / frames, sum_seg_rknn_ms_ / frames,
                sum_seg_post_ms_ / frames,
                ocr_run_count_ > 0 ? sum_ocr_ms_ / static_cast<double>(ocr_run_count_) : 0.0,
                static_cast<unsigned long>(ocr_run_count_), sum_decision_ms_ / frames,
                sum_publish_ms_ / frames, last_det_count_, seg_model_conf_avg,
                static_cast<unsigned long>(sum_seg_model_instances_), last_seg_model_conf_mean_,
                last_seg_model_conf_min_, last_seg_model_conf_max_,
                last_seg_model_instances_);

    perf_start_sec_ = now;
    processed_frames_ = 0;
    upstream_frames_ = 0;
    sum_det_rknn_ms_ = 0.0;
    sum_det_post_ms_ = 0.0;
    sum_seg_rknn_ms_ = 0.0;
    sum_seg_post_ms_ = 0.0;
    sum_decision_ms_ = 0.0;
    sum_publish_ms_ = 0.0;
    sum_ocr_ms_ = 0.0;
    ocr_run_count_ = 0;
    sum_seg_model_score_ = 0.0;
    sum_seg_model_instances_ = 0;
  }

  std::string shm_name_;
  std::string encoder_count_topic_{"/chassis/encoder_count"};
  bool enable_flip_{true};
  int flip_code_{0};
  std::string input_format_{"RGB"};
  bool enable_perf_stats_{true};
  double perf_interval_{2.0};
  bool show_window_{false};
  float blend_alpha_{1.0f};
  bool show_branch_debug_{true};
  bool enable_debug_screenshots_{false};
  double debug_screenshot_interval_sec_{0.0};
  bool debug_screenshot_branch_only_{false};
  std::string debug_screenshot_dir_{"/tmp/fused_perception_screenshots"};
  double last_debug_screenshot_sec_{0.0};
  uint64_t debug_screenshot_count_{0};
  bool publish_detections_{true};
  bool publish_lane_state_{true};
  bool enable_status_log_{false};
  bool enable_branch_event_log_{false};

  std::string det_model_path_;
  std::string label_list_path_;
  int det_core_id_{0};
  int det_input_width_{384};
  int det_input_height_{288};
  float det_conf_threshold_{0.5f};
  float det_nms_threshold_{0.45f};
  bool det_raw_output_{false};

  std::string seg_model_path_;
  std::vector<int> seg_core_ids_{1, 2};
  int seg_input_width_{384};
  int seg_input_height_{160};
  float seg_crop_y0_ratio_{0.0f};
  float seg_crop_y1_ratio_{1.0f};
  int seg_pad_value_{114};
  float seg_conf_threshold_{0.45f};
  float seg_nms_threshold_{0.45f};
  float seg_nms_contain_threshold_{0.85f};
  float seg_mask_threshold_{0.45f};
  int seg_max_detections_{30};
  bool seg_raw_output_{false};

  std::unique_ptr<ShmReader> shm_reader_;
  YoloDetector detector_;
  YoloSeg segmenter_;
  LaneDecision lane_decision_;
  std::atomic<bool> busy_{false};

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr detection_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr label_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y07_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y08_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y09_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr heading_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr curvature_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_valid_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_request_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_debug_pub_;
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr encoder_count_sub_;

  uint64_t last_fid_{0};
  uint64_t upstream_frames_{0};
  uint64_t processed_frames_{0};
  double perf_start_sec_{0.0};
  double sum_det_rknn_ms_{0.0};
  double sum_det_post_ms_{0.0};
  double sum_seg_rknn_ms_{0.0};
  double sum_seg_post_ms_{0.0};
  double sum_decision_ms_{0.0};
  double sum_publish_ms_{0.0};
  double sum_ocr_ms_{0.0};
  uint64_t ocr_run_count_{0};
  double sum_seg_model_score_{0.0};
  uint64_t sum_seg_model_instances_{0};
  size_t last_det_count_{0};
  int last_seg_model_instances_{0};
  float last_seg_model_conf_mean_{0.0f};
  float last_seg_model_conf_min_{0.0f};
  float last_seg_model_conf_max_{0.0f};
  double last_status_log_sec_{0.0};
  bool last_guideboard_seen_{false};
  bool last_branch_detected_{false};
  int last_branch_score_{-1};
  std::string last_road_state_;
  std::string last_branch_side_;
  std::string last_task_state_;

  bool enable_guideboard_ocr_{true};
  bool guideboard_ocr_ready_{false};
  std::string ocr_det_model_path_;
  std::string ocr_rec_model_path_;
  float ocr_min_score_{0.75f};
  float ocr_crop_padding_ratio_{0.25f};
  int ocr_vote_window_{3};
  float lane_guideboard_y0_ratio_{0.2f};
  float lane_guideboard_y1_ratio_{0.7f};
  PPOCRDirectionSystem guideboard_ocr_;
  std::future<OcrTaskResult> guideboard_ocr_future_;
  int ocr_task_crop_width_{0};
  int ocr_task_crop_height_{0};
  std::deque<int> guideboard_ocr_votes_;
  std::string stable_guideboard_branch_{"left"};
  int stable_guideboard_raw_direction_{0};
  double last_guideboard_ocr_sec_{0.0};
  std::string last_ocr_text_;
  int last_ocr_status_{PPOCR_STATUS_NO_TEXT};
  int last_ocr_raw_direction_{-1};
  float last_ocr_score_{0.0f};
  double last_ocr_time_ms_{0.0};
  bool last_ocr_touch_edge_{false};
};

}  // namespace track_perception_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<track_perception_cpp::FusedPerceptionNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("fused_perception_node"), "%s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
