#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <opencv2/opencv.hpp>

#include "ppocr_direction_system.h"
#include "track_perception_cpp/guideboard_api_client.hpp"
#include "track_perception_cpp/guideboard_recognizer.hpp"
#include "track_perception_cpp/guideboard_route_policy.hpp"
#include "track_perception_cpp/lane_decision.hpp"
#include "track_perception_cpp/shm_reader.hpp"
#include "track_perception_cpp/yolo_detector.hpp"
#include "track_perception_cpp/yolo_seg.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace track_perception_cpp {

namespace {

constexpr int kMinOcrCropSizePx = 20;
constexpr int kRecBoardWidth = 192;
constexpr int kRecBoardHeight = 128;

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

double percentile95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t index = static_cast<size_t>(
      std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

double normalizeCenterX(int center_x, int image_width) {
  if (center_x < 0 || image_width <= 1) {
    return 0.0;
  }
  return (static_cast<double>(center_x) - 0.5 * static_cast<double>(image_width)) /
         (0.5 * static_cast<double>(image_width));
}

uint64_t frameSignature(const cv::Mat& image) {
  if (image.empty() || image.channels() != 3) {
    return 0;
  }

  // Sample a 40x30 grid. Exact equality is extremely unlikely once camera
  // pixels change, while the cost stays negligible beside RKNN inference.
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  const int row_step = std::max(1, image.rows / 30);
  const int col_step = std::max(1, image.cols / 40);
  for (int y = 0; y < image.rows; y += row_step) {
    const auto* row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < image.cols; x += col_step) {
      const auto& pixel = row[x];
      for (int channel = 0; channel < 3; ++channel) {
        hash ^= static_cast<uint64_t>(pixel[channel]);
        hash *= kFnvPrime;
      }
    }
  }
  hash ^= static_cast<uint64_t>(image.rows);
  hash *= kFnvPrime;
  hash ^= static_cast<uint64_t>(image.cols);
  hash *= kFnvPrime;
  return hash == 0 ? 1 : hash;
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
     << "\"car_boundary_active\":" << (debug_info.car_boundary_active ? "true" : "false") << ","
     << "\"car_left_x\":" << debug_info.car_left_x << ","
     << "\"car_filtered_point_count\":" << debug_info.car_filtered_point_count << ","
     << "\"car_boundary_lost_count\":" << debug_info.car_boundary_lost_count << ","
     << "\"car_push_active\":" << (debug_info.car_push_active ? "true" : "false") << ","
     << "\"car_push_target_x\":" << debug_info.car_push_target_x << ","
     << "\"car_push_expand_bottom_y\":" << debug_info.car_push_expand_bottom_y << ","
     << "\"car_pushed_point_count\":" << debug_info.car_pushed_point_count << ","
     << "\"car_deleted_point_count\":" << debug_info.car_deleted_point_count << ","
     << "\"car_bbox\":[" << debug_info.car_bbox.x << "," << debug_info.car_bbox.y << ","
     << debug_info.car_bbox.width << "," << debug_info.car_bbox.height << "],"
     << "\"car_expanded_bbox\":[" << debug_info.car_expanded_bbox.x << ","
     << debug_info.car_expanded_bbox.y << "," << debug_info.car_expanded_bbox.width << ","
     << debug_info.car_expanded_bbox.height << "],"
     << "\"human_passable\":" << (debug_info.human_passable ? "true" : "false") << ","
     << "\"human_line_intersects\":"
     << (debug_info.human_line_intersects ? "true" : "false") << ","
     << "\"human_stop_candidate\":" << (debug_info.human_stop_candidate ? "true" : "false") << ","
     << "\"human_stop_active\":" << (debug_info.human_stop_active ? "true" : "false") << ","
     << "\"human_clear_confirming\":"
     << (debug_info.human_clear_confirming ? "true" : "false") << ","
     << "\"human_raw_area_ratio\":" << debug_info.human_raw_area_ratio << ","
     << "\"human_state\":\"" << jsonEscape(debug_info.human_state) << "\","
     << "\"human_stop_confirm_count\":" << debug_info.human_stop_confirm_count << ","
     << "\"human_clear_confirm_count\":" << debug_info.human_clear_confirm_count << ","
     << "\"human_count_at_stop\":" << debug_info.human_count_at_stop << ","
     << "\"human_valid_count\":" << debug_info.human_valid_count << ","
     << "\"fit_hold_active\":" << (debug_info.fit_hold_active ? "true" : "false") << ","
     << "\"fit_hold_age\":" << debug_info.fit_hold_age << ","
     << "\"fit_order\":" << debug_info.fit_order << ","
     << "\"branch_detected\":" << (debug_info.branch_detected ? "true" : "false") << ","
     << "\"branch_score\":" << debug_info.branch_score << ","
     << "\"guideboard_hint_valid\":" << (debug_info.guideboard_hint_valid ? "true" : "false") << ","
     << "\"guideboard_waiting_for_hint\":"
     << (debug_info.guideboard_waiting_for_hint ? "true" : "false") << ","
     << "\"guideboard_hint_wait_elapsed\":" << debug_info.guideboard_hint_wait_elapsed << ","
     << "\"guideboard_seen_latched\":"
     << (debug_info.guideboard_seen_latched ? "true" : "false") << ","
     << "\"branch_event_armed\":"
     << (debug_info.branch_event_armed ? "true" : "false") << ","
     << "\"branch_event_rearmed\":"
     << (debug_info.branch_event_rearmed ? "true" : "false") << ","
     << "\"branch_lock_event\":"
     << (debug_info.branch_lock_event ? "true" : "false") << ","
     << "\"branch_lock_guideboard\":"
     << (debug_info.branch_lock_guideboard ? "true" : "false") << ","
     << "\"branch_event_id\":" << debug_info.branch_event_id << ","
     << "\"branch_decision_source\":\""
     << jsonEscape(debug_info.branch_decision_source) << "\","
     << "\"encoder_hold\":" << (debug_info.encoder_hold ? "true" : "false") << ","
     << "\"encoder_hold_side\":\"" << jsonEscape(debug_info.encoder_hold_side) << "\","
     << "\"encoder_count\":" << debug_info.encoder_count << ","
     << "\"encoder_hold_delta\":" << debug_info.encoder_hold_delta << ","
     << "\"encoder_hold_target\":" << debug_info.encoder_hold_target << ","
     << "\"encoder_feedback_valid\":" << (debug_info.encoder_feedback_valid ? "true" : "false") << ","
     << "\"encoder_feedback_age\":" << debug_info.encoder_feedback_age << ","
     << "\"centerline_kalman_enabled\":"
     << (debug_info.centerline_kalman_enabled ? "true" : "false") << ","
     << "\"centerline_kalman_point_count\":" << debug_info.centerline_kalman_point_count << ","
     << "\"centerline_kalman_encoder_delta\":" << debug_info.centerline_kalman_encoder_delta << ","
     << "\"centerline_kalman_motion_ratio\":" << debug_info.centerline_kalman_motion_ratio << ","
     << "\"centerline_kalman_process_noise\":" << debug_info.centerline_kalman_process_noise << ","
     << "\"centerline_kalman_steering\":" << debug_info.centerline_kalman_steering << ","
     << "\"centerline_kalman_reset_count\":" << debug_info.centerline_kalman_reset_count << ","
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
  ss << "],\"removed_fit_points\":[";
  for (size_t i = 0; i < debug_info.removed_fit_points.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    const auto& point = debug_info.removed_fit_points[i];
    ss << "[" << point.x << "," << point.y << "," << point.z << "]";
  }
  ss << "],\"pushed_fit_points\":[";
  for (size_t i = 0; i < debug_info.pushed_fit_points.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    const auto& point = debug_info.pushed_fit_points[i];
    ss << "[" << point.x << "," << point.y << "," << point.z << "]";
  }
  ss << "],\"fit_coeffs\":[";
  for (size_t i = 0; i < debug_info.fit_coeffs.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << debug_info.fit_coeffs[i];
  }
  ss << "],\"humans\":[";
  for (size_t i = 0; i < debug_info.humans.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    const auto& human = debug_info.humans[i];
    ss << "{\"raw_bbox\":["
       << human.raw_bbox.x << "," << human.raw_bbox.y << ","
       << human.raw_bbox.width << "," << human.raw_bbox.height << "],"
       << "\"expanded_bbox\":["
       << human.expanded_bbox.x << "," << human.expanded_bbox.y << ","
       << human.expanded_bbox.width << "," << human.expanded_bbox.height << "],"
       << "\"fit_sample_points\":[";
    for (size_t sample_index = 0; sample_index < human.fit_sample_points.size();
         ++sample_index) {
      if (sample_index > 0) {
        ss << ",";
      }
      const auto& sample = human.fit_sample_points[sample_index];
      ss << "[" << sample.x << "," << sample.y << "]";
    }
    ss << "],"
       << "\"raw_area_ratio\":" << human.raw_area_ratio << ","
       << "\"fit_available\":" << (human.fit_available ? "true" : "false") << ","
       << "\"line_intersects\":" << (human.line_intersects ? "true" : "false") << ","
       << "\"passable\":" << (human.passable ? "true" : "false") << ","
       << "\"stop_candidate\":" << (human.stop_candidate ? "true" : "false")
       << "}";
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

    parameter_callback_handle_ = add_on_set_parameters_callback(
        std::bind(&FusedPerceptionNode::parametersCallback, this, std::placeholders::_1));

    timer_ = create_wall_timer(1ms, std::bind(&FusedPerceptionNode::tick, this));
  }

  ~FusedPerceptionNode() override {
    waitForGuideboardOcr();
    waitForGuideboardApi();
  }

 private:
  struct OcrTaskResult {
    int ret{-1};
    PPOCRDirectionResult result{};
    int crop_width{0};
    int crop_height{0};
    uint64_t track_id{0};
    uint64_t route_session_id{0};
    uint64_t branch_event_id{0};
    std::string pipeline;
    std::string fallback_reason;
  };

  struct GuideboardApiTaskResult {
    uint64_t track_id{0};
    uint64_t route_session_id{0};
    uint64_t branch_event_id{0};
    int attempt{0};
    GuideboardApiResult result;
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
    // Ultralytics-style class-aware NMS IoU threshold (predict(..., iou=0.7)).
    declare_parameter<double>("seg_nms_threshold", 0.70);
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
    declare_parameter<int64_t>("encoder_hold_counts", 9000);
    declare_parameter<int64_t>("encoder_hold_right_counts", 20000);
    declare_parameter<double>("encoder_feedback_timeout_sec", 0.30);
    declare_parameter<bool>("enable_guideboard_branch_selection", true);
    declare_parameter<double>("guideboard_detect_y0_ratio", 0.2);
    declare_parameter<double>("guideboard_detect_y1_ratio", 0.7);
    declare_parameter<bool>("enable_guideboard_ocr", true);
    declare_parameter<bool>("enable_guideboard_api", true);
    declare_parameter<std::string>("guideboard_api_key", "");
    declare_parameter<std::string>("guideboard_api_url",
                                   "https://qianfan.baidubce.com/v2/chat/completions");
    declare_parameter<std::string>("guideboard_api_model", "qwen3.5-35b-a3b");
    declare_parameter<double>("guideboard_api_timeout_sec", 1.8);
    declare_parameter<double>("guideboard_api_uncertain_min_confidence", 0.75);
    declare_parameter<int>("guideboard_api_text_history_size", 3);
    declare_parameter<double>("guideboard_api_text_similarity", 0.70);
    declare_parameter<int>("guideboard_api_force_ocr_count_after_stop", 2);
    declare_parameter<double>("guideboard_api_stop_height_ratio", 0.12);
    declare_parameter<bool>("guideboard_api_retry_on_transport_failure", true);
    declare_parameter<int>("guideboard_api_max_attempts", 1);
    declare_parameter<std::string>("ocr_det_model_path", "model/ppocrv4_det.rknn");
    declare_parameter<std::string>("ocr_rec_model_path", "model/ppocrv4_rec.rknn");
    declare_parameter<std::string>("ocr_pipeline_mode", "rec_then_det");
    declare_parameter<bool>("ocr_apply_to_control", false);
    declare_parameter<double>("ocr_rec_interval_sec", 0.05);
    declare_parameter<int>("ocr_fallback_uncertain_count", 2);
    declare_parameter<double>("ocr_min_text_score", 0.35);
    declare_parameter<double>("ocr_evidence_decay", 0.7);
    declare_parameter<double>("ocr_stable_min_evidence", 4.0);
    declare_parameter<double>("ocr_stable_min_margin", 2.0);
    declare_parameter<int>("ocr_stable_frames", 2);
    declare_parameter<double>("ocr_branch_wait_timeout_sec", 0.20);
    declare_parameter<std::string>("ocr_unknown_maneuver", "straight");
    declare_parameter<std::vector<std::string>>(
        "ocr_template_ids",
        {"rough_right", "irony_straight", "phone_straight", "shortcut_right",
         "right_dead_end_straight", "left_rough_right_flat"});
    declare_parameter<std::vector<std::string>>(
        "ocr_template_texts",
        {"右道真的有点崎岖，但是直道真的走不了",
         "右道比直道好走多了？才怪！",
         "他们来电话了，说不让我方向盘往右打！",
         "我就说三个字：抄近道",
         "右道是一条不归路！",
         "左侧道路崎岖，右侧一马平川"});
    declare_parameter<std::vector<std::string>>(
        "ocr_template_maneuvers", {"right", "straight", "straight", "right", "straight", "right"});
    // Legacy parameters remain declared so old config files still load; closed-set OCR does not use them.
    declare_parameter<double>("ocr_min_score", 0.75);
    declare_parameter<double>("ocr_crop_padding_ratio", 0.25);
    declare_parameter<int>("ocr_vote_window", 3);
    declare_parameter<bool>("enable_segment_branch_logic", true);
    declare_parameter<int>("fit_min_points", 5);
    declare_parameter<int>("fit_order", 2);
    declare_parameter<int>("branch_fit_order", 2);
    declare_parameter<bool>("enable_fit_point_jump_filter", false);
    declare_parameter<double>("max_fit_point_dx_ratio", 0.22);
    declare_parameter<double>("max_fit_point_dx_px", 140.0);
    declare_parameter<bool>("enable_fit_point_trend_filter", false);
    declare_parameter<double>("fit_point_trend_residual_ratio", 0.12);
    declare_parameter<double>("fit_point_trend_residual_px", 80.0);
    declare_parameter<double>("fit_point_trend_slope_delta", 0.65);
    declare_parameter<int>("fit_point_trend_min_points", 6);
    declare_parameter<double>("fit_point_trend_min_keep_ratio", 0.75);
    declare_parameter<bool>("enable_obstacle_avoidance", false);
    declare_parameter<std::string>("obstacle_labels", "Car");
    declare_parameter<double>("obstacle_min_confidence", 0.45);
    declare_parameter<double>("obstacle_x_margin_px", 25.0);
    declare_parameter<double>("obstacle_y_margin_px", 20.0);
    declare_parameter<double>("obstacle_max_age", 0.3);
    declare_parameter<double>("obstacle_min_bottom_y_ratio", 0.30);
    declare_parameter<bool>("enable_human_obstacle_stop", true);
    declare_parameter<double>("human_horizontal_expand_px", 25.0);
    declare_parameter<double>("human_horizontal_expand_width_ratio", 0.50);
    declare_parameter<int>("human_line_sample_count", 5);
    declare_parameter<double>("human_stop_raw_area_ratio", 0.004);
    declare_parameter<int>("human_stop_confirm_frames", 2);
    declare_parameter<int>("human_clear_confirm_frames", 2);
    declare_parameter<bool>("enable_car_right_boundary_filter", true);
    declare_parameter<double>("car_boundary_x_margin_px", 0.0);
    declare_parameter<double>("car_boundary_y_margin_px", 0.0);
    declare_parameter<double>("car_boundary_smoothing_alpha", 0.5);
    declare_parameter<int>("car_boundary_lost_frames", 3);
    declare_parameter<double>("car_fit_hold_timeout_sec", 0.20);
    declare_parameter<bool>("enable_car_point_push_avoidance", false);
    declare_parameter<double>("car_push_expand_left_px", 40.0);
    declare_parameter<double>("car_push_expand_bottom_px", 20.0);
    declare_parameter<double>("car_push_expand_right_px", 0.0);
    declare_parameter<double>("car_push_expand_top_px", 0.0);
    declare_parameter<double>("car_push_clearance_px", 3.0);
    declare_parameter<bool>("enable_finish_stop", true);
    declare_parameter<double>("finish_stop_min_confidence", 0.45);
    declare_parameter<double>("finish_stop_arm_y_ratio", 0.70);
    declare_parameter<int>("finish_stop_lost_frames", 3);
    declare_parameter<double>("finish_stop_max_age", 0.5);
    declare_parameter<double>("offset_y07_ratio", 0.70);
    declare_parameter<double>("offset_y08_ratio", 0.80);
    declare_parameter<double>("offset_y09_ratio", 0.90);
    declare_parameter<double>("heading_y_ratio", 0.75);
    declare_parameter<double>("heading_near_ratio", 0.81);
    declare_parameter<double>("heading_mid_ratio", 0.68);
    declare_parameter<double>("heading_far_ratio", 0.55);
    declare_parameter<double>("heading_near_weight", 0.10);
    declare_parameter<double>("heading_mid_weight", 0.50);
    declare_parameter<double>("heading_far_weight", 0.40);
    declare_parameter<double>("heading_local_window_half_ratio", 0.06);
    declare_parameter<int>("heading_local_min_points", 3);
    declare_parameter<bool>("enable_centerline_kalman", true);
    declare_parameter<double>("centerline_kalman_measurement_noise", 0.0036);
    declare_parameter<double>("centerline_kalman_idle_process_noise", 0.0001);
    declare_parameter<double>("centerline_kalman_motion_process_noise", 0.0008);
    declare_parameter<double>("centerline_kalman_turn_process_noise", 0.0008);
    declare_parameter<double>("centerline_kalman_initial_variance", 0.01);
    declare_parameter<double>("centerline_kalman_encoder_reference_counts", 50.0);
    declare_parameter<double>("centerline_kalman_reset_innovation", 0.25);
    declare_parameter<double>("centerline_kalman_state_timeout_sec", 0.30);
    declare_parameter<double>("centerline_kalman_steering_timeout_sec", 0.25);
    declare_parameter<std::string>("centerline_kalman_steering_topic", "/cmd_vel");
    declare_parameter<bool>("enable_left_boundary_template_line", false);
    declare_parameter<std::string>("left_boundary_template_side", "left");
    declare_parameter<std::string>("left_boundary_template_offsets", "");
    declare_parameter<int>("left_boundary_template_min_points", 6);
    declare_parameter<double>("left_boundary_template_weight", 1.0);
    declare_parameter<std::string>("right_boundary_template_offsets", "");
    declare_parameter<int>("right_boundary_template_min_points", 6);
    declare_parameter<double>("right_boundary_template_weight", 1.0);
    declare_parameter<std::string>("outer_side", "left");
    declare_parameter<bool>("enable_result_log", true);
    declare_parameter<bool>("enable_data_log", false);
    declare_parameter<bool>("guideboard_log_only", true);
    declare_parameter<bool>("publish_lane_state", true);
    declare_parameter<std::string>("line_follower_start_service", "/line_follower/start");
    declare_parameter<std::string>("line_follower_stop_service", "/line_follower/stop");
    declare_parameter<double>("human_service_retry_interval_sec", 0.20);
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
    enable_result_log_ = get_parameter("enable_result_log").as_bool();
    enable_data_log_ = get_parameter("enable_data_log").as_bool();
    guideboard_log_only_ = get_parameter("guideboard_log_only").as_bool();
    line_follower_start_service_ = get_parameter("line_follower_start_service").as_string();
    line_follower_stop_service_ = get_parameter("line_follower_stop_service").as_string();
    human_service_retry_interval_sec_ = std::max(
        0.05, get_parameter("human_service_retry_interval_sec").as_double());
    enable_guideboard_ocr_ = get_parameter("enable_guideboard_ocr").as_bool();
    ocr_det_model_path_ = resolveOwnPackagePath(get_parameter("ocr_det_model_path").as_string());
    ocr_rec_model_path_ = resolveOwnPackagePath(get_parameter("ocr_rec_model_path").as_string());
    ocr_pipeline_mode_ = get_parameter("ocr_pipeline_mode").as_string();
    if (ocr_pipeline_mode_ != "rec_only" && ocr_pipeline_mode_ != "det_rec" &&
        ocr_pipeline_mode_ != "rec_then_det") {
      throw std::runtime_error("ocr_pipeline_mode must be rec_only, det_rec, or rec_then_det");
    }
    ocr_apply_to_control_ = get_parameter("ocr_apply_to_control").as_bool();
    enable_guideboard_api_ = get_parameter("enable_guideboard_api").as_bool();
    guideboard_api_url_ = get_parameter("guideboard_api_url").as_string();
    guideboard_api_model_ = get_parameter("guideboard_api_model").as_string();
    guideboard_api_timeout_sec_ = std::max(
        0.1, get_parameter("guideboard_api_timeout_sec").as_double());
    guideboard_api_uncertain_min_confidence_ = std::clamp(
        get_parameter("guideboard_api_uncertain_min_confidence").as_double(), 0.0, 1.0);
    guideboard_api_text_history_size_ = std::max(
        2, static_cast<int>(get_parameter("guideboard_api_text_history_size").as_int()));
    guideboard_api_text_similarity_ = std::clamp(
        get_parameter("guideboard_api_text_similarity").as_double(), 0.0, 1.0);
    guideboard_api_force_ocr_count_after_stop_ = std::max(
        1, static_cast<int>(get_parameter("guideboard_api_force_ocr_count_after_stop").as_int()));
    guideboard_api_stop_height_ratio_ = std::clamp(
        get_parameter("guideboard_api_stop_height_ratio").as_double(), 0.0, 1.0);
    guideboard_api_retry_on_transport_failure_ =
        get_parameter("guideboard_api_retry_on_transport_failure").as_bool();
    guideboard_api_max_attempts_ = std::clamp(
        static_cast<int>(get_parameter("guideboard_api_max_attempts").as_int()), 1, 2);
    const std::string configured_api_key = get_parameter("guideboard_api_key").as_string();
    const char* environment_api_key = std::getenv("QIANFAN_API_KEY");
    api_key_ = !configured_api_key.empty()
                   ? configured_api_key
                   : (environment_api_key == nullptr ? "" : environment_api_key);
    if (enable_guideboard_api_ && api_key_.empty()) {
      RCLCPP_WARN(get_logger(),
                  "GUIDEBOARD_API_CONFIG missing QIANFAN_API_KEY; first signed guide will use fallback");
    }
    ocr_rec_interval_sec_ = std::max(0.0, get_parameter("ocr_rec_interval_sec").as_double());
    ocr_fallback_uncertain_count_ =
        std::max(1, static_cast<int>(get_parameter("ocr_fallback_uncertain_count").as_int()));
    ocr_min_text_score_ = static_cast<float>(get_parameter("ocr_min_text_score").as_double());
    ocr_branch_wait_timeout_sec_ =
        std::max(0.0, get_parameter("ocr_branch_wait_timeout_sec").as_double());
    ocr_unknown_maneuver_ = get_parameter("ocr_unknown_maneuver").as_string();
    if (ocr_unknown_maneuver_ != "straight" && ocr_unknown_maneuver_ != "right") {
      throw std::runtime_error("ocr_unknown_maneuver must be straight or right");
    }
    ocr_crop_padding_ratio_ = static_cast<float>(get_parameter("ocr_crop_padding_ratio").as_double());
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
    lane_cfg.encoder_hold_right_counts = get_parameter("encoder_hold_right_counts").as_int();
    lane_cfg.encoder_feedback_timeout_sec = get_parameter("encoder_feedback_timeout_sec").as_double();
    encoder_count_topic_ = get_parameter("encoder_count_topic").as_string();
    lane_cfg.outer_side = get_parameter("outer_side").as_string();
    lane_cfg.enable_guideboard_branch_selection = get_parameter("enable_guideboard_branch_selection").as_bool();
    lane_cfg.guideboard_detect_y0_ratio = static_cast<float>(get_parameter("guideboard_detect_y0_ratio").as_double());
    lane_cfg.guideboard_detect_y1_ratio = static_cast<float>(get_parameter("guideboard_detect_y1_ratio").as_double());
    lane_cfg.guideboard_require_hint = enable_guideboard_ocr_ && ocr_apply_to_control_;
    lane_cfg.guideboard_unknown_branch = maneuverToBranch(ocr_unknown_maneuver_);
    // Only a signed branch waits. Give its asynchronous OCR/API round and
    // single transport retry enough time before the safe straight fallback.
    lane_cfg.guideboard_hint_wait_timeout_sec = std::max(
        ocr_branch_wait_timeout_sec_,
        guideboard_api_timeout_sec_ * guideboard_api_max_attempts_ + 0.5);
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
    lane_cfg.heading_y_ratio = static_cast<float>(get_parameter("heading_y_ratio").as_double());
    lane_cfg.heading_near_ratio = static_cast<float>(get_parameter("heading_near_ratio").as_double());
    lane_cfg.heading_mid_ratio = static_cast<float>(get_parameter("heading_mid_ratio").as_double());
    lane_cfg.heading_far_ratio = static_cast<float>(get_parameter("heading_far_ratio").as_double());
    heading_near_weight_ = get_parameter("heading_near_weight").as_double();
    heading_mid_weight_ = get_parameter("heading_mid_weight").as_double();
    heading_far_weight_ = get_parameter("heading_far_weight").as_double();
    lane_cfg.heading_near_weight = static_cast<float>(heading_near_weight_);
    lane_cfg.heading_mid_weight = static_cast<float>(heading_mid_weight_);
    lane_cfg.heading_far_weight = static_cast<float>(heading_far_weight_);
    lane_cfg.heading_local_window_half_ratio = static_cast<float>(
        get_parameter("heading_local_window_half_ratio").as_double());
    lane_cfg.heading_local_min_points = static_cast<int>(
        get_parameter("heading_local_min_points").as_int());
    lane_cfg.enable_centerline_kalman = get_parameter("enable_centerline_kalman").as_bool();
    lane_cfg.centerline_kalman_measurement_noise = static_cast<float>(
        get_parameter("centerline_kalman_measurement_noise").as_double());
    lane_cfg.centerline_kalman_idle_process_noise = static_cast<float>(
        get_parameter("centerline_kalman_idle_process_noise").as_double());
    lane_cfg.centerline_kalman_motion_process_noise = static_cast<float>(
        get_parameter("centerline_kalman_motion_process_noise").as_double());
    lane_cfg.centerline_kalman_turn_process_noise = static_cast<float>(
        get_parameter("centerline_kalman_turn_process_noise").as_double());
    lane_cfg.centerline_kalman_initial_variance = static_cast<float>(
        get_parameter("centerline_kalman_initial_variance").as_double());
    lane_cfg.centerline_kalman_encoder_reference_counts = static_cast<float>(
        get_parameter("centerline_kalman_encoder_reference_counts").as_double());
    lane_cfg.centerline_kalman_reset_innovation = static_cast<float>(
        get_parameter("centerline_kalman_reset_innovation").as_double());
    lane_cfg.centerline_kalman_state_timeout_sec =
        get_parameter("centerline_kalman_state_timeout_sec").as_double();
    lane_cfg.centerline_kalman_steering_timeout_sec =
        get_parameter("centerline_kalman_steering_timeout_sec").as_double();
    centerline_kalman_steering_topic_ =
        get_parameter("centerline_kalman_steering_topic").as_string();
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
    lane_cfg.obstacle_min_confidence = static_cast<float>(get_parameter("obstacle_min_confidence").as_double());
    lane_cfg.obstacle_x_margin_px = static_cast<float>(get_parameter("obstacle_x_margin_px").as_double());
    lane_cfg.obstacle_y_margin_px = static_cast<float>(get_parameter("obstacle_y_margin_px").as_double());
    lane_cfg.obstacle_min_bottom_y_ratio = static_cast<float>(get_parameter("obstacle_min_bottom_y_ratio").as_double());
    lane_cfg.enable_human_obstacle_stop = get_parameter("enable_human_obstacle_stop").as_bool();
    lane_cfg.human_horizontal_expand_px =
        static_cast<float>(get_parameter("human_horizontal_expand_px").as_double());
    lane_cfg.human_horizontal_expand_width_ratio =
        static_cast<float>(get_parameter("human_horizontal_expand_width_ratio").as_double());
    lane_cfg.human_line_sample_count =
        static_cast<int>(get_parameter("human_line_sample_count").as_int());
    lane_cfg.human_stop_raw_area_ratio =
        static_cast<float>(get_parameter("human_stop_raw_area_ratio").as_double());
    lane_cfg.human_stop_confirm_frames = static_cast<int>(get_parameter("human_stop_confirm_frames").as_int());
    lane_cfg.human_clear_confirm_frames =
        static_cast<int>(get_parameter("human_clear_confirm_frames").as_int());
    lane_cfg.enable_car_right_boundary_filter =
        get_parameter("enable_car_right_boundary_filter").as_bool();
    lane_cfg.car_boundary_x_margin_px =
        static_cast<float>(get_parameter("car_boundary_x_margin_px").as_double());
    lane_cfg.car_boundary_y_margin_px =
        static_cast<float>(get_parameter("car_boundary_y_margin_px").as_double());
    lane_cfg.car_boundary_smoothing_alpha =
        static_cast<float>(get_parameter("car_boundary_smoothing_alpha").as_double());
    lane_cfg.car_boundary_lost_frames =
        static_cast<int>(get_parameter("car_boundary_lost_frames").as_int());
    lane_cfg.car_fit_hold_timeout_sec = get_parameter("car_fit_hold_timeout_sec").as_double();
    lane_cfg.enable_car_point_push_avoidance =
        get_parameter("enable_car_point_push_avoidance").as_bool();
    lane_cfg.car_push_expand_left_px =
        static_cast<float>(get_parameter("car_push_expand_left_px").as_double());
    lane_cfg.car_push_expand_bottom_px =
        static_cast<float>(get_parameter("car_push_expand_bottom_px").as_double());
    lane_cfg.car_push_expand_right_px =
        static_cast<float>(get_parameter("car_push_expand_right_px").as_double());
    lane_cfg.car_push_expand_top_px =
        static_cast<float>(get_parameter("car_push_expand_top_px").as_double());
    lane_cfg.car_push_clearance_px =
        static_cast<float>(get_parameter("car_push_clearance_px").as_double());
    lane_cfg.enable_finish_stop = get_parameter("enable_finish_stop").as_bool();
    lane_cfg.finish_stop_min_confidence = static_cast<float>(get_parameter("finish_stop_min_confidence").as_double());
    lane_cfg.finish_stop_arm_y_ratio = static_cast<float>(get_parameter("finish_stop_arm_y_ratio").as_double());
    lane_cfg.finish_stop_lost_frames = static_cast<int>(get_parameter("finish_stop_lost_frames").as_int());
    human_stop_confirm_frames_ = std::max(1, lane_cfg.human_stop_confirm_frames);
    human_clear_confirm_frames_ = std::max(1, lane_cfg.human_clear_confirm_frames);
    lane_decision_.configure(lane_cfg);
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
      const std::vector<rclcpp::Parameter>& parameters) {
    double near_weight = heading_near_weight_;
    double mid_weight = heading_mid_weight_;
    double far_weight = heading_far_weight_;
    bool heading_weights_changed = false;

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& parameter : parameters) {
      const auto& name = parameter.get_name();
      if (name != "heading_near_weight" && name != "heading_mid_weight" &&
          name != "heading_far_weight") {
        continue;
      }
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
        result.successful = false;
        result.reason = name + " must be a floating-point number";
        return result;
      }

      heading_weights_changed = true;
      if (name == "heading_near_weight") {
        near_weight = parameter.as_double();
      } else if (name == "heading_mid_weight") {
        mid_weight = parameter.as_double();
      } else {
        far_weight = parameter.as_double();
      }
    }

    if (!heading_weights_changed) {
      return result;
    }
    if (!std::isfinite(near_weight) || !std::isfinite(mid_weight) ||
        !std::isfinite(far_weight)) {
      result.successful = false;
      result.reason = "heading weights must be finite";
      return result;
    }
    if (near_weight < 0.0 || mid_weight < 0.0 || far_weight < 0.0) {
      result.successful = false;
      result.reason = "heading weights must be non-negative";
      return result;
    }
    const double weight_sum = near_weight + mid_weight + far_weight;
    if (weight_sum <= 1e-9) {
      result.successful = false;
      result.reason = "at least one heading weight must be greater than zero";
      return result;
    }

    heading_near_weight_ = near_weight;
    heading_mid_weight_ = mid_weight;
    heading_far_weight_ = far_weight;
    lane_decision_.setHeadingWeights(near_weight, mid_weight, far_weight);
    RCLCPP_INFO(get_logger(),
                "Updated heading weights: near=%.3f mid=%.3f far=%.3f (normalized %.3f/%.3f/%.3f)",
                near_weight, mid_weight, far_weight, near_weight / weight_sum,
                mid_weight / weight_sum, far_weight / weight_sum);
    return result;
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

  cv::Rect makePaddedCropRect(const cv::Rect2f& bbox, const cv::Size& image_size,
                              float padding_ratio) const {
    float pad = std::max(bbox.width, bbox.height) * std::max(0.0f, padding_ratio);
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

  static float bboxIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float intersection = (a & b).area();
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
  }

  std::string maneuverToBranch(const std::string& maneuver) const {
    return maneuver == "right" ? "right" : "left";
  }

  void logGuideboardApiEvent(const std::string& event, const std::string& reason,
                             const std::string& text, double score,
                             const std::string& current_maneuver,
                             const std::string& corrected_text, bool is_opposite) {
    if (!enable_data_log_) {
      return;
    }
    RCLCPP_INFO(
        get_logger(),
        "%s track_id=%lu sequence=%lu height_ratio=%.3f ocr_history=%s "
        "ocr_score=%.3f api_attempts=%d api_latency_ms=%.1f http_status=%d "
        "route_phase=%s encounter=%d current=%s opposite=%d corrected_text=%s "
        "stop_wait_active=%d trigger_reason=%s fallback=%d route_session=%lu",
        event.c_str(), static_cast<unsigned long>(guideboard_track_id_),
        static_cast<unsigned long>(guideboard_sequence_), last_guideboard_height_ratio_,
        text.c_str(), score, api_attempt_count_, last_api_latency_ms_,
        last_api_http_status_, guideboard_route_policy_.phaseName(),
        guideboard_route_policy_.signedEncounterIndex(), current_maneuver.c_str(),
        is_opposite, corrected_text.c_str(), guideboard_stop_wait_active_, reason.c_str(),
        api_fallback_, static_cast<unsigned long>(guideboard_route_session_id_));
  }

  const char* ocrControlMode() const {
    return ocr_apply_to_control_ ? "ocr" : "shadow";
  }

  void logGuideboardResult(const std::string& event, uint64_t track_id,
                           const std::string& sign, const std::string& action,
                           const std::string& branch, const std::string& reason,
                           double model_score, double evidence, double margin,
                           double latency_ms) {
    if (!enable_result_log_) {
      return;
    }
    RCLCPP_INFO(get_logger(),
                "OCR_RESULT event=%s track=%lu sign=%s action=%s branch=%s "
                "model=%.3f evidence=%.3f margin=%.3f latency_ms=%.1f "
                "control=%s reason=%s",
                event.c_str(), static_cast<unsigned long>(track_id), sign.c_str(),
                action.c_str(), branch.c_str(), model_score, evidence, margin,
                latency_ms, ocrControlMode(), reason.c_str());
  }

  void resetGuideboardTrack(bool create_new_track_id = true) {
    if (create_new_track_id) {
      ++guideboard_track_id_;
    }
    has_guideboard_track_ = false;
    guideboard_track_route_eligible_ = false;
    tracked_guideboard_bbox_ = cv::Rect2f{};
    guideboard_last_seen_sec_ = 0.0;
    last_guideboard_height_ratio_ = 0.0;
  }

  void resetGuideboardRecognitionSession() {
    ++guideboard_route_session_id_;
    active_guideboard_branch_event_id_ = 0;
    guideboard_uncertain_count_ = 0;
    guideboard_fallback_used_ = false;
    current_guideboard_maneuver_.clear();
    current_guideboard_branch_.clear();
    current_guideboard_decision_source_.clear();
    current_guideboard_decision_valid_ = false;
    current_guideboard_opposite_ = false;
    guideboard_decision_start_sec_ = 0.0;
    stable_decision_latency_ms_ = -1.0;
    last_guideboard_ocr_sec_ = 0.0;
    guideboard_api_ocr_history_.clear();
    guideboard_api_force_ocr_count_ = 0;
    api_session_attempted_ = false;
    api_fallback_ = false;
    api_trigger_reason_.clear();
    api_attempt_count_ = 0;
    last_api_latency_ms_ = -1.0;
    last_api_http_status_ = 0;
    last_api_confidence_ = 0.0f;
    last_api_uncertain_ = false;
    last_api_accepted_uncertain_ = false;
    last_api_corrected_text_.clear();
    lane_decision_.setGuideboardBranchHint("", false);
  }

  void startGuideboardTrack(const cv::Rect2f& bbox, double now) {
    resetGuideboardTrack(true);
    has_guideboard_track_ = true;
    tracked_guideboard_bbox_ = bbox;
    guideboard_last_seen_sec_ = now;
    ++guideboard_sequence_;
    guideboard_track_route_eligible_ = accept_guideboard_for_route_ &&
                                       lane_decision_.branchEventArmed();
    if (guideboard_track_route_eligible_ &&
        guideboard_route_policy_.prepareKnownDecision()) {
      setCurrentGuideboardDecision(guideboard_route_policy_.preparedAction(),
                                   guideboard_route_policy_.decisionSource());
    }
    if (guideboard_track_route_eligible_) {
      active_guideboard_branch_event_id_ = lane_decision_.branchEventId();
    }
    logGuideboardResult("track_new", guideboard_track_id_, "unknown",
                        current_guideboard_decision_valid_
                            ? current_guideboard_maneuver_
                            : "unknown",
                        current_guideboard_decision_valid_
                            ? current_guideboard_branch_
                            : "pending",
                        guideboard_route_policy_.phaseName(), -1.0, 0.0, 0.0, -1.0);
  }

  void updateGuideboardTrack(const cv::Rect2f& bbox, double now) {
    if (!has_guideboard_track_) {
      startGuideboardTrack(bbox, now);
      return;
    }
    const cv::Point2f previous_center(
        tracked_guideboard_bbox_.x + tracked_guideboard_bbox_.width * 0.5f,
        tracked_guideboard_bbox_.y + tracked_guideboard_bbox_.height * 0.5f);
    const cv::Point2f center(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
    const float center_distance = cv::norm(center - previous_center);
    const float previous_diagonal = std::hypot(tracked_guideboard_bbox_.width,
                                                tracked_guideboard_bbox_.height);
    const float current_diagonal = std::hypot(bbox.width, bbox.height);
    const bool center_jump = center_distance > 0.25f * std::max(previous_diagonal, current_diagonal);
    if (bboxIoU(tracked_guideboard_bbox_, bbox) < 0.2f && center_jump) {
      startGuideboardTrack(bbox, now);
      return;
    }
    tracked_guideboard_bbox_ = bbox;
    guideboard_last_seen_sec_ = now;
  }

  static std::array<cv::Point2f, 4> orderQuad(const std::vector<cv::Point>& points) {
    std::array<cv::Point2f, 4> ordered{};
    auto min_sum = std::min_element(points.begin(), points.end(), [](const auto& a, const auto& b) {
      return a.x + a.y < b.x + b.y;
    });
    auto max_sum = std::max_element(points.begin(), points.end(), [](const auto& a, const auto& b) {
      return a.x + a.y < b.x + b.y;
    });
    auto min_diff = std::min_element(points.begin(), points.end(), [](const auto& a, const auto& b) {
      return a.x - a.y < b.x - b.y;
    });
    auto max_diff = std::max_element(points.begin(), points.end(), [](const auto& a, const auto& b) {
      return a.x - a.y < b.x - b.y;
    });
    ordered[0] = *min_sum;
    ordered[1] = *max_diff;
    ordered[2] = *max_sum;
    ordered[3] = *min_diff;
    return ordered;
  }

  cv::Mat makeRecOnlyInput(const cv::Mat& frame_rgb, const cv::Rect2f& bbox) const {
    const cv::Rect crop_rect = makePaddedCropRect(bbox, frame_rgb.size(), 0.05f);
    if (crop_rect.width < kMinOcrCropSizePx || crop_rect.height < kMinOcrCropSizePx) {
      return {};
    }
    const cv::Mat crop = frame_rgb(crop_rect).clone();
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_RGB2HSV);
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(90, 60, 30), cv::Scalar(140, 255, 255), blue_mask);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blue_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat board;
    if (!contours.empty()) {
      auto largest = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
      if (cv::contourArea(*largest) >= 0.15 * static_cast<double>(crop.total())) {
        const double perimeter = cv::arcLength(*largest, true);
        std::vector<cv::Point> quad;
        cv::approxPolyDP(*largest, quad, 0.03 * perimeter, true);
        if (quad.size() != 4 || !cv::isContourConvex(quad)) {
          cv::Point2f rect_points[4];
          cv::minAreaRect(*largest).points(rect_points);
          quad.clear();
          for (const auto& point : rect_points) {
            quad.emplace_back(cvRound(point.x), cvRound(point.y));
          }
        }
        const auto src_points = orderQuad(quad);
        const std::array<cv::Point2f, 4> dst_points{{
            {0.0f, 0.0f}, {static_cast<float>(kRecBoardWidth - 1), 0.0f},
            {static_cast<float>(kRecBoardWidth - 1), static_cast<float>(kRecBoardHeight - 1)},
            {0.0f, static_cast<float>(kRecBoardHeight - 1)}}};
        cv::warpPerspective(crop, board,
                            cv::getPerspectiveTransform(src_points.data(), dst_points.data()),
                            cv::Size(kRecBoardWidth, kRecBoardHeight), cv::INTER_LINEAR,
                            cv::BORDER_REPLICATE);
      }
    }
    if (board.empty()) {
      cv::resize(crop, board, cv::Size(kRecBoardWidth, kRecBoardHeight), 0.0, 0.0,
                 cv::INTER_LINEAR);
    }

    const int x0 = std::clamp(cvRound(board.cols * 0.045), 0, board.cols - 2);
    const int x1 = std::clamp(cvRound(board.cols * 0.955), x0 + 1, board.cols);
    const int top_y0 = std::clamp(cvRound(board.rows * 0.17), 0, board.rows - 2);
    const int top_y1 = std::clamp(cvRound(board.rows * 0.49), top_y0 + 1, board.rows);
    const int bottom_y0 = std::clamp(cvRound(board.rows * 0.50), 0, board.rows - 2);
    const int bottom_y1 = std::clamp(cvRound(board.rows * 0.83), bottom_y0 + 1, board.rows);
    cv::Mat top = board(cv::Rect(x0, top_y0, x1 - x0, top_y1 - top_y0)).clone();
    cv::Mat bottom = board(cv::Rect(x0, bottom_y0, x1 - x0, bottom_y1 - bottom_y0)).clone();
    if (bottom.rows != top.rows) {
      cv::resize(bottom, bottom, cv::Size(bottom.cols, top.rows), 0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::Mat joined;
    cv::hconcat(top, bottom, joined);
    return joined;
  }

  void applyGuideboardHint() {
    const bool valid = ocr_apply_to_control_ && current_guideboard_decision_valid_ &&
                       accept_guideboard_for_route_;
    lane_decision_.setGuideboardBranchHint(valid ? current_guideboard_branch_ : "", valid,
                                           current_guideboard_decision_source_);
  }

  void publishGuideboardRouteEvent(const std::string& event, uint64_t branch_event_id,
                                   const std::string& source,
                                   const std::string& action,
                                   const std::string& branch) {
    std::ostringstream ss;
    ss << "{\"event\":\"" << jsonEscape(event) << "\""
       << ",\"track_id\":" << guideboard_track_id_
       << ",\"guideboard_sequence\":" << guideboard_sequence_
       << ",\"pipeline\":\"route_policy\""
       << ",\"text\":\"" << jsonEscape(last_ocr_text_) << "\""
       << ",\"model_score\":" << last_ocr_score_
       << ",\"route_phase\":\"" << guideboard_route_policy_.phaseName() << "\""
       << ",\"signed_encounter_index\":"
       << guideboard_route_policy_.signedEncounterIndex()
       << ",\"first_action\":\"" << jsonEscape(guideboard_route_policy_.firstAction())
       << "\""
       << ",\"pending_second_action\":\""
       << jsonEscape(guideboard_route_policy_.pendingSecondAction()) << "\""
       << ",\"branch_event_id\":" << branch_event_id
       << ",\"route_session_id\":" << guideboard_route_session_id_
       << ",\"decision_source\":\"" << jsonEscape(source) << "\""
       << ",\"action\":\"" << jsonEscape(action) << "\""
       << ",\"branch\":\"" << jsonEscape(branch) << "\""
       << ",\"control_enabled\":" << (ocr_apply_to_control_ ? "true" : "false")
       << ",\"api_attempt_count\":" << api_attempt_count_
       << ",\"api_latency_ms\":" << last_api_latency_ms_
       << ",\"api_http_status\":" << last_api_http_status_
       << ",\"api_confidence\":" << last_api_confidence_
       << ",\"api_uncertain\":" << (last_api_uncertain_ ? "true" : "false")
       << ",\"api_accepted_uncertain\":"
       << (last_api_accepted_uncertain_ ? "true" : "false")
       << ",\"ocr_history\":" << guideboardOcrHistoryJson() << "}";
    std_msgs::msg::String recognition_msg;
    recognition_msg.data = ss.str();
    guideboard_recognition_pub_->publish(recognition_msg);
  }

  void setCurrentGuideboardDecision(const std::string& maneuver,
                                    const std::string& reason) {
    if (maneuver != "straight" && maneuver != "right") {
      return;
    }
    const bool is_new_decision = !current_guideboard_decision_valid_ ||
                                 current_guideboard_maneuver_ != maneuver ||
                                 current_guideboard_decision_source_ != reason;
    current_guideboard_maneuver_ = maneuver;
    current_guideboard_branch_ = maneuverToBranch(current_guideboard_maneuver_);
    current_guideboard_decision_source_ = reason;
    current_guideboard_opposite_ = reason == "second_opposite";
    current_guideboard_decision_valid_ = true;
    if (stable_decision_latency_ms_ < 0.0 && guideboard_decision_start_sec_ > 0.0) {
      stable_decision_latency_ms_ =
          std::max(0.0, (nowSeconds() - guideboard_decision_start_sec_) * 1000.0);
      sum_ocr_decision_ms_ += stable_decision_latency_ms_;
      ++ocr_decision_count_;
      ocr_decision_latencies_ms_.push_back(stable_decision_latency_ms_);
    }
    if (is_new_decision) {
      logGuideboardResult("decision", guideboard_track_id_, "api",
                          current_guideboard_maneuver_, current_guideboard_branch_, reason,
                          last_ocr_score_, 1.0, 0.0, stable_decision_latency_ms_);
      publishGuideboardRouteEvent("decision_ready", lane_decision_.branchEventId(),
                                  reason, current_guideboard_maneuver_,
                                  current_guideboard_branch_);
    }
    applyGuideboardHint();
  }

  std::string guideboardOcrHistoryJson() const {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < guideboard_api_ocr_history_.size(); ++i) {
      if (i > 0) {
        ss << ",";
      }
      ss << "{\"text\":\"" << jsonEscape(guideboard_api_ocr_history_[i].text)
         << "\",\"score\":" << guideboard_api_ocr_history_[i].score << "}";
    }
    ss << "]";
    return ss.str();
  }

  std::string guideboardRecognitionJson(const OcrTaskResult& task) const {
    std::ostringstream ss;
    ss << "{\"track_id\":" << task.track_id
       << ",\"guideboard_sequence\":" << guideboard_sequence_
       << ",\"pipeline\":\"" << jsonEscape(task.pipeline) << "\""
       << ",\"text\":\"" << jsonEscape(task.result.text) << "\""
       << ",\"normalized_text\":\""
       << jsonEscape(GuideboardRecognizer::normalizeUtf8(task.result.text)) << "\""
       << ",\"model_score\":" << task.result.ocr_score
       << ",\"sign\":\"api_guideboard\""
       << ",\"maneuver\":\"" << jsonEscape(current_guideboard_maneuver_) << "\""
       << ",\"stable_template\":\"\""
       << ",\"action\":\"" << jsonEscape(current_guideboard_decision_valid_
                                                    ? current_guideboard_maneuver_
                                                    : "unknown") << "\""
       << ",\"eligible\":" << (!guideboard_api_ocr_history_.empty() ? "true" : "false")
       << ",\"stable\":" << (current_guideboard_decision_valid_ ? "true" : "false")
       << ",\"best_score\":0.0"
       << ",\"margin\":0.0"
       << ",\"reason\":\"" << jsonEscape(api_trigger_reason_) << "\""
       << ",\"fallback_reason\":\"" << jsonEscape(task.fallback_reason) << "\""
       << ",\"latency_ms\":" << task.result.time_ms
       << ",\"decision_latency_ms\":" << stable_decision_latency_ms_
       << ",\"control_enabled\":" << (ocr_apply_to_control_ ? "true" : "false")
       << ",\"return_code\":" << task.ret
       << ",\"status\":" << task.result.status
       << ",\"legacy_raw_direction\":" << task.result.raw_direction
       << ",\"route_phase\":\"" << guideboard_route_policy_.phaseName() << "\""
       << ",\"signed_encounter_index\":"
       << guideboard_route_policy_.signedEncounterIndex()
       << ",\"first_action\":\"" << jsonEscape(guideboard_route_policy_.firstAction()) << "\""
       << ",\"pending_second_action\":\""
       << jsonEscape(guideboard_route_policy_.pendingSecondAction()) << "\""
       << ",\"branch_event_id\":" << task.branch_event_id
       << ",\"route_session_id\":" << task.route_session_id
       << ",\"decision_source\":\"" << jsonEscape(current_guideboard_decision_source_)
       << "\""
       << ",\"current_maneuver\":\"" << jsonEscape(current_guideboard_maneuver_)
       << "\""
       << ",\"is_opposite_decision\":"
       << (current_guideboard_opposite_ ? "true" : "false")
       << ",\"api_attempt_count\":" << api_attempt_count_
       << ",\"api_trigger_reason\":\"" << jsonEscape(api_trigger_reason_) << "\""
       << ",\"api_latency_ms\":" << last_api_latency_ms_
       << ",\"api_http_status\":" << last_api_http_status_
       << ",\"api_confidence\":" << last_api_confidence_
       << ",\"api_uncertain\":" << (last_api_uncertain_ ? "true" : "false")
       << ",\"api_accepted_uncertain\":"
       << (last_api_accepted_uncertain_ ? "true" : "false")
       << ",\"api_fallback\":" << (api_fallback_ ? "true" : "false")
       << ",\"stop_wait_active\":"
       << (guideboard_stop_wait_active_ ? "true" : "false")
       << ",\"ocr_history\":" << guideboardOcrHistoryJson() << "}"
       ;
    return ss.str();
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
    last_ocr_score_ = result.ocr_score;
    last_ocr_time_ms_ = result.time_ms;

    if (result.time_ms > 0.0) {
      if (task.pipeline == "rec") {
        sum_ocr_rec_ms_ += result.time_ms;
        ++ocr_rec_run_count_;
        ocr_rec_latencies_ms_.push_back(result.time_ms);
      } else {
        sum_ocr_det_rec_ms_ += result.time_ms;
        ++ocr_det_rec_run_count_;
        ocr_det_rec_latencies_ms_.push_back(result.time_ms);
      }
    }
    // A detector track may be rebuilt while approaching the same signed branch.
    // Route sessions, rather than bounding-box track ids, own OCR/API evidence.
    if (task.route_session_id != guideboard_route_session_id_ ||
        task.branch_event_id != active_guideboard_branch_event_id_ ||
        !guideboard_route_policy_.recognitionRequired()) {
      return;
    }

    if (task.pipeline == "rec") {
      if (task.ret == 0 && result.ocr_score >= ocr_min_text_score_ && !result.text.empty()) {
        guideboard_uncertain_count_ = 0;
      } else {
        ++guideboard_uncertain_count_;
      }
    }
    if (guideboard_stop_wait_active_) {
      // Count OCR completions, not only non-empty strings: the stop policy is
      // explicitly based on two new local OCR attempts.
      ++guideboard_api_force_ocr_count_;
    }
    if (!result.text.empty()) {
      guideboard_api_ocr_history_.push_back({result.text, result.ocr_score});
      while (guideboard_api_ocr_history_.size() >
             static_cast<size_t>(guideboard_api_text_history_size_)) {
        guideboard_api_ocr_history_.pop_front();
      }
    }

    std_msgs::msg::String recognition_msg;
    recognition_msg.data = guideboardRecognitionJson(task);
    guideboard_recognition_pub_->publish(recognition_msg);

    if (enable_data_log_) {
      RCLCPP_INFO(
          get_logger(),
          "GUIDEBOARD_OCR track_id=%lu sequence=%lu pipeline=%s ret=%d text=%s score=%.3f "
          "history=%s current_maneuver=%s api_attempts=%d stop_wait=%d time_ms=%.1f "
          "crop=%dx%d fallback=%s error=%s",
          static_cast<unsigned long>(task.track_id),
          static_cast<unsigned long>(guideboard_sequence_), task.pipeline.c_str(), task.ret,
          last_ocr_text_.c_str(), last_ocr_score_, guideboardOcrHistoryJson().c_str(),
          current_guideboard_maneuver_.c_str(), api_attempt_count_, guideboard_stop_wait_active_,
          last_ocr_time_ms_, task.crop_width, task.crop_height,
          task.fallback_reason.c_str(), result.error.c_str());
    }
    maybeTriggerGuideboardApi();
    applyGuideboardHint();
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

  void waitForGuideboardApi() {
    if (!guideboard_api_future_.valid()) {
      return;
    }
    try {
      guideboard_api_future_.wait();
      guideboard_api_future_.get();
    } catch (...) {
    }
  }

  bool guideboardTextStable() const {
    if (guideboard_api_ocr_history_.size() < 2) {
      return false;
    }
    for (size_t i = 0; i < guideboard_api_ocr_history_.size(); ++i) {
      if (guideboard_api_ocr_history_[i].score < ocr_min_text_score_ ||
          guideboard_api_ocr_history_[i].text.empty()) {
        continue;
      }
      for (size_t j = i + 1; j < guideboard_api_ocr_history_.size(); ++j) {
        if (guideboard_api_ocr_history_[j].score < ocr_min_text_score_ ||
            guideboard_api_ocr_history_[j].text.empty()) {
          continue;
        }
        if (GuideboardRecognizer::normalizedLcsRatio(
                guideboard_api_ocr_history_[i].text,
                guideboard_api_ocr_history_[j].text) >= guideboard_api_text_similarity_) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<GuideboardApiSample> guideboardApiSamples() const {
    std::vector<GuideboardApiSample> samples;
    samples.reserve(guideboard_api_ocr_history_.size());
    for (const auto& sample : guideboard_api_ocr_history_) {
      samples.push_back({sample.text, sample.score});
    }
    return samples;
  }

  void requestGuideboardStop() {
    if (guideboard_stop_active_ || guideboard_stop_call_pending_) {
      return;
    }
    const bool first_stop_request = !guideboard_stop_wait_active_;
    guideboard_stop_wait_active_ = true;
    guideboard_start_after_stop_ = false;
    if (first_stop_request) {
      logGuideboardResult("stop", guideboard_track_id_, "guideboard", "unknown", "pending",
                          "decision_pending_height_threshold", last_ocr_score_, 0.0, 0.0,
                          stable_decision_latency_ms_);
    }
    logGuideboardApiEvent("GUIDEBOARD_STOP_REQUEST", "bbox_height_reached", last_ocr_text_,
                          last_ocr_score_, current_guideboard_maneuver_, "",
                          current_guideboard_opposite_);
    if (!guideboard_stop_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(),
                  "GUIDEBOARD_STOP_REQUEST service not ready: %s; continue OCR/API without pause",
                  guideboard_stop_service_.c_str());
      return;
    }
    guideboard_stop_call_pending_ = true;
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    guideboard_stop_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
          guideboard_stop_call_pending_ = false;
          try {
            const auto response = future.get();
            if (response->success) {
              guideboard_stop_active_ = true;
              RCLCPP_INFO(get_logger(), "GUIDEBOARD_STOP_REQUEST completed: %s",
                          response->message.c_str());
              if (guideboard_start_after_stop_) {
                requestGuideboardStart();
              }
            } else {
              RCLCPP_WARN(get_logger(), "GUIDEBOARD_STOP_REQUEST rejected: %s",
                          response->message.c_str());
            }
          } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "GUIDEBOARD_STOP_REQUEST failed: %s", e.what());
          }
        });
  }

  void maybeStopForPendingGuideboardDecision() {
    if (!enable_guideboard_api_ || !has_guideboard_track_ ||
        !guideboard_track_route_eligible_ ||
        !guideboard_route_policy_.recognitionRequired() ||
        guideboard_route_policy_.hasPreparedDecision() ||
        current_guideboard_decision_valid_) {
      return;
    }
    if (last_guideboard_height_ratio_ >= guideboard_api_stop_height_ratio_) {
      requestGuideboardStop();
    }
  }

  void requestGuideboardStart() {
    guideboard_start_after_stop_ = true;
    logGuideboardApiEvent("GUIDEBOARD_START_REQUEST", "api_result_ready", last_ocr_text_,
                          last_ocr_score_, current_guideboard_maneuver_, last_api_corrected_text_,
                          current_guideboard_opposite_);
    if (guideboard_stop_call_pending_) {
      return;
    }
    if (!guideboard_stop_active_) {
      guideboard_stop_wait_active_ = false;
      return;
    }
    if (guideboard_start_call_pending_) {
      return;
    }
    if (!guideboard_start_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(),
                  "GUIDEBOARD_START_REQUEST service not ready: %s; vehicle remains paused",
                  guideboard_start_service_.c_str());
      return;
    }
    guideboard_start_call_pending_ = true;
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    guideboard_start_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
          guideboard_start_call_pending_ = false;
          try {
            const auto response = future.get();
            if (response->success) {
              guideboard_stop_active_ = false;
              guideboard_stop_wait_active_ = false;
              guideboard_start_after_stop_ = false;
              RCLCPP_INFO(get_logger(), "GUIDEBOARD_START_REQUEST completed: %s",
                          response->message.c_str());
            } else {
              RCLCPP_WARN(get_logger(), "GUIDEBOARD_START_REQUEST rejected: %s",
                          response->message.c_str());
            }
          } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "GUIDEBOARD_START_REQUEST failed: %s", e.what());
          }
        });
  }

  void retryGuideboardStartIfNeeded() {
    if (guideboard_start_after_stop_ && guideboard_stop_active_ &&
        !guideboard_stop_call_pending_ && !guideboard_start_call_pending_) {
      requestGuideboardStart();
    }
  }

  void handleGuideboardBranchEvent(const LaneState& lane_state,
                                   const LaneDebugInfo& debug_info) {
    if (debug_info.branch_event_rearmed) {
      accept_guideboard_for_route_ = true;
      guideboard_track_route_eligible_ = false;
      active_guideboard_branch_event_id_ = 0;
      if (has_guideboard_track_) {
        // A sign track that survived the previous branch cannot represent the
        // next signed encounter. The route session itself remains untouched.
        resetGuideboardTrack(true);
      }
      if (enable_data_log_) {
        RCLCPP_INFO(get_logger(),
                    "BRANCH_EVENT_REARMED event_id=%lu route_phase=%s",
                    static_cast<unsigned long>(debug_info.branch_event_id),
                    guideboard_route_policy_.phaseName());
      }
    }
    if (!debug_info.branch_lock_event) {
      return;
    }

    accept_guideboard_for_route_ = false;
    guideboard_track_route_eligible_ = false;
    active_guideboard_branch_event_id_ = 0;
    if (!debug_info.branch_lock_guideboard) {
      logGuideboardResult("branch_lock", guideboard_track_id_, "none", "straight",
                          lane_state.branch_side.empty() ? "left" : lane_state.branch_side,
                          "no_guideboard_default", -1.0, 0.0, 0.0, -1.0);
      publishGuideboardRouteEvent(
          "branch_locked", debug_info.branch_event_id,
          "no_guideboard_default", "straight",
          lane_state.branch_side.empty() ? "left" : lane_state.branch_side);
      return;
    }

    if (!guideboard_route_policy_.hasPreparedDecision()) {
      guideboard_route_policy_.prepareRecognitionFailure();
      setCurrentGuideboardDecision(guideboard_route_policy_.preparedAction(),
                                   guideboard_route_policy_.decisionSource());
    }
    const std::string action = guideboard_route_policy_.hasPreparedDecision()
                                   ? guideboard_route_policy_.preparedAction()
                                   : "straight";
    const std::string source = guideboard_route_policy_.decisionSource().empty()
                                   ? "signed_default_straight"
                                   : guideboard_route_policy_.decisionSource();
    logGuideboardResult("branch_lock", guideboard_track_id_, "guideboard", action,
                        lane_state.branch_side.empty() ? maneuverToBranch(action)
                                                       : lane_state.branch_side,
                        source, last_ocr_score_, 1.0, 0.0,
                        stable_decision_latency_ms_);

    if (guideboard_route_policy_.commitSignedEncounter()) {
      publishGuideboardRouteEvent(
          "branch_locked", debug_info.branch_event_id, source, action,
          lane_state.branch_side.empty() ? maneuverToBranch(action)
                                         : lane_state.branch_side);
      requestGuideboardStart();
      resetGuideboardRecognitionSession();
    }
  }

  void beginGuideboardApiRequest(const std::string& trigger_reason) {
    if (guideboard_api_request_in_flight_ || api_fallback_ ||
        api_attempt_count_ >= guideboard_api_max_attempts_) {
      return;
    }
    if (!guideboard_route_policy_.recognitionRequired() ||
        !guideboard_track_route_eligible_) {
      return;
    }
    api_session_attempted_ = true;
    api_trigger_reason_ = trigger_reason;
    ++api_attempt_count_;
    const int attempt = api_attempt_count_;
    const uint64_t task_track_id = guideboard_track_id_;
    const uint64_t task_route_session_id = guideboard_route_session_id_;
    const uint64_t task_branch_event_id = active_guideboard_branch_event_id_;
    const auto samples = guideboardApiSamples();
    logGuideboardApiEvent("GUIDEBOARD_API_TRIGGER", trigger_reason, last_ocr_text_,
                          last_ocr_score_, current_guideboard_maneuver_, "",
                          current_guideboard_opposite_);
    if (attempt > 1) {
      logGuideboardApiEvent("GUIDEBOARD_API_RETRY", trigger_reason, last_ocr_text_,
                            last_ocr_score_, current_guideboard_maneuver_, "",
                            current_guideboard_opposite_);
    }
    if (enable_data_log_) {
      RCLCPP_INFO(get_logger(),
                  "GUIDEBOARD_API_REQUEST track_id=%lu sequence=%lu attempt=%d "
                  "model=%s timeout_sec=%.2f uncertain_min_confidence=%.2f "
                  "history=%s stop_wait_active=%d",
                  static_cast<unsigned long>(task_track_id),
                  static_cast<unsigned long>(guideboard_sequence_), attempt,
                  guideboard_api_model_.c_str(), guideboard_api_timeout_sec_,
                  guideboard_api_uncertain_min_confidence_,
                  guideboardOcrHistoryJson().c_str(), guideboard_stop_wait_active_);
    }
    guideboard_api_request_in_flight_ = true;
    guideboard_api_future_ = std::async(
        std::launch::async,
        [url = guideboard_api_url_, key = api_key_, model = guideboard_api_model_, samples,
         timeout_sec = guideboard_api_timeout_sec_,
         uncertain_min_confidence = guideboard_api_uncertain_min_confidence_, task_track_id,
         task_route_session_id, task_branch_event_id, attempt]() {
          GuideboardApiTaskResult task;
          task.track_id = task_track_id;
          task.route_session_id = task_route_session_id;
          task.branch_event_id = task_branch_event_id;
          task.attempt = attempt;
          task.result = GuideboardApiClient::request(
              url, key, model, samples, timeout_sec, uncertain_min_confidence);
          return task;
        });
  }

  void finalizeGuideboardApiFailure(const std::string& reason) {
    api_fallback_ = true;
    api_trigger_reason_ = reason;
    if (guideboard_route_policy_.prepareRecognitionFailure()) {
      setCurrentGuideboardDecision(guideboard_route_policy_.preparedAction(),
                                   guideboard_route_policy_.decisionSource());
    }
    logGuideboardApiEvent("GUIDEBOARD_API_FALLBACK", reason, last_ocr_text_, last_ocr_score_,
                          current_guideboard_maneuver_, "", false);
    applyGuideboardHint();
    requestGuideboardStart();
  }

  void consumeGuideboardApiResult() {
    if (!guideboard_api_future_.valid() ||
        guideboard_api_future_.wait_for(0ms) != std::future_status::ready) {
      return;
    }
    GuideboardApiTaskResult task;
    try {
      task = guideboard_api_future_.get();
    } catch (const std::exception& e) {
      task.result.error = e.what();
    } catch (...) {
      task.result.error = "unknown API async exception";
    }
    guideboard_api_request_in_flight_ = false;
    if (task.route_session_id != guideboard_route_session_id_ ||
        task.branch_event_id != active_guideboard_branch_event_id_ ||
        !guideboard_route_policy_.recognitionRequired()) {
      if (enable_data_log_) {
        RCLCPP_INFO(get_logger(),
                    "GUIDEBOARD_API_STALE task_session=%lu current_session=%lu track=%lu",
                    static_cast<unsigned long>(task.route_session_id),
                    static_cast<unsigned long>(guideboard_route_session_id_),
                    static_cast<unsigned long>(task.track_id));
      }
      return;
    }
    last_api_latency_ms_ = task.result.latency_ms;
    last_api_http_status_ = task.result.http_status;
    last_api_confidence_ = task.result.confidence;
    last_api_uncertain_ = task.result.uncertain;
    last_api_accepted_uncertain_ = task.result.accepted_uncertain;
    last_api_corrected_text_ = task.result.corrected_text;
    if (enable_data_log_) {
      RCLCPP_INFO(get_logger(),
                  "GUIDEBOARD_API_RESPONSE track_id=%lu sequence=%lu attempt=%d valid=%d "
                  "uncertain=%d accepted_uncertain=%d maneuver=%s confidence=%.3f "
                  "corrected_text=%s "
                  "latency_ms=%.1f http_status=%d error=%s",
                  static_cast<unsigned long>(task.track_id),
                  static_cast<unsigned long>(guideboard_sequence_), task.attempt,
                  task.result.valid, task.result.uncertain, task.result.accepted_uncertain,
                  task.result.maneuver.c_str(), task.result.confidence,
                  task.result.corrected_text.c_str(),
                  task.result.latency_ms, task.result.http_status, task.result.error.c_str());
    }

    if (task.result.valid) {
      api_fallback_ = false;
      api_session_attempted_ = true;
      if (guideboard_route_policy_.prepareRecognitionSuccess(task.result.maneuver)) {
        setCurrentGuideboardDecision(guideboard_route_policy_.preparedAction(),
                                     guideboard_route_policy_.decisionSource());
      }
      logGuideboardApiEvent("GUIDEBOARD_API_DECISION_READY", "success", last_ocr_text_,
                            last_ocr_score_, current_guideboard_maneuver_,
                            task.result.corrected_text, current_guideboard_opposite_);
      applyGuideboardHint();
      requestGuideboardStart();
      return;
    }

    const bool retryable = !task.result.uncertain &&
                           guideboard_api_retry_on_transport_failure_ &&
                           api_attempt_count_ < guideboard_api_max_attempts_;
    if (retryable) {
      beginGuideboardApiRequest("transport_or_parse_retry");
      return;
    }
    finalizeGuideboardApiFailure(task.result.error.empty() ? "api_invalid_result"
                                                            : task.result.error);
  }

  void maybeTriggerGuideboardApi() {
    if (!enable_guideboard_api_ || !has_guideboard_track_ ||
        !guideboard_track_route_eligible_ ||
        !guideboard_route_policy_.recognitionRequired() || api_session_attempted_ ||
        api_fallback_ || guideboard_api_request_in_flight_) {
      return;
    }
    const bool stable = guideboardTextStable();
    if (guideboard_stop_wait_active_) {
      if (guideboard_api_force_ocr_count_ >= guideboard_api_force_ocr_count_after_stop_) {
        beginGuideboardApiRequest("forced_after_stop_ocr");
      }
      return;
    }
    if (stable) {
      beginGuideboardApiRequest("stable_text");
      return;
    }
  }

  void updateGuideboardOcr(const cv::Mat& frame_rgb, const std::vector<Detection>& detections) {
    retryGuideboardStartIfNeeded();
    if (!enable_guideboard_ocr_ || !guideboard_ocr_ready_ || frame_rgb.empty()) {
      lane_decision_.setGuideboardBranchHint("", false);
      return;
    }

    consumeGuideboardApiResult();
    consumeGuideboardOcrResult();

    const Detection* guideboard = selectGuideboardForOcr(detections, frame_rgb.rows);
    const double now = nowSeconds();
    if (guideboard == nullptr) {
      if (has_guideboard_track_ && now - guideboard_last_seen_sec_ > 0.25) {
        logGuideboardApiEvent("GUIDEBOARD_TRACK_LOST", "guideboard_missing_0.25s",
                              last_ocr_text_, last_ocr_score_, current_guideboard_maneuver_,
                              last_api_corrected_text_, current_guideboard_opposite_);
        logGuideboardResult("track_lost", guideboard_track_id_, "unknown",
                            current_guideboard_decision_valid_
                                ? current_guideboard_maneuver_
                                : "unknown",
                            "pending", "guideboard_missing_0.25s", last_ocr_score_,
                            0.0, 0.0, -1.0);
        resetGuideboardTrack(true);
      }
      applyGuideboardHint();
      return;
    }
    updateGuideboardTrack(guideboard->bbox, now);
    if (!guideboard_track_route_eligible_ && accept_guideboard_for_route_ &&
        lane_decision_.branchEventArmed()) {
      guideboard_track_route_eligible_ = true;
      active_guideboard_branch_event_id_ = lane_decision_.branchEventId();
    }
    last_guideboard_height_ratio_ =
        frame_rgb.rows > 0 ? static_cast<double>(guideboard->bbox.height) / frame_rgb.rows : 0.0;
    if (guideboard_decision_start_sec_ <= 0.0 &&
        guideboard->bbox.width >= 70.0f && guideboard->bbox.height >= 30.0f) {
      guideboard_decision_start_sec_ = now;
    }
    if (guideboard_track_route_eligible_ &&
        guideboard_route_policy_.prepareKnownDecision()) {
      setCurrentGuideboardDecision(guideboard_route_policy_.preparedAction(),
                                   guideboard_route_policy_.decisionSource());
    }
    applyGuideboardHint();
    maybeStopForPendingGuideboardDecision();

    if (!guideboard_track_route_eligible_ ||
        !guideboard_route_policy_.recognitionRequired() ||
        guideboard_api_request_in_flight_ || guideboard_ocr_future_.valid()) {
      return;
    }
    if (now - last_guideboard_ocr_sec_ < ocr_rec_interval_sec_) {
      return;
    }
    last_guideboard_ocr_sec_ = now;

    std::string pipeline = ocr_pipeline_mode_ == "det_rec" ? "det_rec" : "rec";
    std::string fallback_reason;
    if (ocr_pipeline_mode_ == "rec_then_det" && !guideboard_fallback_used_ &&
        guideboard_uncertain_count_ >= ocr_fallback_uncertain_count_ &&
        guideboard->bbox.width >= 70.0f && guideboard->bbox.height >= 30.0f) {
      pipeline = "det_rec";
      guideboard_fallback_used_ = true;
      fallback_reason = "rec_uncertain_" + std::to_string(guideboard_uncertain_count_);
      logGuideboardResult("fallback", guideboard_track_id_, "unknown", "unknown", "pending",
                          fallback_reason, last_ocr_score_, 0.0, 0.0, -1.0);
    }

    cv::Mat crop;
    if (pipeline == "rec") {
      crop = makeRecOnlyInput(frame_rgb, guideboard->bbox);
    } else {
      const cv::Rect crop_rect =
          makePaddedCropRect(guideboard->bbox, frame_rgb.size(), ocr_crop_padding_ratio_);
      if (crop_rect.width >= kMinOcrCropSizePx && crop_rect.height >= kMinOcrCropSizePx) {
        crop = frame_rgb(crop_rect).clone();
      }
    }
    if (crop.empty()) {
      return;
    }

    ocr_task_crop_width_ = crop.cols;
    ocr_task_crop_height_ = crop.rows;
    const uint64_t task_track_id = guideboard_track_id_;
    const uint64_t task_route_session_id = guideboard_route_session_id_;
    const uint64_t task_branch_event_id = active_guideboard_branch_event_id_;
    guideboard_ocr_future_ = std::async(
        std::launch::async,
        [this, crop = std::move(crop), pipeline, fallback_reason, task_track_id,
         task_route_session_id, task_branch_event_id]() {
      OcrTaskResult task;
      task.crop_width = crop.cols;
      task.crop_height = crop.rows;
      task.track_id = task_track_id;
      task.route_session_id = task_route_session_id;
      task.branch_event_id = task_branch_event_id;
      task.pipeline = pipeline;
      task.fallback_reason = fallback_reason;
      try {
        task.ret = pipeline == "rec"
                       ? guideboard_ocr_.run_rec_mat(crop, &task.result)
                       : guideboard_ocr_.run_det_rec_mat(crop, &task.result);
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
    lane_state_pub_ = create_publisher<std_msgs::msg::String>("/perception/lane_state", 10);
    lane_debug_pub_ = create_publisher<std_msgs::msg::String>("/perception/lane_debug", 10);
    frame_signature_pub_ =
        create_publisher<std_msgs::msg::UInt64>("/perception/frame_signature", sensor_qos);
    guideboard_recognition_pub_ =
        create_publisher<std_msgs::msg::String>("/perception/guideboard_recognition", 10);
    line_follower_start_client_ = create_client<std_srvs::srv::Trigger>(line_follower_start_service_);
    line_follower_stop_client_ = create_client<std_srvs::srv::Trigger>(line_follower_stop_service_);
    // Guideboard OCR pauses use the controller's normal start/stop pair.  The
    // configurable pair above remains dedicated to the Human obstacle state.
    guideboard_start_client_ = create_client<std_srvs::srv::Trigger>(guideboard_start_service_);
    guideboard_stop_client_ = create_client<std_srvs::srv::Trigger>(guideboard_stop_service_);
    encoder_count_sub_ = create_subscription<std_msgs::msg::Int64>(
      encoder_count_topic_, rclcpp::QoS(10).reliable(),
      [this](const std_msgs::msg::Int64::SharedPtr msg) {
        lane_decision_.setEncoderCount(msg->data, nowSeconds());
      });
    steering_command_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      centerline_kalman_steering_topic_, rclcpp::QoS(10).reliable(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        lane_decision_.setSteeringCommand(msg->angular.z, nowSeconds());
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
                         seg_max_detections_, seg_raw_output_)) {
      throw std::runtime_error("failed to initialize segmentation model");
    }
    if (enable_guideboard_ocr_) {
      int ret = ocr_pipeline_mode_ == "rec_only"
                    ? guideboard_ocr_.init_rec(ocr_rec_model_path_.c_str())
                    : guideboard_ocr_.init(ocr_det_model_path_.c_str(), ocr_rec_model_path_.c_str());
      if (ret != 0) {
        throw std::runtime_error("failed to initialize guideboard OCR model");
      }
      guideboard_ocr_ready_ = true;
      RCLCPP_INFO(get_logger(),
                  "guideboard OCR ready: mode=%s control=%d api=%d model=%s timeout_sec=%.2f "
                  "uncertain_min_confidence=%.2f max_attempts=%d det=%s rec=%s",
                  ocr_pipeline_mode_.c_str(), ocr_apply_to_control_,
                  enable_guideboard_api_, guideboard_api_model_.c_str(),
                  guideboard_api_timeout_sec_, guideboard_api_uncertain_min_confidence_,
                  guideboard_api_max_attempts_,
                  ocr_pipeline_mode_ == "rec_only" ? "not_loaded" : ocr_det_model_path_.c_str(),
                  ocr_rec_model_path_.c_str());
    }

    std_msgs::msg::String labels_msg;
    labels_msg.data = joinLabels(detector_.labels());
    label_pub_->publish(labels_msg);

    perf_start_sec_ = nowSeconds();
    if (!guideboard_log_only_) {
      RCLCPP_INFO(
          get_logger(),
          "fused perception ready: det=%s seg=%s shm=/dev/shm/%s show_window=%d blend_alpha=%.2f",
          det_model_path_.c_str(), seg_model_path_.c_str(), shm_name_.c_str(), show_window_,
          blend_alpha_);
    }
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
    const uint64_t frame_signature = frameSignature(frame.image);

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
    handleGuideboardBranchEvent(lane_state, lane_debug);
    logDecisionStatus(lane_state, lane_debug);
    updateHumanExecution(lane_debug);

    refreshDebugParameters();
    if (show_window_ || enable_debug_screenshots_) {
      showDebugWindow(frame_rgb, seg_map, detections, seg_instances, lane_state, lane_debug);
    }

    auto t_pub0 = std::chrono::steady_clock::now();
    publishAll(detections, lane_state, lane_debug, frame_signature);
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

    if (!guideboard_log_only_ && (!det_ok || !seg_ok)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "inference failure: det_ok=%d seg_ok=%d", det_ok, seg_ok);
    }

    logPerfIfNeeded();
  }

  void updateHumanExecution(const LaneDebugInfo& debug_info) {
    // Human state is the only source for this service pair.  Far overlap and
    // no-fit wait states keep the controller running; only a confirmed
    // OBSTACLE_STOP transition pauses it.
    human_stop_desired_ = debug_info.human_state == "OBSTACLE_STOP";
    const double now = nowSeconds();
    if (now - human_service_last_call_sec_ < human_service_retry_interval_sec_) {
      return;
    }

    if (human_stop_desired_) {
      if (human_service_stop_active_ || human_stop_call_pending_ || human_start_call_pending_) {
        return;
      }
      if (!line_follower_stop_client_->service_is_ready()) {
        human_service_action_ = "stop_wait_service";
        human_service_last_call_sec_ = now;
        if (!guideboard_log_only_) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "Human OBSTACLE_STOP active, waiting for %s",
              line_follower_stop_service_.c_str());
        }
        return;
      }

      human_stop_call_pending_ = true;
      human_service_action_ = "stop_pending";
      human_service_last_call_sec_ = now;
      auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
      line_follower_stop_client_->async_send_request(
          request,
          [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
            human_stop_call_pending_ = false;
            try {
              const auto response = future.get();
              if (response->success) {
                human_service_stop_active_ = true;
                human_service_action_ = "stopped";
                if (!guideboard_log_only_) {
                  RCLCPP_WARN(get_logger(),
                              "Human obstacle stop applied through line follower service: %s",
                              response->message.c_str());
                }
              } else {
                human_service_action_ = "stop_failed";
                if (!guideboard_log_only_) {
                  RCLCPP_WARN(get_logger(),
                              "Human obstacle stop rejected by line follower: %s",
                              response->message.c_str());
                }
              }
            } catch (const std::exception& e) {
              human_service_action_ = "stop_failed";
              if (!guideboard_log_only_) {
                RCLCPP_WARN(get_logger(), "Human obstacle stop service failed: %s", e.what());
              }
            }
          });
      return;
    }

    if (!human_service_stop_active_ || human_stop_call_pending_ || human_start_call_pending_) {
      return;
    }
    if (!line_follower_start_client_->service_is_ready()) {
      human_service_action_ = "resume_wait_service";
      human_service_last_call_sec_ = now;
      if (!guideboard_log_only_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Human obstacle cleared, waiting for %s",
            line_follower_start_service_.c_str());
      }
      return;
    }

    human_start_call_pending_ = true;
    human_service_action_ = "resume_pending";
    human_service_last_call_sec_ = now;
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    line_follower_start_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
          human_start_call_pending_ = false;
          try {
            const auto response = future.get();
            if (response->success) {
              human_service_stop_active_ = false;
              human_service_action_ = "cleared";
              if (!guideboard_log_only_) {
                RCLCPP_INFO(get_logger(),
                            "Human obstacle cleared; controller resume service completed: %s",
                            response->message.c_str());
              }
            } else {
              human_service_action_ = "resume_failed";
              if (!guideboard_log_only_) {
                RCLCPP_WARN(get_logger(),
                            "Human obstacle resume rejected by line follower: %s",
                            response->message.c_str());
              }
            }
          } catch (const std::exception& e) {
            human_service_action_ = "resume_failed";
            if (!guideboard_log_only_) {
              RCLCPP_WARN(get_logger(), "Human obstacle resume service failed: %s", e.what());
            }
          }
        });
  }

  void publishAll(const std::vector<Detection>& detections, const LaneState& lane_state,
                  const LaneDebugInfo& lane_debug, uint64_t frame_signature) {
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

    std_msgs::msg::String lane_msg;
    lane_msg.data = laneStateToJson(lane_state);
    if (publish_lane_state_) {
      lane_state_pub_->publish(lane_msg);
    }

    std_msgs::msg::String lane_debug_msg;
    lane_debug_msg.data = laneDebugToJson(lane_debug, seg_input_width_);
    lane_debug_pub_->publish(lane_debug_msg);

    std_msgs::msg::UInt64 signature_msg;
    signature_msg.data = frame_signature;
    frame_signature_pub_->publish(signature_msg);
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
           << " car_filter=" << (debug_info.car_boundary_active ? 1 : 0)
           << " car_x=" << debug_info.car_left_x
           << " car_rm=" << debug_info.car_filtered_point_count
           << " car_push=" << (debug_info.car_push_active ? 1 : 0)
           << " car_ps=" << debug_info.car_pushed_point_count
           << " car_del=" << debug_info.car_deleted_point_count
           << " car_lost=" << debug_info.car_boundary_lost_count
           << " fit_hold=" << (debug_info.fit_hold_active ? 1 : 0)
           << " alpha=" << std::clamp(blend_alpha_, 0.0f, 1.0f);
    cv::putText(vis, status.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    std::ostringstream car_status;
    car_status << "CAR boundary=" << (debug_info.car_boundary_active ? 1 : 0)
               << " left_x=" << debug_info.car_left_x
               << " push=" << (debug_info.car_push_active ? 1 : 0)
               << " pushed=" << debug_info.car_pushed_point_count
               << " deleted=" << debug_info.car_deleted_point_count
               << " target_x=" << debug_info.car_push_target_x
               << " bottom=" << debug_info.car_push_expand_bottom_y
               << " lost=" << debug_info.car_boundary_lost_count
               << " hold=" << (debug_info.fit_hold_active ? 1 : 0)
               << " age=" << debug_info.fit_hold_age;
    cv::putText(vis, car_status.str(), cv::Point(12, 52), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    std::ostringstream human_status;
    human_status << "HUMAN state=" << debug_info.human_state
                 << " pass=" << (debug_info.human_passable ? 1 : 0)
                 << " overlap=" << (debug_info.human_line_intersects ? 1 : 0)
                 << " raw_area=" << debug_info.human_raw_area_ratio
                 << " stop=" << debug_info.human_stop_confirm_count << "/"
                 << human_stop_confirm_frames_
                 << " clear=" << debug_info.human_clear_confirm_count << "/"
                 << human_clear_confirm_frames_
                 << " count=" << debug_info.human_valid_count << "/"
                 << debug_info.human_count_at_stop
                 << " ctrl=" << human_service_action_;
    const cv::Scalar human_status_color = debug_info.human_state == "OBSTACLE_STOP"
                                              ? cv::Scalar(0, 0, 255)
                                              : (debug_info.human_line_intersects ||
                                                         debug_info.human_state == "NO_FIT_WAIT_NEAR"
                                                     ? cv::Scalar(0, 165, 255)
                                                     : cv::Scalar(0, 255, 0));
    cv::putText(vis, human_status.str(), cv::Point(12, 76), cv::FONT_HERSHEY_SIMPLEX,
                0.55, human_status_color, 2, cv::LINE_AA);
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
      if (!guideboard_log_only_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "saved debug screenshot: %s", path.string().c_str());
      }
    } catch (const std::exception& e) {
      if (!guideboard_log_only_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "failed to save debug screenshot: %s", e.what());
      }
    }
  }

  static cv::Scalar detectionColor(const std::string& class_name) {
    static const std::unordered_map<std::string, cv::Scalar> colors = {
        {"Human", cv::Scalar(0, 255, 0)},
        {"Car", cv::Scalar(0, 0, 255)},
        {"Stop", cv::Scalar(0, 165, 255)},
        {"Gold", cv::Scalar(0, 255, 255)},
        {"GuideBoard", cv::Scalar(255, 255, 255)},
    };
    auto it = colors.find(class_name);
    return it == colors.end() ? cv::Scalar(255, 255, 255) : it->second;
  }

  void drawLaneDebug(cv::Mat& vis, const LaneDebugInfo& debug_info) const {
    bool template_active = debug_info.left_boundary_template_active;
    if (debug_info.car_boundary_active && debug_info.car_left_x >= 0.0f) {
      const int car_x = std::clamp(static_cast<int>(std::round(debug_info.car_left_x)),
                                   0, std::max(0, vis.cols - 1));
      cv::line(vis, cv::Point(car_x, 0), cv::Point(car_x, vis.rows - 1),
               cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    }
    if (debug_info.car_push_active && debug_info.car_expanded_bbox.width > 0.0f &&
        debug_info.car_expanded_bbox.height > 0.0f) {
      cv::rectangle(vis, debug_info.car_expanded_bbox, cv::Scalar(0, 165, 255), 2,
                    cv::LINE_AA);
      const int target_x = std::clamp(
          static_cast<int>(std::round(debug_info.car_push_target_x)),
          0, std::max(0, vis.cols - 1));
      const int top_y = std::clamp(
          static_cast<int>(std::round(debug_info.car_expanded_bbox.y)),
          0, std::max(0, vis.rows - 1));
      const int bottom_y = std::clamp(
          static_cast<int>(std::round(debug_info.car_expanded_bbox.y +
                                      debug_info.car_expanded_bbox.height)),
          top_y, std::max(0, vis.rows - 1));
      cv::line(vis, cv::Point(target_x, top_y), cv::Point(target_x, bottom_y),
               cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    }
    for (const auto& human : debug_info.humans) {
      const cv::Scalar expanded_color = debug_info.human_stop_active
                                            ? cv::Scalar(0, 0, 255)
                                            : (human.stop_candidate
                                                   ? cv::Scalar(0, 165, 255)
                                                   : cv::Scalar(0, 255, 255));
      cv::rectangle(vis, human.expanded_bbox, expanded_color, 2, cv::LINE_AA);
      for (const auto& sample : human.fit_sample_points) {
        const bool sample_intersects =
            sample.x >= human.expanded_bbox.x &&
            sample.x <= human.expanded_bbox.x + human.expanded_bbox.width &&
            sample.y >= human.expanded_bbox.y &&
            sample.y <= human.expanded_bbox.y + human.expanded_bbox.height;
        const cv::Scalar sample_color = sample_intersects ? cv::Scalar(0, 0, 255)
                                                           : cv::Scalar(0, 255, 0);
        const int x = std::clamp(static_cast<int>(std::round(sample.x)),
                                 0, std::max(0, vis.cols - 1));
        const int y = std::clamp(static_cast<int>(std::round(sample.y)),
                                 0, std::max(0, vis.rows - 1));
        cv::drawMarker(vis, cv::Point(x, y), sample_color,
                       cv::MARKER_CROSS, 9, 2, cv::LINE_AA);
      }
    }
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

    for (const auto& point : debug_info.removed_fit_points) {
      const cv::Point center(static_cast<int>(std::round(point.x)),
                             static_cast<int>(std::round(point.y)));
      cv::drawMarker(vis, center, cv::Scalar(255, 0, 255), cv::MARKER_TILTED_CROSS,
                     9, 2, cv::LINE_AA);
    }

    const auto& points = debug_info.fit_points;
    for (const auto& point : points) {
      cv::circle(vis, cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))),
                 template_active ? 3 : 1, cv::Scalar(0, 255, 0), -1);
    }
    for (const auto& point : debug_info.pushed_fit_points) {
      cv::drawMarker(vis,
                     cv::Point(static_cast<int>(std::round(point.x)),
                               static_cast<int>(std::round(point.y))),
                     cv::Scalar(0, 165, 255), cv::MARKER_CROSS, 9, 2, cv::LINE_AA);
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
    enable_result_log_ = get_parameter("enable_result_log").as_bool();
    enable_data_log_ = get_parameter("enable_data_log").as_bool();
    guideboard_log_only_ = get_parameter("guideboard_log_only").as_bool();
    enable_perf_stats_ = get_parameter("enable_perf_stats").as_bool();
  }

  void logDecisionStatus(const LaneState& lane_state, const LaneDebugInfo& debug_info) {
    if (enable_data_log_ && !guideboard_log_only_) {
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
                    "branch_detected=%d score=%d event_id=%lu armed=%d sign_latched=%d "
                    "encoder_hold=%d encoder_delta=%ld/%ld "
                    "lb_tpl=%d tpl_side=%s lb_pts=%d lb_reason=%s "
                    "segments=%d raw_points=%d fit_points=%d",
                    debug_info.branch_detected, debug_info.branch_score,
                    static_cast<unsigned long>(debug_info.branch_event_id),
                    debug_info.branch_event_armed, debug_info.guideboard_seen_latched,
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
        RCLCPP_INFO(get_logger(), "task_state=%s human_state=%s human_service=%s",
                    lane_state.task_state.c_str(), debug_info.human_state.c_str(),
                    human_service_action_.c_str());
      }

    }

    // Keep transition memory independent of either terminal log switch.
    last_guideboard_seen_ = debug_info.guideboard_seen;
    last_branch_detected_ = debug_info.branch_detected;
    last_branch_score_ = debug_info.branch_score;
    last_road_state_ = lane_state.road_state;
    last_branch_side_ = lane_state.branch_side;
    last_task_state_ = lane_state.task_state;

    if (!enable_data_log_ || guideboard_log_only_) {
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
    if (!enable_perf_stats_ || guideboard_log_only_) {
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
                "seg=%.2f/%.2f ms ocr_rec=%.2f/%.2f ms/%lu "
                "ocr_det_rec=%.2f/%.2f ms/%lu ocr_stable=%.2f/%.2f ms/%lu "
                "decision=%.2f ms publish=%.2f ms detections=%zu "
                "seg_model_conf_avg=%.3f seg_model_instances=%lu last_seg_conf=%.3f[%.3f,%.3f]/%d",
                dt, upstream_fps, processed_fps, sum_det_rknn_ms_ / frames,
                sum_det_post_ms_ / frames, sum_seg_rknn_ms_ / frames,
                sum_seg_post_ms_ / frames,
                ocr_rec_run_count_ > 0
                    ? sum_ocr_rec_ms_ / static_cast<double>(ocr_rec_run_count_) : 0.0,
                percentile95(ocr_rec_latencies_ms_),
                static_cast<unsigned long>(ocr_rec_run_count_),
                ocr_det_rec_run_count_ > 0
                    ? sum_ocr_det_rec_ms_ / static_cast<double>(ocr_det_rec_run_count_) : 0.0,
                percentile95(ocr_det_rec_latencies_ms_),
                static_cast<unsigned long>(ocr_det_rec_run_count_),
                ocr_decision_count_ > 0
                    ? sum_ocr_decision_ms_ / static_cast<double>(ocr_decision_count_) : 0.0,
                percentile95(ocr_decision_latencies_ms_),
                static_cast<unsigned long>(ocr_decision_count_), sum_decision_ms_ / frames,
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
    sum_ocr_rec_ms_ = 0.0;
    ocr_rec_run_count_ = 0;
    ocr_rec_latencies_ms_.clear();
    sum_ocr_det_rec_ms_ = 0.0;
    ocr_det_rec_run_count_ = 0;
    ocr_det_rec_latencies_ms_.clear();
    sum_ocr_decision_ms_ = 0.0;
    ocr_decision_count_ = 0;
    ocr_decision_latencies_ms_.clear();
    sum_seg_model_score_ = 0.0;
    sum_seg_model_instances_ = 0;
  }

  std::string shm_name_;
  std::string encoder_count_topic_{"/chassis/encoder_count"};
  std::string centerline_kalman_steering_topic_{"/cmd_vel"};
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
  bool enable_result_log_{true};
  bool enable_data_log_{false};
  bool guideboard_log_only_{true};
  std::string line_follower_start_service_{"/line_follower/start"};
  std::string line_follower_stop_service_{"/line_follower/stop"};
  const std::string guideboard_start_service_{"/line_follower/start"};
  const std::string guideboard_stop_service_{"/line_follower/stop"};
  double human_service_retry_interval_sec_{0.20};

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
  float seg_nms_threshold_{0.70f};
  float seg_mask_threshold_{0.45f};
  int seg_max_detections_{30};
  bool seg_raw_output_{false};

  std::unique_ptr<ShmReader> shm_reader_;
  YoloDetector detector_;
  YoloSeg segmenter_;
  LaneDecision lane_decision_;
  std::atomic<bool> busy_{false};

  double heading_near_weight_{0.10};
  double heading_mid_weight_{0.50};
  double heading_far_weight_{0.40};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_handle_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr detection_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr label_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y07_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y08_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_y09_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr heading_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr curvature_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_valid_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_debug_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr frame_signature_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guideboard_recognition_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr line_follower_start_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr line_follower_stop_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr guideboard_start_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr guideboard_stop_client_;
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr encoder_count_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr steering_command_sub_;

  bool human_stop_desired_{false};
  bool human_service_stop_active_{false};
  bool human_stop_call_pending_{false};
  bool human_start_call_pending_{false};
  int human_stop_confirm_frames_{2};
  int human_clear_confirm_frames_{2};
  double human_service_last_call_sec_{0.0};
  std::string human_service_action_{"idle"};

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
  double sum_ocr_rec_ms_{0.0};
  uint64_t ocr_rec_run_count_{0};
  std::vector<double> ocr_rec_latencies_ms_;
  double sum_ocr_det_rec_ms_{0.0};
  uint64_t ocr_det_rec_run_count_{0};
  std::vector<double> ocr_det_rec_latencies_ms_;
  double sum_ocr_decision_ms_{0.0};
  uint64_t ocr_decision_count_{0};
  std::vector<double> ocr_decision_latencies_ms_;
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
  bool enable_guideboard_api_{true};
  std::string guideboard_api_url_{"https://qianfan.baidubce.com/v2/chat/completions"};
  std::string guideboard_api_model_{"qwen3.5-35b-a3b"};
  std::string api_key_;
  double guideboard_api_timeout_sec_{1.8};
  double guideboard_api_uncertain_min_confidence_{0.75};
  int guideboard_api_text_history_size_{3};
  double guideboard_api_text_similarity_{0.70};
  int guideboard_api_force_ocr_count_after_stop_{2};
  double guideboard_api_stop_height_ratio_{0.12};
  bool guideboard_api_retry_on_transport_failure_{true};
  int guideboard_api_max_attempts_{1};
  std::string ocr_det_model_path_;
  std::string ocr_rec_model_path_;
  std::string ocr_pipeline_mode_{"rec_then_det"};
  bool ocr_apply_to_control_{false};
  double ocr_rec_interval_sec_{0.05};
  int ocr_fallback_uncertain_count_{2};
  double ocr_branch_wait_timeout_sec_{0.20};
  std::string ocr_unknown_maneuver_{"straight"};
  float ocr_min_text_score_{0.35f};
  float ocr_crop_padding_ratio_{0.25f};
  float lane_guideboard_y0_ratio_{0.2f};
  float lane_guideboard_y1_ratio_{0.7f};
  PPOCRDirectionSystem guideboard_ocr_;
  std::future<OcrTaskResult> guideboard_ocr_future_;
  std::future<GuideboardApiTaskResult> guideboard_api_future_;
  std::deque<GuideboardApiSample> guideboard_api_ocr_history_;
  GuideboardRoutePolicy guideboard_route_policy_;
  uint64_t guideboard_route_session_id_{1};
  uint64_t active_guideboard_branch_event_id_{0};
  bool accept_guideboard_for_route_{true};
  bool guideboard_track_route_eligible_{false};
  bool guideboard_api_request_in_flight_{false};
  bool api_session_attempted_{false};
  bool api_fallback_{false};
  std::string api_trigger_reason_;
  int api_attempt_count_{0};
  double last_api_latency_ms_{-1.0};
  int last_api_http_status_{0};
  float last_api_confidence_{0.0f};
  bool last_api_uncertain_{false};
  bool last_api_accepted_uncertain_{false};
  std::string last_api_corrected_text_;
  int ocr_task_crop_width_{0};
  int ocr_task_crop_height_{0};
  bool has_guideboard_track_{false};
  uint64_t guideboard_track_id_{0};
  cv::Rect2f tracked_guideboard_bbox_;
  double guideboard_last_seen_sec_{0.0};
  int guideboard_uncertain_count_{0};
  bool guideboard_fallback_used_{false};
  uint64_t guideboard_sequence_{0};
  std::string current_guideboard_maneuver_;
  std::string current_guideboard_branch_;
  std::string current_guideboard_decision_source_;
  bool current_guideboard_decision_valid_{false};
  bool current_guideboard_opposite_{false};
  double last_guideboard_height_ratio_{0.0};
  int guideboard_api_force_ocr_count_{0};
  bool guideboard_stop_wait_active_{false};
  bool guideboard_stop_active_{false};
  bool guideboard_stop_call_pending_{false};
  bool guideboard_start_call_pending_{false};
  bool guideboard_start_after_stop_{false};
  double guideboard_decision_start_sec_{0.0};
  double stable_decision_latency_ms_{-1.0};
  double last_guideboard_ocr_sec_{0.0};
  std::string last_ocr_text_;
  float last_ocr_score_{0.0f};
  double last_ocr_time_ms_{0.0};
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
