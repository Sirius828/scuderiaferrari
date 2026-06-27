#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
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
#include <std_msgs/msg/string.hpp>

#include <opencv2/opencv.hpp>

#include "track_perception_cpp/lane_decision.hpp"
#include "track_perception_cpp/shm_reader.hpp"
#include "track_perception_cpp/yolo_detector.hpp"
#include "track_perception_cpp/yolo_seg.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace track_perception_cpp {

namespace {

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

}  // namespace

class FusedPerceptionNode : public rclcpp::Node {
 public:
  FusedPerceptionNode() : Node("fused_perception_node") {
    declareParameters();
    loadParameters();
    initialize();

    timer_ = create_wall_timer(1ms, std::bind(&FusedPerceptionNode::tick, this));
  }

 private:
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

    declare_parameter<std::string>("det_model_path", "model/yolov8_n_det_split_int8_v2.rknn");
    declare_parameter<std::string>("label_list_path", "model/label_list.txt");
    declare_parameter<int>("det_core_id", 0);
    declare_parameter<int>("det_input_width", 384);
    declare_parameter<int>("det_input_height", 288);
    declare_parameter<double>("det_conf_threshold", 0.5);
    declare_parameter<double>("det_nms_threshold", 0.45);
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
    declare_parameter<int>("seg_pad_value", 0);
    declare_parameter<double>("seg_conf_threshold", 0.15);
    declare_parameter<double>("seg_mask_threshold", 0.45);
    declare_parameter<int>("seg_max_detections", 30);

    declare_parameter<int>("band_count", 13);
    declare_parameter<double>("band_y_min_ratio", 0.65);
    declare_parameter<double>("band_y_max_ratio", 1.0);
    declare_parameter<double>("band_height_ratio", 0.02);
    declare_parameter<int>("min_segment_width_px", 25);
    declare_parameter<int>("min_segment_gap_px", 40);
    declare_parameter<int>("min_pixels_per_band", 80);
    declare_parameter<int>("branch_detect_min_bands", 2);
    declare_parameter<double>("branch_detect_far_band_ratio", 0.7);
    declare_parameter<bool>("enable_guideboard_branch_selection", true);
    declare_parameter<std::string>("guideboard_branch", "right");
    declare_parameter<double>("guideboard_detect_y0_ratio", 0.2);
    declare_parameter<double>("guideboard_detect_y1_ratio", 0.7);
    declare_parameter<bool>("enable_segment_branch_logic", true);
    declare_parameter<bool>("enable_continuity_branch_selection", false);
    declare_parameter<double>("branch_continuity_max_dx_ratio", 0.35);
    declare_parameter<double>("branch_continuity_near_band_ratio", 0.5);
    declare_parameter<bool>("enable_locked_path_continuity", true);
    declare_parameter<double>("locked_path_continuity_after_time", 0.5);
    declare_parameter<double>("locked_path_continuity_max_dx_ratio", 0.28);
    declare_parameter<double>("branch_lock_time", 2.0);
    declare_parameter<double>("min_branch_lock_time", 0.8);
    declare_parameter<int>("exit_single_path_confirm_frames", 5);
    declare_parameter<double>("exit_single_path_min_ratio", 0.8);
    declare_parameter<bool>("enable_merge_wide_segment_logic", true);
    declare_parameter<double>("merge_wide_segment_ratio", 1.35);
    declare_parameter<int>("merge_wide_min_bands", 2);
    declare_parameter<int>("merge_wide_confirm_frames", 2);
    declare_parameter<int>("merge_wide_release_frames", 4);
    declare_parameter<double>("merge_wide_lane_width_alpha", 0.2);
    declare_parameter<int>("fit_min_points", 5);
    declare_parameter<int>("fit_order", 2);
    declare_parameter<int>("branch_fit_order", 2);
    declare_parameter<bool>("enable_fit_point_jump_filter", true);
    declare_parameter<double>("max_fit_point_dx_ratio", 0.22);
    declare_parameter<double>("max_fit_point_dx_px", 140.0);
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
    declare_parameter<bool>("enable_branch_bottom_anchor", true);
    declare_parameter<double>("branch_bottom_anchor_x_ratio", 0.5);
    declare_parameter<double>("branch_bottom_anchor_y_ratio", 0.98);
    declare_parameter<double>("branch_bottom_anchor_weight", 0.6);
    declare_parameter<double>("lookahead_y_ratio", 0.75);
    declare_parameter<bool>("use_heading_term", true);
    declare_parameter<double>("heading_weight", 0.10);
    declare_parameter<double>("near_offset_weight", 0.90);
    declare_parameter<double>("max_offset_jump", 2.0);
    declare_parameter<double>("offset_smoothing_alpha", 0.35);
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
    publish_detections_ = get_parameter("publish_detections").as_bool();
    publish_lane_state_ = get_parameter("publish_lane_state").as_bool();
    enable_status_log_ = get_parameter("enable_status_log").as_bool();
    enable_branch_event_log_ = get_parameter("enable_branch_event_log").as_bool();

    det_model_path_ = resolveTrackPerceptionPath(get_parameter("det_model_path").as_string());
    label_list_path_ = resolveTrackPerceptionPath(get_parameter("label_list_path").as_string());
    det_core_id_ = static_cast<int>(get_parameter("det_core_id").as_int());
    det_input_width_ = static_cast<int>(get_parameter("det_input_width").as_int());
    det_input_height_ = static_cast<int>(get_parameter("det_input_height").as_int());
    det_conf_threshold_ = static_cast<float>(get_parameter("det_conf_threshold").as_double());
    det_nms_threshold_ = static_cast<float>(get_parameter("det_nms_threshold").as_double());

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
    seg_mask_threshold_ = static_cast<float>(get_parameter("seg_mask_threshold").as_double());
    seg_max_detections_ = static_cast<int>(get_parameter("seg_max_detections").as_int());

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
    lane_cfg.branch_detect_far_band_ratio = static_cast<float>(get_parameter("branch_detect_far_band_ratio").as_double());
    lane_cfg.outer_side = get_parameter("outer_side").as_string();
    lane_cfg.enable_guideboard_branch_selection = get_parameter("enable_guideboard_branch_selection").as_bool();
    lane_cfg.guideboard_branch = get_parameter("guideboard_branch").as_string();
    lane_cfg.guideboard_detect_y0_ratio = static_cast<float>(get_parameter("guideboard_detect_y0_ratio").as_double());
    lane_cfg.guideboard_detect_y1_ratio = static_cast<float>(get_parameter("guideboard_detect_y1_ratio").as_double());
    lane_cfg.enable_continuity_branch_selection = get_parameter("enable_continuity_branch_selection").as_bool();
    lane_cfg.branch_continuity_max_dx_ratio = static_cast<float>(get_parameter("branch_continuity_max_dx_ratio").as_double());
    lane_cfg.branch_continuity_near_band_ratio = static_cast<float>(get_parameter("branch_continuity_near_band_ratio").as_double());
    lane_cfg.enable_locked_path_continuity = get_parameter("enable_locked_path_continuity").as_bool();
    lane_cfg.locked_path_continuity_after_time = static_cast<float>(get_parameter("locked_path_continuity_after_time").as_double());
    lane_cfg.locked_path_continuity_max_dx_ratio = static_cast<float>(get_parameter("locked_path_continuity_max_dx_ratio").as_double());
    lane_cfg.branch_lock_time = static_cast<float>(get_parameter("branch_lock_time").as_double());
    lane_cfg.min_branch_lock_time = static_cast<float>(get_parameter("min_branch_lock_time").as_double());
    lane_cfg.exit_single_path_confirm_frames = static_cast<int>(get_parameter("exit_single_path_confirm_frames").as_int());
    lane_cfg.exit_single_path_min_ratio = static_cast<float>(get_parameter("exit_single_path_min_ratio").as_double());
    lane_cfg.enable_merge_wide_segment_logic = get_parameter("enable_merge_wide_segment_logic").as_bool();
    lane_cfg.merge_wide_segment_ratio = static_cast<float>(get_parameter("merge_wide_segment_ratio").as_double());
    lane_cfg.merge_wide_min_bands = static_cast<int>(get_parameter("merge_wide_min_bands").as_int());
    lane_cfg.merge_wide_confirm_frames = static_cast<int>(get_parameter("merge_wide_confirm_frames").as_int());
    lane_cfg.merge_wide_release_frames = static_cast<int>(get_parameter("merge_wide_release_frames").as_int());
    lane_cfg.merge_wide_lane_width_alpha = static_cast<float>(get_parameter("merge_wide_lane_width_alpha").as_double());
    lane_cfg.fit_min_points = static_cast<int>(get_parameter("fit_min_points").as_int());
    lane_cfg.fit_order = static_cast<int>(get_parameter("fit_order").as_int());
    lane_cfg.branch_fit_order = static_cast<int>(get_parameter("branch_fit_order").as_int());
    lane_cfg.enable_fit_point_jump_filter = get_parameter("enable_fit_point_jump_filter").as_bool();
    lane_cfg.max_fit_point_dx_ratio = static_cast<float>(get_parameter("max_fit_point_dx_ratio").as_double());
    lane_cfg.max_fit_point_dx_px = static_cast<float>(get_parameter("max_fit_point_dx_px").as_double());
    lane_cfg.enable_branch_bottom_anchor = get_parameter("enable_branch_bottom_anchor").as_bool();
    lane_cfg.branch_bottom_anchor_x_ratio = static_cast<float>(get_parameter("branch_bottom_anchor_x_ratio").as_double());
    lane_cfg.branch_bottom_anchor_y_ratio = static_cast<float>(get_parameter("branch_bottom_anchor_y_ratio").as_double());
    lane_cfg.branch_bottom_anchor_weight = static_cast<float>(get_parameter("branch_bottom_anchor_weight").as_double());
    lane_cfg.lookahead_y_ratio = static_cast<float>(get_parameter("lookahead_y_ratio").as_double());
    lane_cfg.use_heading_term = get_parameter("use_heading_term").as_bool();
    lane_cfg.heading_weight = static_cast<float>(get_parameter("heading_weight").as_double());
    lane_cfg.near_offset_weight = static_cast<float>(get_parameter("near_offset_weight").as_double());
    lane_cfg.max_offset_jump = static_cast<float>(get_parameter("max_offset_jump").as_double());
    lane_cfg.offset_smoothing_alpha = static_cast<float>(get_parameter("offset_smoothing_alpha").as_double());
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

  void initialize() {
    detection_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/detection/results", 10);
    label_pub_ = create_publisher<std_msgs::msg::String>("/detection/labels", 10);
    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    center_offset_pub_ = create_publisher<std_msgs::msg::Float32>("/segmentation/center_offset", sensor_qos);
    is_valid_pub_ = create_publisher<std_msgs::msg::Bool>("/segmentation/is_valid", sensor_qos);
    stop_request_pub_ = create_publisher<std_msgs::msg::Bool>("/perception/stop_request", 10);
    lane_state_pub_ = create_publisher<std_msgs::msg::String>("/perception/lane_state", 10);

    shm_reader_ = std::make_unique<ShmReader>(shm_name_);
    if (!shm_reader_->connect()) {
      RCLCPP_WARN(get_logger(), "SHM not connected yet: /dev/shm/%s", shm_name_.c_str());
    }

    if (!detector_.init(det_model_path_, label_list_path_, det_core_id_, det_input_width_,
                        det_input_height_, det_conf_threshold_, det_nms_threshold_)) {
      throw std::runtime_error("failed to initialize detector model");
    }
    if (!segmenter_.init(seg_model_path_, seg_core_ids_, seg_input_width_, seg_input_height_,
                         seg_crop_y0_ratio_, seg_crop_y1_ratio_, seg_pad_value_,
                         seg_conf_threshold_, seg_mask_threshold_, seg_max_detections_)) {
      throw std::runtime_error("failed to initialize segmentation model");
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
      double rknn_ms = 0.0;
      double post_ms = 0.0;
      bool ok = segmenter_.infer(frame_rgb, local_seg_map, &rknn_ms, &post_ms);
      return std::make_tuple(ok, local_seg_map, rknn_ms, post_ms);
    });

    auto det_result = det_future.get();
    auto seg_result = seg_future.get();
    bool det_ok = std::get<0>(det_result);
    bool seg_ok = std::get<0>(seg_result);
    detections = std::move(std::get<1>(det_result));
    seg_map = std::get<1>(seg_result);
    stats.det_rknn_ms = std::get<2>(det_result);
    stats.det_post_ms = std::get<3>(det_result);
    stats.seg_rknn_ms = std::get<2>(seg_result);
    stats.seg_post_ms = std::get<3>(seg_result);

    auto t_decision0 = std::chrono::steady_clock::now();
    LaneState lane_state = lane_decision_.decide(seg_map, detections);
    auto t_decision1 = std::chrono::steady_clock::now();
    stats.decision_ms = std::chrono::duration<double, std::milli>(t_decision1 - t_decision0).count();
    const LaneDebugInfo& lane_debug = lane_decision_.debugInfo();
    logDecisionStatus(lane_state, lane_debug);

    refreshDebugParameters();
    if (show_window_) {
      showDebugWindow(frame_rgb, seg_map, detections, lane_state, lane_debug);
    }

    auto t_pub0 = std::chrono::steady_clock::now();
    publishAll(detections, lane_state);
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

    if (!det_ok || !seg_ok) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "inference failure: det_ok=%d seg_ok=%d", det_ok, seg_ok);
    }

    logPerfIfNeeded();
  }

  void publishAll(const std::vector<Detection>& detections, const LaneState& lane_state) {
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

    std_msgs::msg::Float32 offset_msg;
    offset_msg.data = lane_state.control_offset;
    center_offset_pub_->publish(offset_msg);

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
  }

  void showDebugWindow(const cv::Mat& frame_rgb, const cv::Mat& seg_map,
                       const std::vector<Detection>& detections, const LaneState& lane_state,
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

      cv::Mat mask_vis(vis.size(), vis.type(), cv::Scalar(0, 0, 0));
      mask_vis.setTo(cv::Scalar(255, 0, 0), mask_u8);
      cv::addWeighted(vis, 1.0f - alpha, mask_vis, alpha, 0.0, vis);
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

    if (show_branch_debug_) {
      drawLaneDebug(vis, debug_info);
    }

    std::ostringstream status;
    status << lane_state.road_state << " offset=" << lane_state.control_offset
           << " valid=" << (lane_state.is_valid ? 1 : 0) << " task=" << lane_state.task_state
           << " alpha=" << std::clamp(blend_alpha_, 0.0f, 1.0f);
    cv::putText(vis, status.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::imshow("fused_perception", vis);
    cv::waitKey(1);
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
    for (const auto& band : debug_info.bands) {
      int y0 = std::clamp(band.y0, 0, std::max(0, vis.rows - 1));
      int y1 = std::clamp(band.y1, y0 + 1, vis.rows);
      cv::rectangle(vis, cv::Point(0, y0), cv::Point(vis.cols - 1, y1 - 1),
                    cv::Scalar(255, 200, 100), 1);

      for (const auto& seg : band.segments) {
        int x0 = std::clamp(seg.x0, 0, std::max(0, vis.cols - 1));
        int x1 = std::clamp(seg.x1, 0, std::max(0, vis.cols - 1));
        int cy = (y0 + y1) / 2;
        cv::line(vis, cv::Point(x0, y0), cv::Point(x0, y1 - 1), cv::Scalar(0, 255, 0), 1);
        cv::line(vis, cv::Point(x1, y0), cv::Point(x1, y1 - 1), cv::Scalar(0, 255, 0), 1);
        cv::circle(vis, cv::Point(std::clamp(seg.center_x, 0, std::max(0, vis.cols - 1)), cy),
                   3, cv::Scalar(0, 255, 255), -1);
      }

      if (band.selected_center_x >= 0) {
        int cy = (y0 + y1) / 2;
        cv::circle(vis, cv::Point(std::clamp(band.selected_center_x, 0, std::max(0, vis.cols - 1)), cy),
                   5, cv::Scalar(0, 128, 255), 2);
      }
    }

    const auto& points = debug_info.fit_points;
    for (const auto& point : points) {
      cv::circle(vis, cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))),
                 4, cv::Scalar(0, 255, 255), -1);
    }

    if (!debug_info.fit_coeffs.empty()) {
      int y0 = debug_info.bands.empty() ? 0 : debug_info.bands.front().y0;
      int y1 = debug_info.bands.empty() ? vis.rows - 1 : debug_info.bands.back().y1;
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
    enable_status_log_ = get_parameter("enable_status_log").as_bool();
    enable_branch_event_log_ = get_parameter("enable_branch_event_log").as_bool();
  }

  void logDecisionStatus(const LaneState& lane_state, const LaneDebugInfo& debug_info) {
    if (enable_branch_event_log_) {
      if (debug_info.guideboard_seen && !last_guideboard_seen_) {
        RCLCPP_INFO(get_logger(),
                    "GuideBoard usable: roi_count=%d total=%d best_conf=%.2f center=(%.1f,%.1f); "
                    "branch_detected=%d score=%d locked=%d side=%s",
                    debug_info.guideboard_roi_count, debug_info.guideboard_count,
                    debug_info.guideboard_best_confidence, debug_info.guideboard_best_center.x,
                    debug_info.guideboard_best_center.y, debug_info.branch_detected,
                    debug_info.branch_score, debug_info.branch_locked,
                    debug_info.locked_branch_side.c_str());
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
        RCLCPP_INFO(get_logger(), "branch_detected=%d score=%d segments=%d raw_points=%d fit_points=%d",
                    debug_info.branch_detected, debug_info.branch_score, debug_info.segment_count,
                    debug_info.raw_point_count, debug_info.fit_point_count);
      }
      if (lane_state.road_state != last_road_state_ ||
          lane_state.branch_side != last_branch_side_) {
        RCLCPP_INFO(get_logger(), "road_state=%s branch_side=%s offset=%.3f valid=%d guideboard=%d",
                    lane_state.road_state.c_str(), lane_state.branch_side.c_str(),
                    lane_state.control_offset, lane_state.is_valid, debug_info.guideboard_seen);
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
    RCLCPP_INFO(get_logger(),
                "status road=%s branch=%s offset=%.3f lateral=%.3f heading=%.3f conf=%.2f valid=%d "
                "branch_detected=%d score=%d guideboard_roi=%d/%d guideboard_best=%.2f@(%.0f,%.0f) "
                "obstacles=%zu segments=%d points=%d/%d task=%s",
                lane_state.road_state.c_str(), lane_state.branch_side.c_str(),
                lane_state.control_offset, lane_state.lateral_offset, lane_state.heading_error,
                lane_state.confidence, lane_state.is_valid, debug_info.branch_detected,
                debug_info.branch_score, debug_info.guideboard_roi_count, debug_info.guideboard_count,
                debug_info.guideboard_best_confidence, debug_info.guideboard_best_center.x,
                debug_info.guideboard_best_center.y, debug_info.obstacle_zones.size(), debug_info.segment_count,
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
    RCLCPP_INFO(get_logger(),
                "perf %.1fs upstream=%.1f fps processed=%.1f fps det=%.2f/%.2f ms "
                "seg=%.2f/%.2f ms decision=%.2f ms publish=%.2f ms detections=%zu",
                dt, upstream_fps, processed_fps, sum_det_rknn_ms_ / frames,
                sum_det_post_ms_ / frames, sum_seg_rknn_ms_ / frames,
                sum_seg_post_ms_ / frames, sum_decision_ms_ / frames,
                sum_publish_ms_ / frames, last_det_count_);

    perf_start_sec_ = now;
    processed_frames_ = 0;
    upstream_frames_ = 0;
    sum_det_rknn_ms_ = 0.0;
    sum_det_post_ms_ = 0.0;
    sum_seg_rknn_ms_ = 0.0;
    sum_seg_post_ms_ = 0.0;
    sum_decision_ms_ = 0.0;
    sum_publish_ms_ = 0.0;
  }

  std::string shm_name_;
  bool enable_flip_{true};
  int flip_code_{0};
  std::string input_format_{"RGB"};
  bool enable_perf_stats_{true};
  double perf_interval_{2.0};
  bool show_window_{false};
  float blend_alpha_{1.0f};
  bool show_branch_debug_{true};
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

  std::string seg_model_path_;
  std::vector<int> seg_core_ids_{1, 2};
  int seg_input_width_{384};
  int seg_input_height_{160};
  float seg_crop_y0_ratio_{0.0f};
  float seg_crop_y1_ratio_{1.0f};
  int seg_pad_value_{0};
  float seg_conf_threshold_{0.15f};
  float seg_mask_threshold_{0.45f};
  int seg_max_detections_{30};

  std::unique_ptr<ShmReader> shm_reader_;
  YoloDetector detector_;
  YoloSeg segmenter_;
  LaneDecision lane_decision_;
  std::atomic<bool> busy_{false};

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr detection_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr label_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr center_offset_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_valid_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_request_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_state_pub_;

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
  size_t last_det_count_{0};
  double last_status_log_sec_{0.0};
  bool last_guideboard_seen_{false};
  bool last_branch_detected_{false};
  int last_branch_score_{-1};
  std::string last_road_state_;
  std::string last_branch_side_;
  std::string last_task_state_;
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
