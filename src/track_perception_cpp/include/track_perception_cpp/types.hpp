#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace track_perception_cpp {

struct Detection {
  int class_id{0};
  std::string class_name;
  float confidence{0.0f};
  cv::Rect2f bbox;
  cv::Point2f center;
};

struct LaneState {
  float control_offset{0.0f};
  float lateral_offset{0.0f};
  float bottom_offset{0.0f};
  float raw_control_offset{0.0f};
  float lookahead_x{0.0f};
  float lookahead_y{0.0f};
  float heading_error{0.0f};
  float curvature{0.0f};
  float confidence{0.0f};
  bool is_valid{false};
  std::string road_state{"LOW_CONFIDENCE"};
  std::string branch_side;
  std::string task_state{"CLEAR"};
  float task_bias{0.0f};
  double timestamp{0.0};
};

struct Frame {
  uint64_t fid{0};
  uint32_t width{0};
  uint32_t height{0};
  cv::Mat image;
};

struct TensorData {
  std::vector<int> dims;
  std::vector<float> data;

  int count() const {
    int n = 1;
    for (int d : dims) {
      n *= d;
    }
    return n;
  }
};

struct PerfStats {
  double det_rknn_ms{0.0};
  double det_post_ms{0.0};
  double seg_rknn_ms{0.0};
  double seg_post_ms{0.0};
  double decision_ms{0.0};
  double publish_ms{0.0};
};

double nowSeconds();
std::string laneStateToJson(const LaneState& state);

}  // namespace track_perception_cpp
