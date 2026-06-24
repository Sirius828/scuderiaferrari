#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "track_perception_cpp/types.hpp"

namespace track_perception_cpp {

class RknnModel {
 public:
  RknnModel() = default;
  ~RknnModel();

  bool load(const std::string& model_path, int core_id);
  void release();
  bool infer(const cv::Mat& input_nhwc_u8, std::vector<TensorData>& outputs, double* rknn_ms);

  int inputWidth() const { return input_width_; }
  int inputHeight() const { return input_height_; }
  int inputChannels() const { return input_channels_; }

 private:
  static std::vector<uint8_t> readFile(const std::string& path);
  static rknn_core_mask coreMask(int core_id);

  rknn_context ctx_{0};
  bool loaded_{false};
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  int input_width_{0};
  int input_height_{0};
  int input_channels_{3};
  std::mutex mutex_;
};

}  // namespace track_perception_cpp
