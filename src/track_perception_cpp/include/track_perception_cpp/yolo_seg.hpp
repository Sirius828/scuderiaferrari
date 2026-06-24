#pragma once

#include <memory>
#include <string>
#include <vector>

#include "track_perception_cpp/rknn_model.hpp"

namespace track_perception_cpp {

class YoloSeg {
 public:
  bool init(const std::string& model_path, const std::vector<int>& core_ids,
            int input_width, int input_height, float crop_y0, float crop_y1,
            int pad_value, float conf_thresh, float mask_thresh, int max_detections);
  bool infer(const cv::Mat& frame_rgb, cv::Mat& seg_map, double* rknn_ms, double* post_ms);

 private:
  struct PreprocessMeta {
    int orig_w{0};
    int orig_h{0};
    int crop_y0{0};
    int crop_y1{0};
    int crop_w{0};
    int crop_h{0};
    int input_w{0};
    int input_h{0};
    int resize_w{0};
    int resize_h{0};
    int pad_x{0};
    int pad_y{0};
  };

  cv::Mat preprocess(const cv::Mat& frame_rgb, PreprocessMeta& meta) const;
  bool postprocess(const std::vector<TensorData>& outputs, const PreprocessMeta& meta, cv::Mat& seg_map) const;
  cv::Mat restoreMask(const cv::Mat& model_mask, const PreprocessMeta& meta) const;
  RknnModel& nextModel();

  std::vector<std::unique_ptr<RknnModel>> models_;
  size_t next_model_{0};
  int input_width_{384};
  int input_height_{160};
  float crop_y0_ratio_{0.0f};
  float crop_y1_ratio_{1.0f};
  int pad_value_{0};
  float conf_thresh_{0.15f};
  float mask_thresh_{0.45f};
  int max_detections_{30};
};

}  // namespace track_perception_cpp
