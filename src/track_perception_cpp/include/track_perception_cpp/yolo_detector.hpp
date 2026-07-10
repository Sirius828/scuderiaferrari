#pragma once

#include <string>
#include <vector>

#include "track_perception_cpp/rknn_model.hpp"

namespace track_perception_cpp {

class YoloDetector {
 public:
  bool init(const std::string& model_path, const std::string& label_path, int core_id,
            int input_width, int input_height, float conf_thresh, float nms_thresh,
            bool raw_output);
  bool infer(const cv::Mat& frame_rgb, std::vector<Detection>& detections,
             double* rknn_ms, double* post_ms);
  const std::vector<std::string>& labels() const { return labels_; }

 private:
  struct LetterboxMeta {
    float scale{1.0f};
    int pad_x{0};
    int pad_y{0};
  };

  cv::Mat preprocess(const cv::Mat& frame_rgb, LetterboxMeta& meta) const;
  void postprocess(const std::vector<TensorData>& outputs, const cv::Size& original_size,
                   const LetterboxMeta& meta, std::vector<Detection>& detections) const;
  void postprocessDecoded(const std::vector<TensorData>& outputs, const cv::Size& original_size,
                          const LetterboxMeta& meta, std::vector<Detection>& detections) const;
  void postprocessRaw(const std::vector<TensorData>& outputs, const cv::Size& original_size,
                      const LetterboxMeta& meta, std::vector<Detection>& detections) const;
  static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

  RknnModel model_;
  std::vector<std::string> labels_;
  int input_width_{384};
  int input_height_{288};
  float conf_thresh_{0.5f};
  float nms_thresh_{0.45f};
  bool raw_output_{false};
};

}  // namespace track_perception_cpp
