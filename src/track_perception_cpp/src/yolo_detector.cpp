#include "track_perception_cpp/yolo_detector.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>

namespace track_perception_cpp {

bool YoloDetector::init(const std::string& model_path, const std::string& label_path, int core_id,
                        int input_width, int input_height, float conf_thresh, float nms_thresh) {
  input_width_ = input_width;
  input_height_ = input_height;
  conf_thresh_ = conf_thresh;
  nms_thresh_ = nms_thresh;

  std::ifstream labels_file(label_path);
  std::string line;
  while (std::getline(labels_file, line)) {
    if (!line.empty()) {
      labels_.push_back(line);
    }
  }
  return model_.load(model_path, core_id);
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame_rgb, LetterboxMeta& meta) const {
  int src_h = frame_rgb.rows;
  int src_w = frame_rgb.cols;
  meta.scale = std::min(static_cast<float>(input_width_) / src_w,
                        static_cast<float>(input_height_) / src_h);
  int resized_w = static_cast<int>(std::round(src_w * meta.scale));
  int resized_h = static_cast<int>(std::round(src_h * meta.scale));
  meta.pad_x = 0;
  meta.pad_y = 0;

  cv::Mat resized;
  cv::resize(frame_rgb, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
  cv::Mat canvas(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(meta.pad_x, meta.pad_y, resized_w, resized_h)));
  return canvas;
}

float YoloDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  float inter = (a & b).area();
  float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

void YoloDetector::postprocess(const std::vector<TensorData>& outputs, const cv::Size& original_size,
                               const LetterboxMeta& meta, std::vector<Detection>& detections) const {
  detections.clear();
  if (outputs.size() < 2) {
    return;
  }

  const TensorData* boxes_tensor = nullptr;
  const TensorData* scores_tensor = nullptr;
  for (const auto& output : outputs) {
    if (output.dims.size() < 2) {
      continue;
    }
    int rows = output.dims[0] == 1 && output.dims.size() >= 3 ? output.dims[1] : output.dims[0];
    int cols = output.dims[0] == 1 && output.dims.size() >= 3 ? output.dims[2] : output.dims[1];
    if (cols == 4 && boxes_tensor == nullptr) {
      boxes_tensor = &output;
    } else if (scores_tensor == nullptr) {
      scores_tensor = &output;
    }
    (void)rows;
  }
  if (boxes_tensor == nullptr || scores_tensor == nullptr) {
    return;
  }

  auto tensorShape = [](const TensorData& t) {
    int rows = t.dims[0] == 1 && t.dims.size() >= 3 ? t.dims[1] : t.dims[0];
    int cols = t.dims[0] == 1 && t.dims.size() >= 3 ? t.dims[2] : t.dims[1];
    return std::pair<int, int>(rows, cols);
  };
  auto [box_rows, box_cols] = tensorShape(*boxes_tensor);
  auto [score_rows, score_cols] = tensorShape(*scores_tensor);
  if (box_cols != 4 || box_rows != score_rows || score_cols <= 0) {
    return;
  }

  std::vector<Detection> candidates;
  for (int i = 0; i < box_rows; ++i) {
    const float* scores = scores_tensor->data.data() + i * score_cols;
    int class_id = static_cast<int>(std::max_element(scores, scores + score_cols) - scores);
    float conf = scores[class_id];
    if (conf < conf_thresh_) {
      continue;
    }

    const float* b = boxes_tensor->data.data() + i * 4;
    float x1 = (b[0] - meta.pad_x) / std::max(meta.scale, 1e-6f);
    float y1 = (b[1] - meta.pad_y) / std::max(meta.scale, 1e-6f);
    float x2 = (b[2] - meta.pad_x) / std::max(meta.scale, 1e-6f);
    float y2 = (b[3] - meta.pad_y) / std::max(meta.scale, 1e-6f);
    x1 = std::clamp(x1, 0.0f, static_cast<float>(original_size.width - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(original_size.height - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(original_size.width));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(original_size.height));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    Detection det;
    det.class_id = class_id;
    det.class_name = class_id >= 0 && class_id < static_cast<int>(labels_.size()) ? labels_[class_id] : "unknown";
    det.confidence = conf;
    det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    det.center = cv::Point2f((x1 + x2) * 0.5f, (y1 + y2) * 0.5f);
    candidates.push_back(det);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  std::vector<bool> removed(candidates.size(), false);
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    detections.push_back(candidates[i]);
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!removed[j] && candidates[i].class_id == candidates[j].class_id &&
          iou(candidates[i].bbox, candidates[j].bbox) > nms_thresh_) {
        removed[j] = true;
      }
    }
  }
}

bool YoloDetector::infer(const cv::Mat& frame_rgb, std::vector<Detection>& detections,
                         double* rknn_ms, double* post_ms) {
  LetterboxMeta meta;
  cv::Mat input = preprocess(frame_rgb, meta);
  std::vector<TensorData> outputs;
  if (!model_.infer(input, outputs, rknn_ms)) {
    return false;
  }
  auto t0 = std::chrono::steady_clock::now();
  postprocess(outputs, frame_rgb.size(), meta, detections);
  auto t1 = std::chrono::steady_clock::now();
  if (post_ms != nullptr) {
    *post_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return true;
}

}  // namespace track_perception_cpp
