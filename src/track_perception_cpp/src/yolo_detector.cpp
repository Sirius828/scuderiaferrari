#include "track_perception_cpp/yolo_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>

namespace track_perception_cpp {

namespace {

constexpr int kPostTopkPerClass = 50;
constexpr int kPostKeepTopk = 30;
constexpr float kClassAgnosticNmsThresh = 0.60f;
constexpr int kRegMax = 16;

std::string trimLabel(std::string text) {
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
  text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), text.end());
  return text;
}

float dflExpected(const float* logits, int reg_max) {
  float max_logit = logits[0];
  for (int i = 1; i < reg_max; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  float sum = 0.0f;
  float weighted = 0.0f;
  for (int i = 0; i < reg_max; ++i) {
    float e = std::exp(logits[i] - max_logit);
    sum += e;
    weighted += e * static_cast<float>(i);
  }
  return sum > 0.0f ? weighted / sum : 0.0f;
}

float sigmoid(float x) {
  x = std::clamp(x, -50.0f, 50.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

bool YoloDetector::init(const std::string& model_path, const std::string& label_path, int core_id,
                        int input_width, int input_height, float conf_thresh, float nms_thresh,
                        bool raw_output) {
  input_width_ = input_width;
  input_height_ = input_height;
  conf_thresh_ = conf_thresh;
  nms_thresh_ = nms_thresh;
  raw_output_ = raw_output;

  std::ifstream labels_file(label_path);
  std::string line;
  while (std::getline(labels_file, line)) {
    line = trimLabel(line);
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
  if (raw_output_) {
    postprocessRaw(outputs, original_size, meta, detections);
  } else {
    postprocessDecoded(outputs, original_size, meta, detections);
  }
}

void YoloDetector::postprocessDecoded(const std::vector<TensorData>& outputs, const cv::Size& original_size,
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

    for (int class_id = 0; class_id < score_cols; ++class_id) {
      float conf = scores[class_id];
      if (conf < conf_thresh_) {
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
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  if (kPostTopkPerClass > 0) {
    std::vector<int> kept_per_class(static_cast<size_t>(std::max(0, score_cols)), 0);
    std::vector<Detection> topk_candidates;
    topk_candidates.reserve(candidates.size());
    for (const auto& det : candidates) {
      if (det.class_id < 0 || det.class_id >= static_cast<int>(kept_per_class.size())) {
        topk_candidates.push_back(det);
        continue;
      }
      if (kept_per_class[det.class_id] >= kPostTopkPerClass) {
        continue;
      }
      ++kept_per_class[det.class_id];
      topk_candidates.push_back(det);
    }
    candidates = std::move(topk_candidates);
  }

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

  if (detections.size() > 1) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<Detection> agnostic;
    agnostic.reserve(detections.size());
    for (const auto& det : detections) {
      bool duplicate = false;
      for (const auto& kept : agnostic) {
        if (iou(det.bbox, kept.bbox) > kClassAgnosticNmsThresh) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        agnostic.push_back(det);
      }
    }
    detections = std::move(agnostic);
  }

  if (kPostKeepTopk > 0 && detections.size() > static_cast<size_t>(kPostKeepTopk)) {
    detections.resize(kPostKeepTopk);
  }
}

void YoloDetector::postprocessRaw(const std::vector<TensorData>& outputs, const cv::Size& original_size,
                                  const LetterboxMeta& meta, std::vector<Detection>& detections) const {
  detections.clear();
  int num_classes = static_cast<int>(labels_.size());
  if (num_classes <= 0) {
    return;
  }

  std::vector<Detection> candidates;
  for (const auto& output : outputs) {
    if (output.dims.size() != 4 || output.dims[0] != 1) {
      continue;
    }
    int channels = output.dims[1];
    int h = output.dims[2];
    int w = output.dims[3];
    int expected_min = 4 * kRegMax + num_classes;
    if (channels < expected_min || h <= 0 || w <= 0) {
      continue;
    }
    float stride_x = static_cast<float>(input_width_) / static_cast<float>(w);
    float stride_y = static_cast<float>(input_height_) / static_cast<float>(h);
    size_t plane = static_cast<size_t>(h) * w;
    const float* data = output.data.data();

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        size_t pos = static_cast<size_t>(y) * w + x;
        int class_id = 0;
        float best_score = sigmoid(data[(4 * kRegMax) * plane + pos]);
        for (int c = 1; c < num_classes; ++c) {
          float s = sigmoid(data[(4 * kRegMax + c) * plane + pos]);
          if (s > best_score) {
            best_score = s;
            class_id = c;
          }
        }
        if (best_score < conf_thresh_) {
          continue;
        }

        float dist[4];
        for (int side = 0; side < 4; ++side) {
          float logits[kRegMax];
          for (int r = 0; r < kRegMax; ++r) {
            logits[r] = data[(side * kRegMax + r) * plane + pos];
          }
          dist[side] = dflExpected(logits, kRegMax);
        }

        float cx = (static_cast<float>(x) + 0.5f) * stride_x;
        float cy = (static_cast<float>(y) + 0.5f) * stride_y;
        float x1 = (cx - dist[0] * stride_x - meta.pad_x) / std::max(meta.scale, 1e-6f);
        float y1 = (cy - dist[1] * stride_y - meta.pad_y) / std::max(meta.scale, 1e-6f);
        float x2 = (cx + dist[2] * stride_x - meta.pad_x) / std::max(meta.scale, 1e-6f);
        float y2 = (cy + dist[3] * stride_y - meta.pad_y) / std::max(meta.scale, 1e-6f);
        x1 = std::clamp(x1, 0.0f, static_cast<float>(original_size.width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(original_size.height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(original_size.width));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(original_size.height));
        if (x2 <= x1 || y2 <= y1) {
          continue;
        }

        Detection det;
        det.class_id = class_id;
        det.class_name = class_id < static_cast<int>(labels_.size()) ? labels_[class_id] : "unknown";
        det.confidence = best_score;
        det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        det.center = cv::Point2f((x1 + x2) * 0.5f, (y1 + y2) * 0.5f);
        candidates.push_back(det);
      }
    }
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

  if (detections.size() > 1) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<Detection> agnostic;
    agnostic.reserve(detections.size());
    for (const auto& det : detections) {
      bool duplicate = false;
      for (const auto& kept : agnostic) {
        if (iou(det.bbox, kept.bbox) > kClassAgnosticNmsThresh) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        agnostic.push_back(det);
      }
    }
    detections = std::move(agnostic);
  }

  if (kPostKeepTopk > 0 && detections.size() > static_cast<size_t>(kPostKeepTopk)) {
    detections.resize(kPostKeepTopk);
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
