#include "track_perception_cpp/yolo_seg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace track_perception_cpp {
namespace {

float sigmoid(float x) {
  x = std::clamp(x, -50.0f, 50.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

struct MatrixView {
  int rows{0};
  int cols{0};
  std::vector<float> data;
};

MatrixView asMatrix(const TensorData& tensor) {
  MatrixView view;
  if (tensor.dims.size() == 3 && tensor.dims[0] == 1) {
    view.rows = tensor.dims[1];
    view.cols = tensor.dims[2];
    view.data = tensor.data;
  } else if (tensor.dims.size() == 2) {
    view.rows = tensor.dims[0];
    view.cols = tensor.dims[1];
    view.data = tensor.data;
  } else {
    return view;
  }

  if (view.rows <= 128 && view.cols > view.rows) {
    std::vector<float> transposed(static_cast<size_t>(view.rows) * view.cols);
    for (int r = 0; r < view.rows; ++r) {
      for (int c = 0; c < view.cols; ++c) {
        transposed[static_cast<size_t>(c) * view.rows + r] =
            view.data[static_cast<size_t>(r) * view.cols + c];
      }
    }
    std::swap(view.rows, view.cols);
    view.data = std::move(transposed);
  }
  return view;
}

bool asProto(const TensorData& tensor, int& mask_dim, int& proto_h, int& proto_w, std::vector<float>& proto) {
  if (tensor.dims.size() != 4) {
    return false;
  }

  int d0 = tensor.dims[0];
  int d1 = tensor.dims[1];
  int d2 = tensor.dims[2];
  int d3 = tensor.dims[3];
  if (d0 == 1 && d1 <= 64) {
    mask_dim = d1;
    proto_h = d2;
    proto_w = d3;
    proto = tensor.data;
    return true;
  }
  if (d0 == 1 && d3 <= 64) {
    mask_dim = d3;
    proto_h = d1;
    proto_w = d2;
    proto.resize(static_cast<size_t>(mask_dim) * proto_h * proto_w);
    for (int y = 0; y < proto_h; ++y) {
      for (int x = 0; x < proto_w; ++x) {
        for (int c = 0; c < mask_dim; ++c) {
          size_t src = ((static_cast<size_t>(y) * proto_w + x) * mask_dim) + c;
          size_t dst = (static_cast<size_t>(c) * proto_h + y) * proto_w + x;
          proto[dst] = tensor.data[src];
        }
      }
    }
    return true;
  }
  return false;
}

}  // namespace

bool YoloSeg::init(const std::string& model_path, const std::vector<int>& core_ids,
                   int input_width, int input_height, float crop_y0, float crop_y1,
                   int pad_value, float conf_thresh, float mask_thresh, int max_detections) {
  input_width_ = input_width;
  input_height_ = input_height;
  crop_y0_ratio_ = crop_y0;
  crop_y1_ratio_ = crop_y1;
  pad_value_ = pad_value;
  conf_thresh_ = conf_thresh;
  mask_thresh_ = mask_thresh;
  max_detections_ = max_detections;

  std::vector<int> cores = core_ids.empty() ? std::vector<int>{1, 2} : core_ids;
  for (int core_id : cores) {
    auto model = std::make_unique<RknnModel>();
    if (!model->load(model_path, core_id)) {
      return false;
    }
    models_.push_back(std::move(model));
  }
  return !models_.empty();
}

RknnModel& YoloSeg::nextModel() {
  RknnModel& model = *models_[next_model_ % models_.size()];
  next_model_++;
  return model;
}

cv::Mat YoloSeg::preprocess(const cv::Mat& frame_rgb, PreprocessMeta& meta) const {
  meta.orig_w = frame_rgb.cols;
  meta.orig_h = frame_rgb.rows;
  meta.input_w = input_width_;
  meta.input_h = input_height_;

  float y0_ratio = std::clamp(crop_y0_ratio_, 0.0f, 1.0f);
  float y1_ratio = std::clamp(crop_y1_ratio_, y0_ratio + 1e-6f, 1.0f);
  meta.crop_y0 = std::clamp(static_cast<int>(std::round(meta.orig_h * y0_ratio)), 0, meta.orig_h - 1);
  meta.crop_y1 = std::clamp(static_cast<int>(std::round(meta.orig_h * y1_ratio)), meta.crop_y0 + 1, meta.orig_h);
  meta.crop_w = meta.orig_w;
  meta.crop_h = meta.crop_y1 - meta.crop_y0;

  cv::Mat crop = frame_rgb(cv::Rect(0, meta.crop_y0, meta.crop_w, meta.crop_h));
  float scale = std::min(static_cast<float>(input_width_) / std::max(1, meta.crop_w),
                         static_cast<float>(input_height_) / std::max(1, meta.crop_h));
  meta.resize_w = std::clamp(static_cast<int>(std::round(meta.crop_w * scale)), 1, input_width_);
  meta.resize_h = std::clamp(static_cast<int>(std::round(meta.crop_h * scale)), 1, input_height_);
  meta.pad_x = (input_width_ - meta.resize_w) / 2;
  meta.pad_y = (input_height_ - meta.resize_h) / 2;

  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(meta.resize_w, meta.resize_h), 0, 0, cv::INTER_LINEAR);
  cv::Mat canvas(input_height_, input_width_, CV_8UC3,
                 cv::Scalar(pad_value_, pad_value_, pad_value_));
  resized.copyTo(canvas(cv::Rect(meta.pad_x, meta.pad_y, meta.resize_w, meta.resize_h)));
  return canvas;
}

cv::Mat YoloSeg::restoreMask(const cv::Mat& model_mask, const PreprocessMeta& meta) const {
  cv::Mat resized_model;
  if (model_mask.cols != meta.input_w || model_mask.rows != meta.input_h) {
    cv::resize(model_mask, resized_model, cv::Size(meta.input_w, meta.input_h), 0, 0, cv::INTER_NEAREST);
  } else {
    resized_model = model_mask;
  }

  cv::Rect valid(meta.pad_x, meta.pad_y, meta.resize_w, meta.resize_h);
  cv::Mat unpadded = resized_model(valid);
  cv::Mat crop_mask;
  cv::resize(unpadded, crop_mask, cv::Size(meta.crop_w, meta.crop_h), 0, 0, cv::INTER_NEAREST);

  cv::Mat full(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
  crop_mask.copyTo(full(cv::Rect(0, meta.crop_y0, meta.crop_w, meta.crop_h)));
  return full;
}

bool YoloSeg::postprocess(const std::vector<TensorData>& outputs, const PreprocessMeta& meta, cv::Mat& seg_map) const {
  int mask_dim = 0;
  int proto_h = 0;
  int proto_w = 0;
  std::vector<float> proto;
  std::vector<MatrixView> tensors;

  for (const auto& output : outputs) {
    int md = 0, ph = 0, pw = 0;
    std::vector<float> p;
    if (asProto(output, md, ph, pw, p)) {
      mask_dim = md;
      proto_h = ph;
      proto_w = pw;
      proto = std::move(p);
    } else {
      MatrixView m = asMatrix(output);
      if (m.rows > 0 && m.cols > 0) {
        tensors.push_back(std::move(m));
      }
    }
  }

  MatrixView boxes;
  MatrixView scores;
  MatrixView coeffs;
  for (const auto& tensor : tensors) {
    if (tensor.cols == 4 && boxes.rows == 0) {
      boxes = tensor;
    } else if (tensor.cols == mask_dim && coeffs.rows == 0) {
      coeffs = tensor;
    } else if (scores.rows == 0) {
      scores = tensor;
    }
  }
  if (mask_dim <= 0 || proto_h <= 0 || proto_w <= 0 || boxes.rows == 0 || scores.rows == 0 ||
      coeffs.rows == 0 || boxes.rows != scores.rows || boxes.rows != coeffs.rows) {
    seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
    return false;
  }

  std::vector<int> keep;
  keep.reserve(boxes.rows);
  for (int i = 0; i < scores.rows; ++i) {
    const float* row = scores.data.data() + static_cast<size_t>(i) * scores.cols;
    float max_score = row[0];
    for (int c = 1; c < scores.cols; ++c) {
      max_score = std::max(max_score, row[c]);
    }
    if (max_score > 1.0f || max_score < 0.0f) {
      max_score = sigmoid(max_score);
    }
    if (max_score >= conf_thresh_) {
      keep.push_back(i);
    }
  }
  if (keep.empty()) {
    seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
    return true;
  }

  std::sort(keep.begin(), keep.end(), [&](int a, int b) {
    auto maxScore = [&](int i) {
      const float* row = scores.data.data() + static_cast<size_t>(i) * scores.cols;
      float s = row[0];
      for (int c = 1; c < scores.cols; ++c) {
        s = std::max(s, row[c]);
      }
      return s > 1.0f || s < 0.0f ? sigmoid(s) : s;
    };
    return maxScore(a) > maxScore(b);
  });
  if (max_detections_ > 0 && static_cast<int>(keep.size()) > max_detections_) {
    keep.resize(max_detections_);
  }

  cv::Mat low_mask(proto_h, proto_w, CV_8UC1, cv::Scalar(0));
  float scale_x = static_cast<float>(proto_w) / std::max(1, input_width_);
  float scale_y = static_cast<float>(proto_h) / std::max(1, input_height_);

  for (int idx : keep) {
    const float* box = boxes.data.data() + static_cast<size_t>(idx) * boxes.cols;
    float x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];
    if (!(x1 > x0 && y1 > y0)) {
      float cx = box[0], cy = box[1], bw = box[2], bh = box[3];
      x0 = cx - bw * 0.5f;
      y0 = cy - bh * 0.5f;
      x1 = cx + bw * 0.5f;
      y1 = cy + bh * 0.5f;
    }
    float coord_abs_max = std::max({std::abs(x0), std::abs(y0), std::abs(x1), std::abs(y1)});
    if (coord_abs_max <= 2.0f) {
      x0 *= input_width_;
      x1 *= input_width_;
      y0 *= input_height_;
      y1 *= input_height_;
    }
    x0 = std::clamp(x0, 0.0f, static_cast<float>(input_width_));
    y0 = std::clamp(y0, 0.0f, static_cast<float>(input_height_));
    x1 = std::clamp(x1, 0.0f, static_cast<float>(input_width_));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(input_height_));

    int px0 = std::clamp(static_cast<int>(std::floor(x0 * scale_x)), 0, proto_w - 1);
    int py0 = std::clamp(static_cast<int>(std::floor(y0 * scale_y)), 0, proto_h - 1);
    int px1 = std::clamp(static_cast<int>(std::ceil(x1 * scale_x)), px0 + 1, proto_w);
    int py1 = std::clamp(static_cast<int>(std::ceil(y1 * scale_y)), py0 + 1, proto_h);

    const float* coeff = coeffs.data.data() + static_cast<size_t>(idx) * coeffs.cols;
    for (int y = py0; y < py1; ++y) {
      for (int x = px0; x < px1; ++x) {
        float v = 0.0f;
        for (int c = 0; c < mask_dim; ++c) {
          v += coeff[c] * proto[(static_cast<size_t>(c) * proto_h + y) * proto_w + x];
        }
        if (sigmoid(v) > mask_thresh_) {
          low_mask.at<uint8_t>(y, x) = 1;
        }
      }
    }
  }

  cv::Mat model_mask;
  cv::resize(low_mask, model_mask, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_NEAREST);
  seg_map = restoreMask(model_mask, meta);
  return true;
}

bool YoloSeg::infer(const cv::Mat& frame_rgb, cv::Mat& seg_map, double* rknn_ms, double* post_ms) {
  if (models_.empty()) {
    return false;
  }
  PreprocessMeta meta;
  cv::Mat input = preprocess(frame_rgb, meta);
  std::vector<TensorData> outputs;
  if (!nextModel().infer(input, outputs, rknn_ms)) {
    return false;
  }
  auto t0 = std::chrono::steady_clock::now();
  bool ok = postprocess(outputs, meta, seg_map);
  auto t1 = std::chrono::steady_clock::now();
  if (post_ms != nullptr) {
    *post_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return ok;
}

}  // namespace track_perception_cpp
