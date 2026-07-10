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

struct MatrixView {
  int rows{0};
  int cols{0};
  std::vector<float> data;
};

struct SegCandidate {
  int index{0};
  int class_id{0};
  float score{0.0f};
  float x0{0.0f};
  float y0{0.0f};
  float x1{0.0f};
  float y1{0.0f};
  std::vector<float> coeff;
};

constexpr int kRegMax = 16;

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

bool decodeBox(const float* box, int input_width, int input_height, SegCandidate& candidate) {
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
    x0 *= input_width;
    x1 *= input_width;
    y0 *= input_height;
    y1 *= input_height;
  }
  candidate.x0 = std::clamp(x0, 0.0f, static_cast<float>(input_width));
  candidate.y0 = std::clamp(y0, 0.0f, static_cast<float>(input_height));
  candidate.x1 = std::clamp(x1, 0.0f, static_cast<float>(input_width));
  candidate.y1 = std::clamp(y1, 0.0f, static_cast<float>(input_height));
  return candidate.x1 > candidate.x0 && candidate.y1 > candidate.y0;
}

float boxIou(const SegCandidate& a, const SegCandidate& b) {
  float ix0 = std::max(a.x0, b.x0);
  float iy0 = std::max(a.y0, b.y0);
  float ix1 = std::min(a.x1, b.x1);
  float iy1 = std::min(a.y1, b.y1);
  float iw = std::max(0.0f, ix1 - ix0);
  float ih = std::max(0.0f, iy1 - iy0);
  float inter = iw * ih;
  float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
  float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
  float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

float boxContainment(const SegCandidate& a, const SegCandidate& b) {
  float ix0 = std::max(a.x0, b.x0);
  float iy0 = std::max(a.y0, b.y0);
  float ix1 = std::min(a.x1, b.x1);
  float iy1 = std::min(a.y1, b.y1);
  float iw = std::max(0.0f, ix1 - ix0);
  float ih = std::max(0.0f, iy1 - iy0);
  float inter = iw * ih;
  float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
  float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
  float smaller = std::min(area_a, area_b);
  return smaller > 0.0f ? inter / smaller : 0.0f;
}

}  // namespace

bool YoloSeg::init(const std::string& model_path, const std::vector<int>& core_ids,
                   int input_width, int input_height, float crop_y0, float crop_y1,
                   int pad_value, float conf_thresh, float nms_thresh, float mask_thresh,
                   float nms_contain_thresh, int max_detections, bool raw_output) {
  input_width_ = input_width;
  input_height_ = input_height;
  crop_y0_ratio_ = crop_y0;
  crop_y1_ratio_ = crop_y1;
  pad_value_ = pad_value;
  conf_thresh_ = conf_thresh;
  nms_thresh_ = nms_thresh;
  mask_thresh_ = mask_thresh;
  nms_contain_thresh_ = nms_contain_thresh;
  max_detections_ = max_detections;
  raw_output_ = raw_output;

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

cv::Mat YoloSeg::restoreProbToOriginal(const cv::Mat& model_prob, const PreprocessMeta& meta) const {
  cv::Mat resized_model;
  if (model_prob.cols != meta.input_w || model_prob.rows != meta.input_h) {
    cv::resize(model_prob, resized_model, cv::Size(meta.input_w, meta.input_h), 0, 0, cv::INTER_LINEAR);
  } else {
    resized_model = model_prob;
  }

  cv::Rect valid(meta.pad_x, meta.pad_y, meta.resize_w, meta.resize_h);
  cv::Mat unpadded = resized_model(valid);
  cv::Mat crop_prob;
  cv::resize(unpadded, crop_prob, cv::Size(meta.crop_w, meta.crop_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat full(meta.orig_h, meta.orig_w, CV_32FC1, cv::Scalar(0.0f));
  crop_prob.copyTo(full(cv::Rect(0, meta.crop_y0, meta.crop_w, meta.crop_h)));
  return full;
}

bool YoloSeg::postprocess(const std::vector<TensorData>& outputs, const PreprocessMeta& meta, cv::Mat& seg_map) {
  last_stats_ = YoloSegStats{};
  last_instances_.clear();
  int mask_dim = 0;
  int proto_h = 0;
  int proto_w = 0;
  std::vector<float> proto;
  std::vector<MatrixView> tensors;
  std::vector<const TensorData*> raw_heads;

  for (const auto& output : outputs) {
    if (raw_output_ && output.dims.size() == 4 && output.dims[0] == 1 && output.dims[1] > 64) {
      raw_heads.push_back(&output);
      continue;
    }
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
  std::vector<SegCandidate> candidates;
  if (mask_dim <= 0 || proto_h <= 0 || proto_w <= 0) {
    seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
    return false;
  }

  if (raw_output_) {
    for (const TensorData* head : raw_heads) {
      if (head == nullptr || head->dims.size() != 4 || head->dims[0] != 1) {
        continue;
      }
      int channels = head->dims[1];
      int h = head->dims[2];
      int w = head->dims[3];
      int class_count = channels - 4 * kRegMax - mask_dim;
      if (class_count <= 0 || h <= 0 || w <= 0) {
        continue;
      }
      float stride_x = static_cast<float>(input_width_) / static_cast<float>(w);
      float stride_y = static_cast<float>(input_height_) / static_cast<float>(h);
      size_t plane = static_cast<size_t>(h) * w;
      const float* data = head->data.data();
      candidates.reserve(candidates.size() + plane);

      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          size_t pos = static_cast<size_t>(y) * w + x;
          int class_id = 0;
          float max_score = sigmoid(data[(4 * kRegMax) * plane + pos]);
          for (int c = 1; c < class_count; ++c) {
            float s = sigmoid(data[(4 * kRegMax + c) * plane + pos]);
            if (s > max_score) {
              max_score = s;
              class_id = c;
            }
          }
          if (max_score < conf_thresh_) {
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
          SegCandidate candidate;
          candidate.index = -1;
          candidate.class_id = class_id;
          candidate.score = max_score;
          candidate.x0 = std::clamp(cx - dist[0] * stride_x, 0.0f, static_cast<float>(input_width_));
          candidate.y0 = std::clamp(cy - dist[1] * stride_y, 0.0f, static_cast<float>(input_height_));
          candidate.x1 = std::clamp(cx + dist[2] * stride_x, 0.0f, static_cast<float>(input_width_));
          candidate.y1 = std::clamp(cy + dist[3] * stride_y, 0.0f, static_cast<float>(input_height_));
          if (candidate.x1 <= candidate.x0 || candidate.y1 <= candidate.y0) {
            continue;
          }
          candidate.coeff.resize(mask_dim);
          int coeff_offset = 4 * kRegMax + class_count;
          for (int m = 0; m < mask_dim; ++m) {
            candidate.coeff[m] = data[(coeff_offset + m) * plane + pos];
          }
          candidates.push_back(std::move(candidate));
        }
      }
    }
  } else {
    for (const auto& tensor : tensors) {
      if (tensor.cols == 4 && boxes.rows == 0) {
        boxes = tensor;
      } else if (tensor.cols == mask_dim && coeffs.rows == 0) {
        coeffs = tensor;
      } else if (scores.rows == 0) {
        scores = tensor;
      }
    }
    if (boxes.rows == 0 || scores.rows == 0 || coeffs.rows == 0 || boxes.rows != scores.rows ||
        boxes.rows != coeffs.rows) {
      seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
      return false;
    }

    candidates.reserve(boxes.rows);
    for (int i = 0; i < scores.rows; ++i) {
      const float* row = scores.data.data() + static_cast<size_t>(i) * scores.cols;
      float max_score = row[0];
      int class_id = 0;
      for (int c = 1; c < scores.cols; ++c) {
        if (row[c] > max_score) {
          max_score = row[c];
          class_id = c;
        }
      }
      if (max_score > 1.0f || max_score < 0.0f) {
        max_score = sigmoid(max_score);
      }
      if (max_score >= conf_thresh_) {
        SegCandidate candidate;
        candidate.index = i;
        candidate.class_id = class_id;
        candidate.score = max_score;
        const float* box = boxes.data.data() + static_cast<size_t>(i) * boxes.cols;
        if (decodeBox(box, input_width_, input_height_, candidate)) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  if (candidates.empty()) {
    seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
    return true;
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });

  std::vector<SegCandidate> selected;
  selected.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    bool suppress = false;
    for (const auto& kept : selected) {
      bool same_class = candidate.class_id == kept.class_id;
      bool high_iou = boxIou(candidate, kept) > nms_thresh_;
      bool high_containment = nms_contain_thresh_ > 0.0f &&
                              boxContainment(candidate, kept) > nms_contain_thresh_;
      if (same_class && (high_iou || high_containment)) {
        suppress = true;
        break;
      }
    }
    if (suppress) {
      continue;
    }
    selected.push_back(candidate);
    if (max_detections_ > 0 && static_cast<int>(selected.size()) >= max_detections_) {
      break;
    }
  }

  if (selected.empty()) {
    seg_map = cv::Mat(meta.orig_h, meta.orig_w, CV_8UC1, cv::Scalar(0));
    return true;
  }

  float sum_score = 0.0f;
  float min_score = selected.front().score;
  float max_score = selected.front().score;
  for (const auto& candidate : selected) {
    sum_score += candidate.score;
    min_score = std::min(min_score, candidate.score);
    max_score = std::max(max_score, candidate.score);
  }
  last_stats_.kept_instances = static_cast<int>(selected.size());
  last_stats_.score_mean = sum_score / static_cast<float>(selected.size());
  last_stats_.score_min = min_score;
  last_stats_.score_max = max_score;
  last_instances_.reserve(selected.size());
  for (const auto& candidate : selected) {
    float x0 = (candidate.x0 - static_cast<float>(meta.pad_x)) /
               std::max(static_cast<float>(meta.resize_w), 1.0f) * meta.crop_w;
    float x1 = (candidate.x1 - static_cast<float>(meta.pad_x)) /
               std::max(static_cast<float>(meta.resize_w), 1.0f) * meta.crop_w;
    float y0 = (candidate.y0 - static_cast<float>(meta.pad_y)) /
                   std::max(static_cast<float>(meta.resize_h), 1.0f) * meta.crop_h +
               meta.crop_y0;
    float y1 = (candidate.y1 - static_cast<float>(meta.pad_y)) /
                   std::max(static_cast<float>(meta.resize_h), 1.0f) * meta.crop_h +
               meta.crop_y0;
    x0 = std::clamp(x0, 0.0f, static_cast<float>(meta.orig_w - 1));
    x1 = std::clamp(x1, 0.0f, static_cast<float>(meta.orig_w));
    y0 = std::clamp(y0, 0.0f, static_cast<float>(meta.orig_h - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(meta.orig_h));
    if (x1 > x0 && y1 > y0) {
      YoloSegInstance instance;
      instance.class_id = candidate.class_id;
      instance.score = candidate.score;
      instance.bbox = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
      last_instances_.push_back(instance);
    }
  }

  cv::Mat full_prob(meta.orig_h, meta.orig_w, CV_32FC1, cv::Scalar(0.0f));
  float scale_x = static_cast<float>(proto_w) / std::max(1, input_width_);
  float scale_y = static_cast<float>(proto_h) / std::max(1, input_height_);
  for (const auto& candidate : selected) {
    cv::Mat low_prob(proto_h, proto_w, CV_32FC1, cv::Scalar(0.0f));
    int px0 = std::clamp(static_cast<int>(std::floor(candidate.x0 * scale_x)), 0, proto_w);
    int py0 = std::clamp(static_cast<int>(std::floor(candidate.y0 * scale_y)), 0, proto_h);
    int px1 = std::clamp(static_cast<int>(std::ceil(candidate.x1 * scale_x)), 0, proto_w);
    int py1 = std::clamp(static_cast<int>(std::ceil(candidate.y1 * scale_y)), 0, proto_h);

    const float* coeff = !candidate.coeff.empty()
                             ? candidate.coeff.data()
                             : coeffs.data.data() + static_cast<size_t>(candidate.index) * coeffs.cols;
    for (int y = py0; y < py1; ++y) {
      for (int x = px0; x < px1; ++x) {
        float v = 0.0f;
        for (int c = 0; c < mask_dim; ++c) {
          v += coeff[c] * proto[(static_cast<size_t>(c) * proto_h + y) * proto_w + x];
        }
        low_prob.at<float>(y, x) = sigmoid(v);
      }
    }

    cv::Mat instance_full = restoreProbToOriginal(low_prob, meta);
    cv::max(full_prob, instance_full, full_prob);
  }

  cv::Mat binary;
  cv::threshold(full_prob, binary, mask_thresh_, 1.0, cv::THRESH_BINARY);
  binary.convertTo(seg_map, CV_8UC1);
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
