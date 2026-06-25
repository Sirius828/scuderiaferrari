#include "track_perception_cpp/lane_decision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

namespace track_perception_cpp {

namespace {

template <typename T>
T clampValue(T v, T lo, T hi) {
  return std::max(lo, std::min(hi, v));
}

double evalPoly(const std::vector<double>& coeffs, double x) {
  double y = 0.0;
  for (double c : coeffs) {
    y = y * x + c;
  }
  return y;
}

std::vector<double> polyDeriv(const std::vector<double>& coeffs, int order = 1) {
  std::vector<double> out = coeffs;
  for (int k = 0; k < order; ++k) {
    if (out.size() <= 1) {
      return {0.0};
    }
    std::vector<double> d;
    int degree = static_cast<int>(out.size()) - 1;
    d.reserve(out.size() - 1);
    for (size_t i = 0; i + 1 < out.size(); ++i) {
      d.push_back(out[i] * (degree - static_cast<int>(i)));
    }
    out = std::move(d);
  }
  return out;
}

bool solveLinearSystem(std::vector<std::vector<double>> a, std::vector<double> b,
                       std::vector<double>* x) {
  int n = static_cast<int>(b.size());
  for (int i = 0; i < n; ++i) {
    int pivot = i;
    for (int r = i + 1; r < n; ++r) {
      if (std::abs(a[r][i]) > std::abs(a[pivot][i])) {
        pivot = r;
      }
    }
    if (std::abs(a[pivot][i]) < 1e-9) {
      return false;
    }
    if (pivot != i) {
      std::swap(a[pivot], a[i]);
      std::swap(b[pivot], b[i]);
    }

    double div = a[i][i];
    for (int c = i; c < n; ++c) {
      a[i][c] /= div;
    }
    b[i] /= div;

    for (int r = 0; r < n; ++r) {
      if (r == i) {
        continue;
      }
      double factor = a[r][i];
      for (int c = i; c < n; ++c) {
        a[r][c] -= factor * a[i][c];
      }
      b[r] -= factor * b[i];
    }
  }
  *x = std::move(b);
  return true;
}

bool weightedPolyfit(const std::vector<cv::Point3f>& points, int order,
                     std::vector<double>* coeffs) {
  int n = static_cast<int>(points.size());
  if (n <= order) {
    return false;
  }
  int terms = order + 1;
  std::vector<std::vector<double>> ata(terms, std::vector<double>(terms, 0.0));
  std::vector<double> atb(terms, 0.0);

  for (const auto& p : points) {
    double y = p.y;
    double x = p.x;
    double w = p.z <= 0.0f ? 1.0 : static_cast<double>(p.z);
    std::vector<double> basis(terms, 1.0);
    for (int i = terms - 2; i >= 0; --i) {
      basis[i] = basis[i + 1] * y;
    }
    for (int r = 0; r < terms; ++r) {
      atb[r] += w * basis[r] * x;
      for (int c = 0; c < terms; ++c) {
        ata[r][c] += w * basis[r] * basis[c];
      }
    }
  }
  return solveLinearSystem(std::move(ata), std::move(atb), coeffs);
}

}  // namespace

double nowSeconds() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string laneStateToJson(const LaneState& state) {
  std::ostringstream ss;
  ss << "{"
     << "\"control_offset\":" << state.control_offset << ","
     << "\"lateral_offset\":" << state.lateral_offset << ","
     << "\"heading_error\":" << state.heading_error << ","
     << "\"curvature\":" << state.curvature << ","
     << "\"confidence\":" << state.confidence << ","
     << "\"is_valid\":" << (state.is_valid ? "true" : "false") << ","
     << "\"road_state\":\"" << state.road_state << "\","
     << "\"branch_side\":\"" << state.branch_side << "\","
     << "\"task_state\":\"" << state.task_state << "\","
     << "\"task_bias\":" << state.task_bias << ","
     << "\"timestamp\":" << state.timestamp
     << "}";
  return ss.str();
}

void LaneDecision::configure(const LaneDecisionConfig& config) {
  cfg_ = config;
  cfg_.band_count = std::max(1, cfg_.band_count);
  cfg_.fit_order = clampValue(cfg_.fit_order, 1, 2);
  cfg_.branch_fit_order = clampValue(cfg_.branch_fit_order, 1, 2);
  locked_branch_side_ = cfg_.outer_side;
  band_lane_widths_.assign(static_cast<size_t>(cfg_.band_count), std::nullopt);
}

LaneState LaneDecision::decide(const cv::Mat& seg_map_in, const std::vector<Detection>& detections) {
  LaneState state;
  state.timestamp = nowSeconds();
  state.branch_side = branch_locked_ ? locked_branch_side_ : "";
  debug_info_ = LaneDebugInfo{};

  if (seg_map_in.empty()) {
    return state;
  }

  cv::Mat seg_map = seg_map_in.channels() == 1 ? seg_map_in : seg_map_in.reshape(1);
  int h = seg_map.rows;
  int w = seg_map.cols;
  double current_time = nowSeconds();
  double center_offset = 0.0;
  double lateral_offset = 0.0;
  double heading_error = 0.0;
  double curvature = 0.0;
  double confidence = 0.0;
  std::string road_state = "LOW_CONFIDENCE";
  bool is_valid = false;
  std::vector<Band> bands;
  std::vector<cv::Point3f> raw_points;
  std::vector<cv::Point3f> fit_points;
  std::vector<double> fit_coeffs;

  if (cfg_.enable_segment_branch_logic) {
    bands = buildBands(seg_map, detections);
    auto [branch_detected, branch_score] = detectBranchFromBands(bands);
    int guideboard_count = 0;
    int guideboard_roi_count = 0;
    float guideboard_best_confidence = 0.0f;
    cv::Point2f guideboard_best_center;
    int guideboard_y0 = static_cast<int>(h * cfg_.guideboard_detect_y0_ratio);
    int guideboard_y1 = static_cast<int>(h * cfg_.guideboard_detect_y1_ratio);
    for (const auto& det : detections) {
      if (det.class_name != "GuideBoard") {
        continue;
      }
      ++guideboard_count;
      if (det.confidence > guideboard_best_confidence) {
        guideboard_best_confidence = det.confidence;
        guideboard_best_center = det.center;
      }
      if (det.center.y >= guideboard_y0 && det.center.y <= guideboard_y1) {
        ++guideboard_roi_count;
      }
    }
    bool guideboard_seen = cfg_.enable_guideboard_branch_selection && guideboard_roi_count > 0;
    debug_info_.branch_detected = branch_detected;
    debug_info_.branch_score = branch_score;
    debug_info_.guideboard_seen = guideboard_seen;
    debug_info_.guideboard_count = guideboard_count;
    debug_info_.guideboard_roi_count = guideboard_roi_count;
    debug_info_.guideboard_best_confidence = guideboard_best_confidence;
    debug_info_.guideboard_best_center = guideboard_best_center;

    if (!branch_locked_) {
      if (branch_detected) {
        double last_center_x = last_offset_ * w / 2.0 + w / 2.0;
        std::string target_branch = cfg_.outer_side;
        if (guideboard_seen) {
          target_branch = cfg_.guideboard_branch;
        } else if (cfg_.enable_continuity_branch_selection) {
          auto continuity_branch = chooseBranchSideByContinuity(bands, w, last_center_x);
          if (continuity_branch) {
            target_branch = *continuity_branch;
          }
        }
        if (target_branch != "left" && target_branch != "right") {
          target_branch = cfg_.outer_side;
        }

        branch_locked_ = true;
        locked_branch_side_ = target_branch;
        lock_start_time_ = current_time;
        exit_confirm_count_ = 0;
        merge_wide_locked_ = false;
        merge_wide_side_.clear();
        merge_wide_confirm_count_ = 0;
        merge_wide_release_count_ = 0;
        (void)branch_score;
      }
    } else {
      int far_bands_count = std::max(1, static_cast<int>(bands.size() * cfg_.branch_detect_far_band_ratio));
      int single_path_count = 0;
      for (int i = 0; i < std::min(far_bands_count, static_cast<int>(bands.size())); ++i) {
        if (bands[i].segments.size() <= 1) {
          ++single_path_count;
        }
      }
      double lock_duration = current_time - lock_start_time_;
      int single_path_required = std::max(
          cfg_.branch_detect_min_bands,
          static_cast<int>(std::ceil(far_bands_count * cfg_.exit_single_path_min_ratio)));
      if (lock_duration >= cfg_.min_branch_lock_time && single_path_count >= single_path_required) {
        ++exit_confirm_count_;
      } else {
        exit_confirm_count_ = 0;
      }
      if ((lock_duration >= cfg_.min_branch_lock_time &&
           exit_confirm_count_ >= cfg_.exit_single_path_confirm_frames) ||
          lock_duration > cfg_.branch_lock_time) {
        branch_locked_ = false;
        locked_branch_side_ = cfg_.outer_side;
        exit_confirm_count_ = 0;
      }
    }

    std::string target_side = branch_locked_ ? locked_branch_side_ : cfg_.outer_side;
    double last_center_x = last_offset_ * w / 2.0 + w / 2.0;
    if (!branch_locked_) {
      updateMergeWideState(bands, last_center_x);
    }
    road_state = currentRoadState();
    raw_points = collectCenterlinePoints(bands, branch_locked_, target_side, last_center_x, w, current_time);
    int fit_order = branch_locked_ ? cfg_.branch_fit_order : cfg_.fit_order;
    fit_points = filterCenterlinePoints(raw_points, w, last_center_x);
    if (branch_locked_ && cfg_.enable_branch_bottom_anchor && fit_order >= 2) {
      fit_points.emplace_back(w * cfg_.branch_bottom_anchor_x_ratio,
                              h * cfg_.branch_bottom_anchor_y_ratio,
                              cfg_.branch_bottom_anchor_weight);
    }

    double raw_offset = 0.0;
    if (fitCenterlineAndComputeOffset(fit_points, h, w, fit_order, &raw_offset, &fit_coeffs,
                                      &lateral_offset, &heading_error, &curvature)) {
      center_offset = smoothOffset(raw_offset);
      is_valid = true;
      confidence = calculateLaneConfidence(fit_points, bands);
    } else {
      cv::Mat bottom_seg = seg_map(cv::Range(static_cast<int>(h * 0.8), h), cv::Range::all());
      center_offset = fallbackCenterOffset(bottom_seg);
      lateral_offset = center_offset;
      is_valid = cv::countNonZero(bottom_seg == 1) > 0;
      confidence = is_valid ? 0.2 : 0.0;
      if (!is_valid && std::abs(last_offset_) > 0.01) {
        center_offset = last_offset_;
      }
      fit_points.clear();
      fit_coeffs.clear();
    }
    populateDebugInfo(bands, getActiveObstacleZones(detections, w, h), fit_points, fit_coeffs);
  } else {
    cv::Mat bottom_seg = seg_map(cv::Range(h / 2, h), cv::Range::all());
    center_offset = fallbackCenterOffset(bottom_seg);
    lateral_offset = center_offset;
    is_valid = cv::countNonZero(bottom_seg == 1) > 0;
    confidence = is_valid ? 0.2 : 0.0;
    road_state = is_valid ? "NORMAL" : "LOW_CONFIDENCE";
  }

  updateTrafficLightStopState(detections, h);
  updateFinishStopState(detections, h);

  state.control_offset = static_cast<float>(center_offset);
  state.lateral_offset = static_cast<float>(lateral_offset);
  state.heading_error = static_cast<float>(heading_error);
  state.curvature = static_cast<float>(curvature);
  state.confidence = static_cast<float>(clampValue(confidence, 0.0, 1.0));
  state.is_valid = is_valid;
  state.road_state = is_valid ? road_state : "LOW_CONFIDENCE";
    state.branch_side = branch_locked_ ? locked_branch_side_ : "";
  state.task_state = taskState();
  state.task_bias = 0.0f;
  state.timestamp = current_time;
  return state;
}

std::vector<LaneDecision::Band> LaneDecision::buildBands(
    const cv::Mat& road_mask, const std::vector<Detection>& detections) {
  int h = road_mask.rows;
  int w = road_mask.cols;
  auto zones = getActiveObstacleZones(detections, w, h);
  int y_min = clampValue(static_cast<int>(h * cfg_.band_y_min_ratio), 0, std::max(0, h - 1));
  int y_max = clampValue(static_cast<int>(h * cfg_.band_y_max_ratio), y_min + 1, h);
  int band_height = std::max(1, static_cast<int>(h * cfg_.band_height_ratio));
  int step = std::max(1, (y_max - y_min) / cfg_.band_count);

  std::vector<Band> bands;
  for (int i = 0; i < cfg_.band_count; ++i) {
    int y0 = y_min + i * step;
    int y1 = std::min(y0 + band_height, y_max);
    if (y1 <= y0 || y0 >= h) {
      continue;
    }
    Band band;
    band.index = i;
    band.y0 = y0;
    band.y1 = y1;
    band.y_center = (y0 + y1) / 2.0;
    cv::Mat band_mask = road_mask(cv::Range(y0, y1), cv::Range::all());
    band.segments = applyObstacleExclusionToSegments(extractSegmentsInBand(band_mask), y0, y1, zones);
    bands.push_back(std::move(band));
  }
  return bands;
}

std::vector<LaneDecision::ObstacleZone> LaneDecision::getActiveObstacleZones(
    const std::vector<Detection>& detections, int image_width, int image_height) const {
  if (!cfg_.enable_obstacle_avoidance) {
    return {};
  }
  std::vector<ObstacleZone> zones;
  double min_bottom_y = image_height * clampValue(cfg_.obstacle_min_bottom_y_ratio, 0.0f, 1.0f);
  for (const auto& det : detections) {
    if (!cfg_.obstacle_labels.count(det.class_name) || det.confidence < cfg_.obstacle_min_confidence) {
      continue;
    }
    if (det.bbox.y + det.bbox.height < min_bottom_y) {
      continue;
    }
    float x0 = std::max(0.0f, det.bbox.x - cfg_.obstacle_x_margin_px);
    float y0 = std::max(0.0f, det.bbox.y - cfg_.obstacle_y_margin_px);
    float x1 = std::min(static_cast<float>(image_width - 1),
                        det.bbox.x + det.bbox.width + cfg_.obstacle_x_margin_px);
    float y1 = std::min(static_cast<float>(image_height - 1),
                        det.bbox.y + det.bbox.height + cfg_.obstacle_y_margin_px);
    if (x1 > x0 && y1 > y0) {
      zones.push_back({cv::Rect2f(x0, y0, x1 - x0, y1 - y0), det.class_name});
    }
  }
  return zones;
}

std::vector<LaneDecision::Segment> LaneDecision::extractSegmentsInBand(const cv::Mat& band_mask) const {
  int w = band_mask.cols;
  std::vector<int> col_counts(w, 0);
  for (int y = 0; y < band_mask.rows; ++y) {
    const uint8_t* row = band_mask.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      if (row[x] == 1) {
        ++col_counts[x];
      }
    }
  }

  std::vector<Segment> segments;
  int start = -1;
  int pixel_count = 0;
  for (int x = 0; x <= w; ++x) {
    bool active = x < w && col_counts[x] > 0;
    if (active) {
      if (start < 0) {
        start = x;
        pixel_count = 0;
      }
      pixel_count += col_counts[x];
    } else if (start >= 0) {
      int end = x;
      int width = end - start;
      if (width >= cfg_.min_segment_width_px && pixel_count >= cfg_.min_pixels_per_band) {
        Segment seg;
        seg.x0 = start;
        seg.x1 = end - 1;
        seg.width = width;
        seg.center_x = (start + end - 1) / 2.0;
        seg.pixel_count = pixel_count;
        segments.push_back(seg);
      }
      start = -1;
    }
  }
  return segments;
}

std::vector<LaneDecision::Segment> LaneDecision::applyObstacleExclusionToSegments(
    const std::vector<Segment>& segments, int band_y0, int band_y1,
    const std::vector<ObstacleZone>& zones) const {
  if (segments.empty() || zones.empty()) {
    return segments;
  }
  std::vector<Segment> split;
  for (const auto& seg : segments) {
    std::vector<std::pair<double, double>> intervals{{static_cast<double>(seg.x0), static_cast<double>(seg.x1)}};
    for (const auto& zone : zones) {
      if (zone.rect.y + zone.rect.height < band_y0 || zone.rect.y > band_y1) {
        continue;
      }
      std::vector<std::pair<double, double>> next;
      for (auto [x0, x1] : intervals) {
        double cut_x0 = std::max(x0, static_cast<double>(zone.rect.x));
        double cut_x1 = std::min(x1, static_cast<double>(zone.rect.x + zone.rect.width));
        if (cut_x1 < x0 || cut_x0 > x1) {
          next.emplace_back(x0, x1);
          continue;
        }
        if (cut_x0 - x0 >= cfg_.min_segment_width_px) {
          next.emplace_back(x0, cut_x0);
        }
        if (x1 - cut_x1 >= cfg_.min_segment_width_px) {
          next.emplace_back(cut_x1, x1);
        }
      }
      intervals = std::move(next);
      if (intervals.empty()) {
        break;
      }
    }
    for (auto [x0, x1] : intervals) {
      double width = x1 - x0;
      if (width < cfg_.min_segment_width_px) {
        continue;
      }
      Segment out;
      out.x0 = static_cast<int>(std::round(x0));
      out.x1 = static_cast<int>(std::round(x1));
      out.width = width;
      out.center_x = (x0 + x1) / 2.0;
      out.pixel_count = std::max(cfg_.min_pixels_per_band,
                                 static_cast<int>(seg.pixel_count * width / std::max(1.0, seg.width)));
      out.obstacle_cut = true;
      split.push_back(out);
    }
  }
  return split;
}

std::pair<bool, int> LaneDecision::detectBranchFromBands(const std::vector<Band>& bands) const {
  int far_count = static_cast<int>(bands.size() * cfg_.branch_detect_far_band_ratio);
  if (far_count <= 0) {
    far_count = static_cast<int>(bands.size());
  }
  int branch_bands = 0;
  for (int i = 0; i < std::min(far_count, static_cast<int>(bands.size())); ++i) {
    int valid = 0;
    const auto& segs = bands[i].segments;
    for (size_t a = 0; a < segs.size(); ++a) {
      bool isolated = true;
      for (size_t b = 0; b < segs.size(); ++b) {
        if (a == b) {
          continue;
        }
        double gap = std::abs(segs[a].center_x - segs[b].center_x) -
                     (segs[a].width / 2.0 + segs[b].width / 2.0);
        if (gap < cfg_.min_segment_gap_px) {
          isolated = false;
          break;
        }
      }
      if (isolated) {
        ++valid;
      }
    }
    if (valid >= 2) {
      ++branch_bands;
    }
  }
  return {branch_bands >= cfg_.branch_detect_min_bands, branch_bands};
}

std::optional<std::string> LaneDecision::chooseBranchSideByContinuity(
    const std::vector<Band>& bands, int image_width, double last_center_x) const {
  if (bands.empty()) {
    return std::nullopt;
  }
  double near_ratio = clampValue(cfg_.branch_continuity_near_band_ratio, 0.1f, 1.0f);
  int near_count = std::max(1, static_cast<int>(std::ceil(bands.size() * near_ratio)));
  double max_dx = std::max(1.0, static_cast<double>(cfg_.branch_continuity_max_dx_ratio) * image_width);
  const Segment* best_seg = nullptr;
  const Band* best_band = nullptr;
  double best_dist = 0.0;

  for (int i = static_cast<int>(bands.size()) - 1; i >= std::max(0, static_cast<int>(bands.size()) - near_count); --i) {
    if (bands[i].segments.size() < 2) {
      continue;
    }
    auto it = std::min_element(bands[i].segments.begin(), bands[i].segments.end(),
                               [&](const Segment& a, const Segment& b) {
                                 return std::abs(a.center_x - last_center_x) < std::abs(b.center_x - last_center_x);
                               });
    double dist = std::abs(it->center_x - last_center_x);
    if (dist > max_dx) {
      continue;
    }
    if (!best_seg || dist < best_dist) {
      best_seg = &(*it);
      best_band = &bands[i];
      best_dist = dist;
    }
  }
  if (!best_seg || !best_band) {
    return std::nullopt;
  }
  auto sorted = best_band->segments;
  std::sort(sorted.begin(), sorted.end(), [](const Segment& a, const Segment& b) {
    return a.center_x < b.center_x;
  });
  int best_index = 0;
  for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
    if (std::abs(sorted[i].center_x - best_seg->center_x) < 1e-3) {
      best_index = i;
      break;
    }
  }
  return best_index < static_cast<int>(sorted.size()) / 2.0 ? "left" : "right";
}

std::optional<LaneDecision::Segment> LaneDecision::chooseTargetSegment(
    const Band& band, const std::string& side) const {
  if (band.segments.empty()) {
    return std::nullopt;
  }
  if (band.segments.size() == 1) {
    Segment seg = band.segments.front();
    double width = seg.x1 - seg.x0;
    double target_x = side == "left" ? seg.x0 + width * 0.20 : seg.x0 + width * 0.80;
    Segment out = seg;
    out.center_x = target_x;
    out.width = std::max(1.0, width * 0.4);
    out.x0 = side == "left" ? seg.x0 : static_cast<int>(target_x);
    out.x1 = side == "left" ? static_cast<int>(target_x) : seg.x1;
    out.pixel_count = std::max(1, seg.pixel_count / 2);
    out.virtual_segment = true;
    return out;
  }
  return side == "left"
             ? *std::min_element(band.segments.begin(), band.segments.end(),
                                 [](const Segment& a, const Segment& b) { return a.center_x < b.center_x; })
             : *std::max_element(band.segments.begin(), band.segments.end(),
                                 [](const Segment& a, const Segment& b) { return a.center_x < b.center_x; });
}

std::optional<LaneDecision::Segment> LaneDecision::chooseLockedSegmentByContinuity(
    const Band& band, int image_width, double last_center_x) const {
  if (band.segments.empty()) {
    return std::nullopt;
  }
  double max_dx = std::max(1.0, static_cast<double>(cfg_.locked_path_continuity_max_dx_ratio) * image_width);
  if (band.segments.size() == 1) {
    Segment seg = band.segments.front();
    if (last_center_x < seg.x0 - max_dx || last_center_x > seg.x1 + max_dx) {
      return std::nullopt;
    }
    double target_x = clampValue(last_center_x, static_cast<double>(seg.x0), static_cast<double>(seg.x1));
    Segment out = seg;
    out.center_x = target_x;
    out.x0 = static_cast<int>(std::max<double>(seg.x0, target_x - seg.width * 0.2));
    out.x1 = static_cast<int>(std::min<double>(seg.x1, target_x + seg.width * 0.2));
    out.width = std::max(1.0, seg.width * 0.4);
    out.virtual_segment = true;
    return out;
  }
  auto it = std::min_element(band.segments.begin(), band.segments.end(),
                             [&](const Segment& a, const Segment& b) {
                               return std::abs(a.center_x - last_center_x) < std::abs(b.center_x - last_center_x);
                             });
  if (std::abs(it->center_x - last_center_x) > max_dx) {
    return std::nullopt;
  }
  return *it;
}

bool LaneDecision::shouldUseLockedPathContinuity(double now) const {
  return cfg_.enable_locked_path_continuity && branch_locked_ &&
         (now - lock_start_time_) >= cfg_.locked_path_continuity_after_time;
}

std::string LaneDecision::currentRoadState() const {
  if (branch_locked_) {
    return "BRANCH_LOCKED";
  }
  if (merge_wide_locked_) {
    return "MERGE_WIDE";
  }
  return "NORMAL";
}

double LaneDecision::calculateLaneConfidence(const std::vector<cv::Point3f>& fit_points,
                                             const std::vector<Band>& bands) const {
  int valid_band_count = 0;
  for (const auto& band : bands) {
    if (!band.segments.empty()) {
      ++valid_band_count;
    }
  }
  if (valid_band_count <= 0) {
    return 0.0;
  }
  return clampValue(static_cast<double>(fit_points.size()) / valid_band_count, 0.0, 1.0);
}

std::optional<double> LaneDecision::getBandLaneWidth(int band_index) const {
  if (band_index < 0 || band_index >= static_cast<int>(band_lane_widths_.size())) {
    return std::nullopt;
  }
  return band_lane_widths_[band_index];
}

void LaneDecision::updateBandLaneWidth(int band_index, double width) {
  if (band_index < 0 || width <= 0.0) {
    return;
  }
  if (band_index >= static_cast<int>(band_lane_widths_.size())) {
    band_lane_widths_.resize(static_cast<size_t>(band_index + 1));
  }
  if (!band_lane_widths_[band_index]) {
    band_lane_widths_[band_index] = width;
    return;
  }
  double alpha = clampValue(cfg_.merge_wide_lane_width_alpha, 0.0f, 1.0f);
  band_lane_widths_[band_index] = (1.0 - alpha) * *band_lane_widths_[band_index] + alpha * width;
}

bool LaneDecision::isMergeWideSegment(const Band& band) const {
  if (!cfg_.enable_merge_wide_segment_logic || band.segments.size() != 1) {
    return false;
  }
  auto lane_width = getBandLaneWidth(band.index);
  if (!lane_width || *lane_width <= 0.0) {
    return false;
  }
  return band.segments.front().width >= *lane_width * std::max(1.05f, cfg_.merge_wide_segment_ratio);
}

void LaneDecision::updateMergeWideState(const std::vector<Band>& bands, double last_center_x) {
  if (!cfg_.enable_merge_wide_segment_logic) {
    merge_wide_locked_ = false;
    merge_wide_side_.clear();
    return;
  }
  std::vector<const Band*> wide_bands;
  for (const auto& band : bands) {
    if (band.segments.size() != 1) {
      continue;
    }
    double width = band.segments.front().width;
    auto lane_width = getBandLaneWidth(band.index);
    if (!lane_width) {
      updateBandLaneWidth(band.index, width);
      continue;
    }
    if (isMergeWideSegment(band)) {
      wide_bands.push_back(&band);
    } else if (!merge_wide_locked_) {
      updateBandLaneWidth(band.index, width);
    }
  }
  if (static_cast<int>(wide_bands.size()) >= std::max(1, cfg_.merge_wide_min_bands)) {
    ++merge_wide_confirm_count_;
    merge_wide_release_count_ = 0;
    if (merge_wide_confirm_count_ >= std::max(1, cfg_.merge_wide_confirm_frames)) {
      if (!merge_wide_locked_) {
        const auto& ref = wide_bands.back()->segments.front();
        merge_wide_side_ = last_center_x <= ref.center_x ? "left" : "right";
      }
      merge_wide_locked_ = true;
    }
    return;
  }
  merge_wide_confirm_count_ = 0;
  if (merge_wide_locked_) {
    ++merge_wide_release_count_;
    if (merge_wide_release_count_ >= std::max(1, cfg_.merge_wide_release_frames)) {
      merge_wide_locked_ = false;
      merge_wide_side_.clear();
      merge_wide_release_count_ = 0;
    }
  }
}

std::optional<LaneDecision::Segment> LaneDecision::chooseMergeWideSegment(
    const Band& band, std::optional<double> last_center_x) {
  if (!merge_wide_locked_ || !isMergeWideSegment(band)) {
    return std::nullopt;
  }
  auto lane_width = getBandLaneWidth(band.index);
  if (!lane_width || *lane_width <= 0.0) {
    return std::nullopt;
  }
  const auto& seg = band.segments.front();
  std::string side = merge_wide_side_;
  if (side != "left" && side != "right") {
    side = last_center_x ? (*last_center_x <= seg.center_x ? "left" : "right") : cfg_.outer_side;
    merge_wide_side_ = side;
  }
  double x0 = seg.x0;
  double x1 = seg.x1;
  double target_x = seg.center_x;
  if (side == "left") {
    x1 = std::min<double>(seg.x1, x0 + *lane_width);
    target_x = x0 + *lane_width * 0.5;
  } else {
    x0 = std::max<double>(seg.x0, x1 - *lane_width);
    target_x = x1 - *lane_width * 0.5;
  }
  Segment out = seg;
  out.x0 = static_cast<int>(std::round(x0));
  out.x1 = static_cast<int>(std::round(x1));
  out.width = std::max(1.0, x1 - x0);
  out.center_x = clampValue(target_x, static_cast<double>(seg.x0), static_cast<double>(seg.x1));
  out.virtual_segment = true;
  return out;
}

std::vector<cv::Point3f> LaneDecision::collectCenterlinePoints(
    std::vector<Band>& bands, bool branch_locked, const std::string& side,
    std::optional<double> last_center_x, int image_width, double now) {
  std::vector<cv::Point3f> points;
  bool use_locked_continuity = branch_locked && last_center_x &&
                               shouldUseLockedPathContinuity(now);
  for (auto& band : bands) {
    if (band.segments.empty()) {
      continue;
    }
    std::optional<Segment> target;
    if (branch_locked) {
      if (use_locked_continuity) {
        target = chooseLockedSegmentByContinuity(band, image_width, *last_center_x);
      }
      if (!target) {
        target = chooseTargetSegment(band, side);
      }
    } else {
      target = chooseMergeWideSegment(band, last_center_x);
      if (!target) {
        if (band.segments.size() == 1) {
          target = band.segments.front();
        } else if (last_center_x) {
          target = *std::min_element(band.segments.begin(), band.segments.end(),
                                     [&](const Segment& a, const Segment& b) {
                                       return std::abs(a.center_x - *last_center_x) <
                                              std::abs(b.center_x - *last_center_x);
                                     });
        } else {
          target = band.segments.front();
        }
      }
    }
    if (target) {
      band.selected_segment = *target;
      points.emplace_back(static_cast<float>(target->center_x), static_cast<float>(band.y_center), 1.0f);
    }
  }
  return points;
}

std::vector<cv::Point3f> LaneDecision::filterCenterlinePoints(
    const std::vector<cv::Point3f>& points, int image_width,
    std::optional<double> last_center_x) const {
  if (!cfg_.enable_fit_point_jump_filter || points.size() < 3) {
    return points;
  }
  double ratio_limit = std::max(0.0f, cfg_.max_fit_point_dx_ratio) * image_width;
  double px_limit = cfg_.max_fit_point_dx_px > 0 ? cfg_.max_fit_point_dx_px : ratio_limit;
  double max_dx = std::max(1.0, std::min(ratio_limit, px_limit));
  std::vector<bool> keep(points.size(), true);

  for (size_t i = 1; i + 1 < points.size(); ++i) {
    double prev_x = points[i - 1].x;
    double cur_x = points[i].x;
    double next_x = points[i + 1].x;
    if (std::abs(cur_x - prev_x) > max_dx &&
        std::abs(cur_x - next_x) > max_dx &&
        std::abs(next_x - prev_x) <= max_dx) {
      keep[i] = false;
    }
  }
  if (points.size() >= 3) {
    if (std::abs(points[0].x - points[1].x) > max_dx &&
        std::abs(points[1].x - points[2].x) <= max_dx) {
      keep[0] = false;
    }
    size_t n = points.size();
    if (std::abs(points[n - 1].x - points[n - 2].x) > max_dx &&
        std::abs(points[n - 2].x - points[n - 3].x) <= max_dx) {
      keep[n - 1] = false;
    }
  }
  std::vector<cv::Point3f> filtered;
  for (size_t i = 0; i < points.size(); ++i) {
    if (keep[i]) {
      filtered.push_back(points[i]);
    }
  }
  if (filtered.size() < 2) {
    return filtered;
  }

  std::vector<std::vector<cv::Point3f>> chains;
  std::vector<cv::Point3f> chain{filtered.front()};
  for (size_t i = 1; i < filtered.size(); ++i) {
    if (std::abs(filtered[i].x - chain.back().x) <= max_dx) {
      chain.push_back(filtered[i]);
    } else {
      chains.push_back(chain);
      chain = {filtered[i]};
    }
  }
  chains.push_back(chain);
  if (chains.size() == 1) {
    return filtered;
  }
  auto score = [&](const std::vector<cv::Point3f>& c) {
    double bottom = 0.0;
    double continuity = 0.0;
    for (const auto& p : c) {
      bottom = std::max(bottom, static_cast<double>(p.y));
      if (last_center_x) {
        continuity = std::min(continuity, -std::abs(p.x - *last_center_x));
      }
    }
    return std::tuple<int, double, double>(static_cast<int>(c.size()), bottom, continuity);
  };
  return *std::max_element(chains.begin(), chains.end(),
                           [&](const auto& a, const auto& b) { return score(a) < score(b); });
}

bool LaneDecision::fitCenterlineAndComputeOffset(const std::vector<cv::Point3f>& points,
                                                 int h, int w, int fit_order,
                                                 double* final_offset,
                                                 std::vector<double>* coeffs,
                                                 double* lateral_offset,
                                                 double* heading_error,
                                                 double* curvature) const {
  if (static_cast<int>(points.size()) < cfg_.fit_min_points || static_cast<int>(points.size()) <= fit_order) {
    return false;
  }
  if (!weightedPolyfit(points, fit_order, coeffs)) {
    return false;
  }
  double near_y = h * cfg_.lookahead_y_ratio;
  double near_x = evalPoly(*coeffs, near_y);
  double near_offset = (near_x - w / 2.0) / (w / 2.0);
  double heading = 0.0;
  double curv = 0.0;
  if (cfg_.use_heading_term && coeffs->size() > 1) {
    auto deriv = polyDeriv(*coeffs);
    double dx_dy = evalPoly(deriv, near_y);
    heading = std::atan(dx_dy) / (M_PI / 2.0);
    if (coeffs->size() > 2) {
      auto second = polyDeriv(*coeffs, 2);
      curv = clampValue(evalPoly(second, near_y) * h, -1.0, 1.0);
    }
  }
  *final_offset = clampValue(cfg_.near_offset_weight * near_offset + cfg_.heading_weight * heading, -1.0, 1.0);
  *lateral_offset = clampValue(near_offset, -1.0, 1.0);
  *heading_error = clampValue(heading, -1.0, 1.0);
  *curvature = curv;
  return true;
}

double LaneDecision::smoothOffset(double raw_offset) {
  double diff = raw_offset - last_offset_;
  if (cfg_.max_offset_jump > 0.0f && std::abs(diff) > cfg_.max_offset_jump) {
    raw_offset = last_offset_ + std::copysign(cfg_.max_offset_jump, diff);
  }
  double alpha = clampValue(cfg_.offset_smoothing_alpha, 0.0f, 1.0f);
  double smoothed = alpha * raw_offset + (1.0 - alpha) * last_offset_;
  last_offset_ = smoothed;
  return smoothed;
}

double LaneDecision::fallbackCenterOffset(const cv::Mat& seg_map) const {
  std::vector<int> xs;
  for (int y = 0; y < seg_map.rows; ++y) {
    const uint8_t* row = seg_map.ptr<uint8_t>(y);
    for (int x = 0; x < seg_map.cols; ++x) {
      if (row[x] == 1) {
        xs.push_back(x);
      }
    }
  }
  if (xs.empty()) {
    return 0.0;
  }
  double mean = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
  return clampValue((mean - seg_map.cols / 2.0) / (seg_map.cols / 2.0), -1.0, 1.0);
}

bool LaneDecision::checkGuideboardInFarRoi(const std::vector<Detection>& detections, int h) const {
  int y0 = static_cast<int>(h * cfg_.guideboard_detect_y0_ratio);
  int y1 = static_cast<int>(h * cfg_.guideboard_detect_y1_ratio);
  for (const auto& det : detections) {
    if (det.class_name == "GuideBoard" && det.center.y >= y0 && det.center.y <= y1) {
      return true;
    }
  }
  return false;
}

void LaneDecision::updateTrafficLightStopState(const std::vector<Detection>& detections, int image_height) {
  if (!cfg_.enable_traffic_light_stop) {
    traffic_stop_active_ = false;
    stop_request_active_ = finish_stop_active_;
    traffic_light_state_ = "CLEAR";
    red_light_confirm_count_ = 0;
    green_light_confirm_count_ = 0;
    return;
  }
  bool has_red = false;
  bool has_green = false;
  bool zebra_reached = false;
  double zebra_stop_y = image_height * clampValue(cfg_.zebra_stop_y_ratio, 0.0f, 1.0f);
  for (const auto& det : detections) {
    if (det.class_name == "green_light" && det.confidence >= cfg_.traffic_light_min_confidence) {
      has_green = true;
    } else if (det.class_name == "red_light" && det.confidence >= cfg_.traffic_light_min_confidence) {
      has_red = true;
    } else if (det.class_name == "Zebra" && det.confidence >= cfg_.zebra_min_confidence &&
               det.bbox.y + det.bbox.height >= zebra_stop_y) {
      zebra_reached = true;
    }
  }
  green_light_confirm_count_ = has_green ? green_light_confirm_count_ + 1 : 0;
  if (green_light_confirm_count_ >= std::max(1, cfg_.green_light_confirm_frames)) {
    traffic_light_state_ = "CLEAR";
    traffic_stop_active_ = false;
    stop_request_active_ = finish_stop_active_;
    red_light_confirm_count_ = 0;
    return;
  }
  red_light_confirm_count_ = has_red ? red_light_confirm_count_ + 1 : 0;
  if (traffic_light_state_ == "WAIT_GREEN") {
    traffic_stop_active_ = true;
    stop_request_active_ = true;
    return;
  }
  if (red_light_confirm_count_ >= std::max(1, cfg_.red_light_confirm_frames)) {
    traffic_light_state_ = "RED_SEEN";
  }
  if (traffic_light_state_ == "RED_SEEN" && zebra_reached) {
    traffic_light_state_ = "WAIT_GREEN";
    traffic_stop_active_ = true;
    stop_request_active_ = true;
  } else {
    traffic_stop_active_ = false;
    stop_request_active_ = finish_stop_active_;
  }
}

void LaneDecision::updateFinishStopState(const std::vector<Detection>& detections, int image_height) {
  if (!cfg_.enable_finish_stop) {
    finish_stop_active_ = false;
    finish_stop_state_ = "CLEAR";
    finish_stop_lost_count_ = 0;
    stop_request_active_ = traffic_stop_active_;
    return;
  }
  if (finish_stop_active_) {
    stop_request_active_ = true;
    return;
  }
  bool stop_seen = false;
  bool stop_reached = false;
  double arm_y = image_height * clampValue(cfg_.finish_stop_arm_y_ratio, 0.0f, 1.0f);
  for (const auto& det : detections) {
    if (det.class_name != "Stop" || det.confidence < cfg_.finish_stop_min_confidence) {
      continue;
    }
    stop_seen = true;
    if (det.bbox.y + det.bbox.height >= arm_y) {
      stop_reached = true;
    }
  }
  if (finish_stop_state_ == "CLEAR" && stop_seen) {
    finish_stop_state_ = "STOP_SEEN";
    finish_stop_lost_count_ = 0;
  }
  if (finish_stop_state_ == "STOP_SEEN") {
    if (stop_reached) {
      finish_stop_state_ = "STOP_ARMED";
      finish_stop_lost_count_ = 0;
    } else if (!stop_seen) {
      finish_stop_state_ = "CLEAR";
      finish_stop_lost_count_ = 0;
    }
  } else if (finish_stop_state_ == "STOP_ARMED") {
    if (stop_seen) {
      finish_stop_lost_count_ = 0;
    } else {
      ++finish_stop_lost_count_;
      if (finish_stop_lost_count_ >= std::max(1, cfg_.finish_stop_lost_frames)) {
        finish_stop_state_ = "FINISH_STOP";
        finish_stop_active_ = true;
      }
    }
  }
  stop_request_active_ = traffic_stop_active_ || finish_stop_active_;
}

std::string LaneDecision::taskState() const {
  if (finish_stop_active_) {
    return "FINISH_STOP";
  }
  if (traffic_stop_active_) {
    return "TRAFFIC_STOP";
  }
  return "CLEAR";
}

void LaneDecision::populateDebugInfo(const std::vector<Band>& bands,
                                     const std::vector<ObstacleZone>& zones,
                                     const std::vector<cv::Point3f>& fit_points,
                                     const std::vector<double>& fit_coeffs) {
  debug_info_.bands.clear();
  debug_info_.obstacle_zones.clear();
  debug_info_.fit_points = fit_points;
  debug_info_.fit_coeffs = fit_coeffs;
  debug_info_.branch_locked = branch_locked_;
  debug_info_.locked_branch_side = locked_branch_side_;
  debug_info_.raw_point_count = 0;
  debug_info_.fit_point_count = static_cast<int>(fit_points.size());
  debug_info_.segment_count = 0;

  for (const auto& zone : zones) {
    debug_info_.obstacle_zones.push_back({zone.rect, zone.label});
  }
  for (const auto& band : bands) {
    LaneBandDebug bd;
    bd.y0 = band.y0;
    bd.y1 = band.y1;
    if (band.selected_segment) {
      bd.selected_center_x = static_cast<int>(std::round(band.selected_segment->center_x));
      ++debug_info_.raw_point_count;
    }
    for (const auto& seg : band.segments) {
      ++debug_info_.segment_count;
      LaneSegmentDebug sd;
      sd.x0 = seg.x0;
      sd.x1 = seg.x1;
      sd.center_x = static_cast<int>(std::round(seg.center_x));
      sd.pixel_count = seg.pixel_count;
      sd.obstacle_cut = seg.obstacle_cut;
      if (band.selected_segment && std::abs(band.selected_segment->center_x - seg.center_x) < 1e-3) {
        sd.selected = true;
      }
      bd.segments.push_back(sd);
    }
    if (band.selected_segment && band.selected_segment->virtual_segment) {
      LaneSegmentDebug sd;
      sd.x0 = band.selected_segment->x0;
      sd.x1 = band.selected_segment->x1;
      sd.center_x = static_cast<int>(std::round(band.selected_segment->center_x));
      sd.pixel_count = band.selected_segment->pixel_count;
      sd.selected = true;
      sd.virtual_segment = true;
      bd.segments.push_back(sd);
    }
    debug_info_.bands.push_back(std::move(bd));
  }
}

}  // namespace track_perception_cpp
