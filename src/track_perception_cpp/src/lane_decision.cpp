#include "track_perception_cpp/lane_decision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

std::vector<double> parseDoubleList(const std::string& text) {
  std::vector<double> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      size_t pos = 0;
      double value = std::stod(item, &pos);
      values.push_back(value);
    } catch (const std::exception&) {
    }
  }
  return values;
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
     << "\"bottom_offset\":" << state.bottom_offset << ","
     << "\"raw_control_offset\":" << state.raw_control_offset << ","
     << "\"lookahead_x\":" << state.lookahead_x << ","
     << "\"lookahead_y\":" << state.lookahead_y << ","
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
  cfg_.branch_confirm_frames = std::max(1, cfg_.branch_confirm_frames);
  cfg_.fit_order = clampValue(cfg_.fit_order, 1, 2);
  cfg_.branch_fit_order = clampValue(cfg_.branch_fit_order, 1, 2);
  cfg_.left_boundary_template_min_points = std::max(1, cfg_.left_boundary_template_min_points);
  cfg_.left_boundary_template_weight = std::max(0.01f, cfg_.left_boundary_template_weight);
  cfg_.right_boundary_template_min_points = std::max(1, cfg_.right_boundary_template_min_points);
  cfg_.right_boundary_template_weight = std::max(0.01f, cfg_.right_boundary_template_weight);
  if (cfg_.near_split_template_side != "left" && cfg_.near_split_template_side != "right") {
    cfg_.near_split_template_side = cfg_.outer_side;
  }
  cfg_.near_split_unlock_min_lock_time = std::max(0.0f, cfg_.near_split_unlock_min_lock_time);
  cfg_.near_split_exit_recent_frames = std::max(0, cfg_.near_split_exit_recent_frames);
  cfg_.near_split_exit_bottom_shift_norm = std::max(0.0f, cfg_.near_split_exit_bottom_shift_norm);
  cfg_.near_split_exit_residual_px = std::max(0.0f, cfg_.near_split_exit_residual_px);
  left_boundary_template_offsets_ = parseDoubleList(cfg_.left_boundary_template_offsets);
  right_boundary_template_offsets_ = parseDoubleList(cfg_.right_boundary_template_offsets);
  locked_branch_side_ = cfg_.outer_side;
}

void LaneDecision::setGuideboardBranchHint(const std::string& branch, bool valid) {
  if (!valid) {
    guideboard_branch_hint_valid_ = false;
    guideboard_branch_hint_ = "left";
    return;
  }
  if (branch == "left" || branch == "right") {
    guideboard_branch_hint_ = branch;
    guideboard_branch_hint_valid_ = true;
  }
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
  double bottom_offset = 0.0;
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
    auto [near_split_detected, near_split_score] = detectNearSplitFromBands(bands);
    if (near_split_detected) {
      near_split_hold_count_ = std::max(0, cfg_.near_split_hold_frames);
      near_split_recent_count_ = std::max(near_split_recent_count_, cfg_.near_split_exit_recent_frames);
    } else if (near_split_hold_count_ > 0) {
      --near_split_hold_count_;
    }
    if (!near_split_detected && near_split_recent_count_ > 0) {
      --near_split_recent_count_;
    }
    bool near_split_active = near_split_detected || near_split_hold_count_ > 0;
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
    debug_info_.near_split_detected = near_split_active;
    debug_info_.near_split_score = near_split_score;
    debug_info_.guideboard_seen = guideboard_seen;
    debug_info_.guideboard_count = guideboard_count;
    debug_info_.guideboard_roi_count = guideboard_roi_count;
    debug_info_.guideboard_best_confidence = guideboard_best_confidence;
    debug_info_.guideboard_best_center = guideboard_best_center;
    bool near_split_overrides_branch = near_split_active && !guideboard_seen &&
                                       near_split_score >= branch_score;
    bool branch_lock_candidate = !near_split_overrides_branch &&
                                 (branch_detected || (guideboard_seen && branch_score > 0));
    if (branch_locked_ && near_split_active &&
        locked_branch_side_ != cfg_.near_split_template_side &&
        current_time - lock_start_time_ >= cfg_.near_split_unlock_min_lock_time &&
        near_split_score >= branch_score) {
      branch_locked_ = false;
      locked_branch_side_ = cfg_.outer_side;
      branch_confirm_count_ = 0;
      exit_confirm_count_ = 0;
      debug_info_.branch_transition_reason = "near_split_unlock";
    }
    if (branch_locked_ && guideboard_seen && guideboard_branch_hint_valid_ &&
        locked_branch_side_ != guideboard_branch_hint_) {
      locked_branch_side_ = guideboard_branch_hint_;
      lock_start_time_ = current_time;
      exit_confirm_count_ = 0;
      debug_info_.branch_transition_reason = "guideboard_ocr_override";
    }

    if (!branch_locked_) {
      if (branch_lock_candidate) {
        ++branch_confirm_count_;
      } else {
        branch_confirm_count_ = 0;
      }
      if (branch_confirm_count_ >= cfg_.branch_confirm_frames) {
        double last_center_x = last_offset_ * w / 2.0 + w / 2.0;
        std::string target_branch = cfg_.outer_side;
        if (guideboard_seen) {
          target_branch = guideboard_branch_hint_valid_ ? guideboard_branch_hint_ : cfg_.guideboard_branch;
        } else if (isBoundaryTemplateReady(cfg_.outer_side)) {
          target_branch = cfg_.outer_side;
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
        branch_confirm_count_ = 0;
        exit_confirm_count_ = 0;
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
      bool branch_fully_lost = !branch_detected && branch_score == 0;
      if (lock_duration >= cfg_.min_branch_lock_time && branch_fully_lost &&
          single_path_count >= single_path_required) {
        ++exit_confirm_count_;
      } else {
        exit_confirm_count_ = 0;
      }
      if (lock_duration >= cfg_.min_branch_lock_time &&
          exit_confirm_count_ >= cfg_.exit_single_path_confirm_frames) {
        branch_locked_ = false;
        locked_branch_side_ = cfg_.outer_side;
        branch_confirm_count_ = 0;
        exit_confirm_count_ = 0;
      }
    }

    std::string target_side = branch_locked_
                                  ? locked_branch_side_
                                  : (near_split_active ? cfg_.near_split_template_side : cfg_.outer_side);
    double last_center_x = last_offset_ * w / 2.0 + w / 2.0;
    raw_points = collectCenterlinePoints(bands, branch_locked_, target_side, last_center_x, w, current_time);
    bool near_split_residual = !branch_detected && !branch_locked_ &&
                               detectNearSplitResidual(raw_points, w);
    bool near_split_exit_continuation = !branch_detected && !branch_locked_ &&
                                        !near_split_active && near_split_recent_count_ > 0 &&
                                        detectNearSplitExitContinuation(raw_points, w);
    if (near_split_residual) {
      near_split_active = true;
      target_side = cfg_.near_split_template_side;
      near_split_hold_count_ = std::max(near_split_hold_count_, std::max(0, cfg_.near_split_hold_frames / 2));
      near_split_recent_count_ = std::max(near_split_recent_count_, cfg_.near_split_exit_recent_frames / 2);
      debug_info_.near_split_detected = true;
    } else if (near_split_exit_continuation) {
      near_split_active = true;
      target_side = cfg_.near_split_template_side;
      debug_info_.near_split_detected = true;
      debug_info_.branch_transition_reason = "near_split_exit";
    }
    RoadClass road_class = classifyRoadGeometry(bands, raw_points, branch_detected, branch_score,
                                                near_split_active, w);
    road_state = roadClassName(road_class);
    debug_info_.asym_wide_detected = road_class == RoadClass::AsymWide;
    debug_info_.asym_wide_band_count = countAsymWideBands(bands);
    debug_info_.center_residual_px = static_cast<float>(calculateCenterResidual(raw_points, w));
    bool template_trigger = branch_detected || branch_locked_ || near_split_active;
    bool template_active = shouldUseBoundaryTemplate(target_side, template_trigger);
    if (template_active) {
      int template_min_band_index = near_split_active && !branch_locked_
                                        ? std::max(0, cfg_.near_split_template_skip_bands)
                                        : 0;
      auto template_points = collectBoundaryTemplatePoints(bands, w, target_side,
                                                           template_min_band_index);
      if (static_cast<int>(template_points.size()) >= boundaryTemplateMinPoints(target_side)) {
        raw_points = std::move(template_points);
        if (near_split_active && !branch_locked_) {
          road_state = "NEAR_SPLIT";
        } else {
          road_state = "BRANCH";
        }
        debug_info_.left_boundary_template_active = true;
        debug_info_.boundary_template_side = target_side;
        debug_info_.left_boundary_template_reason = "active";
        debug_info_.branch_entry_transition_active = false;
        debug_info_.branch_transition_reason = target_side + "_boundary_template";
      } else {
        debug_info_.left_boundary_template_reason = "few_points";
      }
    } else if (cfg_.enable_left_boundary_template_line) {
      (void)0;
    }
    int fit_order = (branch_locked_ || debug_info_.left_boundary_template_active) ? cfg_.branch_fit_order : cfg_.fit_order;
    if (debug_info_.left_boundary_template_active) {
      fit_points = raw_points;
    } else {
      fit_points = filterCenterlinePoints(raw_points, w, last_center_x);
    }
    if (!debug_info_.left_boundary_template_active && road_class == RoadClass::AsymWide) {
      fit_points = filterAsymWidePoints(fit_points, bands);
    }
    if (!debug_info_.left_boundary_template_active) {
      appendDetectionFitPoints(fit_points, detections, h);
    }

    double raw_offset = 0.0;
    if (fitCenterlineAndComputeOffset(fit_points, h, w, fit_order, &raw_offset, &fit_coeffs,
                                      &lateral_offset, &heading_error, &curvature)) {
      center_offset = smoothOffset(raw_offset);
      const double lookahead_y = h * cfg_.lookahead_y_ratio;
      const double bottom_y = std::max(0.0, static_cast<double>(h - 1));
      const double bottom_x = evalPoly(fit_coeffs, bottom_y);
      bottom_offset = clampValue((bottom_x - w / 2.0) / (w / 2.0), -1.0, 1.0);
      debug_info_.raw_control_offset = static_cast<float>(raw_offset);
      debug_info_.lookahead_x = static_cast<float>(evalPoly(fit_coeffs, lookahead_y));
      debug_info_.lookahead_y = static_cast<float>(lookahead_y);
      debug_info_.bottom_offset = static_cast<float>(bottom_offset);
      debug_info_.image_width = w;
      is_valid = true;
      confidence = calculateLaneConfidence(fit_points, bands);
    } else {
      cv::Mat bottom_seg = seg_map(cv::Range(static_cast<int>(h * 0.8), h), cv::Range::all());
      center_offset = fallbackCenterOffset(bottom_seg);
      lateral_offset = center_offset;
      bottom_offset = center_offset;
      const bool has_fallback_pixels = cv::countNonZero(bottom_seg == 1) > 0;
      is_valid = false;
      confidence = 0.0;
      if (!has_fallback_pixels && std::abs(last_offset_) > 0.01) {
        center_offset = last_offset_;
      }
      road_state = "LOW_CONFIDENCE";
      fit_points.clear();
      fit_coeffs.clear();
      debug_info_.raw_control_offset = static_cast<float>(center_offset);
      debug_info_.bottom_offset = static_cast<float>(bottom_offset);
      debug_info_.image_width = w;
    }
    populateDebugInfo(
      bands, getActiveObstacleZones(detections, w, h), raw_points, fit_points, fit_coeffs);
  } else {
    cv::Mat bottom_seg = seg_map(cv::Range(h / 2, h), cv::Range::all());
    center_offset = fallbackCenterOffset(bottom_seg);
    lateral_offset = center_offset;
    bottom_offset = center_offset;
    debug_info_.raw_control_offset = static_cast<float>(center_offset);
    debug_info_.bottom_offset = static_cast<float>(bottom_offset);
    debug_info_.image_width = w;
    is_valid = false;
    confidence = 0.0;
    road_state = "LOW_CONFIDENCE";
  }

  updateTrafficLightStopState(detections, h);
  updateFinishStopState(detections, h);
  updateObstacleStopState(detections, h);
  updateStartBoostState(detections, h);

  state.control_offset = static_cast<float>(center_offset);
  state.lateral_offset = static_cast<float>(lateral_offset);
  state.bottom_offset = static_cast<float>(bottom_offset);
  state.raw_control_offset = debug_info_.raw_control_offset;
  state.lookahead_x = debug_info_.lookahead_x;
  state.lookahead_y = debug_info_.lookahead_y;
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

std::pair<bool, int> LaneDecision::detectNearSplitFromBands(const std::vector<Band>& bands) const {
  if (bands.empty()) {
    return {false, 0};
  }
  int near_count = static_cast<int>(
      std::ceil(static_cast<double>(bands.size()) *
                clampValue(cfg_.near_split_detect_near_band_ratio, 0.1f, 1.0f)));
  near_count = std::max(1, std::min(near_count, static_cast<int>(bands.size())));
  int start = std::max(0, static_cast<int>(bands.size()) - near_count);
  int split_bands = 0;
  for (int i = start; i < static_cast<int>(bands.size()); ++i) {
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
      ++split_bands;
    }
  }
  return {split_bands >= cfg_.near_split_detect_min_bands, split_bands};
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
    return band.segments.front();
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
    return seg;
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

bool LaneDecision::detectNearSplitResidual(const std::vector<cv::Point3f>& raw_points,
                                           int image_width) const {
  if (raw_points.size() < 6 || image_width <= 1) {
    return false;
  }
  auto ordered = raw_points;
  std::sort(ordered.begin(), ordered.end(), [](const cv::Point3f& a, const cv::Point3f& b) {
    return a.y < b.y;
  });
  double top_x = ordered.front().x;
  double bottom_x = ordered.back().x;
  double half_w = image_width / 2.0;
  double slope_norm = (top_x - bottom_x) / half_w;
  double bottom_norm = (bottom_x - half_w) / half_w;
  double top_norm = (top_x - half_w) / half_w;
  bool right_sweep = slope_norm >= cfg_.near_split_residual_slope_norm &&
                     bottom_norm >= -cfg_.near_split_residual_bottom_norm;
  bool bottom_near_center = std::abs(bottom_norm) <= cfg_.near_split_residual_bottom_norm;
  return right_sweep && bottom_near_center;
}

bool LaneDecision::detectNearSplitExitContinuation(const std::vector<cv::Point3f>& raw_points,
                                                   int image_width) const {
  if (raw_points.size() < 6 || image_width <= 1) {
    return false;
  }
  auto ordered = raw_points;
  std::sort(ordered.begin(), ordered.end(), [](const cv::Point3f& a, const cv::Point3f& b) {
    return a.y < b.y;
  });
  double top_x = ordered.front().x;
  double bottom_x = ordered.back().x;
  double half_w = image_width / 2.0;
  double slope_norm = (top_x - bottom_x) / half_w;
  double bottom_norm = (bottom_x - half_w) / half_w;
  double residual = calculateCenterResidual(raw_points, image_width);
  bool shifted_near = std::abs(bottom_norm) >= cfg_.near_split_exit_bottom_shift_norm;
  bool still_morphing = std::abs(slope_norm) >= cfg_.near_split_residual_slope_norm ||
                        residual >= cfg_.near_split_exit_residual_px;
  return shifted_near && still_morphing;
}

LaneDecision::RoadClass LaneDecision::classifyRoadGeometry(
    const std::vector<Band>& bands, const std::vector<cv::Point3f>& raw_points,
    bool branch_detected, int branch_score, bool near_split_detected, int image_width) const {
  if (branch_locked_) {
    return RoadClass::Branch;
  }
  int valid_band_count = 0;
  for (const auto& band : bands) {
    if (!band.segments.empty()) {
      ++valid_band_count;
    }
  }
  if (valid_band_count <= 0 || static_cast<int>(raw_points.size()) < cfg_.fit_min_points) {
    return RoadClass::LowConfidence;
  }
  if (near_split_detected && !branch_detected && branch_score == 0) {
    return RoadClass::NearSplit;
  }
  int wide_bands = countAsymWideBands(bands);
  double residual = calculateCenterResidual(raw_points, image_width);
  double residual_ratio_limit = std::max(0.0f, cfg_.asym_wide_center_residual_ratio) * image_width;
  double residual_px_limit = cfg_.asym_wide_center_residual_px > 0.0f
                                 ? cfg_.asym_wide_center_residual_px
                                 : residual_ratio_limit;
  double residual_limit = std::max(1.0, std::min(residual_ratio_limit, residual_px_limit));
  (void)branch_detected;
  bool branch_like = branch_score > 0;
  bool wide = wide_bands >= std::max(1, cfg_.asym_wide_min_bands);
  bool unstable_center = residual > residual_limit;
  bool too_few_normal_points = static_cast<int>(raw_points.size()) < std::max(cfg_.fit_min_points,
                                                                              cfg_.asym_wide_min_normal_points);
  if (wide || branch_like || unstable_center || too_few_normal_points) {
    return RoadClass::AsymWide;
  }
  return RoadClass::Normal;
}

std::string LaneDecision::roadClassName(RoadClass road_class) const {
  switch (road_class) {
    case RoadClass::Branch:
      return "BRANCH";
    case RoadClass::NearSplit:
      return "NEAR_SPLIT";
    case RoadClass::AsymWide:
      return "ASYM_WIDE";
    case RoadClass::LowConfidence:
      return "LOW_CONFIDENCE";
    case RoadClass::Normal:
    default:
      return "NORMAL";
  }
}

int LaneDecision::countAsymWideBands(const std::vector<Band>& bands) const {
  std::vector<double> single_widths;
  single_widths.reserve(bands.size());
  for (const auto& band : bands) {
    if (band.segments.size() == 1) {
      single_widths.push_back(band.segments.front().width);
    }
  }
  if (single_widths.size() < 3) {
    return 0;
  }
  std::sort(single_widths.begin(), single_widths.end());
  double median_width = single_widths[single_widths.size() / 2];
  if (median_width <= 1.0) {
    return 0;
  }
  double wide_threshold = median_width * std::max(1.05f, cfg_.asym_wide_segment_ratio);
  int wide_count = 0;
  int run = 0;
  int best_run = 0;
  for (const auto& band : bands) {
    if (band.segments.size() == 1 && band.segments.front().width >= wide_threshold) {
      ++wide_count;
      ++run;
      best_run = std::max(best_run, run);
    } else {
      run = 0;
    }
  }
  return std::max(wide_count, best_run);
}

double LaneDecision::calculateCenterResidual(const std::vector<cv::Point3f>& points,
                                             int image_width) const {
  if (points.size() < 4) {
    return 0.0;
  }
  std::vector<double> coeffs;
  int order = points.size() >= 5 ? 2 : 1;
  if (!weightedPolyfit(points, order, &coeffs)) {
    return static_cast<double>(image_width);
  }
  double sum = 0.0;
  for (const auto& p : points) {
    sum += std::abs(static_cast<double>(p.x) - evalPoly(coeffs, p.y));
  }
  return sum / static_cast<double>(points.size());
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

bool LaneDecision::isBoundaryTemplateReady(const std::string& side) const {
  if (!cfg_.enable_left_boundary_template_line) {
    return false;
  }
  if (side == "left") {
    return !left_boundary_template_offsets_.empty();
  }
  if (side == "right") {
    return !right_boundary_template_offsets_.empty();
  }
  return false;
}

int LaneDecision::boundaryTemplateMinPoints(const std::string& side) const {
  return side == "right" ? cfg_.right_boundary_template_min_points
                         : cfg_.left_boundary_template_min_points;
}

bool LaneDecision::shouldUseBoundaryTemplate(const std::string& side, bool template_trigger) {
  if (!cfg_.enable_left_boundary_template_line) {
    debug_info_.left_boundary_template_reason = "disabled";
    return false;
  }
  if (side != "left" && side != "right") {
    debug_info_.left_boundary_template_reason = "bad_side";
    return false;
  }
  if (!isBoundaryTemplateReady(side)) {
    debug_info_.left_boundary_template_reason = "no_offsets";
    return false;
  }
  if (!template_trigger) {
    debug_info_.left_boundary_template_reason = "no_template_trigger";
    return false;
  }
  debug_info_.boundary_template_side = side;
  debug_info_.left_boundary_template_reason = "ready";
  return true;
}

std::vector<cv::Point3f> LaneDecision::collectBoundaryTemplatePoints(
    std::vector<Band>& bands, int image_width, const std::string& side,
    int min_band_index) {
  std::vector<cv::Point3f> points;
  const std::vector<double>* offsets = nullptr;
  float weight = 1.0f;
  if (side == "left") {
    offsets = &left_boundary_template_offsets_;
    weight = std::max(0.01f, cfg_.left_boundary_template_weight);
  } else if (side == "right") {
    offsets = &right_boundary_template_offsets_;
    weight = std::max(0.01f, cfg_.right_boundary_template_weight);
  }
  if (!offsets || offsets->empty()) {
    debug_info_.left_boundary_template_reason = "no_offsets";
    return points;
  }
  for (auto& band : bands) {
    band.selected_segment.reset();
  }
  std::optional<double> last_template_x;
  for (auto it_band = bands.rbegin(); it_band != bands.rend(); ++it_band) {
    auto& band = *it_band;
    if (band.segments.empty() || band.index < min_band_index ||
        band.index >= static_cast<int>(offsets->size())) {
      continue;
    }
    double offset = (*offsets)[band.index];
    if (!std::isfinite(offset) || offset <= 0.0) {
      continue;
    }
    auto target_for_segment = [&](const Segment& seg) {
      double x = side == "right" ? static_cast<double>(seg.x1) - offset
                                 : static_cast<double>(seg.x0) + offset;
      double margin = std::min(3.0, std::max(0.0, seg.width * 0.25));
      return clampValue(x, static_cast<double>(seg.x0) + margin,
                        static_cast<double>(seg.x1) - margin);
    };
    const Segment* source_segment = nullptr;
    if (last_template_x) {
      auto it = std::min_element(
          band.segments.begin(), band.segments.end(),
          [&](const Segment& a, const Segment& b) {
            return std::abs(target_for_segment(a) - *last_template_x) <
                   std::abs(target_for_segment(b) - *last_template_x);
          });
      if (it != band.segments.end()) {
        source_segment = &(*it);
      }
    }
    if (!source_segment) {
      auto it = side == "right"
                    ? std::max_element(band.segments.begin(), band.segments.end(),
                                       [](const Segment& a, const Segment& b) { return a.x1 < b.x1; })
                    : std::min_element(band.segments.begin(), band.segments.end(),
                                       [](const Segment& a, const Segment& b) { return a.x0 < b.x0; });
      if (it != band.segments.end()) {
        source_segment = &(*it);
      }
    }
    if (!source_segment) {
      continue;
    }
    double target_x = target_for_segment(*source_segment);
    target_x = clampValue(target_x, 0.0, static_cast<double>(std::max(0, image_width - 1)));
    Segment virtual_seg;
    virtual_seg.x0 = static_cast<int>(std::round(target_x));
    virtual_seg.x1 = virtual_seg.x0;
    virtual_seg.center_x = target_x;
    virtual_seg.width = 1.0;
    virtual_seg.pixel_count = 1;
    virtual_seg.virtual_segment = true;
    band.selected_segment = virtual_seg;
    points.emplace_back(static_cast<float>(target_x), static_cast<float>(band.y_center), weight);
    last_template_x = target_x;
  }
  debug_info_.left_boundary_template_points = static_cast<int>(points.size());
  debug_info_.boundary_template_side = side;
  if (points.empty()) {
    debug_info_.left_boundary_template_reason = "no_valid_boundary";
  }
  return points;
}

std::vector<cv::Point3f> LaneDecision::collectCenterlinePoints(
    std::vector<Band>& bands, bool branch_locked, const std::string& side,
    std::optional<double> last_center_x, int image_width, double now) {
  std::vector<cv::Point3f> points;
  if (branch_locked && cfg_.enable_branch_entry_transition) {
    auto transition_points = collectBranchGeometryLinePoints(bands, side, image_width);
    if (!transition_points.empty()) {
      debug_info_.branch_entry_transition_active = true;
      return transition_points;
    }
  }
  bool use_locked_continuity = branch_locked && last_center_x &&
                               shouldUseLockedPathContinuity(now);
  for (auto& band : bands) {
    if (band.segments.empty()) {
      continue;
    }
    std::optional<Segment> target;
    if (branch_locked) {
      if (band.segments.size() >= 2) {
        target = chooseTargetSegment(band, side);
      } else {
        if (use_locked_continuity) {
          target = chooseLockedSegmentByContinuity(band, image_width, *last_center_x);
        }
        if (!target) {
          target = chooseTargetSegment(band, side);
        }
      }
    } else {
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
    if (target) {
      band.selected_segment = *target;
      points.emplace_back(static_cast<float>(target->center_x), static_cast<float>(band.y_center), 1.0f);
    }
  }
  return points;
}

std::vector<cv::Point3f> LaneDecision::collectBranchGeometryLinePoints(
    std::vector<Band>& bands, const std::string& side, int image_width) {
  if (bands.empty() || side.empty()) {
    debug_info_.branch_transition_reason = "no_bands";
    return {};
  }

  int n = static_cast<int>(bands.size());
  int near_count = std::max(1, static_cast<int>(
      std::ceil(n * clampValue(cfg_.branch_transition_near_main_ratio, 0.0f, 1.0f))));
  std::vector<cv::Point3f> branch_points;
  std::vector<cv::Point3f> near_main_points;
  std::vector<int> branch_indices;
  branch_points.reserve(bands.size());
  near_main_points.reserve(bands.size());
  branch_indices.reserve(bands.size());

  for (int i = 0; i < n; ++i) {
    const auto& band = bands[i];
    if (band.segments.empty()) {
      continue;
    }
    if (band.segments.size() >= 2) {
      auto target = chooseTargetSegment(band, side);
      if (target) {
        branch_points.emplace_back(static_cast<float>(target->center_x),
                                   static_cast<float>(band.y_center), 1.0f);
        branch_indices.push_back(i);
      }
    }
  }

  if (!branch_indices.empty()) {
    int bottom_branch_index = *std::max_element(branch_indices.begin(), branch_indices.end());
    for (int i = bottom_branch_index + 1; i < n && static_cast<int>(near_main_points.size()) < near_count; ++i) {
      const auto& band = bands[i];
      if (band.segments.size() != 1) {
        break;
      }
      const auto& seg = band.segments.front();
      near_main_points.emplace_back(static_cast<float>(seg.center_x),
                                    static_cast<float>(band.y_center), 1.0f);
    }
  }

  debug_info_.branch_transition_near_single_bands = static_cast<int>(near_main_points.size());
  debug_info_.branch_transition_branch_points = static_cast<int>(branch_points.size());
  if (static_cast<int>(branch_points.size()) <
      std::max(1, cfg_.branch_transition_min_branch_points)) {
    debug_info_.branch_transition_reason = "few_branch_points";
    return {};
  }
  if (near_main_points.empty() || !cfg_.branch_transition_use_when_near_single_path) {
    debug_info_.branch_transition_reason = "branch_only";
    for (const auto& p : branch_points) {
      for (auto& band : bands) {
        if (std::abs(static_cast<double>(p.y) - band.y_center) > 1.0 || band.segments.size() < 2) {
          continue;
        }
        auto target = chooseTargetSegment(band, side);
        if (target) {
          band.selected_segment = *target;
        }
        break;
      }
    }
    return branch_points;
  }

  auto avg_point = [](const std::vector<cv::Point3f>& points) {
    double sx = 0.0;
    double sy = 0.0;
    for (const auto& p : points) {
      sx += p.x;
      sy += p.y;
    }
    double inv = 1.0 / static_cast<double>(points.size());
    return cv::Point2d(sx * inv, sy * inv);
  };

  auto slope_or_zero = [](const std::vector<cv::Point3f>& points) {
    if (points.size() < 2) {
      return 0.0;
    }
    std::vector<double> coeffs;
    if (weightedPolyfit(points, 1, &coeffs) && coeffs.size() >= 2 &&
        std::isfinite(coeffs[0])) {
      return coeffs[0];
    }
    return 0.0;
  };

  cv::Point2d p0 = avg_point(near_main_points);
  cv::Point2d p1 = avg_point(branch_points);
  double min_dy = std::max(8.0, static_cast<double>(image_width) * 0.04);
  if (p0.y <= p1.y + min_dy) {
    debug_info_.branch_transition_reason = "bad_anchor_order";
    return {};
  }

  double dy = p1.y - p0.y;
  double main_slope = slope_or_zero(near_main_points);
  double branch_slope = slope_or_zero(branch_points);
  double tangent_scale = 0.45;
  cv::Point2d m0(main_slope * dy * tangent_scale, dy * tangent_scale);
  cv::Point2d m1(branch_slope * dy * tangent_scale, dy * tangent_scale);

  int samples = std::max(4, cfg_.branch_transition_samples);
  float weight = std::max(0.01f, cfg_.branch_transition_weight);
  std::vector<cv::Point3f> out;
  out.reserve(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i) {
    double t = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
    double tt = t * t;
    double ttt = tt * t;
    double h00 = 2.0 * ttt - 3.0 * tt + 1.0;
    double h10 = ttt - 2.0 * tt + t;
    double h01 = -2.0 * ttt + 3.0 * tt;
    double h11 = ttt - tt;
    double x = h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x;
    double y = h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y;
    x = clampValue(x, 0.0, static_cast<double>(std::max(0, image_width - 1)));
    out.emplace_back(static_cast<float>(x), static_cast<float>(y), weight);
  }

  for (auto& band : bands) {
    if (band.segments.empty()) {
      continue;
    }
    auto nearest = std::min_element(out.begin(), out.end(), [&](const cv::Point3f& a,
                                                                const cv::Point3f& b) {
      return std::abs(static_cast<double>(a.y) - band.y_center) <
             std::abs(static_cast<double>(b.y) - band.y_center);
    });
    if (nearest == out.end()) {
      continue;
    }
    Segment seg;
    seg.center_x = nearest->x;
    seg.width = 1.0;
    seg.x0 = static_cast<int>(std::round(nearest->x));
    seg.x1 = seg.x0;
    seg.pixel_count = 1;
    seg.virtual_segment = true;
    band.selected_segment = seg;
  }

  debug_info_.branch_transition_reason = "split_entry_to_branch";
  return out;
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
    if (!cfg_.enable_fit_point_trend_filter || filtered.size() < 4) {
      return filtered;
    }
  } else {
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
    filtered = *std::max_element(chains.begin(), chains.end(),
                                 [&](const auto& a, const auto& b) { return score(a) < score(b); });
    if (!cfg_.enable_fit_point_trend_filter || filtered.size() < 4) {
      return filtered;
    }
  }

  double trend_ratio_limit = std::max(0.0f, cfg_.fit_point_trend_residual_ratio) * image_width;
  double trend_px_limit = cfg_.fit_point_trend_residual_px > 0 ? cfg_.fit_point_trend_residual_px : trend_ratio_limit;
  double residual_limit = std::max(1.0, std::min(trend_ratio_limit, trend_px_limit));
  double slope_delta_limit = std::max(0.0f, cfg_.fit_point_trend_slope_delta);
  int min_trend_points = std::max(3, cfg_.fit_point_trend_min_points);

  std::vector<cv::Point3f> ordered = filtered;
  std::sort(ordered.begin(), ordered.end(), [](const cv::Point3f& a, const cv::Point3f& b) {
    return a.y > b.y;
  });

  std::vector<cv::Point3f> trend;
  trend.reserve(ordered.size());
  trend.push_back(ordered.front());
  double last_slope = 0.0;
  bool has_slope = false;
  for (size_t i = 1; i < ordered.size(); ++i) {
    const auto& p = ordered[i];
    bool accept = true;
    if (trend.size() == 1) {
      double dy = std::max(1.0f, std::abs(p.y - trend.back().y));
      double dx = std::abs(p.x - trend.back().x);
      accept = dx <= std::max(max_dx, residual_limit * (dy / std::max(1.0, static_cast<double>(image_width) * 0.05)));
      if (accept) {
        last_slope = (p.x - trend.back().x) / (p.y - trend.back().y);
        has_slope = std::isfinite(last_slope);
      }
    } else {
      const auto& p0 = trend[trend.size() - 2];
      const auto& p1 = trend[trend.size() - 1];
      double dy01 = p1.y - p0.y;
      double slope = std::abs(dy01) > 1e-3 ? (p1.x - p0.x) / dy01 : last_slope;
      double pred_x = p1.x + slope * (p.y - p1.y);
      double residual = std::abs(p.x - pred_x);
      double dy = p.y - p1.y;
      double candidate_slope = std::abs(dy) > 1e-3 ? (p.x - p1.x) / dy : slope;
      double slope_delta = has_slope ? std::abs(candidate_slope - last_slope) : 0.0;
      accept = residual <= residual_limit && slope_delta <= slope_delta_limit;
      if (accept) {
        last_slope = 0.6 * slope + 0.4 * candidate_slope;
        has_slope = std::isfinite(last_slope);
      }
    }
    if (accept) {
      trend.push_back(p);
    }
  }

  if (static_cast<int>(trend.size()) < min_trend_points ||
      trend.size() < static_cast<size_t>(std::max(3, cfg_.fit_min_points))) {
    return filtered;
  }
  double min_keep_ratio = clampValue(cfg_.fit_point_trend_min_keep_ratio, 0.0f, 1.0f);
  double keep_ratio = static_cast<double>(trend.size()) / static_cast<double>(std::max<size_t>(1, filtered.size()));
  if (keep_ratio < min_keep_ratio) {
    return filtered;
  }
  std::sort(trend.begin(), trend.end(), [](const cv::Point3f& a, const cv::Point3f& b) {
    return a.y < b.y;
  });
  return trend;
}

std::vector<cv::Point3f> LaneDecision::filterAsymWidePoints(
    const std::vector<cv::Point3f>& points, const std::vector<Band>& bands) const {
  if (points.empty() || bands.empty()) {
    return points;
  }
  std::vector<double> single_widths;
  single_widths.reserve(bands.size());
  for (const auto& band : bands) {
    if (band.segments.size() == 1) {
      single_widths.push_back(band.segments.front().width);
    }
  }
  if (single_widths.size() < 3) {
    return points;
  }
  std::sort(single_widths.begin(), single_widths.end());
  double median_width = single_widths[single_widths.size() / 2];
  if (median_width <= 1.0) {
    return points;
  }
  double wide_threshold = median_width * std::max(1.05f, cfg_.asym_wide_segment_ratio);
  std::vector<double> wide_band_y;
  for (const auto& band : bands) {
    if (band.segments.size() == 1 && band.segments.front().width >= wide_threshold) {
      wide_band_y.push_back(band.y_center);
    }
  }
  if (wide_band_y.empty()) {
    return points;
  }

  auto is_wide_point = [&](const cv::Point3f& p) {
    return std::any_of(wide_band_y.begin(), wide_band_y.end(), [&](double y) {
      return std::abs(static_cast<double>(p.y) - y) <= 1.0;
    });
  };

  std::vector<cv::Point3f> kept;
  kept.reserve(points.size());
  for (const auto& p : points) {
    if (!is_wide_point(p)) {
      kept.push_back(p);
    }
  }
  if (static_cast<int>(kept.size()) >= cfg_.fit_min_points) {
    return kept;
  }

  std::vector<cv::Point3f> weighted = points;
  for (auto& p : weighted) {
    if (is_wide_point(p)) {
      p.z = std::max(0.05f, p.z * 0.35f);
    }
  }
  return weighted;
}

void LaneDecision::appendDetectionFitPoints(std::vector<cv::Point3f>& points,
                                            const std::vector<Detection>& detections,
                                            int image_height) const {
  if (!cfg_.enable_label_fit_points || cfg_.fit_point_labels.empty()) {
    return;
  }

  float y0 = image_height * clampValue(cfg_.fit_point_y0_ratio, 0.0f, 1.0f);
  float y1 = image_height * clampValue(cfg_.fit_point_y1_ratio, 0.0f, 1.0f);
  if (y1 < y0) {
    std::swap(y0, y1);
  }
  float weight = std::max(0.01f, cfg_.fit_point_weight);

  for (const auto& det : detections) {
    if (!cfg_.fit_point_labels.count(det.class_name) || det.confidence < cfg_.fit_point_min_confidence) {
      continue;
    }
    if (det.center.y < y0 || det.center.y > y1) {
      continue;
    }
    points.emplace_back(det.center.x, det.center.y, weight);
  }
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
  if (coeffs->size() > 1) {
    auto deriv = polyDeriv(*coeffs);
    double dx_dy = evalPoly(deriv, near_y);
    heading = std::atan(dx_dy) / (M_PI / 2.0);
    if (coeffs->size() > 2) {
      auto second = polyDeriv(*coeffs, 2);
      curv = clampValue(evalPoly(second, near_y) * h, -1.0, 1.0);
    }
  }
  const double heading_term = cfg_.use_heading_term ? cfg_.heading_weight * heading : 0.0;
  *final_offset = clampValue(cfg_.near_offset_weight * near_offset + heading_term, -1.0, 1.0);
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

void LaneDecision::updateObstacleStopState(const std::vector<Detection>& detections, int image_height) {
  if (!cfg_.enable_obstacle_avoidance || !cfg_.enable_obstacle_stop) {
    obstacle_stop_active_ = false;
    obstacle_stop_state_ = "CLEAR";
    obstacle_stop_confirm_count_ = 0;
    obstacle_stop_lost_count_ = 0;
    stop_request_active_ = traffic_stop_active_ || finish_stop_active_;
    return;
  }

  double stop_y = image_height * clampValue(cfg_.obstacle_stop_bottom_y_ratio, 0.0f, 1.0f);
  bool obstacle_reached = false;
  for (const auto& det : detections) {
    if (!cfg_.obstacle_stop_labels.count(det.class_name) || det.confidence < cfg_.obstacle_min_confidence) {
      continue;
    }
    if (det.bbox.y + det.bbox.height >= stop_y) {
      obstacle_reached = true;
      break;
    }
  }

  if (obstacle_reached) {
    obstacle_stop_confirm_count_++;
    obstacle_stop_lost_count_ = 0;
  } else {
    obstacle_stop_confirm_count_ = 0;
    if (obstacle_stop_active_) {
      obstacle_stop_lost_count_++;
    }
  }

  if (!obstacle_stop_active_ &&
      obstacle_stop_confirm_count_ >= std::max(1, cfg_.obstacle_stop_confirm_frames)) {
    obstacle_stop_active_ = true;
    obstacle_stop_state_ = "OBSTACLE_STOP";
  }

  if (obstacle_stop_active_ &&
      obstacle_stop_lost_count_ >= std::max(1, cfg_.obstacle_stop_lost_frames)) {
    obstacle_stop_active_ = false;
    obstacle_stop_state_ = "CLEAR";
    obstacle_stop_lost_count_ = 0;
  }

  stop_request_active_ = traffic_stop_active_ || finish_stop_active_ || obstacle_stop_active_;
}

void LaneDecision::updateStartBoostState(const std::vector<Detection>& detections, int image_height) {
  if (!cfg_.enable_start_boost_trigger || start_boost_used_) {
    start_boost_active_ = false;
    start_boost_lost_count_ = 0;
    return;
  }

  float y0 = image_height * clampValue(cfg_.start_boost_y0_ratio, 0.0f, 1.0f);
  float y1 = image_height * clampValue(cfg_.start_boost_y1_ratio, 0.0f, 1.0f);
  if (y1 < y0) {
    std::swap(y0, y1);
  }

  std::unordered_set<std::string> seen;
  for (const auto& det : detections) {
    if (!cfg_.start_boost_labels.count(det.class_name) ||
        det.confidence < cfg_.start_boost_min_confidence) {
      continue;
    }
    if (det.center.y < y0 || det.center.y > y1) {
      continue;
    }
    seen.insert(det.class_name);
  }

  bool all_seen = !cfg_.start_boost_labels.empty();
  for (const auto& label : cfg_.start_boost_labels) {
    if (!seen.count(label)) {
      all_seen = false;
      break;
    }
  }

  if (all_seen) {
    start_boost_active_ = true;
    start_boost_lost_count_ = 0;
    return;
  }

  if (start_boost_active_) {
    ++start_boost_lost_count_;
    if (start_boost_lost_count_ >= std::max(1, cfg_.start_boost_lost_frames)) {
      start_boost_active_ = false;
      start_boost_used_ = true;
      start_boost_lost_count_ = 0;
    }
  }
}

std::string LaneDecision::taskState() const {
  if (finish_stop_active_) {
    return "FINISH_STOP";
  }
  if (traffic_stop_active_) {
    return "TRAFFIC_STOP";
  }
  if (obstacle_stop_active_) {
    return "OBSTACLE_STOP";
  }
  if (start_boost_active_) {
    return "START_BOOST";
  }
  return "CLEAR";
}

void LaneDecision::populateDebugInfo(const std::vector<Band>& bands,
                                     const std::vector<ObstacleZone>& zones,
                                     const std::vector<cv::Point3f>& raw_points,
                                     const std::vector<cv::Point3f>& fit_points,
                                     const std::vector<double>& fit_coeffs) {
  debug_info_.bands.clear();
  debug_info_.obstacle_zones.clear();
  debug_info_.raw_points = raw_points;
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
