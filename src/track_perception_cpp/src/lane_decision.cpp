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
     << "\"offset_y07\":" << state.offset_y07 << ","
     << "\"offset_y08\":" << state.offset_y08 << ","
     << "\"offset_y09\":" << state.offset_y09 << ","
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
  cfg_.encoder_hold_counts = std::max<int64_t>(0, cfg_.encoder_hold_counts);
  cfg_.encoder_hold_right_counts = std::max<int64_t>(0, cfg_.encoder_hold_right_counts);
  cfg_.encoder_feedback_timeout_sec = std::max(0.0, cfg_.encoder_feedback_timeout_sec);
  cfg_.guideboard_hint_wait_timeout_sec =
      std::max(0.0, cfg_.guideboard_hint_wait_timeout_sec);
  cfg_.car_boundary_x_margin_px = std::max(0.0f, cfg_.car_boundary_x_margin_px);
  cfg_.car_boundary_y_margin_px = std::max(0.0f, cfg_.car_boundary_y_margin_px);
  cfg_.car_boundary_smoothing_alpha = clampValue(cfg_.car_boundary_smoothing_alpha, 0.0f, 1.0f);
  cfg_.car_boundary_lost_frames = std::max(1, cfg_.car_boundary_lost_frames);
  cfg_.car_fit_hold_timeout_sec = std::max(0.0, cfg_.car_fit_hold_timeout_sec);
  cfg_.car_push_expand_left_px = std::max(0.0f, cfg_.car_push_expand_left_px);
  cfg_.car_push_expand_bottom_px = std::max(0.0f, cfg_.car_push_expand_bottom_px);
  cfg_.car_push_expand_right_px = std::max(0.0f, cfg_.car_push_expand_right_px);
  cfg_.car_push_expand_top_px = std::max(0.0f, cfg_.car_push_expand_top_px);
  cfg_.car_push_clearance_px = std::max(0.0f, cfg_.car_push_clearance_px);
  cfg_.human_left_expand_px = std::max(0.0f, cfg_.human_left_expand_px);
  cfg_.human_left_expand_width_ratio = std::max(0.0f, cfg_.human_left_expand_width_ratio);
  cfg_.human_line_margin_px = std::max(0.0f, cfg_.human_line_margin_px);
  cfg_.human_stop_effective_area_ratio =
      clampValue(cfg_.human_stop_effective_area_ratio, 0.0f, 1.0f);
  cfg_.human_stop_confirm_frames = std::max(1, cfg_.human_stop_confirm_frames);
  cfg_.human_clear_confirm_frames = std::max(1, cfg_.human_clear_confirm_frames);
  cfg_.human_right_edge_pass_margin_px =
      std::max(0.0f, cfg_.human_right_edge_pass_margin_px);
  if (cfg_.guideboard_unknown_branch != "left" && cfg_.guideboard_unknown_branch != "right") {
    cfg_.guideboard_unknown_branch = cfg_.outer_side;
  }
  cfg_.offset_y07_ratio = clampValue(cfg_.offset_y07_ratio, 0.0f, 1.0f);
  cfg_.offset_y08_ratio = clampValue(cfg_.offset_y08_ratio, 0.0f, 1.0f);
  cfg_.offset_y09_ratio = clampValue(cfg_.offset_y09_ratio, 0.0f, 1.0f);
  cfg_.heading_y_ratio = clampValue(cfg_.heading_y_ratio, 0.0f, 1.0f);
  cfg_.centerline_kalman_measurement_noise =
      std::max(1e-9f, cfg_.centerline_kalman_measurement_noise);
  cfg_.centerline_kalman_idle_process_noise =
      std::max(0.0f, cfg_.centerline_kalman_idle_process_noise);
  cfg_.centerline_kalman_motion_process_noise =
      std::max(0.0f, cfg_.centerline_kalman_motion_process_noise);
  cfg_.centerline_kalman_turn_process_noise =
      std::max(0.0f, cfg_.centerline_kalman_turn_process_noise);
  cfg_.centerline_kalman_initial_variance =
      std::max(1e-9f, cfg_.centerline_kalman_initial_variance);
  cfg_.centerline_kalman_encoder_reference_counts =
      std::max(1.0f, cfg_.centerline_kalman_encoder_reference_counts);
  cfg_.centerline_kalman_reset_innovation =
      clampValue(cfg_.centerline_kalman_reset_innovation, 0.01f, 2.0f);
  cfg_.centerline_kalman_state_timeout_sec =
      std::max(0.0, cfg_.centerline_kalman_state_timeout_sec);
  cfg_.centerline_kalman_steering_timeout_sec =
      std::max(0.0, cfg_.centerline_kalman_steering_timeout_sec);
  last_offsets_.fill(0.0);
  resetCenterlineKalman();
  latest_steering_command_ = 0.0;
  last_steering_command_time_ = 0.0;
  has_steering_command_ = false;
  left_boundary_template_offsets_ = parseDoubleList(cfg_.left_boundary_template_offsets);
  right_boundary_template_offsets_ = parseDoubleList(cfg_.right_boundary_template_offsets);
  branch_locked_ = false;
  locked_branch_side_ = cfg_.outer_side;
  guideboard_branch_hint_ = cfg_.guideboard_unknown_branch;
  guideboard_branch_hint_source_ = "guideboard_hint";
  guideboard_branch_hint_valid_ = false;
  guideboard_hint_wait_start_sec_ = 0.0;
  branch_confirm_count_ = 0;
  guideboard_seen_latched_ = false;
  branch_event_armed_ = true;
  branch_event_rearmed_ = false;
  branch_clear_count_ = 0;
  branch_event_id_ = 1;
  encoder_hold_active_ = false;
  encoder_hold_baseline_valid_ = false;
  encoder_hold_delta_ = 0;
  encoder_hold_target_ = cfg_.encoder_hold_counts;
  car_boundary_active_ = false;
  car_left_x_ = -1.0;
  car_right_x_ = 0.0;
  car_top_y_ = 0.0;
  car_bottom_y_ = 0.0;
  car_boundary_lost_count_ = 0;
  fit_hold_active_ = false;
  fit_hold_age_ = 0.0;
  has_last_valid_fit_ = false;
  last_valid_fit_coeffs_.clear();
  last_valid_offsets_.fill(0.0);
  last_valid_raw_offsets_.fill(0.0);
  last_valid_fit_heading_ = 0.0;
  last_valid_fit_curvature_ = 0.0;
  last_valid_fit_confidence_ = 0.0;
  last_valid_fit_time_ = 0.0;
  last_valid_road_state_ = "NORMAL";
  human_left_seen_latched_ = false;
  human_stop_active_ = false;
  human_state_ = "NONE";
  human_right_clear_confirm_count_ = 0;
  human_stop_confirm_count_ = 0;
}

void LaneDecision::setGuideboardBranchHint(const std::string& branch, bool valid,
                                           const std::string& decision_source) {
  if (!valid) {
    guideboard_branch_hint_valid_ = false;
    return;
  }
  if (branch == "left" || branch == "right") {
    guideboard_branch_hint_ = branch;
    guideboard_branch_hint_source_ = decision_source.empty()
                                         ? "guideboard_hint"
                                         : decision_source;
    guideboard_branch_hint_valid_ = true;
  }
}

void LaneDecision::setEncoderCount(int64_t count, double timestamp) {
  latest_encoder_count_ = count;
  has_encoder_count_ = true;
  last_encoder_update_sec_ = timestamp;
}

void LaneDecision::setSteeringCommand(double steering_ratio, double timestamp) {
  latest_steering_command_ = clampValue(steering_ratio, -1.0, 1.0);
  last_steering_command_time_ = timestamp;
  has_steering_command_ = true;
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
  fit_hold_active_ = false;
  fit_hold_age_ = 0.0;
  std::array<double, 3> offsets{{0.0, 0.0, 0.0}};
  std::array<double, 3> raw_offsets{{0.0, 0.0, 0.0}};
  double heading_error = 0.0;
  double curvature = 0.0;
  double confidence = 0.0;
  std::string road_state = "LOW_CONFIDENCE";
  bool is_valid = false;
  std::vector<Band> bands;
  std::vector<cv::Point3f> raw_points;
  std::vector<cv::Point3f> fit_points;
  std::vector<cv::Point3f> removed_fit_points;
  std::vector<cv::Point3f> pushed_fit_points;
  std::vector<double> fit_coeffs;

  updateCarBoundaryState(detections, w, h);

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
    branch_event_rearmed_ = false;
    if (!branch_locked_ && !branch_event_armed_) {
      if (!branch_detected) {
        ++branch_clear_count_;
        if (branch_clear_count_ >= cfg_.branch_confirm_frames) {
          branch_event_armed_ = true;
          branch_event_rearmed_ = true;
          branch_clear_count_ = 0;
          guideboard_seen_latched_ = false;
          ++branch_event_id_;
        }
      } else {
        branch_clear_count_ = 0;
      }
    }
    if (!branch_locked_ && branch_event_armed_ && guideboard_seen) {
      guideboard_seen_latched_ = true;
    }
    debug_info_.branch_detected = branch_detected;
    debug_info_.branch_score = branch_score;
    debug_info_.guideboard_seen = guideboard_seen;
    debug_info_.guideboard_count = guideboard_count;
    debug_info_.guideboard_roi_count = guideboard_roi_count;
    debug_info_.guideboard_best_confidence = guideboard_best_confidence;
    debug_info_.guideboard_best_center = guideboard_best_center;
    debug_info_.guideboard_hint_valid = guideboard_branch_hint_valid_;
    debug_info_.guideboard_seen_latched = guideboard_seen_latched_;
    debug_info_.branch_event_armed = branch_event_armed_;
    debug_info_.branch_event_rearmed = branch_event_rearmed_;
    debug_info_.branch_event_id = branch_event_id_;
    if (!branch_locked_ && branch_event_armed_) {
      const bool hint_wait_active = guideboard_hint_wait_start_sec_ > 0.0;
      if (branch_detected || (guideboard_seen_latched_ && branch_score > 0) || hint_wait_active) {
        ++branch_confirm_count_;
      } else {
        branch_confirm_count_ = 0;
        guideboard_hint_wait_start_sec_ = 0.0;
      }
      if (branch_confirm_count_ >= cfg_.branch_confirm_frames) {
        const bool waiting_for_hint = cfg_.guideboard_require_hint &&
                                      guideboard_seen_latched_ &&
                                      !guideboard_branch_hint_valid_;
        if (waiting_for_hint && guideboard_hint_wait_start_sec_ <= 0.0) {
          guideboard_hint_wait_start_sec_ = current_time;
        }
        const double hint_wait_elapsed = waiting_for_hint
                                             ? std::max(0.0, current_time - guideboard_hint_wait_start_sec_)
                                             : 0.0;
        const bool hint_wait_timed_out = waiting_for_hint &&
                                         hint_wait_elapsed >= cfg_.guideboard_hint_wait_timeout_sec;
        debug_info_.guideboard_waiting_for_hint = waiting_for_hint && !hint_wait_timed_out;
        debug_info_.guideboard_hint_wait_elapsed = hint_wait_elapsed;
        if (waiting_for_hint && !hint_wait_timed_out) {
          branch_confirm_count_ = cfg_.branch_confirm_frames;
        } else {
          std::string target_branch = cfg_.guideboard_unknown_branch;
          std::string decision_source = "no_guideboard_default";
          if (guideboard_seen_latched_ && guideboard_branch_hint_valid_) {
            target_branch = guideboard_branch_hint_;
            decision_source = guideboard_branch_hint_source_;
          } else if (guideboard_seen_latched_ && cfg_.guideboard_require_hint) {
            target_branch = guideboard_branch_hint_valid_
                                ? guideboard_branch_hint_
                                : cfg_.guideboard_unknown_branch;
            decision_source = "guideboard_timeout";
          }
          if (target_branch != "left" && target_branch != "right") {
            target_branch = cfg_.outer_side;
          }

          const bool locked_with_guideboard = guideboard_seen_latched_;
          branch_locked_ = cfg_.enable_encoder_branch_hold;
          locked_branch_side_ = target_branch;
          encoder_hold_target_ = target_branch == "right"
                                     ? cfg_.encoder_hold_right_counts
                                     : cfg_.encoder_hold_counts;
          lock_start_time_ = current_time;
          branch_confirm_count_ = 0;
          branch_event_armed_ = false;
          branch_clear_count_ = 0;
          encoder_hold_active_ = branch_locked_;
          encoder_hold_baseline_valid_ = false;
          encoder_hold_delta_ = 0;
          guideboard_hint_wait_start_sec_ = 0.0;
          guideboard_seen_latched_ = false;
          guideboard_branch_hint_valid_ = false;
          debug_info_.branch_lock_event = true;
          debug_info_.branch_lock_guideboard = locked_with_guideboard;
          debug_info_.branch_decision_source = decision_source;
          debug_info_.branch_event_armed = false;
          if (has_encoder_count_) {
            encoder_hold_start_count_ = latest_encoder_count_;
            encoder_hold_baseline_valid_ = true;
          }
        }
      }
    } else if (branch_locked_) {
      guideboard_hint_wait_start_sec_ = 0.0;
    }

    if (branch_locked_ && encoder_hold_active_) {
      if (has_encoder_count_ && !encoder_hold_baseline_valid_) {
        encoder_hold_start_count_ = latest_encoder_count_;
        encoder_hold_baseline_valid_ = true;
      }
      if (has_encoder_count_ && encoder_hold_baseline_valid_) {
        encoder_hold_delta_ = latest_encoder_count_ - encoder_hold_start_count_;
        if (encoder_hold_delta_ >= encoder_hold_target_) {
          branch_locked_ = false;
          encoder_hold_active_ = false;
          encoder_hold_baseline_valid_ = false;
          encoder_hold_delta_ = 0;
          locked_branch_side_ = cfg_.outer_side;
        }
      }
    }

    std::string target_side = branch_locked_ ? locked_branch_side_ : cfg_.outer_side;
    double last_center_x = last_offsets_[2] * w / 2.0 + w / 2.0;
    raw_points = collectCenterlinePoints(bands, branch_locked_, target_side, last_center_x, w);
    RoadClass road_class = classifyRoadGeometry(bands, raw_points);
    road_state = roadClassName(road_class);
    bool template_trigger = branch_locked_;
    bool template_active = shouldUseBoundaryTemplate(target_side, template_trigger);
    if (template_active) {
      auto template_points = collectBoundaryTemplatePoints(bands, w, target_side, 0);
      if (static_cast<int>(template_points.size()) >= boundaryTemplateMinPoints(target_side)) {
        raw_points = std::move(template_points);
        road_state = "BRANCH";
        debug_info_.left_boundary_template_active = true;
        debug_info_.boundary_template_side = target_side;
        debug_info_.left_boundary_template_reason = "active";
      } else {
        debug_info_.left_boundary_template_reason = "few_points";
      }
    }
    int fit_order = (branch_locked_ || debug_info_.left_boundary_template_active) ? cfg_.branch_fit_order : cfg_.fit_order;
    fit_points = raw_points;
    if (cfg_.enable_car_point_push_avoidance) {
      fit_points = applyCarPointPushAvoidance(
          fit_points, w, h, &pushed_fit_points, &removed_fit_points);
    } else {
      fit_points = filterCarRightBoundaryPoints(fit_points, &removed_fit_points);
    }
    fit_points = filterCenterlinePoints(fit_points, w, last_center_x);
    fit_points = filterCenterlinePointsKalman(
        fit_points, bands, w, h, current_time, target_side, template_active);

    const bool fit_success = static_cast<int>(fit_points.size()) >= cfg_.fit_min_points &&
                             fitCenterlineAndComputeGeometry(fit_points, h, fit_order, &fit_coeffs,
                                                             &heading_error, &curvature);
    if (fit_success) {
      const std::array<float, 3> ratios{{
          cfg_.offset_y07_ratio, cfg_.offset_y08_ratio, cfg_.offset_y09_ratio}};
      for (size_t i = 0; i < ratios.size(); ++i) {
        raw_offsets[i] = offsetAtY(fit_coeffs, h * ratios[i], w);
        offsets[i] = raw_offsets[i];
        last_offsets_[i] = offsets[i];
      }
      debug_info_.offset_y07 = static_cast<float>(offsets[0]);
      debug_info_.offset_y08 = static_cast<float>(offsets[1]);
      debug_info_.offset_y09 = static_cast<float>(offsets[2]);
      debug_info_.raw_offset_y07 = static_cast<float>(raw_offsets[0]);
      debug_info_.raw_offset_y08 = static_cast<float>(raw_offsets[1]);
      debug_info_.raw_offset_y09 = static_cast<float>(raw_offsets[2]);
      debug_info_.image_width = w;
      is_valid = true;
      confidence = calculateLaneConfidence(fit_points, bands);
      has_last_valid_fit_ = true;
      last_valid_fit_coeffs_ = fit_coeffs;
      last_valid_offsets_ = offsets;
      last_valid_raw_offsets_ = raw_offsets;
      last_valid_fit_heading_ = heading_error;
      last_valid_fit_curvature_ = curvature;
      last_valid_fit_confidence_ = confidence;
      last_valid_fit_time_ = current_time;
      last_valid_road_state_ = road_state;
    } else {
      fit_hold_age_ = has_last_valid_fit_ ? std::max(0.0, current_time - last_valid_fit_time_) : 0.0;
      if (has_last_valid_fit_ && fit_hold_age_ <= cfg_.car_fit_hold_timeout_sec) {
        fit_hold_active_ = true;
        fit_coeffs = last_valid_fit_coeffs_;
        offsets = last_valid_offsets_;
        raw_offsets = last_valid_raw_offsets_;
        heading_error = last_valid_fit_heading_;
        curvature = last_valid_fit_curvature_;
        confidence = last_valid_fit_confidence_;
        road_state = last_valid_road_state_;
        is_valid = true;
      } else {
        cv::Mat bottom_seg = seg_map(cv::Range(static_cast<int>(h * 0.8), h), cv::Range::all());
        const double fallback_offset = fallbackCenterOffset(bottom_seg);
        offsets.fill(fallback_offset);
        raw_offsets.fill(fallback_offset);
        const bool has_fallback_pixels = cv::countNonZero(bottom_seg == 1) > 0;
        is_valid = false;
        confidence = 0.0;
        if (!has_fallback_pixels && std::abs(last_offsets_[2]) > 0.01) {
          offsets.fill(last_offsets_[2]);
          raw_offsets.fill(last_offsets_[2]);
        }
        road_state = "LOW_CONFIDENCE";
        fit_coeffs.clear();
      }
      debug_info_.offset_y07 = static_cast<float>(offsets[0]);
      debug_info_.offset_y08 = static_cast<float>(offsets[1]);
      debug_info_.offset_y09 = static_cast<float>(offsets[2]);
      debug_info_.raw_offset_y07 = static_cast<float>(raw_offsets[0]);
      debug_info_.raw_offset_y08 = static_cast<float>(raw_offsets[1]);
      debug_info_.raw_offset_y09 = static_cast<float>(raw_offsets[2]);
      debug_info_.image_width = w;
    }
    populateDebugInfo(
      bands, getActiveObstacleZones(detections, w, h), raw_points, fit_points,
      removed_fit_points, pushed_fit_points, w, h, fit_coeffs);
  } else {
    cv::Mat bottom_seg = seg_map(cv::Range(h / 2, h), cv::Range::all());
    offsets.fill(fallbackCenterOffset(bottom_seg));
    raw_offsets = offsets;
    debug_info_.offset_y07 = static_cast<float>(offsets[0]);
    debug_info_.offset_y08 = static_cast<float>(offsets[1]);
    debug_info_.offset_y09 = static_cast<float>(offsets[2]);
    debug_info_.raw_offset_y07 = static_cast<float>(raw_offsets[0]);
    debug_info_.raw_offset_y08 = static_cast<float>(raw_offsets[1]);
    debug_info_.raw_offset_y09 = static_cast<float>(raw_offsets[2]);
    debug_info_.image_width = w;
    is_valid = false;
    confidence = 0.0;
    road_state = "LOW_CONFIDENCE";
  }

  updateFinishStopState(detections, h);
  updateHumanStopState(detections, w, h, fit_coeffs, is_valid);

  state.offset_y07 = static_cast<float>(offsets[0]);
  state.offset_y08 = static_cast<float>(offsets[1]);
  state.offset_y09 = static_cast<float>(offsets[2]);
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
    // Human is handled after fitting by the side-memory state machine. It must
    // never remove road-mask segments or fit points through the legacy path.
    if (det.class_name == "Human") {
      continue;
    }
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

LaneDecision::RoadClass LaneDecision::classifyRoadGeometry(
    const std::vector<Band>& bands, const std::vector<cv::Point3f>& raw_points) const {
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
  return RoadClass::Normal;
}

std::string LaneDecision::roadClassName(RoadClass road_class) const {
  switch (road_class) {
    case RoadClass::Branch:
      return "BRANCH";
    case RoadClass::LowConfidence:
      return "LOW_CONFIDENCE";
    case RoadClass::Normal:
    default:
      return "NORMAL";
  }
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
    std::optional<double> last_center_x, int image_width) {
  std::vector<cv::Point3f> points;
  for (auto& band : bands) {
    if (band.segments.empty()) {
      continue;
    }
    std::optional<Segment> target;
    if (branch_locked) {
      if (band.segments.size() >= 2) {
        target = chooseTargetSegment(band, side);
      } else {
        target = chooseTargetSegment(band, side);
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

void LaneDecision::resetCenterlineKalman() {
  centerline_kalman_states_.assign(
      static_cast<size_t>(std::max(1, cfg_.band_count)), CenterlineKalmanState{});
  centerline_kalman_context_valid_ = false;
  centerline_kalman_image_width_ = 0;
  centerline_kalman_image_height_ = 0;
  centerline_kalman_branch_locked_ = false;
  centerline_kalman_template_active_ = false;
  centerline_kalman_target_side_.clear();
  centerline_kalman_encoder_baseline_valid_ = false;
  centerline_kalman_last_encoder_count_ = 0;
}

std::vector<cv::Point3f> LaneDecision::filterCenterlinePointsKalman(
    const std::vector<cv::Point3f>& points, const std::vector<Band>& bands,
    int image_width, int image_height, double timestamp,
    const std::string& target_side, bool template_active) {
  debug_info_.centerline_kalman_enabled = cfg_.enable_centerline_kalman;
  debug_info_.centerline_kalman_point_count = 0;
  debug_info_.centerline_kalman_encoder_delta = 0;
  debug_info_.centerline_kalman_motion_ratio = 0.0;
  debug_info_.centerline_kalman_process_noise = 0.0;
  debug_info_.centerline_kalman_steering = 0.0;
  debug_info_.centerline_kalman_reset_count = 0;

  if (!cfg_.enable_centerline_kalman || image_width <= 1 || image_height <= 1) {
    return points;
  }

  const bool context_changed =
      !centerline_kalman_context_valid_ ||
      centerline_kalman_image_width_ != image_width ||
      centerline_kalman_image_height_ != image_height ||
      centerline_kalman_branch_locked_ != branch_locked_ ||
      centerline_kalman_template_active_ != template_active ||
      centerline_kalman_target_side_ != target_side ||
      centerline_kalman_states_.size() != static_cast<size_t>(cfg_.band_count);
  if (context_changed) {
    resetCenterlineKalman();
    centerline_kalman_context_valid_ = true;
    centerline_kalman_image_width_ = image_width;
    centerline_kalman_image_height_ = image_height;
    centerline_kalman_branch_locked_ = branch_locked_;
    centerline_kalman_template_active_ = template_active;
    centerline_kalman_target_side_ = target_side;
    ++debug_info_.centerline_kalman_reset_count;
  }

  int64_t encoder_delta = 0;
  const bool encoder_fresh = has_encoder_count_ &&
      timestamp - last_encoder_update_sec_ <= cfg_.encoder_feedback_timeout_sec;
  if (encoder_fresh) {
    if (centerline_kalman_encoder_baseline_valid_) {
      const long double difference =
          static_cast<long double>(latest_encoder_count_) -
          static_cast<long double>(centerline_kalman_last_encoder_count_);
      const long double restart_limit =
          static_cast<long double>(cfg_.centerline_kalman_encoder_reference_counts) * 1000.0L;
      if (std::abs(difference) <= restart_limit) {
        encoder_delta = static_cast<int64_t>(difference);
      } else {
        // A chassis-node restart resets the accumulated counter. Ignore that
        // discontinuity instead of turning it into a large process noise.
        centerline_kalman_encoder_baseline_valid_ = false;
      }
    }
    centerline_kalman_last_encoder_count_ = latest_encoder_count_;
    centerline_kalman_encoder_baseline_valid_ = true;
  }

  const double motion_ratio = std::clamp(
      std::abs(static_cast<double>(encoder_delta)) /
          static_cast<double>(cfg_.centerline_kalman_encoder_reference_counts),
      0.0, 3.0);
  const bool steering_fresh = has_steering_command_ &&
      timestamp - last_steering_command_time_ <= cfg_.centerline_kalman_steering_timeout_sec;
  const double steering = steering_fresh ? std::abs(latest_steering_command_) : 0.0;
  const double process_noise =
      static_cast<double>(cfg_.centerline_kalman_idle_process_noise) +
      static_cast<double>(cfg_.centerline_kalman_motion_process_noise) * motion_ratio +
      static_cast<double>(cfg_.centerline_kalman_turn_process_noise) * motion_ratio * steering;

  debug_info_.centerline_kalman_encoder_delta = encoder_delta;
  debug_info_.centerline_kalman_motion_ratio = motion_ratio;
  debug_info_.centerline_kalman_process_noise = process_noise;
  debug_info_.centerline_kalman_steering = steering;

  std::vector<cv::Point3f> filtered = points;
  for (auto& point : filtered) {
    const auto nearest_band = std::min_element(
        bands.begin(), bands.end(), [&](const Band& a, const Band& b) {
          return std::abs(a.y_center - point.y) < std::abs(b.y_center - point.y);
        });
    if (nearest_band == bands.end() || nearest_band->index < 0 ||
        nearest_band->index >= static_cast<int>(centerline_kalman_states_.size())) {
      continue;
    }

    auto& filter = centerline_kalman_states_[static_cast<size_t>(nearest_band->index)];
    const double measurement = clampValue(
        (static_cast<double>(point.x) - image_width * 0.5) / (image_width * 0.5),
        -1.0, 1.0);
    const double age = filter.initialized ? timestamp - filter.last_update_time : 0.0;
    const bool expired = filter.initialized &&
        cfg_.centerline_kalman_state_timeout_sec > 0.0 &&
        age > cfg_.centerline_kalman_state_timeout_sec;

    if (!filter.initialized || expired) {
      filter.initialized = true;
      filter.x_normalized = measurement;
      filter.variance = cfg_.centerline_kalman_initial_variance;
      if (expired) {
        ++debug_info_.centerline_kalman_reset_count;
      }
    } else {
      const double time_scale = std::clamp(age / 0.05, 0.5, 6.0);
      const double predicted_variance = filter.variance + process_noise * time_scale;
      const double innovation = measurement - filter.x_normalized;
      if (std::abs(innovation) > cfg_.centerline_kalman_reset_innovation) {
        // A large, spatially valid jump is more likely to be a genuine bend or
        // route transition than usable history. Reinitialize without lag.
        filter.x_normalized = measurement;
        filter.variance = cfg_.centerline_kalman_initial_variance;
        ++debug_info_.centerline_kalman_reset_count;
      } else {
        const double measurement_noise =
            static_cast<double>(cfg_.centerline_kalman_measurement_noise) /
            clampValue(static_cast<double>(point.z), 0.25, 4.0);
        const double gain = predicted_variance / (predicted_variance + measurement_noise);
        filter.x_normalized += gain * innovation;
        filter.variance = std::max(1e-12, (1.0 - gain) * predicted_variance);
      }
    }

    filter.x_normalized = clampValue(filter.x_normalized, -1.0, 1.0);
    filter.last_update_time = timestamp;
    point.x = static_cast<float>(clampValue(
        (filter.x_normalized + 1.0) * image_width * 0.5,
        0.0, static_cast<double>(image_width - 1)));
    ++debug_info_.centerline_kalman_point_count;
  }
  return filtered;
}

void LaneDecision::updateCarBoundaryState(const std::vector<Detection>& detections,
                                           int image_width, int image_height) {
  if (!cfg_.enable_car_right_boundary_filter && !cfg_.enable_car_point_push_avoidance) {
    car_boundary_active_ = false;
    car_left_x_ = -1.0;
    car_right_x_ = 0.0;
    car_top_y_ = 0.0;
    car_bottom_y_ = 0.0;
    car_boundary_lost_count_ = 0;
    return;
  }

  const double min_bottom_y = image_height *
                              clampValue(cfg_.obstacle_min_bottom_y_ratio, 0.0f, 1.0f);
  bool car_seen = false;
  double detected_left_x = std::numeric_limits<double>::max();
  double detected_right_x = 0.0;
  double detected_top_y = 0.0;
  double detected_bottom_y = 0.0;
  for (const auto& det : detections) {
    if (det.class_name != "Car" || det.confidence < cfg_.obstacle_min_confidence) {
      continue;
    }
    const double bbox_left = det.bbox.x;
    const double bbox_bottom = det.bbox.y + det.bbox.height;
    if (bbox_bottom < min_bottom_y || det.bbox.width <= 0.0f ||
        det.bbox.height <= 0.0f) {
      continue;
    }
    if (bbox_left < detected_left_x) {
      detected_left_x = bbox_left;
      detected_right_x = det.bbox.x + det.bbox.width;
      detected_top_y = det.bbox.y;
      detected_bottom_y = bbox_bottom;
      car_seen = true;
    }
  }

  if (car_seen) {
    const double alpha = clampValue(static_cast<double>(cfg_.car_boundary_smoothing_alpha), 0.0, 1.0);
    if (!car_boundary_active_ || car_left_x_ < 0.0) {
      car_left_x_ = detected_left_x;
      car_right_x_ = detected_right_x;
      car_top_y_ = detected_top_y;
      car_bottom_y_ = detected_bottom_y;
    } else {
      car_left_x_ = alpha * detected_left_x + (1.0 - alpha) * car_left_x_;
      car_right_x_ = alpha * detected_right_x + (1.0 - alpha) * car_right_x_;
      car_top_y_ = alpha * detected_top_y + (1.0 - alpha) * car_top_y_;
      car_bottom_y_ = alpha * detected_bottom_y + (1.0 - alpha) * car_bottom_y_;
    }
    car_left_x_ = clampValue(car_left_x_, 0.0, static_cast<double>(std::max(0, image_width - 1)));
    car_right_x_ = clampValue(car_right_x_, 0.0, static_cast<double>(std::max(0, image_width - 1)));
    car_top_y_ = clampValue(car_top_y_, 0.0, static_cast<double>(std::max(0, image_height - 1)));
    car_bottom_y_ = clampValue(car_bottom_y_, 0.0, static_cast<double>(std::max(0, image_height - 1)));
    if (car_right_x_ < car_left_x_) {
      car_right_x_ = car_left_x_;
    }
    if (car_bottom_y_ < car_top_y_) {
      car_bottom_y_ = car_top_y_;
    }
    car_boundary_active_ = true;
    car_boundary_lost_count_ = 0;
    return;
  }

  if (car_boundary_active_) {
    ++car_boundary_lost_count_;
    if (car_boundary_lost_count_ >= std::max(1, cfg_.car_boundary_lost_frames)) {
      car_boundary_active_ = false;
      car_left_x_ = -1.0;
      car_right_x_ = 0.0;
      car_top_y_ = 0.0;
      car_bottom_y_ = 0.0;
    }
  }
}

std::vector<cv::Point3f> LaneDecision::filterCarRightBoundaryPoints(
    const std::vector<cv::Point3f>& points,
    std::vector<cv::Point3f>* removed_points) const {
  if (removed_points) {
    removed_points->clear();
  }
  if (!car_boundary_active_ || car_left_x_ < 0.0) {
    return points;
  }

  const double car_left_limit = car_left_x_ - cfg_.car_boundary_x_margin_px;
  const double car_bottom_limit = car_bottom_y_ + cfg_.car_boundary_y_margin_px;
  std::vector<cv::Point3f> filtered;
  filtered.reserve(points.size());
  for (const auto& point : points) {
    const bool on_car_side = point.y <= car_bottom_limit && point.x > car_left_limit;
    if (on_car_side) {
      if (removed_points) {
        removed_points->push_back(point);
      }
    } else {
      filtered.push_back(point);
    }
  }
  return filtered;
}

std::vector<cv::Point3f> LaneDecision::applyCarPointPushAvoidance(
    const std::vector<cv::Point3f>& points, int image_width, int image_height,
    std::vector<cv::Point3f>* pushed_points,
    std::vector<cv::Point3f>* removed_points) {
  if (pushed_points) {
    pushed_points->clear();
  }
  if (removed_points) {
    removed_points->clear();
  }
  if (!car_boundary_active_ || car_left_x_ < 0.0 || image_width <= 0 || image_height <= 0) {
    return points;
  }

  const double max_x = static_cast<double>(image_width - 1);
  const double max_y = static_cast<double>(image_height - 1);
  const double original_left = clampValue(car_left_x_, 0.0, max_x);
  const double original_right = clampValue(car_right_x_, 0.0, max_x);
  const double original_top = clampValue(car_top_y_, 0.0, max_y);
  const double original_bottom = clampValue(car_bottom_y_, 0.0, max_y);
  const double expanded_left = clampValue(
      original_left - cfg_.car_push_expand_left_px, 0.0, max_x);
  const double expanded_right = clampValue(
      original_right + cfg_.car_push_expand_right_px, 0.0, max_x);
  const double expanded_top = clampValue(
      original_top - cfg_.car_push_expand_top_px, 0.0, max_y);
  const double expanded_bottom = clampValue(
      original_bottom + cfg_.car_push_expand_bottom_px, 0.0, max_y);
  const double push_target_x = clampValue(
      expanded_left - cfg_.car_push_clearance_px, 0.0, max_x);

  const auto inside = [](const cv::Point3f& point, double left, double right,
                         double top, double bottom) {
    return static_cast<double>(point.x) >= left &&
           static_cast<double>(point.x) <= right &&
           static_cast<double>(point.y) >= top &&
           static_cast<double>(point.y) <= bottom;
  };

  std::vector<cv::Point3f> filtered;
  filtered.reserve(points.size());
  for (const auto& point : points) {
    if (inside(point, expanded_left, expanded_right, expanded_top, expanded_bottom)) {
      cv::Point3f pushed = point;
      pushed.x = static_cast<float>(push_target_x);
      filtered.push_back(pushed);
      if (pushed_points) {
        pushed_points->push_back(pushed);
      }
      continue;
    }

    const bool outside_original_box =
        !inside(point, original_left, original_right, original_top, original_bottom);
    if (static_cast<double>(point.x) > original_left && outside_original_box) {
      if (removed_points) {
        removed_points->push_back(point);
      }
      continue;
    }
    filtered.push_back(point);
  }
  return filtered;
}

bool LaneDecision::fitCenterlineAndComputeGeometry(const std::vector<cv::Point3f>& points,
                                                   int h, int fit_order,
                                                   std::vector<double>* coeffs,
                                                   double* heading_error,
                                                   double* curvature) const {
  if (static_cast<int>(points.size()) < cfg_.fit_min_points || static_cast<int>(points.size()) <= fit_order) {
    return false;
  }
  if (!weightedPolyfit(points, fit_order, coeffs)) {
    return false;
  }
  const double heading_y = h * cfg_.heading_y_ratio;
  const double curvature_y = h * cfg_.offset_y09_ratio;
  double heading = 0.0;
  double curv = 0.0;
  if (coeffs->size() > 1) {
    // Estimate heading from local line fits at near/mid/far lookahead
    // positions.  Fitting each window independently prevents the weighted
    // result from being mathematically equivalent to one derivative sample
    // on the global quadratic fit.
    const std::array<double, 3> sample_ratios{{
        clampValue(static_cast<double>(cfg_.heading_near_ratio), 0.0, 1.0),
        clampValue(static_cast<double>(cfg_.heading_mid_ratio), 0.0, 1.0),
        clampValue(static_cast<double>(cfg_.heading_far_ratio), 0.0, 1.0)}};
    const std::array<double, 3> sample_weights{{
        std::max(0.0, static_cast<double>(cfg_.heading_near_weight)),
        std::max(0.0, static_cast<double>(cfg_.heading_mid_weight)),
        std::max(0.0, static_cast<double>(cfg_.heading_far_weight))}};
    const double half_window = std::max(
        1.0, clampValue(static_cast<double>(cfg_.heading_local_window_half_ratio), 0.01, 0.25) * h);
    const int min_local_points = std::max(2, cfg_.heading_local_min_points);
    double heading_sum = 0.0;
    double weight_sum = 0.0;

    for (size_t i = 0; i < sample_ratios.size(); ++i) {
      if (sample_weights[i] <= 0.0) {
        continue;
      }
      const double sample_y = h * sample_ratios[i];
      std::vector<cv::Point3f> local_points;
      local_points.reserve(points.size());
      for (const auto& point : points) {
        if (std::abs(static_cast<double>(point.y) - sample_y) <= half_window) {
          // Center y around the sample to keep the local linear solve well
          // conditioned even when image coordinates are large.
          local_points.emplace_back(
              point.x, static_cast<float>(static_cast<double>(point.y) - sample_y), point.z);
        }
      }
      if (static_cast<int>(local_points.size()) < min_local_points) {
        continue;
      }
      std::vector<double> local_coeffs;
      if (!weightedPolyfit(local_points, 1, &local_coeffs) || local_coeffs.size() < 2) {
        continue;
      }
      const double local_slope = local_coeffs[0];
      if (!std::isfinite(local_slope)) {
        continue;
      }
      const double local_heading = std::atan(local_slope) / (M_PI / 2.0);
      heading_sum += sample_weights[i] * local_heading;
      weight_sum += sample_weights[i];
    }

    if (weight_sum > 1e-9) {
      heading = heading_sum / weight_sum;
    } else {
      // Preserve a safe fallback when a frame has too few points in every
      // local window.
      auto deriv = polyDeriv(*coeffs);
      const double dx_dy = evalPoly(deriv, heading_y);
      heading = std::atan(dx_dy) / (M_PI / 2.0);
    }
    if (coeffs->size() > 2) {
      auto second = polyDeriv(*coeffs, 2);
      curv = clampValue(evalPoly(second, curvature_y) * h, -1.0, 1.0);
    }
  }
  *heading_error = clampValue(heading, -1.0, 1.0);
  *curvature = curv;
  return true;
}

double LaneDecision::offsetAtY(const std::vector<double>& coeffs, double y, int image_width) const {
  if (image_width <= 1) {
    return 0.0;
  }
  const double x = evalPoly(coeffs, y);
  return clampValue((x - image_width / 2.0) / (image_width / 2.0), -1.0, 1.0);
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

void LaneDecision::updateFinishStopState(const std::vector<Detection>& detections, int image_height) {
  if (!cfg_.enable_finish_stop) {
    finish_stop_active_ = false;
    finish_stop_state_ = "CLEAR";
    finish_stop_lost_count_ = 0;
    return;
  }
  if (finish_stop_active_) {
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
}

void LaneDecision::updateHumanStopState(const std::vector<Detection>& detections,
                                         int image_width, int image_height,
                                         const std::vector<double>& fit_coeffs,
                                         bool fit_valid) {
  debug_info_.humans.clear();
  debug_info_.human_left_seen_latched = human_left_seen_latched_;
  debug_info_.human_passable = false;
  debug_info_.human_right_clear_confirming = false;
  debug_info_.human_stop_candidate = false;
  debug_info_.human_effective_left_x = -1.0f;
  debug_info_.human_fit_line_limit_x = -1.0f;
  debug_info_.human_effective_area_ratio = 0.0f;
  debug_info_.human_right_edge_passable = false;
  debug_info_.human_right_edge_limit_x = -1.0f;
  debug_info_.human_right_edge_pass_count = 0;
  debug_info_.human_state = human_state_;
  debug_info_.human_right_clear_confirm_count = human_right_clear_confirm_count_;
  debug_info_.human_stop_confirm_count = human_stop_confirm_count_;

  if (!cfg_.enable_human_obstacle_stop || image_width <= 0 || image_height <= 0) {
    human_left_seen_latched_ = false;
    human_stop_active_ = false;
    human_state_ = "NONE";
    human_right_clear_confirm_count_ = 0;
    human_stop_confirm_count_ = 0;
    debug_info_.human_left_seen_latched = false;
    debug_info_.human_state = human_state_;
    return;
  }

  const bool fit_available = fit_valid && fit_coeffs.size() >= 2;
  const double area_threshold = clampValue(
      static_cast<double>(cfg_.human_stop_effective_area_ratio), 0.0, 1.0);
  const double image_area = static_cast<double>(image_width) * image_height;
  const double right_edge_limit_x = cfg_.enable_human_right_edge_pass
                                        ? clampValue(
                                            static_cast<double>(image_width) -
                                                cfg_.human_right_edge_pass_margin_px,
                                            0.0, static_cast<double>(image_width))
                                        : -1.0;

  bool has_valid_human = false;
  bool all_current_humans_passable = true;
  bool any_nonpassable_human = false;
  bool any_stop_candidate = false;
  bool any_right_edge_passable = false;
  bool any_no_fit_human = false;
  int right_edge_pass_count = 0;
  bool have_focus = false;
  double focus_gap = std::numeric_limits<double>::infinity();

  for (const auto& det : detections) {
    if (det.class_name != "Human" ||
        det.confidence < cfg_.obstacle_min_confidence ||
        det.bbox.width <= 0.0f || det.bbox.height <= 0.0f) {
      continue;
    }

    const double expand_px = std::max(
        static_cast<double>(cfg_.human_left_expand_px),
        static_cast<double>(det.bbox.width) * cfg_.human_left_expand_width_ratio);
    const double raw_left = det.bbox.x;
    const double raw_top = det.bbox.y;
    const double raw_right = det.bbox.x + det.bbox.width;
    const double raw_bottom = det.bbox.y + det.bbox.height;
    const double effective_left = clampValue(raw_left - expand_px, 0.0,
                                             static_cast<double>(image_width));
    const double effective_right = clampValue(raw_right, 0.0,
                                              static_cast<double>(image_width));
    const double effective_top = clampValue(raw_top, 0.0,
                                             static_cast<double>(image_height));
    const double effective_bottom = clampValue(raw_bottom, 0.0,
                                               static_cast<double>(image_height));
    if (effective_right <= effective_left || effective_bottom <= effective_top) {
      continue;
    }

    has_valid_human = true;
    const double effective_width = effective_right - effective_left;
    const double effective_height = effective_bottom - effective_top;
    const double effective_area_ratio = image_area > 0.0
                                            ? effective_width * effective_height / image_area
                                            : 0.0;

    LaneHumanDebug human_debug;
    human_debug.raw_bbox = det.bbox;
    human_debug.effective_bbox = cv::Rect2f(
        static_cast<float>(effective_left), static_cast<float>(effective_top),
        static_cast<float>(effective_width), static_cast<float>(effective_height));
    human_debug.effective_area_ratio = static_cast<float>(effective_area_ratio);
    human_debug.fit_available = fit_available;
    const bool right_edge_passable = right_edge_limit_x >= 0.0 &&
                                     effective_right >= right_edge_limit_x;
    human_debug.right_edge_passable = right_edge_passable;

    // Compare the midpoint of the expanded box's left edge with the fitted
    // point at exactly the same image height.  This avoids using a farther
    // or nearer point on a curved fit to decide whether Human is passable.
    const double effective_center_y = effective_top + effective_height * 0.5;
    double line_limit_x = -std::numeric_limits<double>::infinity();
    bool line_samples_valid = fit_available;
    if (fit_available) {
      line_limit_x = evalPoly(fit_coeffs, effective_center_y);
      if (!std::isfinite(line_limit_x)) {
        line_samples_valid = false;
      }
    }
    human_debug.fit_available = line_samples_valid;
    human_debug.fit_line_limit_x = line_samples_valid
                                       ? static_cast<float>(line_limit_x)
                                       : -1.0f;

    const bool passable = right_edge_passable ||
                          (line_samples_valid &&
                           effective_left > line_limit_x + cfg_.human_line_margin_px);
    const bool stop_candidate = !passable && effective_area_ratio >= area_threshold;
    human_debug.passable = passable;
    human_debug.stop_candidate = stop_candidate;
    debug_info_.humans.push_back(human_debug);

    all_current_humans_passable = all_current_humans_passable && passable;
    any_nonpassable_human = any_nonpassable_human || !passable;
    any_stop_candidate = any_stop_candidate || stop_candidate;
    any_right_edge_passable = any_right_edge_passable || right_edge_passable;
    if (right_edge_passable) {
      ++right_edge_pass_count;
    }
    any_no_fit_human = any_no_fit_human || !line_samples_valid;

    const double gap = line_samples_valid
                           ? effective_left - line_limit_x
                           : -std::numeric_limits<double>::infinity();
    if (!have_focus || gap < focus_gap) {
      have_focus = true;
      focus_gap = gap;
      debug_info_.human_effective_left_x = static_cast<float>(effective_left);
      debug_info_.human_fit_line_limit_x = line_samples_valid
                                               ? static_cast<float>(line_limit_x)
                                               : -1.0f;
      debug_info_.human_effective_area_ratio = static_cast<float>(effective_area_ratio);
    }
  }

  if (any_nonpassable_human) {
    human_left_seen_latched_ = true;
    human_right_clear_confirm_count_ = 0;
  } else if (has_valid_human && all_current_humans_passable) {
    if (human_left_seen_latched_ && any_right_edge_passable) {
      human_left_seen_latched_ = false;
      human_right_clear_confirm_count_ = 0;
    } else if (human_left_seen_latched_) {
      ++human_right_clear_confirm_count_;
      if (human_right_clear_confirm_count_ >= cfg_.human_clear_confirm_frames) {
        human_left_seen_latched_ = false;
        human_right_clear_confirm_count_ = 0;
      }
    }
  } else {
    // No detection is never evidence that a previously left-side Human is safe.
    human_right_clear_confirm_count_ = 0;
  }

  if (any_stop_candidate) {
    ++human_stop_confirm_count_;
  } else if (!human_stop_active_) {
    human_stop_confirm_count_ = 0;
  }
  if (!human_stop_active_ &&
      human_stop_confirm_count_ >= cfg_.human_stop_confirm_frames) {
    human_stop_active_ = true;
  }

  // A stop can only be released by stable right-side visual evidence. A lost
  // detection, including a lost detection after stopping, never releases it.
  if (human_stop_active_ && !human_left_seen_latched_ && has_valid_human &&
      all_current_humans_passable) {
    human_stop_active_ = false;
    human_stop_confirm_count_ = 0;
  }

  if (human_stop_active_) {
    human_state_ = "OBSTACLE_STOP";
  } else if (human_left_seen_latched_) {
    human_state_ = any_no_fit_human ? "NO_FIT_LATCHED" : "LEFT_LATCHED_WAIT_AREA";
  } else if (has_valid_human && all_current_humans_passable) {
    human_state_ = "PASSABLE";
  } else {
    human_state_ = "NONE";
  }

  debug_info_.human_left_seen_latched = human_left_seen_latched_;
  debug_info_.human_passable = has_valid_human && all_current_humans_passable &&
                               !human_left_seen_latched_;
  debug_info_.human_right_clear_confirming = human_left_seen_latched_ &&
                                             has_valid_human &&
                                             all_current_humans_passable;
  debug_info_.human_stop_candidate = any_stop_candidate;
  debug_info_.human_right_edge_passable = any_right_edge_passable;
  debug_info_.human_right_edge_limit_x = static_cast<float>(right_edge_limit_x);
  debug_info_.human_right_edge_pass_count = right_edge_pass_count;
  debug_info_.human_state = human_state_;
  debug_info_.human_right_clear_confirm_count = human_right_clear_confirm_count_;
  debug_info_.human_stop_confirm_count = human_stop_confirm_count_;
}

std::string LaneDecision::taskState() const {
  if (finish_stop_active_) {
    return "FINISH_STOP";
  }
  if (human_stop_active_) {
    return "OBSTACLE_STOP";
  }
  return "CLEAR";
}

void LaneDecision::populateDebugInfo(const std::vector<Band>& bands,
                                     const std::vector<ObstacleZone>& zones,
                                     const std::vector<cv::Point3f>& raw_points,
                                     const std::vector<cv::Point3f>& fit_points,
                                     const std::vector<cv::Point3f>& removed_fit_points,
                                     const std::vector<cv::Point3f>& pushed_fit_points,
                                     int image_width, int image_height,
                                     const std::vector<double>& fit_coeffs) {
  debug_info_.bands.clear();
  debug_info_.obstacle_zones.clear();
  debug_info_.raw_points = raw_points;
  debug_info_.fit_points = fit_points;
  debug_info_.removed_fit_points = removed_fit_points;
  debug_info_.pushed_fit_points = pushed_fit_points;
  debug_info_.fit_coeffs = fit_coeffs;
  debug_info_.car_boundary_active = car_boundary_active_;
  debug_info_.car_left_x = static_cast<float>(car_left_x_);
  debug_info_.car_filtered_point_count = static_cast<int>(removed_fit_points.size());
  debug_info_.car_boundary_lost_count = car_boundary_lost_count_;
  debug_info_.car_push_active = cfg_.enable_car_point_push_avoidance && car_boundary_active_;
  debug_info_.car_pushed_point_count = static_cast<int>(pushed_fit_points.size());
  debug_info_.car_deleted_point_count = static_cast<int>(removed_fit_points.size());
  if (car_boundary_active_) {
    const double max_x = static_cast<double>(std::max(0, image_width - 1));
    const double max_y = static_cast<double>(std::max(0, image_height - 1));
    const double bbox_left = clampValue(car_left_x_, 0.0, max_x);
    const double bbox_right = clampValue(car_right_x_, 0.0, max_x);
    const double bbox_top = clampValue(car_top_y_, 0.0, max_y);
    const double bbox_bottom = clampValue(car_bottom_y_, 0.0, max_y);
    const double expanded_left = clampValue(
        bbox_left - cfg_.car_push_expand_left_px, 0.0, max_x);
    const double expanded_right = clampValue(
        bbox_right + cfg_.car_push_expand_right_px, 0.0, max_x);
    const double expanded_top = clampValue(
        bbox_top - cfg_.car_push_expand_top_px, 0.0, max_y);
    const double expanded_bottom = clampValue(
        bbox_bottom + cfg_.car_push_expand_bottom_px, 0.0, max_y);
    const double target_x = clampValue(
        expanded_left - cfg_.car_push_clearance_px, 0.0, max_x);
    debug_info_.car_push_target_x = static_cast<float>(target_x);
    debug_info_.car_push_expand_bottom_y = static_cast<float>(
        expanded_bottom);
    debug_info_.car_bbox = cv::Rect2f(
        static_cast<float>(bbox_left), static_cast<float>(bbox_top),
        static_cast<float>(std::max(0.0, bbox_right - bbox_left)),
        static_cast<float>(std::max(0.0, bbox_bottom - bbox_top)));
    debug_info_.car_expanded_bbox = cv::Rect2f(
        static_cast<float>(expanded_left), static_cast<float>(expanded_top),
        static_cast<float>(std::max(0.0, expanded_right - expanded_left)),
        static_cast<float>(std::max(0.0, expanded_bottom - expanded_top)));
  }
  debug_info_.fit_hold_active = fit_hold_active_;
  debug_info_.fit_hold_age = fit_hold_age_;
  debug_info_.fit_order = fit_coeffs.empty() ? 0 : static_cast<int>(fit_coeffs.size()) - 1;
  if (fit_points.empty()) {
    debug_info_.fit_y_min = 0;
    debug_info_.fit_y_max = 0;
    debug_info_.fit_y_span = 0;
  } else {
    const auto y_range = std::minmax_element(
      fit_points.begin(), fit_points.end(),
      [](const cv::Point3f& a, const cv::Point3f& b) { return a.y < b.y; });
    debug_info_.fit_y_min = static_cast<int>(std::round(y_range.first->y));
    debug_info_.fit_y_max = static_cast<int>(std::round(y_range.second->y));
    debug_info_.fit_y_span = debug_info_.fit_y_max - debug_info_.fit_y_min;
  }
  debug_info_.encoder_hold = encoder_hold_active_ && branch_locked_;
  debug_info_.encoder_hold_side = debug_info_.encoder_hold ? locked_branch_side_ : "";
  debug_info_.encoder_count = latest_encoder_count_;
  debug_info_.encoder_hold_delta = encoder_hold_delta_;
  debug_info_.encoder_hold_target = encoder_hold_target_;
  debug_info_.encoder_feedback_valid = has_encoder_count_ &&
      (nowSeconds() - last_encoder_update_sec_) <= cfg_.encoder_feedback_timeout_sec;
  debug_info_.encoder_feedback_age = has_encoder_count_
      ? std::max(0.0, nowSeconds() - last_encoder_update_sec_) : 0.0;
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
