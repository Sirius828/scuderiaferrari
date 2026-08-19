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

struct PolynomialClosestPoint {
  bool valid{false};
  double x{0.0};
  double y{0.0};
  double distance{0.0};
  double signed_distance{0.0};
  double slope{0.0};
};

std::pair<double, double> fitYBounds(
    const std::vector<cv::Point3f>& fit_points, int image_height) {
  double y_min = 0.0;
  double y_max = std::max(0, image_height - 1);
  if (!fit_points.empty()) {
    const auto bounds = std::minmax_element(
        fit_points.begin(), fit_points.end(),
        [](const cv::Point3f& lhs, const cv::Point3f& rhs) {
          return lhs.y < rhs.y;
        });
    y_min = clampValue(static_cast<double>(bounds.first->y), 0.0, y_max);
    y_max = clampValue(static_cast<double>(bounds.second->y), y_min, y_max);
  }
  return {y_min, y_max};
}

PolynomialClosestPoint closestPointOnPolynomial(
    const std::vector<double>& coeffs, double point_x, double point_y,
    double y_min, double y_max) {
  PolynomialClosestPoint result;
  if (coeffs.size() < 2 || !std::isfinite(point_x) ||
      !std::isfinite(point_y) || !std::isfinite(y_min) ||
      !std::isfinite(y_max) || y_max < y_min) {
    return result;
  }

  const auto distance_squared = [&](double y) {
    const double x = evalPoly(coeffs, y);
    if (!std::isfinite(x)) {
      return std::numeric_limits<double>::infinity();
    }
    const double dx = x - point_x;
    const double dy = y - point_y;
    return dx * dx + dy * dy;
  };

  constexpr int kCoarseSegments = 32;
  double best_y = y_min;
  double best_distance_squared = distance_squared(best_y);
  for (int i = 1; i <= kCoarseSegments; ++i) {
    const double ratio = static_cast<double>(i) / kCoarseSegments;
    const double y = y_min + (y_max - y_min) * ratio;
    const double candidate_distance_squared = distance_squared(y);
    if (candidate_distance_squared < best_distance_squared) {
      best_y = y;
      best_distance_squared = candidate_distance_squared;
    }
  }

  const double segment_span = (y_max - y_min) / kCoarseSegments;
  double search_left = std::max(y_min, best_y - segment_span);
  double search_right = std::min(y_max, best_y + segment_span);
  constexpr double kGoldenRatio = 0.6180339887498948482;
  double left_probe = search_right -
                      (search_right - search_left) * kGoldenRatio;
  double right_probe = search_left +
                       (search_right - search_left) * kGoldenRatio;
  double left_distance_squared = distance_squared(left_probe);
  double right_distance_squared = distance_squared(right_probe);
  constexpr int kRefineIterations = 14;
  for (int iteration = 0; iteration < kRefineIterations; ++iteration) {
    if (left_distance_squared <= right_distance_squared) {
      search_right = right_probe;
      right_probe = left_probe;
      right_distance_squared = left_distance_squared;
      left_probe = search_right -
                   (search_right - search_left) * kGoldenRatio;
      left_distance_squared = distance_squared(left_probe);
    } else {
      search_left = left_probe;
      left_probe = right_probe;
      left_distance_squared = right_distance_squared;
      right_probe = search_left +
                    (search_right - search_left) * kGoldenRatio;
      right_distance_squared = distance_squared(right_probe);
    }
  }
  const double refined_y = 0.5 * (search_left + search_right);
  const double refined_distance_squared = distance_squared(refined_y);
  if (refined_distance_squared < best_distance_squared) {
    best_y = refined_y;
    best_distance_squared = refined_distance_squared;
  }
  if (!std::isfinite(best_distance_squared)) {
    return result;
  }

  const double best_x = evalPoly(coeffs, best_y);
  const auto derivative = polyDeriv(coeffs);
  const double slope = evalPoly(derivative, best_y);
  if (!std::isfinite(best_x) || !std::isfinite(slope)) {
    return result;
  }
  const double normal_scale = std::sqrt(1.0 + slope * slope);
  const double normal_projection =
      ((point_x - best_x) - slope * (point_y - best_y)) / normal_scale;
  const double distance = std::sqrt(std::max(0.0, best_distance_squared));
  const double side_value = std::abs(normal_projection) > 1e-6
                                ? normal_projection
                                : point_x - best_x;

  result.valid = true;
  result.x = best_x;
  result.y = best_y;
  result.distance = distance;
  result.signed_distance = std::copysign(distance, side_value);
  result.slope = slope;
  return result;
}

double smoothStep(double value) {
  const double t = clampValue(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
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
     << "\"global_offset\":" << state.global_offset << ","
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
  cfg_.car_side_confirm_frames = std::max(1, cfg_.car_side_confirm_frames);
  cfg_.car_side_fit_downward_extension_px = std::max(
      0, cfg_.car_side_fit_downward_extension_px);
  cfg_.car_side_connectivity_strip_width_px = std::max(
      1, cfg_.car_side_connectivity_strip_width_px);
  cfg_.car_side_connectivity_gap_px = std::max(
      0, cfg_.car_side_connectivity_gap_px);
  cfg_.car_side_connectivity_x_margin_px = std::max(
      cfg_.car_side_connectivity_strip_width_px +
          cfg_.car_side_connectivity_gap_px,
      cfg_.car_side_connectivity_x_margin_px);
  cfg_.car_side_connectivity_y_start_ratio = clampValue(
      cfg_.car_side_connectivity_y_start_ratio, 0.0f, 1.0f);
  cfg_.car_side_connectivity_bottom_extend_height_ratio = std::max(
      0.0f, cfg_.car_side_connectivity_bottom_extend_height_ratio);
  cfg_.car_side_connectivity_min_seed_pixels = std::max(
      1, cfg_.car_side_connectivity_min_seed_pixels);
  cfg_.car_avoidance_min_height_ratio =
      clampValue(cfg_.car_avoidance_min_height_ratio, 0.0f, 1.0f);
  cfg_.car_avoidance_min_height_px =
      std::max(1, cfg_.car_avoidance_min_height_px);
  cfg_.car_encoder_detour_counts =
      std::max<int64_t>(0, cfg_.car_encoder_detour_counts);
  cfg_.car_encoder_return_counts =
      std::max<int64_t>(1, cfg_.car_encoder_return_counts);
  cfg_.car_rearm_clear_frames = std::max(1, cfg_.car_rearm_clear_frames);
  cfg_.car_template_min_points = std::max(3, cfg_.car_template_min_points);
  cfg_.car_template_fit_order = 2;
  cfg_.car_template_weight = std::max(0.01f, cfg_.car_template_weight);
  cfg_.car_encoder_fault_hold_sec =
      std::max(0.0, cfg_.car_encoder_fault_hold_sec);
  cfg_.human_horizontal_expand_px = std::max(0.0f, cfg_.human_horizontal_expand_px);
  cfg_.human_horizontal_expand_width_ratio =
      std::max(0.0f, cfg_.human_horizontal_expand_width_ratio);
  cfg_.human_line_sample_count = std::max(2, cfg_.human_line_sample_count);
  cfg_.human_stop_raw_area_ratio =
      clampValue(cfg_.human_stop_raw_area_ratio, 0.0f, 1.0f);
  cfg_.human_stop_confirm_frames = std::max(1, cfg_.human_stop_confirm_frames);
  cfg_.human_clear_confirm_frames = std::max(1, cfg_.human_clear_confirm_frames);
  cfg_.coin_min_confidence = clampValue(cfg_.coin_min_confidence, 0.0f, 1.0f);
  cfg_.coin_evaluate_min_y_ratio =
      clampValue(cfg_.coin_evaluate_min_y_ratio, 0.0f, 1.0f);
  cfg_.coin_near_committed_y_ratio = clampValue(
      cfg_.coin_near_committed_y_ratio,
      cfg_.coin_evaluate_min_y_ratio, 1.0f);
  cfg_.coin_car_half_width_area_scale =
      std::max(0.0f, cfg_.coin_car_half_width_area_scale);
  cfg_.coin_car_half_width_min_px =
      std::max(0.0f, cfg_.coin_car_half_width_min_px);
  cfg_.coin_car_half_width_max_px = std::max(
      cfg_.coin_car_half_width_min_px, cfg_.coin_car_half_width_max_px);
  cfg_.coin_hit_margin_px = std::max(0.0f, cfg_.coin_hit_margin_px);
  cfg_.coin_reachable_extra_area_scale =
      std::max(0.0f, cfg_.coin_reachable_extra_area_scale);
  cfg_.coin_reachable_extra_min_px =
      std::max(0.0f, cfg_.coin_reachable_extra_min_px);
  cfg_.coin_reachable_extra_max_px = std::max(
      cfg_.coin_reachable_extra_min_px, cfg_.coin_reachable_extra_max_px);
  cfg_.coin_obstacle_expand_px = std::max(0.0f, cfg_.coin_obstacle_expand_px);
  cfg_.coin_obstacle_lookahead_ratio =
      clampValue(cfg_.coin_obstacle_lookahead_ratio, 0.0f, 1.0f);
  cfg_.coin_side_clear_frames = std::max(1, cfg_.coin_side_clear_frames);
  cfg_.coin_route_max_targets = std::max(1, cfg_.coin_route_max_targets);
  cfg_.coin_route_sample_count = std::max(
      std::max(5, cfg_.fit_min_points), cfg_.coin_route_sample_count);
  cfg_.coin_route_approach_span_ratio = clampValue(
      cfg_.coin_route_approach_span_ratio, 0.02f, 1.0f);
  cfg_.coin_route_return_span_ratio = clampValue(
      cfg_.coin_route_return_span_ratio, 0.02f, 1.0f);
  cfg_.coin_route_overlap_margin_px =
      std::max(0.0f, cfg_.coin_route_overlap_margin_px);
  cfg_.coin_route_base_weight = std::max(0.01f, cfg_.coin_route_base_weight);
  cfg_.coin_route_target_weight = std::max(
      cfg_.coin_route_base_weight, cfg_.coin_route_target_weight);
  cfg_.coin_route_endpoint_weight = std::max(
      cfg_.coin_route_base_weight, cfg_.coin_route_endpoint_weight);
  cfg_.coin_route_max_offset_px = std::max(1.0f, cfg_.coin_route_max_offset_px);
  cfg_.coin_route_max_abs_heading = clampValue(
      cfg_.coin_route_max_abs_heading, 0.0f, 1.0f);
  cfg_.coin_route_max_abs_curvature = clampValue(
      cfg_.coin_route_max_abs_curvature, 0.0f, 1.0f);
  cfg_.coin_route_obstacle_clearance_px =
      std::max(0.0f, cfg_.coin_route_obstacle_clearance_px);
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
  car_right_template_offsets_ = parseDoubleList(cfg_.car_right_template_offsets);
  car_left_template_offsets_ = parseDoubleList(cfg_.car_left_template_offsets);
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
  resetCarAvoidanceState();
  has_last_underlying_fit_ = false;
  last_underlying_fit_coeffs_.clear();
  last_underlying_fit_heading_ = 0.0;
  last_underlying_fit_curvature_ = 0.0;
  last_underlying_fit_confidence_ = 0.0;
  human_stop_active_ = false;
  human_state_ = "NONE";
  human_stop_confirm_count_ = 0;
  human_clear_confirm_count_ = 0;
  human_count_at_stop_ = 0;
  coin_selected_side_ = "NONE";
  coin_side_clear_count_ = 0;
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

void LaneDecision::setHeadingWeights(double near_weight, double mid_weight,
                                     double far_weight) {
  cfg_.heading_near_weight = static_cast<float>(std::max(0.0, near_weight));
  cfg_.heading_mid_weight = static_cast<float>(std::max(0.0, mid_weight));
  cfg_.heading_far_weight = static_cast<float>(std::max(0.0, far_weight));
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
  std::array<double, 3> offsets{{0.0, 0.0, 0.0}};
  std::array<double, 3> raw_offsets{{0.0, 0.0, 0.0}};
  double global_offset = 0.0;
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
    bands = buildBands(seg_map);
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
          if (cfg_.branch_extend_while_detected && branch_detected) {
            // Continue the same physical branch event without briefly falling
            // back to the normal fit or retriggering GuideBoard processing.
            encoder_hold_start_count_ = latest_encoder_count_;
            encoder_hold_delta_ = 0;
            lock_start_time_ = current_time;
          } else {
            branch_locked_ = false;
            encoder_hold_active_ = false;
            encoder_hold_baseline_valid_ = false;
            encoder_hold_delta_ = 0;
            locked_branch_side_ = cfg_.outer_side;
          }
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
    fit_points = filterCenterlinePoints(fit_points, w, last_center_x);
    fit_points = filterCenterlinePointsKalman(
        fit_points, bands, w, h, current_time, target_side, template_active);
    // Car side detection uses the current frame's selected centerline points
    // (the yellow band points in the debug image), before any Car template
    // replaces the output line.
    updateCarAvoidanceState(detections, raw_points, seg_map, w, h);
    const std::vector<cv::Point3f> underlying_points = fit_points;
    std::vector<double> underlying_coeffs;
    double underlying_heading = 0.0;
    double underlying_curvature = 0.0;
    const bool underlying_fit_success =
        static_cast<int>(underlying_points.size()) >= cfg_.fit_min_points &&
        fitCenterlineAndComputeGeometry(
            underlying_points, h, fit_order, &underlying_coeffs,
            &underlying_heading, &underlying_curvature);
    const double underlying_confidence = underlying_fit_success
        ? calculateLaneConfidence(underlying_points, bands) : 0.0;
    if (underlying_fit_success) {
      has_last_underlying_fit_ = true;
      last_underlying_fit_coeffs_ = underlying_coeffs;
      last_underlying_fit_heading_ = underlying_heading;
      last_underlying_fit_curvature_ = underlying_curvature;
      last_underlying_fit_confidence_ = underlying_confidence;
    }

    auto refresh_car_template = [&]() {
      if (car_template_side_ != "LEFT" && car_template_side_ != "RIGHT") {
        return false;
      }
      auto template_points = collectCarBoundaryTemplatePoints(
          bands, w, car_template_side_);
      car_template_point_count_ = static_cast<int>(template_points.size());
      if (car_template_point_count_ < cfg_.car_template_min_points) {
        return false;
      }
      std::vector<double> coeffs;
      double template_heading = 0.0;
      double template_curvature = 0.0;
      if (!fitCenterlineAndComputeGeometry(
              template_points, h, cfg_.car_template_fit_order, &coeffs,
              &template_heading, &template_curvature)) {
        return false;
      }
      car_template_cached_points_ = std::move(template_points);
      car_template_cached_coeffs_ = std::move(coeffs);
      car_template_cached_heading_ = template_heading;
      car_template_cached_curvature_ = template_curvature;
      car_template_cached_confidence_ =
          calculateLaneConfidence(car_template_cached_points_, bands);
      return true;
    };

    if (car_template_state_ == CarTemplateState::WaitClear) {
      if (car_detection_active_) {
        car_rearm_clear_count_ = 0;
      } else {
        ++car_rearm_clear_count_;
        if (car_rearm_clear_count_ >= cfg_.car_rearm_clear_frames) {
          resetCarAvoidanceState();
        }
      }
    }

    const bool encoder_fresh = has_encoder_count_ &&
        current_time - last_encoder_update_sec_ <= cfg_.encoder_feedback_timeout_sec;
    if (car_template_state_ == CarTemplateState::Idle &&
        cfg_.enable_car_obstacle_avoidance && car_detection_active_ &&
        (car_side_ == "LEFT" || car_side_ == "RIGHT") &&
        car_side_confirm_count_ >= cfg_.car_side_confirm_frames &&
        encoder_fresh) {
      car_template_side_ = car_side_;
      if (refresh_car_template()) {
        car_template_state_ = CarTemplateState::Detour;
        car_avoidance_active_ = true;
        car_encoder_start_count_ = latest_encoder_count_;
        car_encoder_baseline_valid_ = true;
        car_encoder_delta_ = 0;
        car_encoder_return_delta_ = 0;
        car_encoder_return_progress_ = 0.0;
        car_encoder_fault_start_sec_ = 0.0;
        car_encoder_fault_age_ = 0.0;
        car_fault_exit_active_ = false;
        car_rearm_clear_count_ = 0;
      } else {
        car_template_side_ = "UNKNOWN";
      }
    }

    if (car_template_state_ == CarTemplateState::Detour ||
        car_template_state_ == CarTemplateState::Return) {
      car_avoidance_active_ = true;
      bool template_refreshed = false;
      if ((car_side_ == "LEFT" || car_side_ == "RIGHT") &&
          car_side_ != car_template_side_) {
        const std::string previous_template_side = car_template_side_;
        const int previous_template_point_count = car_template_point_count_;
        car_template_side_ = car_side_;
        template_refreshed = refresh_car_template();
        if (!template_refreshed) {
          // Keep publishing the last valid template until the newly confirmed
          // side can produce a complete quadratic template.
          car_template_side_ = previous_template_side;
          car_template_point_count_ = previous_template_point_count;
        }
      }
      if (!template_refreshed) {
        (void)refresh_car_template();
      }

      const bool encoder_progress_valid = encoder_fresh &&
          car_encoder_baseline_valid_ &&
          latest_encoder_count_ >= car_encoder_start_count_;
      if (encoder_progress_valid && !car_fault_exit_active_) {
        car_encoder_fault_start_sec_ = 0.0;
        car_encoder_fault_age_ = 0.0;
        car_encoder_delta_ = latest_encoder_count_ - car_encoder_start_count_;
        const int64_t total_counts = cfg_.car_encoder_detour_counts +
                                     cfg_.car_encoder_return_counts;
        if (car_encoder_delta_ >= total_counts) {
          car_template_state_ = CarTemplateState::WaitClear;
          car_avoidance_active_ = false;
          car_encoder_return_delta_ = cfg_.car_encoder_return_counts;
          car_encoder_return_progress_ = 1.0;
        } else if (car_encoder_delta_ >= cfg_.car_encoder_detour_counts) {
          car_template_state_ = CarTemplateState::Return;
          car_encoder_return_delta_ =
              car_encoder_delta_ - cfg_.car_encoder_detour_counts;
          car_encoder_return_progress_ = clampValue(
              static_cast<double>(car_encoder_return_delta_) /
                  static_cast<double>(cfg_.car_encoder_return_counts),
              0.0, 1.0);
        } else {
          car_template_state_ = CarTemplateState::Detour;
          car_encoder_return_delta_ = 0;
          car_encoder_return_progress_ = 0.0;
        }
      } else if (!car_fault_exit_active_) {
        if (car_encoder_fault_start_sec_ <= 0.0) {
          car_encoder_fault_start_sec_ = current_time;
        }
        car_encoder_fault_age_ =
            std::max(0.0, current_time - car_encoder_fault_start_sec_);
        if (car_encoder_fault_age_ >= cfg_.car_encoder_fault_hold_sec) {
          if (has_last_underlying_fit_) {
            car_fault_exit_active_ = true;
            car_fault_exit_start_sec_ = current_time;
          } else {
            car_template_state_ = CarTemplateState::WaitClear;
            car_avoidance_active_ = false;
          }
        }
      }
    }

    bool fit_success = underlying_fit_success;
    fit_coeffs = underlying_coeffs;
    heading_error = underlying_heading;
    curvature = underlying_curvature;
    confidence = underlying_confidence;

    const auto quadraticCoeffs = [](const std::vector<double>& coeffs) {
      if (coeffs.size() == 3) {
        return coeffs;
      }
      if (coeffs.size() == 2) {
        return std::vector<double>{0.0, coeffs[0], coeffs[1]};
      }
      return std::vector<double>{};
    };
    auto use_car_template = [&]() {
      if (car_template_cached_coeffs_.empty()) {
        fit_success = false;
        return;
      }
      raw_points = car_template_cached_points_;
      fit_points = car_template_cached_points_;
      fit_coeffs = car_template_cached_coeffs_;
      heading_error = car_template_cached_heading_;
      curvature = car_template_cached_curvature_;
      confidence = car_template_cached_confidence_;
      fit_success = true;
    };
    auto blend_from_car = [&](const std::vector<double>& target_coeffs,
                              double target_heading, double target_curvature,
                              double target_confidence, double progress) {
      const auto car_coeffs = quadraticCoeffs(car_template_cached_coeffs_);
      const auto target = quadraticCoeffs(target_coeffs);
      if (car_coeffs.size() != 3 || target.size() != 3) {
        use_car_template();
        return;
      }
      progress = clampValue(progress, 0.0, 1.0);
      fit_coeffs.resize(3);
      for (size_t i = 0; i < fit_coeffs.size(); ++i) {
        fit_coeffs[i] = car_coeffs[i] * (1.0 - progress) +
                        target[i] * progress;
      }
      raw_points = car_template_cached_points_;
      fit_points = car_template_cached_points_;
      heading_error = car_template_cached_heading_ * (1.0 - progress) +
                      target_heading * progress;
      curvature = car_template_cached_curvature_ * (1.0 - progress) +
                  target_curvature * progress;
      confidence = car_template_cached_confidence_ * (1.0 - progress) +
                   target_confidence * progress;
      fit_success = true;
    };

    if (car_fault_exit_active_) {
      constexpr double kFaultExitBlendSec = 0.5;
      const double progress = clampValue(
          (current_time - car_fault_exit_start_sec_) / kFaultExitBlendSec,
          0.0, 1.0);
      blend_from_car(last_underlying_fit_coeffs_,
                     last_underlying_fit_heading_,
                     last_underlying_fit_curvature_,
                     last_underlying_fit_confidence_, progress);
      car_encoder_return_progress_ = progress;
      if (progress >= 1.0) {
        car_fault_exit_active_ = false;
        car_template_state_ = CarTemplateState::WaitClear;
        car_avoidance_active_ = false;
      }
    } else if (car_template_state_ == CarTemplateState::Detour) {
      use_car_template();
    } else if (car_template_state_ == CarTemplateState::Return) {
      if (underlying_fit_success) {
        blend_from_car(underlying_coeffs, underlying_heading,
                       underlying_curvature, underlying_confidence,
                       car_encoder_return_progress_);
      } else {
        use_car_template();
      }
    }
    // Coin classification and route generation always use the final road/car
    // fit as their immutable base.  The optional controller takeover happens
    // only after the candidate has passed all route checks.
    evaluateCoinsShadow(detections, w, h, fit_points, fit_coeffs, fit_success);
    buildCoinRoutePreview(detections, w, h, fit_points, fit_coeffs, fit_success);
    if (cfg_.enable_coin_route_control &&
        debug_info_.coin_route_candidate_valid) {
      fit_points = debug_info_.coin_route_points;
      fit_coeffs = debug_info_.coin_route_coeffs;
      heading_error = debug_info_.coin_route_heading;
      curvature = debug_info_.coin_route_curvature;
      debug_info_.coin_route_control_active = true;
    }

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
      // Average the final fitted path at several fixed lookahead rows.  This
      // captures a whole-line translation (including an avoidance template)
      // without depending on a single offset sample or on line slope.
      constexpr std::array<float, 5> kGlobalOffsetRatios{{0.68f, 0.75f, 0.82f, 0.89f, 0.96f}};
      double global_offset_sum = 0.0;
      for (const float ratio : kGlobalOffsetRatios) {
        global_offset_sum += offsetAtY(fit_coeffs, h * ratio, w);
      }
      global_offset = clampValue(
          global_offset_sum / static_cast<double>(kGlobalOffsetRatios.size()), -1.0, 1.0);
      debug_info_.global_offset = static_cast<float>(global_offset);
      debug_info_.image_width = w;
      is_valid = true;
    } else {
      cv::Mat bottom_seg = seg_map(
          cv::Range(static_cast<int>(h * 0.8), h), cv::Range::all());
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
      debug_info_.offset_y07 = static_cast<float>(offsets[0]);
      debug_info_.offset_y08 = static_cast<float>(offsets[1]);
      debug_info_.offset_y09 = static_cast<float>(offsets[2]);
      debug_info_.raw_offset_y07 = static_cast<float>(raw_offsets[0]);
      debug_info_.raw_offset_y08 = static_cast<float>(raw_offsets[1]);
      debug_info_.raw_offset_y09 = static_cast<float>(raw_offsets[2]);
      global_offset = clampValue(
          (offsets[0] + offsets[1] + offsets[2]) / 3.0, -1.0, 1.0);
      debug_info_.global_offset = static_cast<float>(global_offset);
      debug_info_.image_width = w;
    }
    populateDebugInfo(bands, raw_points, fit_points, w, h, fit_coeffs);
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
    global_offset = clampValue(
        (offsets[0] + offsets[1] + offsets[2]) / 3.0, -1.0, 1.0);
    debug_info_.global_offset = static_cast<float>(global_offset);
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
  state.global_offset = static_cast<float>(global_offset);
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

std::vector<LaneDecision::Band> LaneDecision::buildBands(const cv::Mat& road_mask) {
  int h = road_mask.rows;
  int w = road_mask.cols;
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
    band.segments = extractSegmentsInBand(band_mask);
    bands.push_back(std::move(band));
  }
  return bands;
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

void LaneDecision::resetCarAvoidanceState() {
  car_detection_active_ = false;
  car_avoidance_active_ = false;
  car_side_ = "UNKNOWN";
  car_side_candidate_ = "UNKNOWN";
  car_side_confirm_count_ = 0;
  car_left_x_ = -1.0;
  car_right_x_ = 0.0;
  car_top_y_ = 0.0;
  car_bottom_y_ = 0.0;
  car_side_fit_valid_ = false;
  car_side_fit_x_ = -1.0;
  car_side_fit_y_ = -1.0;
  car_side_fit_relation_ = "UNKNOWN";
  car_mask_connectivity_ = "UNKNOWN";
  car_side_source_ = "NONE";
  car_left_seed_pixels_ = 0;
  car_right_seed_pixels_ = 0;
  car_common_component_pixels_ = 0;
  car_connectivity_roi_ = cv::Rect();
  car_left_seed_roi_ = cv::Rect();
  car_right_seed_roi_ = cv::Rect();
  car_template_state_ = CarTemplateState::Idle;
  car_template_side_ = "UNKNOWN";
  car_template_point_count_ = 0;
  car_template_cached_points_.clear();
  car_template_cached_coeffs_.clear();
  car_template_cached_heading_ = 0.0;
  car_template_cached_curvature_ = 0.0;
  car_template_cached_confidence_ = 0.0;
  car_encoder_baseline_valid_ = false;
  car_encoder_start_count_ = 0;
  car_encoder_delta_ = 0;
  car_encoder_return_delta_ = 0;
  car_encoder_return_progress_ = 0.0;
  car_encoder_fault_start_sec_ = 0.0;
  car_encoder_fault_age_ = 0.0;
  car_fault_exit_active_ = false;
  car_fault_exit_start_sec_ = 0.0;
  car_rearm_clear_count_ = 0;
}

void LaneDecision::updateCarAvoidanceState(
    const std::vector<Detection>& detections,
    const std::vector<cv::Point3f>& fit_points,
    const cv::Mat& seg_map,
    int image_width, int image_height) {
  if (!cfg_.enable_car_obstacle_avoidance || image_width <= 0 ||
      image_height <= 0) {
    resetCarAvoidanceState();
    return;
  }

  const double min_height = std::max(
      static_cast<double>(cfg_.car_avoidance_min_height_px),
      static_cast<double>(image_height) *
          static_cast<double>(cfg_.car_avoidance_min_height_ratio));
  const Detection* selected_car = nullptr;
  double selected_height = -std::numeric_limits<double>::infinity();
  double selected_bottom = -std::numeric_limits<double>::infinity();
  double selected_area = 0.0;
  for (const auto& det : detections) {
    if (det.class_name != "Car" ||
        det.confidence < cfg_.obstacle_min_confidence ||
        det.bbox.width <= 0.0f || det.bbox.height <= 0.0f) {
      continue;
    }
    const double height = det.bbox.height;
    if (height < min_height) {
      continue;
    }
    const double bottom = det.bbox.y + det.bbox.height;
    const double area = static_cast<double>(det.bbox.width) * det.bbox.height;
    if (!selected_car || height > selected_height ||
        (std::abs(height - selected_height) < 1e-6 &&
         (bottom > selected_bottom ||
          (std::abs(bottom - selected_bottom) < 1e-6 && area > selected_area)))) {
      selected_car = &det;
      selected_height = height;
      selected_bottom = bottom;
      selected_area = area;
    }
  }

  if (!selected_car) {
    car_detection_active_ = false;
    car_side_candidate_ = "UNKNOWN";
    car_side_fit_valid_ = false;
    car_side_fit_x_ = -1.0;
    car_side_fit_y_ = -1.0;
    car_side_fit_relation_ = "UNKNOWN";
    car_mask_connectivity_ = "UNKNOWN";
    car_side_source_ = "NONE";
    car_left_seed_pixels_ = 0;
    car_right_seed_pixels_ = 0;
    car_common_component_pixels_ = 0;
    car_connectivity_roi_ = cv::Rect();
    car_left_seed_roi_ = cv::Rect();
    car_right_seed_roi_ = cv::Rect();
    car_left_x_ = -1.0;
    car_right_x_ = 0.0;
    car_top_y_ = 0.0;
    car_bottom_y_ = 0.0;
    if (car_template_state_ == CarTemplateState::Idle) {
      car_side_ = "UNKNOWN";
    }
    // A detection gap is not evidence for switching sides.  Keep the active
    // template direction, but require a fresh consecutive confirmation when
    // the Car becomes observable again.
    car_side_confirm_count_ = 0;
    return;
  }

  const double max_x = static_cast<double>(image_width - 1);
  const double max_y = static_cast<double>(image_height - 1);
  const double raw_left = clampValue(
      static_cast<double>(selected_car->bbox.x), 0.0, max_x);
  const double raw_right = clampValue(
      static_cast<double>(selected_car->bbox.x + selected_car->bbox.width),
      0.0, max_x);
  const double raw_top = clampValue(
      static_cast<double>(selected_car->bbox.y), 0.0, max_y);
  const double raw_bottom = clampValue(
      static_cast<double>(selected_car->bbox.y + selected_car->bbox.height),
      0.0, max_y);

  car_left_x_ = raw_left;
  car_right_x_ = raw_right;
  car_top_y_ = raw_top;
  car_bottom_y_ = raw_bottom;
  car_detection_active_ = true;

  car_side_fit_valid_ = false;
  car_side_fit_x_ = -1.0;
  car_side_fit_y_ = -1.0;
  car_side_fit_relation_ = "UNKNOWN";
  car_mask_connectivity_ = "UNKNOWN";
  car_side_source_ = "NONE";
  car_left_seed_pixels_ = 0;
  car_right_seed_pixels_ = 0;
  car_common_component_pixels_ = 0;
  car_connectivity_roi_ = cv::Rect();
  car_left_seed_roi_ = cv::Rect();
  car_right_seed_roi_ = cv::Rect();

  // On the fixed clockwise right-turn track, a Car on the left leaves the
  // nearby road mask connected around its bbox.  A Car on the right occludes
  // the right-bending road and splits the road visible at its two vertical
  // sides.  Test this topology in a small ROI instead of comparing mask area.
  std::string mask_candidate = "UNKNOWN";
  if (!seg_map.empty() && seg_map.channels() == 1 &&
      seg_map.rows == image_height && seg_map.cols == image_width) {
    const int bbox_left_px = std::clamp(
        static_cast<int>(std::floor(selected_car->bbox.x)), 0,
        image_width - 1);
    const int bbox_right_px = std::clamp(
        static_cast<int>(std::ceil(
            selected_car->bbox.x + selected_car->bbox.width)),
        bbox_left_px + 1, image_width);
    const int bbox_top_px = std::clamp(
        static_cast<int>(std::floor(selected_car->bbox.y)), 0,
        image_height - 1);
    const int bbox_bottom_px = std::clamp(
        static_cast<int>(std::ceil(
            selected_car->bbox.y + selected_car->bbox.height)),
        bbox_top_px + 1, image_height);
    const int gap = cfg_.car_side_connectivity_gap_px;
    const int strip_width = cfg_.car_side_connectivity_strip_width_px;
    const int left_seed_x1 = bbox_left_px - gap;
    const int left_seed_x0 = left_seed_x1 - strip_width;
    const int right_seed_x0 = bbox_right_px + gap;
    const int right_seed_x1 = right_seed_x0 + strip_width;
    const int seed_y0 = std::clamp(
        static_cast<int>(std::floor(
            selected_car->bbox.y + selected_car->bbox.height *
                cfg_.car_side_connectivity_y_start_ratio)),
        0, image_height - 1);
    const int seed_y1 = bbox_bottom_px;
    const bool seed_geometry_valid =
        left_seed_x0 >= 0 && right_seed_x1 <= image_width &&
        seed_y1 > seed_y0;

    const int roi_x0 = std::max(
        0, bbox_left_px - cfg_.car_side_connectivity_x_margin_px);
    const int roi_x1 = std::min(
        image_width,
        bbox_right_px + cfg_.car_side_connectivity_x_margin_px);
    const int roi_y0 = seed_y0;
    const int roi_y1 = std::clamp(
        static_cast<int>(std::ceil(
            selected_car->bbox.y + selected_car->bbox.height *
                (1.0 + cfg_.car_side_connectivity_bottom_extend_height_ratio))),
        seed_y1, image_height);
    if (roi_x1 > roi_x0 && roi_y1 > roi_y0) {
      car_connectivity_roi_ = cv::Rect(
          roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);
    }

    if (seed_geometry_valid && car_connectivity_roi_.area() > 0) {
      car_left_seed_roi_ = cv::Rect(
          left_seed_x0, seed_y0, strip_width, seed_y1 - seed_y0);
      car_right_seed_roi_ = cv::Rect(
          right_seed_x0, seed_y0, strip_width, seed_y1 - seed_y0);

      cv::Mat road_binary;
      cv::compare(seg_map(car_connectivity_roi_), 0, road_binary,
                  cv::CMP_GT);
      cv::morphologyEx(
          road_binary, road_binary, cv::MORPH_CLOSE,
          cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

      // Never allow a segmentation leak through the detected Car to make the
      // two sides appear connected.
      const int clear_x0 = std::max(bbox_left_px, roi_x0) - roi_x0;
      const int clear_x1 = std::min(bbox_right_px, roi_x1) - roi_x0;
      const int clear_y0 = std::max(bbox_top_px, roi_y0) - roi_y0;
      const int clear_y1 = std::min(bbox_bottom_px, roi_y1) - roi_y0;
      if (clear_x1 > clear_x0 && clear_y1 > clear_y0) {
        road_binary(cv::Rect(clear_x0, clear_y0,
                             clear_x1 - clear_x0,
                             clear_y1 - clear_y0)).setTo(0);
      }

      cv::Mat labels;
      cv::Mat stats;
      cv::Mat centroids;
      const int component_count = cv::connectedComponentsWithStats(
          road_binary, labels, stats, centroids, 8, CV_32S);
      std::vector<uint8_t> left_labels(
          static_cast<size_t>(component_count), 0);
      std::vector<uint8_t> right_labels(
          static_cast<size_t>(component_count), 0);

      const auto collect_seed = [&](const cv::Rect& absolute_seed,
                                    std::vector<uint8_t>* touched_labels) {
        const cv::Rect local_seed(
            absolute_seed.x - roi_x0, absolute_seed.y - roi_y0,
            absolute_seed.width, absolute_seed.height);
        int pixels = 0;
        for (int y = local_seed.y; y < local_seed.y + local_seed.height; ++y) {
          const int* label_row = labels.ptr<int>(y);
          for (int x = local_seed.x; x < local_seed.x + local_seed.width; ++x) {
            const int label = label_row[x];
            if (label > 0) {
              ++pixels;
              (*touched_labels)[static_cast<size_t>(label)] = 1;
            }
          }
        }
        return pixels;
      };
      car_left_seed_pixels_ = collect_seed(
          car_left_seed_roi_, &left_labels);
      car_right_seed_pixels_ = collect_seed(
          car_right_seed_roi_, &right_labels);

      int common_label = 0;
      for (int label = 1; label < component_count; ++label) {
        if (left_labels[static_cast<size_t>(label)] == 0 ||
            right_labels[static_cast<size_t>(label)] == 0) {
          continue;
        }
        if (common_label == 0 ||
            stats.at<int>(label, cv::CC_STAT_AREA) >
                stats.at<int>(common_label, cv::CC_STAT_AREA)) {
          common_label = label;
        }
      }
      if (common_label > 0) {
        car_common_component_pixels_ =
            stats.at<int>(common_label, cv::CC_STAT_AREA);
      }

      const int min_seed_pixels =
          cfg_.car_side_connectivity_min_seed_pixels;
      if (car_left_seed_pixels_ >= min_seed_pixels &&
          car_right_seed_pixels_ >= min_seed_pixels) {
        if (common_label > 0) {
          car_mask_connectivity_ = "CONNECTED";
          mask_candidate = "LEFT";
        } else {
          car_mask_connectivity_ = "SPLIT";
          mask_candidate = "RIGHT";
        }
      }
    }
  }

  // Use centerline points at the Car's vertical height.  The point does not
  // need to be inside the bbox: a point on the Car's left means the Car
  // occupies the road's right side, and vice versa.
  std::vector<cv::Point3f> side_points;
  const double sample_bottom = std::min(
      raw_bottom + static_cast<double>(cfg_.car_side_fit_downward_extension_px),
      max_y);
  for (const auto& point : fit_points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        point.y < raw_top || point.y > sample_bottom) {
      continue;
    }
    if (point.x < raw_left || point.x > raw_right) {
      side_points.push_back(point);
    }
  }

  std::string fit_candidate = "UNKNOWN";
  if (!side_points.empty()) {
    // Prefer the point with the largest y in the Car's vertical span: it is
    // the fitting point at the height closest to the Car's lower edge.
    const double sample_y = std::max_element(
        side_points.begin(), side_points.end(),
        [](const cv::Point3f& a, const cv::Point3f& b) {
          return a.y < b.y;
        })->y;
    std::vector<cv::Point3f> lowest_height_points;
    for (const auto& point : side_points) {
      if (std::abs(static_cast<double>(point.y) - sample_y) <= 1.0) {
        lowest_height_points.push_back(point);
      }
    }
    bool has_left_point = false;
    bool has_right_point = false;
    for (const auto& point : lowest_height_points) {
      has_left_point = has_left_point || point.x < raw_left;
      has_right_point = has_right_point || point.x > raw_right;
    }
    if (has_left_point && has_right_point) {
      car_side_fit_relation_ = "AMBIGUOUS_BOTH_SIDES";
    } else {
      const cv::Point3f& selected_point = lowest_height_points.front();
      car_side_fit_valid_ = true;
      car_side_fit_x_ = selected_point.x;
      car_side_fit_y_ = selected_point.y;
      if (has_left_point) {
        car_side_fit_relation_ = "LEFT_OF_CAR";
        fit_candidate = "RIGHT";
      } else {
        car_side_fit_relation_ = "RIGHT_OF_CAR";
        fit_candidate = "LEFT";
      }
    }
  }

  std::string candidate = mask_candidate;
  if (candidate == "LEFT" || candidate == "RIGHT") {
    car_side_source_ = "MASK_CONNECTIVITY";
  } else if (car_template_state_ == CarTemplateState::Idle &&
             (fit_candidate == "LEFT" || fit_candidate == "RIGHT")) {
    // The old fit-point relation is retained only to acquire an initial side
    // when mask topology has insufficient evidence.  It cannot switch an
    // active DETOUR/RETURN template.
    candidate = fit_candidate;
    car_side_source_ = "FIT_FALLBACK";
  }

  if (car_template_state_ == CarTemplateState::WaitClear) {
    car_side_candidate_ = candidate;
    return;
  }

  if (candidate == "UNKNOWN") {
    car_side_candidate_ = "UNKNOWN";
    car_side_confirm_count_ = 0;
    if (car_template_state_ == CarTemplateState::Idle) {
      car_side_ = "UNKNOWN";
    }
    return;
  }

  if (candidate == car_side_candidate_) {
    car_side_confirm_count_ = std::min(
        car_side_confirm_count_ + 1, cfg_.car_side_confirm_frames);
  } else {
    car_side_candidate_ = candidate;
    car_side_confirm_count_ = 1;
  }

  if (car_side_confirm_count_ < cfg_.car_side_confirm_frames) {
    car_side_ = "UNKNOWN";
    return;
  }

  // During DETOUR/RETURN only the encoder baseline is latched.  A newly
  // confirmed side may replace the Car template without restarting distance
  // accumulation.
  car_side_ = candidate;
}

std::vector<cv::Point3f> LaneDecision::collectCarBoundaryTemplatePoints(
    std::vector<Band>& bands, int image_width,
    const std::string& car_side) const {
  std::vector<cv::Point3f> points;
  if (car_side != "LEFT" && car_side != "RIGHT") {
    return points;
  }

  const bool car_is_on_right = car_side == "RIGHT";
  const auto& offsets = car_is_on_right ? car_right_template_offsets_
                                        : car_left_template_offsets_;
  std::optional<double> last_template_x;
  for (auto it_band = bands.rbegin(); it_band != bands.rend(); ++it_band) {
    auto& band = *it_band;
    if (band.segments.empty() || band.index < 0 ||
        band.index >= static_cast<int>(offsets.size())) {
      continue;
    }
    const double offset = offsets[band.index];
    if (!std::isfinite(offset)) {
      continue;
    }
    // Both Car directions use the continuous road left boundary as the only
    // geometric reference.  The per-band offset is empirical and is applied
    // directly; it is intentionally not normalized by image height, segment
    // width, or any other detected geometry.
    const auto boundary_x = [&](const Segment& segment) {
      return static_cast<double>(segment.x0) + offset;
    };

    const Segment* selected = nullptr;
    if (last_template_x) {
      const auto it = std::min_element(
          band.segments.begin(), band.segments.end(),
          [&](const Segment& a, const Segment& b) {
            return std::abs(boundary_x(a) - *last_template_x) <
                   std::abs(boundary_x(b) - *last_template_x);
          });
      if (it != band.segments.end()) {
        selected = &(*it);
      }
    } else {
      const auto it = std::min_element(
          band.segments.begin(), band.segments.end(),
          [&](const Segment& a, const Segment& b) {
            return boundary_x(a) < boundary_x(b);
          });
      if (it != band.segments.end()) {
        selected = &(*it);
      }
    }
    if (!selected) {
      continue;
    }

    const double x = clampValue(
        boundary_x(*selected), 0.0,
        static_cast<double>(std::max(0, image_width - 1)));
    Segment virtual_segment;
    virtual_segment.x0 = static_cast<int>(std::round(x));
    virtual_segment.x1 = virtual_segment.x0;
    virtual_segment.width = 1.0;
    virtual_segment.center_x = x;
    virtual_segment.pixel_count = 1;
    virtual_segment.virtual_segment = true;
    band.selected_segment = virtual_segment;
    points.emplace_back(static_cast<float>(x),
                        static_cast<float>(band.y_center),
                        cfg_.car_template_weight);
    last_template_x = x;
  }
  return points;
}

std::string LaneDecision::carTemplateStateName() const {
  switch (car_template_state_) {
    case CarTemplateState::Detour:
      return "DETOUR";
    case CarTemplateState::Return:
      return "RETURN";
    case CarTemplateState::WaitClear:
      return "WAIT_CLEAR";
    case CarTemplateState::Idle:
    default:
      return "IDLE";
  }
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
  debug_info_.human_passable = false;
  debug_info_.human_line_intersects = false;
  debug_info_.human_stop_candidate = false;
  debug_info_.human_stop_active = human_stop_active_;
  debug_info_.human_clear_confirming = false;
  debug_info_.human_raw_area_ratio = 0.0f;
  debug_info_.human_state = human_state_;
  debug_info_.human_stop_confirm_count = human_stop_confirm_count_;
  debug_info_.human_clear_confirm_count = human_clear_confirm_count_;
  debug_info_.human_count_at_stop = human_count_at_stop_;
  debug_info_.human_valid_count = 0;
  debug_info_.human_candidate_count = 0;

  if (!cfg_.enable_human_obstacle_stop || image_width <= 0 || image_height <= 0) {
    human_stop_active_ = false;
    human_state_ = "NONE";
    human_stop_confirm_count_ = 0;
    human_clear_confirm_count_ = 0;
    human_count_at_stop_ = 0;
    debug_info_.human_stop_active = false;
    debug_info_.human_state = human_state_;
    debug_info_.human_stop_confirm_count = 0;
    debug_info_.human_clear_confirm_count = 0;
    debug_info_.human_count_at_stop = 0;
    debug_info_.human_candidate_count = 0;
    return;
  }

  const bool fit_available = fit_valid && fit_coeffs.size() >= 2;
  const double area_threshold = clampValue(
      static_cast<double>(cfg_.human_stop_raw_area_ratio), 0.0, 1.0);
  const double image_area = static_cast<double>(image_width) * image_height;

  bool has_valid_human = false;
  bool all_current_humans_passable = true;
  bool any_line_intersects = false;
  bool any_stop_candidate = false;
  bool any_no_fit_human = false;
  int valid_human_count = 0;
  double maximum_raw_area_ratio = 0.0;

  struct HumanCandidate {
    const Detection* detection{nullptr};
    double raw_bottom{0.0};
    double raw_area_ratio{0.0};
    double expanded_left{0.0};
    double expanded_right{0.0};
    double expanded_top{0.0};
    double expanded_bottom{0.0};
  };
  std::vector<HumanCandidate> candidates;

  for (const auto& det : detections) {
    if (det.class_name != "Human" ||
        det.confidence < cfg_.obstacle_min_confidence ||
        det.bbox.width <= 0.0f || det.bbox.height <= 0.0f) {
      continue;
    }

    const double expand_px = std::max(
        static_cast<double>(cfg_.human_horizontal_expand_px),
        static_cast<double>(det.bbox.width) * cfg_.human_horizontal_expand_width_ratio);
    const double raw_left = det.bbox.x;
    const double raw_top = det.bbox.y;
    const double raw_right = det.bbox.x + det.bbox.width;
    const double raw_bottom = det.bbox.y + det.bbox.height;
    const double expanded_left = clampValue(raw_left - expand_px, 0.0,
                                            static_cast<double>(image_width));
    const double expanded_right = clampValue(raw_right + expand_px, 0.0,
                                             static_cast<double>(image_width));
    const double expanded_top = clampValue(raw_top, 0.0,
                                           static_cast<double>(image_height));
    const double expanded_bottom = clampValue(raw_bottom, 0.0,
                                              static_cast<double>(image_height));
    if (expanded_right <= expanded_left || expanded_bottom <= expanded_top) {
      continue;
    }

    const double raw_area_ratio = image_area > 0.0
                                      ? clampValue(static_cast<double>(det.bbox.width) *
                                                       det.bbox.height / image_area,
                                                   0.0, 1.0)
                                      : 0.0;
    candidates.push_back({
        &det, raw_bottom, raw_area_ratio, expanded_left, expanded_right,
        expanded_top, expanded_bottom});
  }

  debug_info_.human_candidate_count = static_cast<int>(candidates.size());
  if (!candidates.empty()) {
    const auto nearest_it = std::max_element(
        candidates.begin(), candidates.end(),
        [](const HumanCandidate& a, const HumanCandidate& b) {
          if (a.raw_bottom != b.raw_bottom) {
            return a.raw_bottom < b.raw_bottom;
          }
          if (a.raw_area_ratio != b.raw_area_ratio) {
            return a.raw_area_ratio < b.raw_area_ratio;
          }
          return a.detection->confidence < b.detection->confidence;
        });
    const size_t nearest_index = static_cast<size_t>(nearest_it - candidates.begin());
    has_valid_human = true;
    valid_human_count = 1;

    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
      const auto& candidate = candidates[candidate_index];
      const auto& det = *candidate.detection;
      const bool active = candidate_index == nearest_index;
      const double expanded_width = candidate.expanded_right - candidate.expanded_left;
      const double expanded_height = candidate.expanded_bottom - candidate.expanded_top;

      LaneHumanDebug human_debug;
      human_debug.raw_bbox = det.bbox;
      human_debug.expanded_bbox = cv::Rect2f(
          static_cast<float>(candidate.expanded_left),
          static_cast<float>(candidate.expanded_top),
          static_cast<float>(expanded_width),
          static_cast<float>(expanded_height));
      human_debug.raw_area_ratio = static_cast<float>(candidate.raw_area_ratio);
      human_debug.active = active;

      bool line_samples_valid = fit_available;
      bool line_intersects = false;
      if (line_samples_valid) {
        human_debug.fit_sample_points.reserve(
            static_cast<size_t>(cfg_.human_line_sample_count));
        const double sample_denominator =
            static_cast<double>(cfg_.human_line_sample_count - 1);
        for (int sample_index = 0; sample_index < cfg_.human_line_sample_count;
             ++sample_index) {
          const double ratio = static_cast<double>(sample_index) / sample_denominator;
          const double sample_y = candidate.expanded_top + expanded_height * ratio;
          const double sample_x = evalPoly(fit_coeffs, sample_y);
          if (!std::isfinite(sample_x)) {
            line_samples_valid = false;
            human_debug.fit_sample_points.clear();
            break;
          }
          human_debug.fit_sample_points.emplace_back(static_cast<float>(sample_x),
                                                      static_cast<float>(sample_y));
          if (sample_x >= candidate.expanded_left &&
              sample_x <= candidate.expanded_right) {
            line_intersects = true;
          }
        }
      }
      human_debug.fit_available = line_samples_valid;
      human_debug.line_intersects = line_samples_valid && line_intersects;

      const bool passable = line_samples_valid && !line_intersects;
      const bool stop_candidate = !passable && candidate.raw_area_ratio >= area_threshold;
      human_debug.passable = passable;
      human_debug.stop_candidate = stop_candidate;
      debug_info_.humans.push_back(human_debug);

      if (active) {
        maximum_raw_area_ratio = candidate.raw_area_ratio;
        all_current_humans_passable = passable;
        any_line_intersects = human_debug.line_intersects;
        any_stop_candidate = stop_candidate;
        any_no_fit_human = !line_samples_valid;
      }
    }
  }

  if (!human_stop_active_) {
    human_clear_confirm_count_ = 0;
    if (any_stop_candidate) {
      human_stop_confirm_count_ = std::min(
          human_stop_confirm_count_ + 1, cfg_.human_stop_confirm_frames);
    } else {
      human_stop_confirm_count_ = 0;
    }
    if (human_stop_confirm_count_ >= cfg_.human_stop_confirm_frames) {
      human_stop_active_ = true;
      human_count_at_stop_ = valid_human_count;
    }
  } else {
    // Detection loss is now treated as clearance.  Keep the stop latched
    // while a Human is still observed but unsafe; once no valid Human remains
    // in the current frame, let the execution layer resume the controller.
    if (!has_valid_human) {
      human_stop_active_ = false;
      human_stop_confirm_count_ = 0;
      human_clear_confirm_count_ = 0;
      human_count_at_stop_ = 0;
    } else {
      const bool clear_evidence = fit_available && all_current_humans_passable &&
                                  valid_human_count >= human_count_at_stop_;
      if (clear_evidence) {
        ++human_clear_confirm_count_;
        if (human_clear_confirm_count_ >= cfg_.human_clear_confirm_frames) {
          human_stop_active_ = false;
          human_stop_confirm_count_ = 0;
          human_clear_confirm_count_ = 0;
          human_count_at_stop_ = 0;
        }
      } else {
        human_clear_confirm_count_ = 0;
      }
    }
  }

  if (human_stop_active_) {
    human_state_ = "OBSTACLE_STOP";
  } else if (has_valid_human && all_current_humans_passable) {
    human_state_ = "PASSABLE";
  } else if (has_valid_human && any_no_fit_human) {
    human_state_ = "NO_FIT_WAIT_NEAR";
  } else if (has_valid_human && any_line_intersects) {
    human_state_ = "OVERLAP_WAIT_NEAR";
  } else {
    human_state_ = "NONE";
  }

  debug_info_.human_passable = has_valid_human && all_current_humans_passable &&
                               !human_stop_active_;
  debug_info_.human_line_intersects = any_line_intersects;
  debug_info_.human_stop_candidate = any_stop_candidate;
  debug_info_.human_stop_active = human_stop_active_;
  debug_info_.human_clear_confirming = human_stop_active_ &&
                                        human_clear_confirm_count_ > 0;
  debug_info_.human_raw_area_ratio = static_cast<float>(maximum_raw_area_ratio);
  debug_info_.human_state = human_state_;
  debug_info_.human_stop_confirm_count = human_stop_confirm_count_;
  debug_info_.human_clear_confirm_count = human_clear_confirm_count_;
  debug_info_.human_count_at_stop = human_count_at_stop_;
  debug_info_.human_valid_count = valid_human_count;
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

void LaneDecision::evaluateCoinsShadow(
    const std::vector<Detection>& detections, int image_width,
    int image_height, const std::vector<cv::Point3f>& fit_points,
    const std::vector<double>& fit_coeffs,
    bool fit_valid) {
  debug_info_.coins.clear();
  debug_info_.coin_on_route_count = 0;
  debug_info_.coin_reachable_count = 0;
  debug_info_.coin_too_far_count = 0;
  debug_info_.coin_blocked_count = 0;
  debug_info_.coin_no_fit_count = 0;
  debug_info_.coin_wait_far_count = 0;
  debug_info_.coin_near_committed_count = 0;

  if (!cfg_.enable_coin_shadow_evaluation || image_width <= 1 ||
      image_height <= 1) {
    return;
  }
  const auto [fit_y_min, fit_y_max] = fitYBounds(fit_points, image_height);

  for (const auto& coin : detections) {
    if (coin.class_name != "Gold" || coin.confidence < cfg_.coin_min_confidence ||
        coin.bbox.width <= 0.0f || coin.bbox.height <= 0.0f) {
      continue;
    }

    LaneCoinDebug coin_debug;
    coin_debug.bbox = coin.bbox;
    coin_debug.confidence = coin.confidence;
    const double sqrt_bbox_area = std::sqrt(
        static_cast<double>(coin.bbox.width) * coin.bbox.height);
    coin_debug.sqrt_bbox_area = static_cast<float>(sqrt_bbox_area);
    const double coin_x = clampValue(
        static_cast<double>(coin.bbox.x + coin.bbox.width * 0.5f),
        0.0, static_cast<double>(image_width - 1));
    const double coin_y = clampValue(
        static_cast<double>(coin.bbox.y + coin.bbox.height),
        0.0, static_cast<double>(image_height - 1));
    coin_debug.ground_point = cv::Point2f(
        static_cast<float>(coin_x), static_cast<float>(coin_y));
    const double coin_y_ratio = coin_y / static_cast<double>(image_height - 1);

    // Limit evaluation to the part of the image where the fitted path is
    // useful. Far coins wait without consuming obstacle checks; near coins
    // are already committed/passing and must not turn grey due to projection.
    if (coin_y_ratio < cfg_.coin_evaluate_min_y_ratio) {
      coin_debug.classification = "WAIT_FAR";
      ++debug_info_.coin_wait_far_count;
      debug_info_.coins.push_back(std::move(coin_debug));
      continue;
    }
    if (coin_y_ratio > cfg_.coin_near_committed_y_ratio) {
      coin_debug.classification = "NEAR_COMMITTED";
      ++debug_info_.coin_near_committed_count;
      debug_info_.coins.push_back(std::move(coin_debug));
      continue;
    }

    if (!fit_valid || fit_coeffs.size() < 2) {
      coin_debug.classification = "NO_FIT";
      ++debug_info_.coin_no_fit_count;
      debug_info_.coins.push_back(std::move(coin_debug));
      continue;
    }

    const PolynomialClosestPoint closest = closestPointOnPolynomial(
        fit_coeffs, coin_x, coin_y, fit_y_min, fit_y_max);
    if (!closest.valid) {
      coin_debug.classification = "NO_FIT";
      ++debug_info_.coin_no_fit_count;
      debug_info_.coins.push_back(std::move(coin_debug));
      continue;
    }

    // Use the true shortest Euclidean distance to the bounded fitted curve.
    // The coarse scan avoids the wrong local minimum; a short golden-section
    // refinement keeps the endpoint on the polynomial even in a sharp bend.
    const double normal_distance = closest.distance;
    const double signed_normal_distance = closest.signed_distance;
    const double car_half_width = clampValue(
        static_cast<double>(cfg_.coin_car_half_width_area_scale) * sqrt_bbox_area,
        static_cast<double>(cfg_.coin_car_half_width_min_px),
        static_cast<double>(cfg_.coin_car_half_width_max_px));
    double direction_x = 1.0;
    double direction_y = 0.0;
    if (normal_distance > 1e-6) {
      direction_x = (coin_x - closest.x) / normal_distance;
      direction_y = (coin_y - closest.y) / normal_distance;
    }
    const double coin_half_width_normal = 0.5 * (
        std::abs(direction_x) * coin.bbox.width +
        std::abs(direction_y) * coin.bbox.height);
    const double hit_radius = car_half_width + coin_half_width_normal +
                              cfg_.coin_hit_margin_px;
    const double extra_distance = std::max(0.0, normal_distance - hit_radius);
    const double reachable_extra = clampValue(
        static_cast<double>(cfg_.coin_reachable_extra_area_scale) * sqrt_bbox_area,
        static_cast<double>(cfg_.coin_reachable_extra_min_px),
        static_cast<double>(cfg_.coin_reachable_extra_max_px));

    coin_debug.local_path_point = cv::Point2f(
        static_cast<float>(closest.x), static_cast<float>(closest.y));
    coin_debug.path_slope = static_cast<float>(closest.slope);
    coin_debug.normal_distance = static_cast<float>(normal_distance);
    coin_debug.hit_radius = static_cast<float>(hit_radius);
    coin_debug.extra_distance = static_cast<float>(extra_distance);
    coin_debug.reachable_extra = static_cast<float>(reachable_extra);
    coin_debug.side = signed_normal_distance < 0.0 ? "LEFT" : "RIGHT";

    const bool geometrically_reachable = extra_distance <= reachable_extra;
    if (geometrically_reachable) {
      // Approximate the smallest approach corridor from the fitted path to the
      // coin. Only Car/Human boxes between the vehicle and the coin, or just
      // beyond the coin within the configured lookahead, may veto it.
      const double lookahead = cfg_.coin_obstacle_lookahead_ratio * image_height;
      const double approach_top = std::max(0.0, coin_y - lookahead);
      double approach_min_x = coin_x;
      double approach_max_x = coin_x;
      constexpr int kApproachSamples = 7;
      for (int sample_index = 0; sample_index < kApproachSamples; ++sample_index) {
        const double sample_ratio = static_cast<double>(sample_index) /
                                    static_cast<double>(kApproachSamples - 1);
        const double sample_y = approach_top +
                                (image_height - 1 - approach_top) * sample_ratio;
        const double sample_x = evalPoly(fit_coeffs, sample_y);
        if (std::isfinite(sample_x)) {
          approach_min_x = std::min(approach_min_x, sample_x);
          approach_max_x = std::max(approach_max_x, sample_x);
        }
      }
      const double approach_left = approach_min_x - car_half_width -
                                   cfg_.coin_obstacle_expand_px;
      const double approach_right = approach_max_x + car_half_width +
                                    cfg_.coin_obstacle_expand_px;

      for (const auto& obstacle : detections) {
        if ((obstacle.class_name != "Car" && obstacle.class_name != "Human") ||
            obstacle.confidence < cfg_.obstacle_min_confidence ||
            obstacle.bbox.width <= 0.0f || obstacle.bbox.height <= 0.0f) {
          continue;
        }
        const double obstacle_left = obstacle.bbox.x - cfg_.coin_obstacle_expand_px;
        const double obstacle_right = obstacle.bbox.x + obstacle.bbox.width +
                                      cfg_.coin_obstacle_expand_px;
        const double obstacle_top = obstacle.bbox.y;
        const double obstacle_bottom = obstacle.bbox.y + obstacle.bbox.height;
        const bool vertical_overlap = obstacle_bottom >= approach_top &&
                                      obstacle_top <= image_height;
        const bool horizontal_overlap = obstacle_right >= approach_left &&
                                        obstacle_left <= approach_right;
        if (vertical_overlap && horizontal_overlap) {
          coin_debug.obstacle_blocked = true;
          coin_debug.blocked_by = obstacle.class_name;
          break;
        }
      }
    }

    if (coin_debug.obstacle_blocked) {
      coin_debug.classification = "BLOCKED";
      ++debug_info_.coin_blocked_count;
    } else if (extra_distance <= 0.0) {
      coin_debug.classification = "ON_ROUTE";
      ++debug_info_.coin_on_route_count;
    } else if (geometrically_reachable) {
      coin_debug.classification = "REACHABLE";
      ++debug_info_.coin_reachable_count;
    } else {
      coin_debug.classification = "TOO_FAR";
      ++debug_info_.coin_too_far_count;
    }
    debug_info_.coins.push_back(std::move(coin_debug));
  }

  double left_route_score = 0.0;
  double right_route_score = 0.0;
  for (auto& coin : debug_info_.coins) {
    if (coin.classification != "REACHABLE") {
      continue;
    }
    const double distance_quality = 1.0 - clampValue(
        static_cast<double>(coin.extra_distance) /
            std::max(1.0, static_cast<double>(coin.reachable_extra)),
        0.0, 1.0);
    const double image_ratio = clampValue(
        static_cast<double>(coin.ground_point.y) /
            static_cast<double>(image_height - 1),
        0.0, 1.0);
    const double window_span = std::max(
        1e-6, static_cast<double>(cfg_.coin_near_committed_y_ratio -
                                  cfg_.coin_evaluate_min_y_ratio));
    const double progress = clampValue(
        (image_ratio - cfg_.coin_evaluate_min_y_ratio) / window_span,
        0.0, 1.0);
    coin.route_score = static_cast<float>(
        1.0 + 0.5 * distance_quality + 0.25 * progress);
    if (coin.side == "LEFT") {
      left_route_score += coin.route_score;
    } else if (coin.side == "RIGHT") {
      right_route_score += coin.route_score;
    }
  }

  const auto bestAvailableSide = [&]() {
    if (left_route_score <= 0.0 && right_route_score <= 0.0) {
      return std::string("NONE");
    }
    if (std::abs(left_route_score - right_route_score) <= 1e-6) {
      return cfg_.outer_side == "right" ? std::string("RIGHT")
                                         : std::string("LEFT");
    }
    return left_route_score > right_route_score ? std::string("LEFT")
                                                 : std::string("RIGHT");
  };

  if (coin_selected_side_ != "LEFT" && coin_selected_side_ != "RIGHT") {
    coin_selected_side_ = bestAvailableSide();
    coin_side_clear_count_ = 0;
  } else {
    const double selected_score = coin_selected_side_ == "LEFT"
                                      ? left_route_score : right_route_score;
    if (selected_score > 0.0) {
      coin_side_clear_count_ = 0;
    } else {
      coin_side_clear_count_ = std::min(
          coin_side_clear_count_ + 1, cfg_.coin_side_clear_frames);
      if (coin_side_clear_count_ >= cfg_.coin_side_clear_frames) {
        coin_selected_side_ = bestAvailableSide();
        coin_side_clear_count_ = 0;
      }
    }
  }

  for (auto& coin : debug_info_.coins) {
    coin.selected_for_route = coin.classification == "REACHABLE" &&
                              coin.side == coin_selected_side_;
  }
  debug_info_.coin_selected_side = coin_selected_side_;
  debug_info_.coin_left_route_score = static_cast<float>(left_route_score);
  debug_info_.coin_right_route_score = static_cast<float>(right_route_score);
  debug_info_.coin_side_clear_count = coin_side_clear_count_;
}

void LaneDecision::buildCoinRoutePreview(
    const std::vector<Detection>& detections, int image_width,
    int image_height, const std::vector<cv::Point3f>& base_fit_points,
    const std::vector<double>& base_fit_coeffs, bool fit_valid) {
  debug_info_.coin_route_points.clear();
  debug_info_.coin_route_coeffs.clear();
  debug_info_.coin_route_base_coeffs = base_fit_coeffs;
  debug_info_.coin_route_candidate_valid = false;
  debug_info_.coin_route_control_active = false;
  debug_info_.coin_route_target_count = 0;
  debug_info_.coin_route_heading = 0.0f;
  debug_info_.coin_route_curvature = 0.0f;
  debug_info_.coin_route_max_offset_px = 0.0f;
  debug_info_.coin_route_reject_reason = "DISABLED";
  for (auto& coin : debug_info_.coins) {
    coin.route_targeted = false;
    coin.route_target_point = cv::Point2f();
  }

  if ((!cfg_.enable_coin_route_preview && !cfg_.enable_coin_route_control) ||
      image_width <= 1 || image_height <= 1) {
    return;
  }
  if (!fit_valid || base_fit_coeffs.size() < 2) {
    debug_info_.coin_route_reject_reason = "NO_FIT";
    return;
  }
  if (car_avoidance_active_) {
    debug_info_.coin_route_reject_reason = "CAR_AVOIDANCE_ACTIVE";
    return;
  }
  if (finish_stop_active_ || human_stop_active_) {
    debug_info_.coin_route_reject_reason = "TASK_STOP_ACTIVE";
    return;
  }

  std::vector<LaneCoinDebug*> selected_coins;
  for (auto& coin : debug_info_.coins) {
    if (coin.selected_for_route && coin.classification == "REACHABLE") {
      selected_coins.push_back(&coin);
    }
  }
  if (selected_coins.empty()) {
    debug_info_.coin_route_reject_reason = "NO_SELECTED_TARGET";
    return;
  }
  std::sort(selected_coins.begin(), selected_coins.end(),
            [](const LaneCoinDebug* lhs, const LaneCoinDebug* rhs) {
              if (std::abs(lhs->ground_point.y - rhs->ground_point.y) > 1e-3f) {
                return lhs->ground_point.y > rhs->ground_point.y;
              }
              return lhs->route_score > rhs->route_score;
            });
  if (static_cast<int>(selected_coins.size()) > cfg_.coin_route_max_targets) {
    selected_coins.resize(cfg_.coin_route_max_targets);
  }

  const auto [fit_y_min, fit_y_max] = fitYBounds(
      base_fit_points, image_height);
  if (fit_y_max - fit_y_min < 2.0) {
    debug_info_.coin_route_reject_reason = "SHORT_FIT_RANGE";
    return;
  }

  struct RouteTarget {
    LaneCoinDebug* coin{nullptr};
    double anchor_x{0.0};
    double anchor_y{0.0};
    double delta_x{0.0};
    double support_low_y{0.0};
    double support_high_y{0.0};
  };
  std::vector<RouteTarget> targets;
  targets.reserve(selected_coins.size());
  const double approach_span = std::max(
      2.0, static_cast<double>(cfg_.coin_route_approach_span_ratio) *
               image_height);
  const double return_span = std::max(
      2.0, static_cast<double>(cfg_.coin_route_return_span_ratio) *
               image_height);

  for (LaneCoinDebug* coin : selected_coins) {
    const double path_x = coin->local_path_point.x;
    const double path_y = coin->local_path_point.y;
    const double coin_x = coin->ground_point.x;
    const double coin_y = coin->ground_point.y;
    const double distance = std::hypot(coin_x - path_x, coin_y - path_y);
    if (!std::isfinite(distance) || distance <= 1e-6) {
      continue;
    }
    const double desired_distance = std::max(
        0.0, static_cast<double>(coin->hit_radius) -
                 cfg_.coin_route_overlap_margin_px);
    const double shift = std::max(0.0, distance - desired_distance);
    const double anchor_x = path_x + (coin_x - path_x) * shift / distance;
    const double anchor_y = clampValue(
        path_y + (coin_y - path_y) * shift / distance,
        fit_y_min, fit_y_max);
    const double base_x_at_anchor = evalPoly(base_fit_coeffs, anchor_y);
    if (!std::isfinite(anchor_x) || !std::isfinite(anchor_y) ||
        !std::isfinite(base_x_at_anchor)) {
      continue;
    }

    RouteTarget target;
    target.coin = coin;
    target.anchor_x = anchor_x;
    target.anchor_y = anchor_y;
    target.delta_x = anchor_x - base_x_at_anchor;
    target.support_low_y = std::max(fit_y_min, anchor_y - return_span);
    target.support_high_y = std::min(fit_y_max, anchor_y + approach_span);
    targets.push_back(target);
    coin->route_targeted = true;
    coin->route_target_point = cv::Point2f(
        static_cast<float>(anchor_x), static_cast<float>(anchor_y));
  }
  if (targets.empty()) {
    debug_info_.coin_route_reject_reason = "NO_VALID_ANCHOR";
    return;
  }
  debug_info_.coin_route_target_count = static_cast<int>(targets.size());

  auto routeOffsetAtY = [&](double y) {
    double selected_offset = 0.0;
    for (const auto& target : targets) {
      double blend = 0.0;
      if (y >= target.support_low_y && y <= target.anchor_y) {
        const double span = std::max(1e-6,
                                     target.anchor_y - target.support_low_y);
        blend = smoothStep((y - target.support_low_y) / span);
      } else if (y > target.anchor_y && y <= target.support_high_y) {
        const double span = std::max(1e-6,
                                     target.support_high_y - target.anchor_y);
        blend = smoothStep((target.support_high_y - y) / span);
      }
      const double candidate = target.delta_x * blend;
      if (std::abs(candidate) > std::abs(selected_offset)) {
        selected_offset = candidate;
      }
    }
    return selected_offset;
  };

  std::vector<cv::Point3f> route_points;
  route_points.reserve(cfg_.coin_route_sample_count + targets.size() + 2);
  for (int index = 0; index < cfg_.coin_route_sample_count; ++index) {
    const double ratio = static_cast<double>(index) /
                         static_cast<double>(cfg_.coin_route_sample_count - 1);
    const double y = fit_y_min + (fit_y_max - fit_y_min) * ratio;
    const double x = evalPoly(base_fit_coeffs, y) + routeOffsetAtY(y);
    const float weight = (index == 0 ||
                          index == cfg_.coin_route_sample_count - 1)
                             ? cfg_.coin_route_endpoint_weight
                             : cfg_.coin_route_base_weight;
    route_points.emplace_back(static_cast<float>(x), static_cast<float>(y),
                              weight);
  }
  for (const auto& target : targets) {
    route_points.emplace_back(
        static_cast<float>(target.anchor_x),
        static_cast<float>(target.anchor_y), cfg_.coin_route_target_weight);
  }

  std::vector<double> route_coeffs;
  double route_heading = 0.0;
  double route_curvature = 0.0;
  if (!fitCenterlineAndComputeGeometry(
          route_points, image_height, 2, &route_coeffs,
          &route_heading, &route_curvature)) {
    debug_info_.coin_route_reject_reason = "ROUTE_FIT_FAILED";
    return;
  }
  debug_info_.coin_route_points = route_points;
  debug_info_.coin_route_coeffs = route_coeffs;
  debug_info_.coin_route_heading = static_cast<float>(route_heading);
  debug_info_.coin_route_curvature = static_cast<float>(route_curvature);

  double maximum_offset = 0.0;
  constexpr int kValidationSamples = 48;
  for (int index = 0; index <= kValidationSamples; ++index) {
    const double ratio = static_cast<double>(index) / kValidationSamples;
    const double y = fit_y_min + (fit_y_max - fit_y_min) * ratio;
    maximum_offset = std::max(
        maximum_offset,
        std::abs(evalPoly(route_coeffs, y) -
                 evalPoly(base_fit_coeffs, y)));
  }
  debug_info_.coin_route_max_offset_px = static_cast<float>(maximum_offset);
  if (maximum_offset > cfg_.coin_route_max_offset_px) {
    debug_info_.coin_route_reject_reason = "OFFSET_LIMIT";
    return;
  }
  if (std::abs(route_heading) > cfg_.coin_route_max_abs_heading) {
    debug_info_.coin_route_reject_reason = "HEADING_LIMIT";
    return;
  }
  if (std::abs(route_curvature) > cfg_.coin_route_max_abs_curvature) {
    debug_info_.coin_route_reject_reason = "CURVATURE_LIMIT";
    return;
  }

  for (const auto& target : targets) {
    const PolynomialClosestPoint route_closest = closestPointOnPolynomial(
        route_coeffs, target.coin->ground_point.x,
        target.coin->ground_point.y, fit_y_min, fit_y_max);
    if (!route_closest.valid ||
        route_closest.distance > target.coin->hit_radius + 1e-3) {
      debug_info_.coin_route_reject_reason = "MISSES_TARGET";
      return;
    }
  }

  double route_check_min_y = fit_y_max;
  for (const auto& target : targets) {
    route_check_min_y = std::min(route_check_min_y, target.support_low_y);
  }
  for (const auto& obstacle : detections) {
    if ((obstacle.class_name != "Car" && obstacle.class_name != "Human") ||
        obstacle.confidence < cfg_.obstacle_min_confidence ||
        obstacle.bbox.width <= 0.0f || obstacle.bbox.height <= 0.0f) {
      continue;
    }
    const double clearance = cfg_.coin_route_obstacle_clearance_px;
    const double obstacle_left = obstacle.bbox.x - clearance;
    const double obstacle_right = obstacle.bbox.x + obstacle.bbox.width + clearance;
    const double obstacle_top = obstacle.bbox.y - clearance;
    const double obstacle_bottom = obstacle.bbox.y + obstacle.bbox.height + clearance;
    for (int index = 0; index <= kValidationSamples; ++index) {
      const double ratio = static_cast<double>(index) / kValidationSamples;
      const double y = route_check_min_y +
                       (fit_y_max - route_check_min_y) * ratio;
      if (y < obstacle_top || y > obstacle_bottom) {
        continue;
      }
      const double x = evalPoly(route_coeffs, y);
      if (x >= obstacle_left && x <= obstacle_right) {
        debug_info_.coin_route_reject_reason = "ROUTE_BLOCKED_" +
                                               obstacle.class_name;
        return;
      }
    }
  }

  debug_info_.coin_route_candidate_valid = true;
  debug_info_.coin_route_reject_reason = "VALID";
}

void LaneDecision::populateDebugInfo(const std::vector<Band>& bands,
                                     const std::vector<cv::Point3f>& raw_points,
                                     const std::vector<cv::Point3f>& fit_points,
                                     int image_width, int image_height,
                                     const std::vector<double>& fit_coeffs) {
  debug_info_.bands.clear();
  debug_info_.raw_points = raw_points;
  debug_info_.fit_points = fit_points;
  debug_info_.fit_coeffs = fit_coeffs;
  debug_info_.car_avoidance_active = car_avoidance_active_;
  debug_info_.car_side = car_side_;
  debug_info_.car_side_candidate = car_side_candidate_;
  debug_info_.car_side_fit_valid = car_side_fit_valid_;
  debug_info_.car_side_fit_x = static_cast<float>(car_side_fit_x_);
  debug_info_.car_side_fit_y = static_cast<float>(car_side_fit_y_);
  debug_info_.car_side_fit_relation = car_side_fit_relation_;
  debug_info_.car_mask_connectivity = car_mask_connectivity_;
  debug_info_.car_side_source = car_side_source_;
  debug_info_.car_left_seed_pixels = car_left_seed_pixels_;
  debug_info_.car_right_seed_pixels = car_right_seed_pixels_;
  debug_info_.car_common_component_pixels = car_common_component_pixels_;
  debug_info_.car_connectivity_roi = car_connectivity_roi_;
  debug_info_.car_left_seed_roi = car_left_seed_roi_;
  debug_info_.car_right_seed_roi = car_right_seed_roi_;
  debug_info_.car_side_confirm_count = car_side_confirm_count_;
  debug_info_.car_template_state = carTemplateStateName();
  debug_info_.car_template_active = car_avoidance_active_;
  debug_info_.car_template_side = car_template_side_;
  debug_info_.car_template_point_count = car_template_point_count_;
  debug_info_.car_template_cache_active = !car_template_cached_coeffs_.empty();
  debug_info_.car_encoder_start_count = car_encoder_start_count_;
  debug_info_.car_encoder_delta = car_encoder_delta_;
  debug_info_.car_encoder_detour_target = cfg_.car_encoder_detour_counts;
  debug_info_.car_encoder_return_delta = car_encoder_return_delta_;
  debug_info_.car_encoder_return_progress =
      static_cast<float>(car_encoder_return_progress_);
  debug_info_.car_encoder_fault_age = car_encoder_fault_age_;
  debug_info_.car_rearm_clear_count = car_rearm_clear_count_;
  if (car_detection_active_) {
    debug_info_.car_bbox = cv::Rect2f(
        static_cast<float>(car_left_x_), static_cast<float>(car_top_y_),
        static_cast<float>(std::max(0.0, car_right_x_ - car_left_x_)),
        static_cast<float>(std::max(0.0, car_bottom_y_ - car_top_y_)));
  }
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
