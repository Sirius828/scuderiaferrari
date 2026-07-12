#!/usr/bin/env python3
"""Line follower tuning helper.

This script avoids ros2cli daemon issues by using rclpy directly.
It can:
- apply the isolation tuning parameter set to /line_follower_controller
- watch lane_state, line_follower/debug, and cmd_vel
- optionally enable/disable line following through /line_follower/set_enabled
"""

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


ISOLATION_PARAMS = {
    "Kp": 0.82,
    "Ki": 0.0,
    "Kd": 0.045,
    "enable_error_gain_schedule": False,
    "linear_speed": 0.60,
    "enable_dynamic_speed": False,
    "min_linear_speed": 0.60,
    "enable_curve_adaptive_control": False,
    "max_steering": 0.82,
    "steering_slew_rate": 4.5,
    "steering_return_slew_rate": 0.0,
    "offset_deadband": 0.020,
    "heading_gain": 0.0,
    "heading_term_limit": 0.0,
    "lookahead_heading_gain": 0.0,
    "lookahead_error_limit": 0.0,
    "curvature_gain": 0.0,
    "curvature_term_limit": 0.0,
    "lane_term_offset_fade_start": 0.45,
    "lane_term_offset_fade_full": 0.80,
    "enable_start_boost": False,
    "controller_log_mode": "pid_tuning",
    "pid_tuning_log_hz": 20.0,
}


FLOAT_RE = re.compile(r"([A-Za-z_]+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
BOOL_RE = re.compile(r"([A-Za-z_]+)=(True|False|true|false)")


@dataclass
class Sample:
    t: float
    offset: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    conf: float = 0.0
    steering: float = 0.0
    wheel_rps: float = 0.0
    speed_mps: float = 0.0
    curve_factor: float = 0.0
    max_steer: float = 0.0
    mode: str = ""
    fit_points: int = 0
    raw_points: int = 0
    branch_detected: bool = False
    branch_score: int = 0
    transition: bool = False


@dataclass
class Window:
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample):
        self.samples.append(sample)
        cutoff = sample.t - 20.0
        while self.samples and self.samples[0].t < cutoff:
            self.samples.pop(0)

    def summary(self) -> str:
        if not self.samples:
            return "no samples yet"
        offsets = [s.offset for s in self.samples]
        headings = [s.heading for s in self.samples]
        steers = [s.steering for s in self.samples]
        speeds = [s.speed_mps for s in self.samples]
        confs = [s.conf for s in self.samples]
        fit_points = [s.fit_points for s in self.samples if s.fit_points > 0]
        modes = {}
        for s in self.samples:
            modes[s.mode] = modes.get(s.mode, 0) + 1

        sat_count = sum(
            1 for s in self.samples
            if s.max_steer > 0.0 and abs(s.steering) >= 0.95 * s.max_steer
        )
        within_040 = sum(1 for value in offsets if abs(value) <= 0.40) / len(offsets)
        max_abs_offset = max(abs(value) for value in offsets)
        offset_jumps = sum(
            1 for previous, current in zip(offsets, offsets[1:])
            if abs(current - previous) > 0.15
        )
        heading_leads = sum(
            1 for s in self.samples
            if abs(s.heading) > 0.35 and abs(s.offset) < 0.10
        )
        opposite = sum(
            1 for s in self.samples
            if abs(s.offset) > 0.04 and abs(s.steering) > 0.04 and s.offset * s.steering < 0.0
        )
        low_conf = sum(1 for s in self.samples if s.conf < 0.35)
        high_offset = sum(1 for s in self.samples if abs(s.offset) > 0.60)
        branch_frames = sum(1 for s in self.samples if s.branch_detected or s.branch_score > 0)
        transition_frames = sum(1 for s in self.samples if s.transition)
        modes_text = ",".join(
            f"{mode or '?'}:{count}" for mode, count in sorted(modes.items(), key=lambda item: -item[1])[:4]
        )

        return (
            f"n={len(self.samples)} "
            f"modes={modes_text} "
            f"offset avg={statistics.fmean(offsets):+.3f} "
            f"abs95={percentile([abs(v) for v in offsets], 95):.3f} "
            f"within040={within_040 * 100:.1f}% max_abs={max_abs_offset:.3f} jumps15={offset_jumps} "
            f"heading abs95={percentile([abs(v) for v in headings], 95):.3f} "
            f"steer abs95={percentile([abs(v) for v in steers], 95):.3f} "
            f"maxS avg={statistics.fmean([s.max_steer for s in self.samples]):.2f} "
            f"speed avg/min/max={statistics.fmean(speeds):.2f}/{min(speeds):.2f}/{max(speeds):.2f} "
            f"conf min={min(confs):.2f} low_conf={low_conf} "
            f"high_offset={high_offset} "
            f"fit_min={min(fit_points) if fit_points else 0} "
            f"branch={branch_frames} "
            f"transition={transition_frames} "
            f"sat={sat_count} "
            f"heading_leads_offset={heading_leads} "
            f"opposite_sign={opposite}"
        )


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def parameter_value(value):
    pv = ParameterValue()
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = value
    elif isinstance(value, float):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = value
    elif isinstance(value, int):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = value
    elif isinstance(value, str):
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = value
    else:
        raise TypeError(f"unsupported parameter type for {value!r}")
    return pv


class LineFollowDiag(Node):
    def __init__(self):
        super().__init__("line_follow_diag")
        self.window = Window()
        self.latest = Sample(t=time.time())
        self.create_subscription(String, "/perception/lane_state", self.on_lane_state, 10)
        self.create_subscription(String, "/perception/lane_debug", self.on_lane_debug, 10)
        self.create_subscription(String, "/line_follower/debug", self.on_debug, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.param_client = self.create_client(
            SetParameters,
            "/line_follower_controller_cpp/set_parameters",
        )
        self.enable_client = self.create_client(
            SetBool,
            "/line_follower/set_enabled",
        )

    def on_lane_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        offset_y07 = data.get("offset_y07")
        offset_y08 = data.get("offset_y08")
        offset_y09 = data.get("offset_y09")
        if offset_y07 is not None and offset_y08 is not None and offset_y09 is not None:
            self.latest.offset = (
                0.2 * float(offset_y07) +
                0.3 * float(offset_y08) +
                0.5 * float(offset_y09))
        self.latest.heading = float(data.get("heading_error", self.latest.heading))
        self.latest.curvature = float(data.get("curvature", self.latest.curvature))
        self.latest.conf = float(data.get("confidence", self.latest.conf))
        self.record()

    def on_debug(self, msg):
        fields = dict(FLOAT_RE.findall(msg.data))
        bools = dict(BOOL_RE.findall(msg.data))
        self.latest.mode = parse_token(msg.data, "mode", self.latest.mode)
        self.latest.speed_mps = float(fields.get("speed_mps", self.latest.speed_mps))
        self.latest.curve_factor = float(fields.get("curve_factor", self.latest.curve_factor))
        self.latest.max_steer = float(fields.get("max_steer", self.latest.max_steer))
        self.latest.steering = float(fields.get("steering_cmd", self.latest.steering))
        if "weighted_offset" in fields:
            self.latest.offset = float(fields["weighted_offset"])
        if "enabled" in bools and bools["enabled"].lower() == "false":
            self.latest.mode = "disabled"
        self.record()

    def on_lane_debug(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        fit_points = data.get("fit_points", self.latest.fit_points)
        raw_points = data.get("raw_points", self.latest.raw_points)
        # fused_perception publishes the actual point arrays. Older diagnostics
        # expected integer counts, so accept both wire formats.
        self.latest.fit_points = len(fit_points) if isinstance(fit_points, list) else int(fit_points)
        self.latest.raw_points = len(raw_points) if isinstance(raw_points, list) else int(raw_points)
        self.latest.branch_detected = bool(data.get("branch_detected", self.latest.branch_detected))
        self.latest.branch_score = int(data.get("branch_score", self.latest.branch_score))
        self.latest.transition = bool(data.get("transition", self.latest.transition))
        self.record()

    def on_cmd_vel(self, msg):
        self.latest.wheel_rps = float(msg.linear.x)
        self.latest.steering = float(msg.angular.z)
        self.record()

    def record(self):
        now = time.time()
        sample = Sample(**{**self.latest.__dict__, "t": now})
        self.window.add(sample)

    def set_isolation_params(self):
        if not self.param_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("parameter service not available")
        request = SetParameters.Request()
        for name, value in ISOLATION_PARAMS.items():
            request.parameters.append(Parameter(name=name, value=parameter_value(value)))
        future = self.param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("parameter service call timed out")
        failed = [
            result.reason or "unknown"
            for result in future.result().results
            if not result.successful
        ]
        if failed:
            raise RuntimeError("parameter update failed: " + "; ".join(failed))

    def set_enabled(self, enabled):
        if not self.enable_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("enable service not available")
        request = SetBool.Request()
        request.data = bool(enabled)
        future = self.enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("enable service call timed out")
        return future.result()


def parse_token(text, name, default=""):
    prefix = f"{name}="
    for part in text.split():
        if part.startswith(prefix):
            return part[len(prefix):]
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-isolation", action="store_true")
    parser.add_argument("--heading-gain", type=float)
    parser.add_argument("--lookahead-heading-gain", type=float)
    parser.add_argument("--lookahead-error-limit", type=float)
    parser.add_argument("--curvature-gain", type=float)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--dynamic-speed", action="store_true")
    parser.add_argument("--fixed-speed", action="store_true")
    parser.add_argument("--linear-speed", type=float)
    parser.add_argument("--min-linear-speed", type=float)
    parser.add_argument("--speed-curve-offset-limit", type=float)
    parser.add_argument("--speed-curve-exponent", type=float)
    parser.add_argument("--speed-accel-rate", type=float)
    parser.add_argument("--speed-decel-rate", type=float)
    parser.add_argument("--max-steering", type=float)
    parser.add_argument("--steering-slew-rate", type=float)
    parser.add_argument("--steering-return-slew-rate", type=float)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--disable-after-watch", action="store_true")
    parser.add_argument("--watch", type=float, default=0.0)
    args = parser.parse_args()

    rclpy.init()
    node = LineFollowDiag()
    try:
        if args.apply_isolation:
            if args.kp is not None:
                ISOLATION_PARAMS["Kp"] = args.kp
            if args.kd is not None:
                ISOLATION_PARAMS["Kd"] = args.kd
            if args.dynamic_speed:
                ISOLATION_PARAMS["enable_dynamic_speed"] = True
            if args.fixed_speed:
                ISOLATION_PARAMS["enable_dynamic_speed"] = False
            if args.linear_speed is not None:
                ISOLATION_PARAMS["linear_speed"] = args.linear_speed
            if args.min_linear_speed is not None:
                ISOLATION_PARAMS["min_linear_speed"] = args.min_linear_speed
            if args.speed_curve_offset_limit is not None:
                ISOLATION_PARAMS["speed_curve_offset_limit"] = args.speed_curve_offset_limit
            if args.speed_curve_exponent is not None:
                ISOLATION_PARAMS["speed_curve_exponent"] = args.speed_curve_exponent
            if args.speed_accel_rate is not None:
                ISOLATION_PARAMS["speed_accel_rate"] = args.speed_accel_rate
            if args.speed_decel_rate is not None:
                ISOLATION_PARAMS["speed_decel_rate"] = args.speed_decel_rate
            if args.heading_gain is not None:
                ISOLATION_PARAMS["heading_gain"] = args.heading_gain
            if args.lookahead_heading_gain is not None:
                ISOLATION_PARAMS["lookahead_heading_gain"] = args.lookahead_heading_gain
            if args.lookahead_error_limit is not None:
                ISOLATION_PARAMS["lookahead_error_limit"] = args.lookahead_error_limit
            if args.curvature_gain is not None:
                ISOLATION_PARAMS["curvature_gain"] = args.curvature_gain
            if args.max_steering is not None:
                ISOLATION_PARAMS["max_steering"] = args.max_steering
            if args.steering_slew_rate is not None:
                ISOLATION_PARAMS["steering_slew_rate"] = args.steering_slew_rate
            if args.steering_return_slew_rate is not None:
                ISOLATION_PARAMS["steering_return_slew_rate"] = args.steering_return_slew_rate
            node.set_isolation_params()
            print("applied isolation parameters")
        if args.enable:
            result = node.set_enabled(True)
            print(f"enable: success={result.success} message={result.message}")
        if args.disable:
            result = node.set_enabled(False)
            print(f"disable: success={result.success} message={result.message}")
        if args.watch > 0.0:
            end_time = time.time() + args.watch
            next_print = time.time() + 1.0
            while rclpy.ok() and time.time() < end_time:
                rclpy.spin_once(node, timeout_sec=0.05)
                if time.time() >= next_print:
                    print(node.window.summary(), flush=True)
                    next_print += 1.0
            if args.disable_after_watch:
                result = node.set_enabled(False)
                print(f"disable_after_watch: success={result.success} message={result.message}")
    finally:
        if args.disable_after_watch:
            try:
                node.set_enabled(False)
            except RuntimeError:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
