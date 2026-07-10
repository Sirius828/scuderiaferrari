#!/usr/bin/env python3
"""Static steering sweep for chassis mapping checks.

Publishes /cmd_vel with linear.x=0.0 and selected angular.z values.
The car should not drive forward; only steering should move.
"""

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def expected_pwm(steering, center=3000, left_max=2300, right_max=3700):
    if steering >= 0.0:
        return int(center - (center - left_max) * steering)
    return int(center + (right_max - center) * abs(steering))


class SteeringSweep(Node):
    def __init__(self):
        super().__init__("steering_sweep")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def publish_steering(self, steering, duration, rate):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = float(steering)
        end_time = time.time() + duration
        period = 1.0 / rate
        print(
            f"steering={steering:+.2f} expected_pwm={expected_pwm(steering)} "
            f"expected={'left' if steering > 0 else 'right' if steering < 0 else 'center'}"
        )
        while rclpy.ok() and time.time() < end_time:
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--values",
        default="0,0.5,1.0,0,-0.5,-1.0,0",
        help="Comma-separated angular.z values",
    )
    args = parser.parse_args()

    values = [float(value.strip()) for value in args.values.split(",") if value.strip()]
    rclpy.init()
    node = SteeringSweep()
    try:
        time.sleep(0.3)
        for value in values:
            node.publish_steering(value, args.duration, args.rate)
            time.sleep(0.2)
    finally:
        node.publish_steering(0.0, 0.5, args.rate)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
