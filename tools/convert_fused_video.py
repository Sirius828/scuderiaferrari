#!/usr/bin/env python3
"""Batch-convert a video to an annotated MP4 using the existing fused node.

The fused node is kept unchanged.  This driver feeds one frame at a time into
its shared-memory input, waits for /perception/frame_signature, then copies the
node's latest debug render into the output video.  Consequently the conversion
is not real-time and does not lose frames when inference is slower than the
source video.

By default the driver starts and stops fused_perception_node itself.  Run it
from a shell where the ROS environment and this workspace are sourced.

Input frames are resized to 640x480 by default before inference.  Pass
``--resize original`` to keep the source resolution.
"""

import argparse
import os
import re
import signal
import struct
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt64

from replay_fused_video import HEADER_SIZE, open_video, positive_fps, write_frame


HEADER = struct.Struct("<QII")
STOP_REQUESTED = False


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_resize(value):
    if value is None or value.strip().lower() in ("", "none", "original"):
        return None
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise ValueError("--resize must use WIDTHxHEIGHT, for example 640x480")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("--resize dimensions must be positive")
    return width, height


def prepare_frame(frame, resize_size):
    if resize_size is None:
        return frame
    width, height = resize_size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class FrameAckNode(Node):
    def __init__(self):
        super().__init__("fused_video_mp4_converter")
        qos = QoSProfile(depth=1)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.received_count = 0
        self.last_signature = 0
        self.create_subscription(
            UInt64,
            "/perception/frame_signature",
            self._on_frame_signature,
            qos,
        )

    def _on_frame_signature(self, message):
        self.received_count += 1
        self.last_signature = int(message.data)

    def wait_for_publisher(self, node_process, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while self.count_publishers("/perception/frame_signature") == 0:
            if node_process is not None and node_process.poll() is not None:
                raise RuntimeError(
                    f"fused node exited before publishing frame signatures "
                    f"(returncode={node_process.returncode})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for /perception/frame_signature; "
                    "is fused_perception_node running and built?"
                )
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_next_frame(self, previous_count: int, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while self.received_count <= previous_count:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for fused node to process a frame; "
                    "check the fused node log and screenshot parameters"
                )
            rclpy.spin_once(self, timeout_sec=0.05)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def configure_isolated_ros_domain(args) -> None:
    if args.no_start_node:
        return
    if args.ros_domain_id is not None:
        domain_id = args.ros_domain_id
    else:
        try:
            current_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        except ValueError:
            current_domain = 0
        domain_id = current_domain + 1
        if domain_id > 232:
            domain_id = 1
    if domain_id < 0 or domain_id > 232:
        raise ValueError("--ros-domain-id must be between 0 and 232")
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    print(f"using isolated ROS_DOMAIN_ID={domain_id}")


def start_fused_node(args):
    command = [
        "ros2",
        "run",
        "track_perception_cpp",
        "fused_perception_node",
        "--ros-args",
        "--params-file",
        str(args.params_file),
        "-p",
        f"shm_name:={args.shm_name}",
        "-p",
        "show_window:=false",
        "-p",
        "enable_debug_screenshots:=true",
        "-p",
        "debug_screenshot_interval_sec:=0.000001",
        "-p",
        "debug_screenshot_branch_only:=false",
        "-p",
        f"debug_screenshot_dir:={args.screenshot_dir}",
        "-p",
        "enable_guideboard_api:=false",
        "-p",
        "enable_human_obstacle_stop:=false",
        "-p",
        "enable_encoder_branch_hold:=false",
    ]
    print("starting fused node")
    return subprocess.Popen(command)


def screenshot_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def wait_for_new_screenshot(path: Path, previous_mtime: int, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        current_mtime = screenshot_mtime(path)
        if current_mtime != previous_mtime:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None and image.size > 0:
                return image, current_mtime
        time.sleep(0.005)
    raise TimeoutError(
        f"timed out waiting for debug screenshot {path}; "
        "start fused node with debug screenshots enabled"
    )


def remove_newest_auto_screenshot(directory: Path) -> None:
    candidates = list(directory.glob("fused_perception_auto_*.jpg"))
    if not candidates:
        return
    newest = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    try:
        newest.unlink()
    except FileNotFoundError:
        pass


def convert(args) -> int:
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not video_path.is_file():
        raise RuntimeError(f"video file does not exist: {video_path}")
    if output_path == video_path:
        raise RuntimeError("input and output paths must be different")
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    if not args.params_file.is_file() and not args.no_start_node:
        raise RuntimeError(f"params file does not exist: {args.params_file}")

    args.screenshot_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = open_video(video_path, args.start_sec)
    source_fps = positive_fps(capture.get(cv2.CAP_PROP_FPS))
    output_fps = positive_fps(args.fps, source_fps)
    resize_size = parse_resize(args.resize)
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"video contains no readable frame: {video_path}")

    first_frame = prepare_frame(first_frame, resize_size)
    height, width = first_frame.shape[:2]
    shm_size = HEADER_SIZE + width * height * 3
    try:
        shm = shared_memory.SharedMemory(
            name=args.shm_name,
            create=True,
            size=shm_size,
        )
    except FileExistsError as exc:
        capture.release()
        raise RuntimeError(
            f"shared memory '{args.shm_name}' already exists; choose another "
            "--shm-name or stop the previous replay first"
        ) from exc

    struct.pack_into("<QII", shm.buf, 0, 0, width, height)
    node_process = None
    ack_node = None
    writer = None
    processed = 0
    started_at = time.monotonic()
    frame_index = 0

    try:
        configure_isolated_ros_domain(args)
        if not args.no_start_node:
            node_process = start_fused_node(args)

        rclpy.init(args=[])
        ack_node = FrameAckNode()
        ack_node.wait_for_publisher(node_process, args.node_timeout)

        print(
            f"input={video_path} size={width}x{height} "
            f"source_fps={source_fps:.3f} output_fps={output_fps:.3f} "
            f"resize={args.resize}"
        )
        print(f"output={output_path}")

        while not STOP_REQUESTED:
            frame = first_frame if frame_index == 0 else None
            if frame is None:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame = prepare_frame(frame, resize_size)

            if args.end_sec > 0.0 and frame_index / output_fps >= args.end_sec:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError("video resolution changed during conversion")

            previous_ack_count = ack_node.received_count
            previous_mtime = screenshot_mtime(args.screenshot_dir / "latest.jpg")
            frame_index += 1
            write_frame(shm, frame_index, frame, "RGB")
            ack_node.wait_for_next_frame(previous_ack_count, args.frame_timeout)

            visual, _ = wait_for_new_screenshot(
                args.screenshot_dir / "latest.jpg",
                previous_mtime,
                args.frame_timeout,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*args.fourcc),
                    output_fps,
                    (visual.shape[1], visual.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"cannot open MP4 writer with fourcc={args.fourcc}: {output_path}"
                    )

            writer.write(visual)
            remove_newest_auto_screenshot(args.screenshot_dir)
            processed += 1

            if processed % max(1, args.report_every) == 0:
                elapsed = max(time.monotonic() - started_at, 1e-6)
                print(
                    f"processed={processed} elapsed={elapsed:.1f}s "
                    f"speed={processed / elapsed:.2f} fps"
                )

        if processed == 0:
            raise RuntimeError("no frames were converted")
        print(f"conversion finished: {processed} frames -> {output_path}")
        return 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if ack_node is not None:
            ack_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stop_process(node_process)
        shm.close()
        shm.unlink()


def parse_args():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert a video to an annotated MP4 with fused perception"
    )
    parser.add_argument("video", help="input .mov/.mp4 path")
    parser.add_argument("output", help="output annotated .mp4 path")
    parser.add_argument(
        "--params-file",
        type=Path,
        default=workspace / "src/track_perception_cpp/config/fused_perception.yaml",
        help="fused node parameter file",
    )
    parser.add_argument(
        "--shm-name", default="fused_mp4_input", help="temporary shared-memory name"
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("/tmp/fused_mp4_frames"),
        help="directory used by the fused node for its latest debug render",
    )
    parser.add_argument(
        "--fps", type=float, default=0.0, help="output FPS; default uses input FPS"
    )
    parser.add_argument(
        "--resize",
        default="640x480",
        help="resize every input frame before inference; use original to disable",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0, help="start position in seconds"
    )
    parser.add_argument(
        "--end-sec", type=float, default=0.0, help="stop after this many seconds; 0 means EOF"
    )
    parser.add_argument(
        "--frame-timeout", type=float, default=30.0, help="per-frame processing timeout"
    )
    parser.add_argument(
        "--node-timeout", type=float, default=60.0, help="node startup timeout"
    )
    parser.add_argument(
        "--report-every", type=int, default=10, help="print progress every N frames"
    )
    parser.add_argument(
        "--fourcc", default="mp4v", help="OpenCV video codec, default mp4v"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    parser.add_argument(
        "--no-start-node",
        action="store_true",
        help="use an already-running fused node instead of starting one",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help="ROS domain for the temporary replay; default is current domain + 1",
    )
    args = parser.parse_args()
    if len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    if args.fps < 0.0 or args.start_sec < 0.0 or args.end_sec < 0.0:
        parser.error("--fps, --start-sec and --end-sec must be >= 0")
    if args.frame_timeout <= 0.0 or args.node_timeout <= 0.0:
        parser.error("timeouts must be > 0")
    return args


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        return convert(parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"conversion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
