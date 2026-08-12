#!/usr/bin/env python3
"""Replay a video file through the existing fused perception node.

The fused node reads frames from a shared-memory object named ``shm_ar_video``
by default.  This helper creates a separate replay object and writes frames in
the same format, so the node source does not need to be changed.

Shared-memory layout (little-endian):
    uint64 fid, uint32 width, uint32 height, RGB uint8 image bytes

Typical use on the orangepi:

    python3 tools/replay_fused_video.py /path/to/video.mov
    ros2 run track_perception_cpp fused_perception_node --ros-args \
      -p shm_name:=replay_fused_video -p show_window:=true

The replay process may be started before or after the node, but start the node
first when the complete video beginning must be processed.  The video
resolution must remain fixed for the lifetime of the shared-memory object.
"""

import argparse
import math
import signal
import struct
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2


HEADER = struct.Struct("<QII")
HEADER_SIZE = HEADER.size
STOP_REQUESTED = False


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def positive_fps(value: float, fallback: float = 30.0) -> float:
    return value if math.isfinite(value) and value > 0.001 else fallback


def write_frame(shm, fid: int, frame_bgr, output_format: str) -> None:
    if output_format == "RGB":
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame = frame_bgr

    height, width = frame.shape[:2]
    image_size = width * height * 3
    if HEADER_SIZE + image_size > len(shm.buf):
        raise RuntimeError(
            f"frame size {width}x{height} does not fit in shared memory "
            f"({len(shm.buf)} bytes)"
        )

    # fid=0 is the in-progress marker understood by ShmReader.  Publish the
    # new fid only after the complete image has been copied.
    struct.pack_into("<QII", shm.buf, 0, 0, width, height)
    shm.buf[HEADER_SIZE:HEADER_SIZE + image_size] = frame.tobytes()
    struct.pack_into("<QII", shm.buf, 0, fid, width, height)


def open_video(path: Path, start_sec: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    if start_sec > 0.0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    return capture


def replay(args) -> int:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise RuntimeError(f"video file does not exist: {video_path}")

    capture = open_video(video_path, args.start_sec)
    source_fps = positive_fps(capture.get(cv2.CAP_PROP_FPS))
    fps = positive_fps(args.fps, source_fps)

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"video contains no readable frame: {video_path}")

    height, width = first_frame.shape[:2]
    image_size = width * height * 3
    shm_size = HEADER_SIZE + image_size
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

    frame_number = 0
    written = 0
    started_at = time.monotonic()
    next_deadline = started_at
    last_report = started_at
    last_report_written = 0

    try:
        print(
            f"replay video={video_path} size={width}x{height} "
            f"source_fps={source_fps:.3f} output_fps={fps:.3f} "
            f"shm=/dev/shm/{args.shm_name} format={args.output_format}"
        )
        print("Press Ctrl+C to stop.")

        while not STOP_REQUESTED:
            frame = first_frame if frame_number == 0 else None
            if frame is None:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if not args.loop:
                        break
                    capture.release()
                    capture = open_video(video_path, args.start_sec)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"cannot restart video: {video_path}")

            if args.end_sec > 0.0 and frame_number / fps >= args.end_sec:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError("video resolution changed during replay")

            frame_number += 1
            written += 1
            write_frame(shm, written, frame, args.output_format)

            if args.realtime:
                next_deadline += 1.0 / fps
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)

            now = time.monotonic()
            if now - last_report >= 1.0:
                interval_frames = written - last_report_written
                interval_time = max(now - last_report, 1e-6)
                print(
                    f"replayed={written} elapsed={now - started_at:.1f}s "
                    f"rate={interval_frames / interval_time:.1f} fps"
                )
                last_report = now
                last_report_written = written

    finally:
        capture.release()
        shm.close()
        shm.unlink()

    print(f"replay finished: written_frames={written}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feed a video into the existing fused perception node via SHM"
    )
    parser.add_argument("video", help="input video path, e.g. the .mov file")
    parser.add_argument(
        "--shm-name",
        default="replay_fused_video",
        help="shared-memory name; pass the same value to fused_perception_node",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="replay FPS; default uses the video's FPS",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0, help="start position in seconds"
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=0.0,
        help="stop after this many replay seconds; 0 means until EOF",
    )
    parser.add_argument(
        "--loop", action="store_true", help="restart the video after reaching EOF"
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="write frames as fast as possible",
    )
    parser.set_defaults(realtime=True)
    parser.add_argument(
        "--output-format",
        choices=("RGB", "BGR"),
        default="RGB",
        help="pixel format written to SHM; match fused node input_format",
    )
    args = parser.parse_args()
    if args.fps < 0.0:
        parser.error("--fps must be >= 0")
    if args.start_sec < 0.0 or args.end_sec < 0.0:
        parser.error("--start-sec and --end-sec must be >= 0")
    return args


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        return replay(parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"replay failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
