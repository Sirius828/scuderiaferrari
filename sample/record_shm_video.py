#!/usr/bin/env python3
"""
Record the shared-memory camera stream to a video file.

The SHM layout matches track_perception:
    uint64 fid, uint32 width, uint32 height, then width*height*3 uint8 image bytes

Run from the workspace root:
    python3 sample/record_shm_video.py

Stop recording with Ctrl+C.
"""

import argparse
import os
import struct
import time
from datetime import datetime
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np


SHM_HEADER_SIZE = 16


def unregister_from_resource_tracker(shm_name):
    """Avoid Python resource_tracker unlinking a SHM segment owned by another process."""
    try:
        resource_tracker.unregister('/' + shm_name, 'shared_memory')
    except Exception:
        pass


def read_header(shm):
    header = bytes(shm.buf[:SHM_HEADER_SIZE])
    return struct.unpack('QII', header)


def make_output_path(output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    if output_name:
        return output_name if os.path.isabs(output_name) else os.path.join(output_dir, output_name)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_dir, f'shm_record_{timestamp}.mp4')


def create_writer(path, width, height, fps, fourcc_name):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for '{path}' with fourcc={fourcc_name}, "
            f"size={width}x{height}, fps={fps}"
        )
    return writer


def convert_for_video(frame, input_format):
    fmt = input_format.upper()
    if fmt == 'RGB':
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if fmt == 'BGR':
        return frame
    raise ValueError("--input-format must be 'RGB' or 'BGR'")


def record_shm_video(
    shm_name,
    output_dir,
    output_name,
    duration_s,
    output_fps,
    input_format,
    fourcc_name,
    poll_interval_s,
):
    shm = shared_memory.SharedMemory(name=shm_name)
    unregister_from_resource_tracker(shm_name)

    writer = None
    output_path = None
    last_fid = None
    start_fid = None
    written_frames = 0
    skipped_frames = 0
    duplicate_polls = 0
    polls = 0

    try:
        start_time = time.monotonic()
        last_report_time = start_time

        print(
            f"Recording SHM '{shm_name}' at original resolution, "
            f"encoded as {output_fps:.1f} FPS"
        )

        try:
            while True:
                now = time.monotonic()
                if duration_s > 0 and now - start_time >= duration_s:
                    break

                fid, width, height = read_header(shm)
                polls += 1

                if width <= 0 or height <= 0:
                    time.sleep(poll_interval_s)
                    continue

                if start_fid is None:
                    start_fid = fid
                    last_fid = fid - 1

                fid_delta = int(fid - last_fid)
                if fid_delta == 0:
                    duplicate_polls += 1
                    time.sleep(poll_interval_s)
                    continue
                if fid_delta > 1:
                    skipped_frames += fid_delta - 1

                size = int(width) * int(height) * 3
                frame_view = np.ndarray(
                    (int(height), int(width), 3),
                    dtype=np.uint8,
                    buffer=shm.buf[SHM_HEADER_SIZE:SHM_HEADER_SIZE + size],
                )
                frame = frame_view.copy()
                del frame_view

                if writer is None:
                    output_path = make_output_path(output_dir, output_name)
                    writer = create_writer(output_path, width, height, output_fps, fourcc_name)
                    print(f"Output: {output_path}")
                    print(f"Input size: {width}x{height}")
                    if duration_s <= 0:
                        print("Recording until Ctrl+C...")

                writer.write(convert_for_video(frame, input_format))
                written_frames += 1
                last_fid = fid

                if now - last_report_time >= 1.0:
                    elapsed = max(now - start_time, 1e-9)
                    input_frames = max(0, int(fid - start_fid + 1))
                    print(
                        f"t={elapsed:6.2f}s fid={fid:<8d} "
                        f"input_fps={input_frames / elapsed:6.2f} "
                        f"written={written_frames:<6d} skipped={skipped_frames:<5d} "
                        f"duplicates={duplicate_polls}"
                    )
                    last_report_time = now

                time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            print("\nStop requested by user.")

        elapsed = max(time.monotonic() - start_time, 1e-9)
        print("\nSummary")
        print(f"  output:          {output_path}")
        print(f"  elapsed_s:       {elapsed:.3f}")
        print(f"  written_frames:  {written_frames}")
        print(f"  encoded_fps:     {output_fps:.2f}")
        print(f"  video_duration:  {written_frames / output_fps:.3f}s")
        print(f"  skipped_frames:  {skipped_frames}")
        print(f"  duplicate_polls: {duplicate_polls}")
        print(f"  poll_count:      {polls}")

    finally:
        if writer is not None:
            writer.release()
        shm.close()


def main():
    parser = argparse.ArgumentParser(description='Record shared-memory video to sample/.')
    parser.add_argument('--shm-name', default='shm_ar_video', help='Shared memory name')
    parser.add_argument('--duration', type=float, default=0.0, help='Record seconds; <=0 records until Ctrl+C')
    parser.add_argument('--fps', type=float, default=60.0, help='Encoded video FPS')
    parser.add_argument('--input-format', default='RGB', choices=['RGB', 'BGR'], help='SHM image color order')
    parser.add_argument('--output-dir', default='sample', help='Directory for the output video')
    parser.add_argument('--output-name', default='', help='Output filename; default uses timestamp')
    parser.add_argument('--fourcc', default='mp4v', help='OpenCV fourcc, e.g. mp4v, avc1, MJPG')
    parser.add_argument('--poll', type=float, default=0.001, help='Polling interval in seconds')
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit('--fps must be > 0')
    if args.poll <= 0:
        raise SystemExit('--poll must be > 0')

    try:
        record_shm_video(
            shm_name=args.shm_name,
            output_dir=args.output_dir,
            output_name=args.output_name,
            duration_s=args.duration,
            output_fps=args.fps,
            input_format=args.input_format,
            fourcc_name=args.fourcc,
            poll_interval_s=args.poll,
        )
    except FileNotFoundError:
        raise SystemExit(
            f"SHM '{args.shm_name}' not found. Start the camera/AR receiver first."
        )


if __name__ == '__main__':
    main()
