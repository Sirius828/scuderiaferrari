#!/usr/bin/env python3
"""
Measure the live FPS of the shared-memory video stream.

This reads only the SHM header used by track_perception:
    uint64 fid, uint32 width, uint32 height

Run from the workspace root:
    python3 src/track_perception/test/shm_fps_test.py
"""

import argparse
import statistics
import struct
import time
from multiprocessing import resource_tracker, shared_memory


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


def measure_shm_fps(shm_name, duration_s, report_interval_s, poll_interval_s):
    shm = shared_memory.SharedMemory(name=shm_name)
    unregister_from_resource_tracker(shm_name)

    try:
        start_fid, width, height = read_header(shm)
        last_fid = start_fid
        last_report_fid = start_fid
        start_time = time.monotonic()
        last_report_time = start_time
        report_fps_values = []
        duplicate_polls = 0
        skipped_frames = 0
        polls = 0

        print(
            f"Connected to SHM '{shm_name}': "
            f"start_fid={start_fid}, size={width}x{height}, "
            f"duration={duration_s:.1f}s"
        )
        print("time_s  fid       size       fps_window  fps_total  skipped  duplicate_polls")

        while True:
            now = time.monotonic()
            if now - start_time >= duration_s:
                break

            fid, width, height = read_header(shm)
            polls += 1

            fid_delta = int(fid - last_fid)
            if fid_delta == 0:
                duplicate_polls += 1
            elif fid_delta > 1:
                skipped_frames += fid_delta - 1
            last_fid = fid

            if now - last_report_time >= report_interval_s:
                elapsed_window = now - last_report_time
                elapsed_total = now - start_time
                frames_window = max(0, int(fid - last_report_fid))
                frames_total = max(0, int(fid - start_fid))
                fps_window = frames_window / elapsed_window if elapsed_window > 0 else 0.0
                fps_total = frames_total / elapsed_total if elapsed_total > 0 else 0.0
                report_fps_values.append(fps_window)

                print(
                    f"{elapsed_total:6.2f}  {fid:<8d}  "
                    f"{width:4d}x{height:<4d}  "
                    f"{fps_window:10.2f}  {fps_total:9.2f}  "
                    f"{skipped_frames:7d}  {duplicate_polls:15d}"
                )

                last_report_time = now
                last_report_fid = fid

            time.sleep(poll_interval_s)

        end_time = time.monotonic()
        end_fid, width, height = read_header(shm)
        elapsed = max(end_time - start_time, 1e-9)
        total_frames = max(0, int(end_fid - start_fid))
        total_fps = total_frames / elapsed

        print("\nSummary")
        print(f"  shm_name:        {shm_name}")
        print(f"  final_size:      {width}x{height}")
        print(f"  elapsed_s:       {elapsed:.3f}")
        print(f"  fid_delta:       {total_frames}")
        print(f"  shm_fps_total:   {total_fps:.2f}")
        print(f"  skipped_frames:  {skipped_frames}")
        print(f"  duplicate_polls: {duplicate_polls}")
        print(f"  poll_count:      {polls}")
        if report_fps_values:
            print(f"  fps_min_window:  {min(report_fps_values):.2f}")
            print(f"  fps_max_window:  {max(report_fps_values):.2f}")
            print(f"  fps_mean_window: {statistics.mean(report_fps_values):.2f}")

    finally:
        shm.close()


def main():
    parser = argparse.ArgumentParser(description='Measure shared-memory video stream FPS.')
    parser.add_argument('--shm-name', default='shm_ar_video', help='Shared memory name')
    parser.add_argument('--duration', type=float, default=10.0, help='Measurement duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Report interval in seconds')
    parser.add_argument('--poll', type=float, default=0.005, help='Header polling interval in seconds')
    args = parser.parse_args()

    if args.duration <= 0:
        raise SystemExit('--duration must be > 0')
    if args.interval <= 0:
        raise SystemExit('--interval must be > 0')
    if args.poll <= 0:
        raise SystemExit('--poll must be > 0')

    try:
        measure_shm_fps(args.shm_name, args.duration, args.interval, args.poll)
    except FileNotFoundError:
        raise SystemExit(
            f"SHM '{args.shm_name}' not found. Start the camera/AR receiver that creates it first."
        )


if __name__ == '__main__':
    main()
