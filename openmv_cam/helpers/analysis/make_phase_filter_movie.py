#!/usr/bin/env python3
"""
Generate side-by-side comparison video of raw vs. phase-filtered event frames.

Requires: numpy, opencv-python
"""
import argparse
from pathlib import Path

import numpy as np

W = 320
H = 320
FPS = 30


def load_xyt_points(file_path: Path) -> np.ndarray:
    """Load x/y/t TSV file with robust handling."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    points = np.loadtxt(file_path, dtype=np.int64, delimiter="\t")

    if points.ndim == 1:
        if points.size != 3:
            raise RuntimeError(f"Expected 3 columns: x, y, t. Got {points.size}.")
        points = points.reshape(1, 3)

    if points.shape[1] != 3:
        raise RuntimeError(f"Expected 3 columns: x, y, t. Got {points.shape[1]}.")

    return points


def filter_valid_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Filter events to valid image bounds."""
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    xs = points[:, 0]
    ys = points[:, 1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return points[valid]


def crop_time_interval(
    points: np.ndarray, start_sec: float = None, end_sec: float = None
) -> np.ndarray:
    """Crop events to optional time interval."""
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    if start_sec is None and end_sec is None:
        return points

    ts_us = points[:, 2].astype(np.float64)
    ts_rel_s = (ts_us - ts_us.min()) * 1e-6

    start_s = 0.0 if start_sec is None else start_sec
    end_s = np.inf if end_sec is None else end_sec

    mask = (ts_rel_s >= start_s) & (ts_rel_s <= end_s)
    return points[mask]


def estimate_global_fft_peak(
    points: np.ndarray, bin_ms: float, max_freq: float
) -> float:
    """Estimate dominant frequency from global event-rate FFT."""
    if len(points) == 0:
        return 1.0

    ts_us = points[:, 2].astype(np.float64)
    ts_s = (ts_us - ts_us.min()) * 1e-6

    duration_s = ts_s.max()
    if duration_s <= 0:
        return 1.0

    bin_s = bin_ms * 1e-3
    num_bins = int(np.ceil(duration_s / bin_s)) + 1

    counts, _ = np.histogram(ts_s, bins=num_bins, range=(0, num_bins * bin_s))
    rate_hz = counts / bin_s

    rate_centered = rate_hz - rate_hz.mean()

    freqs = np.fft.rfftfreq(len(rate_centered), d=bin_s)
    fft_mag = np.abs(np.fft.rfft(rate_centered))

    valid_band = (freqs >= 1.0) & (freqs <= max_freq)
    if np.any(valid_band):
        best_idx = np.argmax(fft_mag[valid_band])
        return float(freqs[valid_band][best_idx])
    else:
        return 1.0


def tilewise_phase_filter(
    points: np.ndarray,
    freq_hz: float,
    tile_size: int,
    tile_min_events: int,
    width: int,
    height: int,
    phase_strength_threshold: float = 0.18,
    num_phase_bins: int = 64,
) -> tuple:
    """
    Tile-wise phase filtering: reject events in tiles with strong phase coherence.

    Returns:
        (filtered_points, num_rejected, num_kept)
    """
    if len(points) == 0 or freq_hz <= 0:
        return points, 0, 0

    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)
    ts_us = points[:, 2].astype(np.float64)
    ts_s = (ts_us - ts_us.min()) * 1e-6

    tile_x = xs // tile_size
    tile_y = ys // tile_size

    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size

    # Build per-tile phase histograms
    tile_id = tile_y * n_tiles_x + tile_x
    num_tiles = n_tiles_x * n_tiles_y

    tile_event_counts = np.bincount(tile_id, minlength=num_tiles)
    tiles_to_filter = tile_event_counts >= tile_min_events

    # Compute phase for all events
    phase_rad = 2.0 * np.pi * ((ts_s * freq_hz) % 1.0)
    phase_bin = np.floor((phase_rad / (2.0 * np.pi)) * num_phase_bins).astype(np.int32)
    phase_bin = np.clip(phase_bin, 0, num_phase_bins - 1)

    # Find peak phase bin per tile and compute phase strength
    peak_phase_bin = np.full(num_tiles, -1, dtype=np.int32)
    phase_strength = np.zeros(num_tiles, dtype=np.float32)

    for tid in np.where(tiles_to_filter)[0]:
        tile_mask = tile_id == tid
        tile_phases = phase_bin[tile_mask]
        hist, _ = np.histogram(tile_phases, bins=num_phase_bins, range=(0, num_phase_bins))
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist_normalized = hist / hist_sum
            peak_idx = np.argmax(hist_normalized)
            peak_phase_bin[tid] = peak_idx
            phase_strength[tid] = hist_normalized[peak_idx]

    # Decide which tiles to filter
    filter_tile = tiles_to_filter & (phase_strength > phase_strength_threshold)

    # Reject events in filtered tiles that are within 2 bins of peak phase
    keep_mask = np.ones(len(points), dtype=bool)
    num_rejected = 0

    for tid in np.where(filter_tile)[0]:
        tile_mask = tile_id == tid
        peak_bin = peak_phase_bin[tid]
        if peak_bin < 0:
            continue

        # Circular distance: within 2 bins of peak
        dist_to_peak = np.abs(phase_bin[tile_mask] - peak_bin)
        dist_to_peak = np.minimum(dist_to_peak, num_phase_bins - dist_to_peak)
        reject_local = dist_to_peak <= 2

        tile_indices = np.where(tile_mask)[0]
        reject_global = np.zeros(len(points), dtype=bool)
        reject_global[tile_indices] = reject_local

        keep_mask &= ~reject_global
        num_rejected += reject_local.sum()

    filtered_points = points[keep_mask]
    num_kept = keep_mask.sum()

    return filtered_points, num_rejected, num_kept


def events_to_frame(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Accumulate events into a 2D count map."""
    counts = np.zeros((height, width), dtype=np.int32)
    if len(points) == 0:
        return counts

    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)

    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    np.add.at(counts, (ys, xs), 1)
    return counts


def render_frame(counts: np.ndarray) -> np.ndarray:
    """Render count map to uint8 image."""
    # Use log scaling for dynamic range
    log_counts = np.log1p(counts.astype(np.float32))

    # Normalize by 99th percentile for robust scaling
    if np.any(log_counts > 0):
        p99 = np.percentile(log_counts, 99)
        if p99 > 0:
            normalized = (log_counts / p99 * 255).astype(np.uint8)
        else:
            normalized = log_counts.astype(np.uint8)
    else:
        normalized = np.zeros_like(counts, dtype=np.uint8)

    return normalized


def write_comparison_movie(
    raw_points: np.ndarray,
    filtered_points: np.ndarray,
    save_path: Path,
    fps: int,
    width: int,
    height: int,
) -> int:
    """Generate side-by-side comparison video of raw vs. filtered events."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required. Install with: pip install opencv-python"
        )

    if len(raw_points) == 0:
        print("[WARN] No raw events; generating empty video.")
        raw_points = np.zeros((0, 3), dtype=np.int64)

    if len(filtered_points) == 0:
        print("[WARN] No filtered events in output.")
        filtered_points = np.zeros((0, 3), dtype=np.int64)

    # Determine video duration from raw events (full recording)
    if len(raw_points) > 0:
        t_start_us = raw_points[0, 2]
        t_end_us = raw_points[-1, 2]
    else:
        t_start_us = 0
        t_end_us = 0

    duration_s = (t_end_us - t_start_us) * 1e-6
    if duration_s <= 0:
        print("[WARN] Recording duration <= 0; no frames generated.")
        return 0

    num_frames = max(1, int(np.ceil(duration_s * fps)))
    frame_duration_us = int(1e6 / fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_width = width * 2
    output_height = height
    writer = cv2.VideoWriter(
        str(save_path), fourcc, fps, (output_width, output_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {save_path}")

    frames_written = 0

    for frame_idx in range(num_frames):
        frame_start_us = t_start_us + frame_idx * frame_duration_us
        frame_end_us = frame_start_us + frame_duration_us

        # Extract events in this frame interval
        raw_mask = (raw_points[:, 2] >= frame_start_us) & (
            raw_points[:, 2] < frame_end_us
        )
        filtered_mask = (filtered_points[:, 2] >= frame_start_us) & (
            filtered_points[:, 2] < frame_end_us
        )

        raw_frame_events = raw_points[raw_mask]
        filtered_frame_events = filtered_points[filtered_mask]

        # Convert to images
        raw_counts = events_to_frame(raw_frame_events, width, height)
        filtered_counts = events_to_frame(filtered_frame_events, width, height)

        raw_img = render_frame(raw_counts)
        filtered_img = render_frame(filtered_counts)

        # Convert grayscale to BGR for OpenCV
        raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        filtered_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)  # White text

        cv2.putText(raw_bgr, "raw", (10, 25), font, font_scale, color, thickness)
        cv2.putText(
            filtered_bgr, "phase filtered", (10, 25), font, font_scale, color, thickness
        )

        # Concatenate left and right
        comparison = np.hstack([raw_bgr, filtered_bgr])

        writer.write(comparison)
        frames_written += 1

    writer.release()
    return frames_written


def main():
    ap = argparse.ArgumentParser(
        description="Generate side-by-side phase-filter comparison video."
    )
    ap.add_argument(
        "--input",
        default="events_xyt.tsv",
        help="Path to x/y/t TSV file.",
    )
    ap.add_argument(
        "--bin-ms",
        type=float,
        default=1.0,
        help="Temporal bin size for FFT estimation.",
    )
    ap.add_argument(
        "--max-freq",
        type=float,
        default=500.0,
        help="Maximum frequency for FFT peak search.",
    )
    ap.add_argument(
        "--phase-min-count",
        type=int,
        default=10,
        help="(Compatibility argument, not used in movie generation.)",
    )
    ap.add_argument(
        "--phase-freq",
        type=float,
        default=None,
        help="Optional fixed phase frequency [Hz]. If not set, estimate from FFT.",
    )
    ap.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Tile size in pixels for phase filtering.",
    )
    ap.add_argument(
        "--tile-max-freq",
        type=float,
        default=None,
        help="(Compatibility argument, not used in movie generation.)",
    )
    ap.add_argument(
        "--tile-min-events",
        type=int,
        default=50,
        help="Minimum events per tile for phase filtering.",
    )
    ap.add_argument(
        "--time-window-sec",
        type=float,
        default=None,
        help="(Compatibility argument, not used in movie generation.)",
    )
    ap.add_argument(
        "--analysis-start-sec",
        type=float,
        default=None,
        help="Analysis start time [s] relative to first timestamp.",
    )
    ap.add_argument(
        "--analysis-end-sec",
        type=float,
        default=None,
        help="Analysis end time [s] relative to first timestamp.",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="Output video path. Default: phase_filter_compare.mp4 next to input.",
    )
    args = ap.parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute() and input_path.parent == Path("."):
        # Relative path with no parent: resolve relative to script directory
        script_dir = Path(__file__).parent
        input_path = script_dir / input_path

    print(f"[INFO] Loading {input_path}")
    points = load_xyt_points(input_path)
    points = filter_valid_points(points, W, H)

    print(f"[INFO] Total loaded events: {len(points)}")

    # Apply analysis interval
    if args.analysis_start_sec is not None or args.analysis_end_sec is not None:
        print(
            f"[INFO] Analysis interval: "
            f"{args.analysis_start_sec if args.analysis_start_sec is not None else 0.0:.3f} "
            f"to {args.analysis_end_sec if args.analysis_end_sec is not None else 'end'} s"
        )
    points = crop_time_interval(
        points, start_sec=args.analysis_start_sec, end_sec=args.analysis_end_sec
    )

    if len(points) == 0:
        print("[WARN] No events in selected analysis interval.")
        return

    print(f"[INFO] Events in analysis interval: {len(points)}")

    # Re-normalize timestamps to first event
    if len(points) > 0:
        t_min = points[0, 2]
        points_copy = points.copy()
        points_copy[:, 2] = points_copy[:, 2] - t_min + points[0, 2]
        points = points_copy

    # Determine phase frequency
    if args.phase_freq is not None:
        phase_freq_hz = args.phase_freq
        print(f"[INFO] Using provided phase frequency: {phase_freq_hz:.2f} Hz")
    else:
        phase_freq_hz = estimate_global_fft_peak(
            points, args.bin_ms, args.max_freq
        )
        print(f"[INFO] Estimated phase frequency from FFT: {phase_freq_hz:.2f} Hz")

    # Apply phase filtering
    print("[INFO] Applying tile-wise phase filtering...")
    filtered_points, num_rejected, num_kept = tilewise_phase_filter(
        points,
        phase_freq_hz,
        args.tile_size,
        args.tile_min_events,
        width=W,
        height=H,
    )

    print(f"[INFO] Phase filtering results:")
    print(f"  Total events: {len(points)}")
    print(f"  Rejected: {num_rejected}")
    print(f"  Kept: {num_kept}")

    # Determine output path
    if args.save is None:
        output_path = input_path.parent / "phase_filter_compare.mp4"
    else:
        output_path = Path(args.save)

    print(f"[INFO] Generating comparison video to {output_path}...")
    frames_written = write_comparison_movie(
        points, filtered_points, output_path, FPS, W, H
    )

    print(f"[INFO] Video complete: {frames_written} frames written.")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()
