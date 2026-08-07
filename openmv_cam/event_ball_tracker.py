"""Small ROS-independent event-time ball tracker."""

import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .evt1_protocol import EventPacket


@dataclass
class TrackerDetection:
    bin_start_us: int
    bin_end_us: int
    parent_packet_id: int
    event_count: int
    x_px: float = 0.0
    y_px: float = 0.0
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0
    speed_px_s: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    velocity_valid: bool = False
    blob_area_px: int = 0
    blob_event_count: int = 0
    circularity: float = 0.0
    candidate_count: int = 0
    start_steady_ns: int = 0
    end_steady_ns: int = 0


@dataclass(frozen=True)
class TrackerDebugSnapshot:
    """Immutable inputs needed to render one annotated debug image."""

    activity: np.ndarray
    threshold_mask: np.ndarray
    detection: TrackerDetection
    candidate_contours: tuple
    accepted_candidate_contours: tuple
    selected_contour: object
    predicted_position: object
    trajectory: tuple
    x_crop: tuple


class EventBallTracker:
    """Bin raw EVT1 events and select a simple activity blob."""

    def __init__(self, *, width=320, height=320, bin_ms=1.0,
                 history_limit_ms=100.0, min_event_count=3,
                 min_blob_area_px=2, max_blob_area_px=500,
                 activity_threshold=1, morphology_kernel=0,
                 morphology_iterations=0, use_circularity=False,
                 min_circularity=0.1, max_jump_px=100.0,
                 velocity_history_size=5, velocity_min_span_ms=3.0,
                 stats_history_size=512, x_crop=None):
        self.width, self.height = int(width), int(height)
        self.bin_us = int(round(float(bin_ms) * 1000.0))
        if self.bin_us <= 0:
            raise ValueError("bin_ms must be positive")
        self.history_limit_us = max(self.bin_us, int(history_limit_ms * 1000.0))
        self.min_event_count = int(min_event_count)
        self.min_blob_area_px = int(min_blob_area_px)
        self.max_blob_area_px = int(max_blob_area_px)
        self.activity_threshold = max(1, int(activity_threshold))
        self.morphology_kernel = int(morphology_kernel)
        self.morphology_iterations = int(morphology_iterations)
        self.use_circularity = bool(use_circularity)
        self.min_circularity = float(min_circularity)
        self.max_jump_px = float(max_jump_px)
        if x_crop is None:
            x_crop = (0, self.width)
        if len(x_crop) != 2:
            raise ValueError("x_crop must contain lower and upper bounds")
        self.x_crop = (int(x_crop[0]), int(x_crop[1]))
        if not 0 <= self.x_crop[0] < self.x_crop[1] <= self.width:
            raise ValueError(
                f"x_crop must satisfy 0 <= lower < upper <= {self.width}")
        self.velocity_min_span_us = float(velocity_min_span_ms) * 1000.0
        self.detections = deque(maxlen=max(2, int(velocity_history_size)))
        self._last_velocity = (0.0, 0.0)
        self.pending = defaultdict(list)
        self.next_bin_start_us = None
        self.max_seen_timestamp_us = None
        self.previous_position = None
        self.counters = defaultdict(int)
        timing_names = (
            "map_build", "blob_detection", "computation", "velocity_fit",
            "total_tracker_update")
        self.timings = {
            name: deque(maxlen=max(1, int(stats_history_size)))
            for name in timing_names}
        self.started_ns = time.monotonic_ns()
        self._debug_lock = threading.Lock()
        self._latest_debug_snapshot = None

    def build_activity_map(self, events):
        activity = np.zeros((self.height, self.width), dtype=np.uint16)
        if len(events):
            events = np.asarray(events)
            x, y = events[:, 4].astype(np.int64), events[:, 5].astype(np.int64)
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            valid &= (x >= self.x_crop[0]) & (x < self.x_crop[1])
            np.add.at(activity, (y[valid], x[valid]), 1)
        return activity

    def _candidates(self, activity):
        mask = (activity >= self.activity_threshold).astype(np.uint8) * 255
        if self.morphology_kernel > 0 and self.morphology_iterations > 0:
            kernel = np.ones(
                (self.morphology_kernel, self.morphology_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=self.morphology_iterations)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            area = int(cv2.countNonZero(cv2.drawContours(
                np.zeros_like(mask), [contour], -1, 255, thickness=-1)))
            if area < self.min_blob_area_px or area > self.max_blob_area_px:
                continue
            component = np.zeros_like(mask)
            cv2.drawContours(component, [contour], -1, 1, thickness=-1)
            weights = activity.astype(np.float64) * component
            count = int(weights.sum())
            if count < self.min_event_count:
                continue
            ys, xs = np.nonzero(weights)
            x_com = float((weights[ys, xs] * xs).sum() / count)
            y_com = float((weights[ys, xs] * ys).sum() / count)
            perimeter = float(cv2.arcLength(contour, True))
            contour_area = float(cv2.contourArea(contour))
            circularity = (
                4.0 * math.pi * contour_area / (perimeter ** 2)
                if perimeter > 0 else 0.0)
            if self.use_circularity and circularity < self.min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            result.append(dict(x=x_com, y=y_com, area=area, count=count,
                               bbox=(x, y, w, h), perimeter=perimeter,
                               circularity=circularity, contour=contour))
        return result, contours, mask

    def _predicted_position(self, timestamp_us):
        target = self.previous_position
        if len(self.detections) >= 2:
            t0, x0, y0 = self.detections[-1]
            dt = (timestamp_us - t0) / 1e6
            target = (x0 + self._last_velocity[0] * dt,
                      y0 + self._last_velocity[1] * dt)
        return target

    def _select(self, candidates, target):
        if not candidates:
            return None
        if target is None:
            return min(candidates, key=lambda c: (-c["count"], c["y"], c["x"]))
        ranked = sorted(candidates, key=lambda c: (
            math.hypot(c["x"] - target[0], c["y"] - target[1]),
            -c["count"], c["y"], c["x"]))
        distance = math.hypot(
            ranked[0]["x"] - target[0], ranked[0]["y"] - target[1])
        return ranked[0] if distance <= self.max_jump_px else None

    def _velocity(self):
        if len(self.detections) < 2:
            return 0.0, 0.0, False
        samples = np.asarray(self.detections, dtype=np.float64)
        span = samples[-1, 0] - samples[0, 0]
        if span < self.velocity_min_span_us:
            return 0.0, 0.0, False
        t = (samples[:, 0] - samples[-1, 0]) / 1e6
        vx = float(np.polyfit(t, samples[:, 1], 1)[0])
        vy = float(np.polyfit(t, samples[:, 2], 1)[0])
        self._last_velocity = (vx, vy)
        return vx, vy, True

    def _process_bin(self, start_us, events, parent_id):
        total_start = time.monotonic_ns()
        t = time.monotonic_ns()
        activity = self.build_activity_map(events)
        self.timings["map_build"].append((time.monotonic_ns() - t) / 1e6)
        t = time.monotonic_ns()
        candidates, threshold_contours, threshold_mask = self._candidates(activity)
        self.timings["blob_detection"].append((time.monotonic_ns() - t) / 1e6)
        self.counters["candidate_blob_count"] += len(candidates)
        predicted_position = self._predicted_position(start_us)
        selected = self._select(candidates, predicted_position)
        detection = TrackerDetection(start_us, start_us + self.bin_us, parent_id,
                                     int(activity.sum()), candidate_count=len(candidates),
                                     start_steady_ns=total_start)
        if selected is not None:
            detection.valid = True
            detection.x_px, detection.y_px = selected["x"], selected["y"]
            detection.blob_area_px = selected["area"]
            detection.blob_event_count = selected["count"]
            detection.circularity = selected["circularity"]
            denominator = max(1.0, self.min_event_count * 2.0)
            detection.confidence = min(1.0, selected["count"] / denominator)
            self.previous_position = (detection.x_px, detection.y_px)
            self.detections.append((start_us, detection.x_px, detection.y_px))
            t = time.monotonic_ns()
            vx, vy, ready = self._velocity()
            self.timings["velocity_fit"].append((time.monotonic_ns() - t) / 1e6)
            detection.vx_px_s, detection.vy_px_s, detection.velocity_valid = vx, vy, ready
            detection.speed_px_s = math.hypot(vx, vy)
            self.counters["valid_detections"] += 1
            self.counters["velocity_ready_count"] += int(ready)
        else:
            self.counters["invalid_detections"] += 1
        detection.end_steady_ns = time.monotonic_ns()
        elapsed = (detection.end_steady_ns - total_start) / 1e6
        self.timings["computation"].append(elapsed)
        self.counters["processed_1ms_bins"] += 1
        self.counters["empty_bins"] += int(detection.event_count == 0)
        snapshot = TrackerDebugSnapshot(
            activity=activity.copy(), threshold_mask=threshold_mask.copy(),
            detection=detection,
            candidate_contours=tuple(
                contour.copy() for contour in threshold_contours),
            accepted_candidate_contours=tuple(
                candidate["contour"].copy() for candidate in candidates),
            selected_contour=(
                selected["contour"].copy() if selected is not None else None),
            predicted_position=predicted_position,
            trajectory=tuple(
                (float(x), float(y)) for _, x, y in self.detections),
            x_crop=self.x_crop)
        with self._debug_lock:
            self._latest_debug_snapshot = snapshot
        return detection

    def latest_debug_snapshot(self):
        """Return the latest complete-bin snapshot, or None before the first bin."""
        with self._debug_lock:
            return self._latest_debug_snapshot

    def update(self, packet: EventPacket):
        update_start = time.monotonic_ns()
        self.counters["packets_received"] += 1
        self.counters["packets_with_events"] += int(packet.event_count > 0)
        self.counters["events_received"] += packet.event_count
        if packet.event_count == 0:
            elapsed_ms = (time.monotonic_ns() - update_start) / 1e6
            self.timings["total_tracker_update"].append(elapsed_ms)
            return []
        for event, timestamp in zip(packet.events, packet.timestamps_us):
            bin_start = int(timestamp // self.bin_us * self.bin_us)
            if self.next_bin_start_us is not None and bin_start < self.next_bin_start_us:
                self.counters["late_events_or_bins"] += 1
                continue
            self.pending[bin_start].append(event)
        packet_max = packet.last_event_timestamp_us
        if self.next_bin_start_us is None:
            first_bin = packet.first_event_timestamp_us // self.bin_us
            self.next_bin_start_us = int(first_bin * self.bin_us)
        self.max_seen_timestamp_us = max(packet_max, self.max_seen_timestamp_us or packet_max)
        oldest = self.max_seen_timestamp_us - self.history_limit_us
        for key in list(self.pending):
            if key < oldest and key < self.next_bin_start_us:
                del self.pending[key]
        output = []
        while self.next_bin_start_us + self.bin_us <= self.max_seen_timestamp_us:
            rows = self.pending.pop(self.next_bin_start_us, [])
            output.append(self._process_bin(self.next_bin_start_us, rows, packet.packet_id))
            self.next_bin_start_us += self.bin_us
        elapsed_ms = (time.monotonic_ns() - update_start) / 1e6
        self.timings["total_tracker_update"].append(elapsed_ms)
        return output

    def statistics(self):
        elapsed = max((time.monotonic_ns() - self.started_ns) / 1e9, 1e-9)
        result = dict(self.counters)
        result["event_rate_hz"] = result.get("events_received", 0) / elapsed
        result["position_output_rate_hz"] = result.get("valid_detections", 0) / elapsed
        result["detection_rate_hz"] = result.get("processed_1ms_bins", 0) / elapsed
        for name, values in self.timings.items():
            array = np.asarray(values, dtype=float)
            result[name] = (
                float(np.percentile(array, 50)),
                float(np.percentile(array, 95)), float(array.max())
            ) if array.size else (0.0, 0.0, 0.0)
        return result


def trace_detail_json(detection: TrackerDetection) -> str:
    """Build strict finite JSON; sensor time belongs here, never in ROS stamps."""
    detail = {"sensor_timestamp_domain": "genx320_microseconds",
              "bin_start_us": detection.bin_start_us, "bin_end_us": detection.bin_end_us,
              "event_count": detection.event_count, "x_px": detection.x_px,
              "y_px": detection.y_px, "vx_px_s": detection.vx_px_s,
              "vy_px_s": detection.vy_px_s, "speed_px_s": detection.speed_px_s,
              "blob_area_px": detection.blob_area_px,
              "blob_event_count": detection.blob_event_count,
              "circularity": detection.circularity,
              "candidate_count": detection.candidate_count,
              "velocity_valid": detection.velocity_valid}
    return json.dumps(detail, allow_nan=False, separators=(",", ":"))


def _rotate_debug_layer(image, rotation_degrees):
    """Rotate graphics while leaving text to be added by the caller."""
    rotations = {
        0: None,
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        -90: cv2.ROTATE_90_CLOCKWISE,
    }
    if rotation_degrees not in rotations:
        raise ValueError("debug rotation must be one of -90, 0, 90, or 180")
    return (cv2.rotate(image, rotations[rotation_degrees])
            if rotations[rotation_degrees] is not None else image)


def _debug_text(image, snapshot, stage):
    """Add upright stage and tracker metadata."""
    detection = snapshot.detection
    line1 = (
        f"{stage} bin [{detection.bin_start_us},{detection.bin_end_us}) us "
        f"events={detection.event_count} candidates={detection.candidate_count}")
    velocity = (
        f"v=({detection.vx_px_s:.1f},{detection.vy_px_s:.1f}) px/s"
        if detection.velocity_valid else "v=not ready")
    line2 = (
        f"valid={str(detection.valid).lower()} {velocity} "
        f"x=[{snapshot.x_crop[0]},{snapshot.x_crop[1]})")
    cv2.putText(image, line1, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, line2, (4, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_debug_images(snapshot: TrackerDebugSnapshot, clip_count=16,
                        velocity_scale_s=0.02,
                        rotation_degrees=90):
    """Render synchronized BGR images for each tracker stage."""
    clip_count = max(1, int(clip_count))
    gray = np.rint(
        np.minimum(snapshot.activity, clip_count) * (255.0 / clip_count)
    ).astype(np.uint8)
    activity = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    threshold = cv2.cvtColor(snapshot.threshold_mask, cv2.COLOR_GRAY2BGR)
    contours = activity.copy()
    tracking = activity.copy()

    accepted_ids = {
        contour.tobytes() for contour in snapshot.accepted_candidate_contours}
    for contour in snapshot.candidate_contours:
        color = (0, 0, 255) if contour.tobytes() in accepted_ids else (0, 128, 255)
        cv2.drawContours(contours, [contour], -1, color, 1)

    selected = snapshot.selected_contour
    for contour in snapshot.candidate_contours:
        if selected is not None and np.array_equal(contour, selected):
            continue
        cv2.drawContours(tracking, [contour], -1, (0, 128, 255), 1)
    if selected is not None:
        cv2.drawContours(tracking, [selected], -1, (0, 255, 0), 1)

    detection = snapshot.detection
    if snapshot.predicted_position is not None:
        px, py = (int(round(value)) for value in snapshot.predicted_position)
        cv2.drawMarker(tracking, (px, py), (0, 255, 255),
                       cv2.MARKER_CROSS, 9, 1)
    if len(snapshot.trajectory) >= 2:
        points = np.rint(snapshot.trajectory).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(tracking, [points], False, (255, 0, 0), 1, cv2.LINE_AA)
    if detection.valid:
        center = (int(round(detection.x_px)), int(round(detection.y_px)))
        cv2.circle(tracking, center, 3, (0, 255, 0), thickness=-1)
        if detection.velocity_valid:
            tip = (
                int(round(detection.x_px + detection.vx_px_s * velocity_scale_s)),
                int(round(detection.y_px + detection.vy_px_s * velocity_scale_s)))
            cv2.arrowedLine(tracking, center, tip, (255, 255, 0), 1,
                            cv2.LINE_AA, tipLength=0.25)

    images = {"activity": activity, "threshold": threshold,
              "contours": contours, "tracking": tracking}
    lower_x, upper_x = snapshot.x_crop
    for image in images.values():
        cv2.line(image, (lower_x, 0), (lower_x, image.shape[0] - 1),
                 (255, 0, 255), 1)
        cv2.line(image, (upper_x - 1, 0),
                 (upper_x - 1, image.shape[0] - 1), (255, 0, 255), 1)
    return {
        stage: _debug_text(
            _rotate_debug_layer(image, rotation_degrees), snapshot, stage)
        for stage, image in images.items()}


def render_debug_image(snapshot: TrackerDebugSnapshot, clip_count=16,
                       velocity_scale_s=0.02,
                       rotation_degrees=90) -> np.ndarray:
    """Render the backwards-compatible combined tracking debug image."""
    return render_debug_images(
        snapshot, clip_count, velocity_scale_s, rotation_degrees)["tracking"]
