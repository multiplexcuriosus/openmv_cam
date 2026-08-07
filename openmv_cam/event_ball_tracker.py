"""Small ROS-independent event-time ball tracker."""

import json
import math
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


class EventBallTracker:
    """Bin raw EVT1 events and select a simple activity blob."""

    def __init__(self, *, width=320, height=320, bin_ms=1.0,
                 history_limit_ms=100.0, min_event_count=3,
                 min_blob_area_px=2, max_blob_area_px=500,
                 activity_threshold=1, morphology_kernel=0,
                 morphology_iterations=0, use_circularity=False,
                 min_circularity=0.1, max_jump_px=100.0,
                 velocity_history_size=5, velocity_min_span_ms=3.0,
                 stats_history_size=512):
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

    def build_activity_map(self, events):
        activity = np.zeros((self.height, self.width), dtype=np.uint16)
        if len(events):
            events = np.asarray(events)
            x, y = events[:, 4].astype(np.int64), events[:, 5].astype(np.int64)
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
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
                               circularity=circularity))
        return result

    def _select(self, candidates, timestamp_us):
        if not candidates:
            return None
        target = self.previous_position
        if len(self.detections) >= 2:
            t0, x0, y0 = self.detections[-1]
            dt = (timestamp_us - t0) / 1e6
            target = (x0 + self._last_velocity[0] * dt,
                      y0 + self._last_velocity[1] * dt)
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
        candidates = self._candidates(activity)
        self.timings["blob_detection"].append((time.monotonic_ns() - t) / 1e6)
        self.counters["candidate_blob_count"] += len(candidates)
        selected = self._select(candidates, start_us)
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
        self.counters["empty_bins"] += int(not events)
        return detection

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
