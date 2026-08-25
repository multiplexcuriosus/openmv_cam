"""ROS-independent sliding-window event ball tracker."""

import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .event_frame_contract import render_event_frame_from_arrays
from .evt1_protocol import EventPacket
from .raw_evt2_protocol import sequence_gap


@dataclass
class TrackerDetection:
    """Result of one packet-rate sliding-window tracker update."""

    bin_start_us: int
    bin_end_us: int
    parent_packet_id: int
    event_count: int
    tracker_update_id: int = 0
    source_packet_id: int = 0
    source_packet_id_valid: bool = False
    window_start_us: int = 0
    window_end_us: int = 0
    window_event_count: int = 0
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
    blob_width_px: int = 0
    blob_height_px: int = 0
    blob_perimeter_px: float = 0.0
    circularity: float = 0.0
    candidate_count: int = 0
    rejection_reason: str = ""
    start_steady_ns: int = 0
    end_steady_ns: int = 0


@dataclass(frozen=True)
class TrackerDebugSnapshot:
    """Immutable inputs needed to render synchronized debug images."""

    activity: np.ndarray
    threshold_mask: np.ndarray
    detection: TrackerDetection
    candidate_contours: tuple
    accepted_candidate_contours: tuple
    selected_contour: object
    predicted_position: object
    trajectory: tuple
    x_crop: tuple
    y_crop: tuple = ()
    candidate_details: tuple = ()
    removed_component_contours: tuple = ()
    debug_events: object = None


class EventBallTracker:
    """Track raw event activity using one sliding-window update per packet."""

    MORPHOLOGY_OPERATIONS = ("none", "close", "dilate")

    def __init__(self, *, width=320, height=320, bin_ms=1.0,
                 accumulation_window_ms=10.0, history_limit_ms=100.0,
                 min_event_count=3, min_blob_area_px=2,
                 max_blob_area_px=2000, min_blob_width_px=1,
                 max_blob_width_px=320, min_blob_height_px=1,
                 max_blob_height_px=320, activity_threshold=1,
                 spatial_filter_enabled=False,
                 spatial_filter_min_neighbors=1,
                 spatial_filter_min_component_area_px=1,
                 morphology_operation="close", morphology_kernel=3,
                 morphology_iterations=1, use_circularity=False,
                 min_circularity=0.1, max_jump_px=100.0,
                 reacquire_after_misses=3, velocity_history_size=5,
                 velocity_min_span_ms=3.0, stats_history_size=512,
                 x_crop=None, y_crop=None,
                 debug_event_frame_window_ms=None):
        self.width, self.height = int(width), int(height)
        self.bin_us = int(round(float(bin_ms) * 1000.0))
        self.accumulation_window_us = int(round(
            float(accumulation_window_ms) * 1000.0))
        if self.bin_us <= 0:
            raise ValueError("bin_ms must be positive")
        if self.accumulation_window_us < self.bin_us:
            raise ValueError("accumulation_window_ms must be at least bin_ms")
        configured_history_us = int(float(history_limit_ms) * 1000.0)
        self.history_limit_us = max(
            self.accumulation_window_us, configured_history_us)
        self.debug_event_frame_window_us = (
            None if debug_event_frame_window_ms is None else
            int(round(float(debug_event_frame_window_ms) * 1000.0)))
        if (self.debug_event_frame_window_us is not None and
                self.debug_event_frame_window_us <= 0):
            raise ValueError("debug_event_frame_window_ms must be positive")
        if self.debug_event_frame_window_us is not None:
            self.history_limit_us = max(
                self.history_limit_us, self.debug_event_frame_window_us)
        self.min_event_count = int(min_event_count)
        self.min_blob_area_px = int(min_blob_area_px)
        self.max_blob_area_px = int(max_blob_area_px)
        self.min_blob_width_px = int(min_blob_width_px)
        self.max_blob_width_px = int(max_blob_width_px)
        self.min_blob_height_px = int(min_blob_height_px)
        self.max_blob_height_px = int(max_blob_height_px)
        self.activity_threshold = max(1, int(activity_threshold))
        self.spatial_filter_enabled = bool(spatial_filter_enabled)
        self.spatial_filter_min_neighbors = int(
            spatial_filter_min_neighbors)
        if not 0 <= self.spatial_filter_min_neighbors <= 8:
            raise ValueError(
                "spatial_filter_min_neighbors must be between 0 and 8")
        self.spatial_filter_min_component_area_px = int(
            spatial_filter_min_component_area_px)
        if self.spatial_filter_min_component_area_px < 1:
            raise ValueError(
                "spatial_filter_min_component_area_px must be positive")
        operation = str(morphology_operation).strip().lower()
        if operation not in self.MORPHOLOGY_OPERATIONS:
            raise ValueError(
                f"morphology_operation must be one of "
                f"{self.MORPHOLOGY_OPERATIONS}")
        self.morphology_operation = operation
        self.morphology_kernel = int(morphology_kernel)
        self.morphology_iterations = int(morphology_iterations)
        self.use_circularity = bool(use_circularity)
        self.min_circularity = float(min_circularity)
        self.max_jump_px = float(max_jump_px)
        self.reacquire_after_misses = max(1, int(reacquire_after_misses))
        self.missed_updates = 0
        if x_crop is None:
            x_crop = (0, self.width)
        if len(x_crop) != 2:
            raise ValueError("x_crop must contain lower and upper bounds")
        self.x_crop = (int(x_crop[0]), int(x_crop[1]))
        if not 0 <= self.x_crop[0] < self.x_crop[1] <= self.width:
            raise ValueError(
                f"x_crop must satisfy 0 <= lower < upper <= {self.width}")
        if y_crop is None:
            y_crop = (0, self.height, 0, self.height)
        if len(y_crop) != 4:
            raise ValueError(
                "y_crop must contain left lower/upper and right lower/upper bounds")
        self.y_crop = tuple(int(value) for value in y_crop)
        left_lower, left_upper, right_lower, right_upper = self.y_crop
        if not (0 <= left_lower < left_upper <= self.height and
                0 <= right_lower < right_upper <= self.height):
            raise ValueError(
                "y_crop endpoint pairs must satisfy "
                f"0 <= lower < upper <= {self.height}")

        self.velocity_min_span_us = float(velocity_min_span_ms) * 1000.0
        self.detections = deque(maxlen=max(2, int(velocity_history_size)))
        self._last_velocity = (0.0, 0.0)
        self.previous_position = None

        # Incomplete bins accept current/future events. Completed bins retain
        # exact timestamps so non-bin-aligned sliding-window edges stay exact.
        self.pending = defaultdict(list)
        self.pending_timestamps = defaultdict(list)
        self.pending_packet_ids = defaultdict(list)
        self.pending_source_packet_ids = defaultdict(list)
        self.pending_source_packet_valid = defaultdict(list)
        self.max_history_bins = (
            int(math.ceil(self.history_limit_us / self.bin_us)) + 2)
        self.history_bins = deque(maxlen=self.max_history_bins)
        self.next_bin_start_us = None
        self.max_seen_timestamp_us = None

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
        self._last_removed_component_contours = ()
        self._tracker_update_id = 0
        self._last_source_packet_id = None

    def _valid_event_coordinates(self, events):
        x = events[:, 4].astype(np.int64)
        y = events[:, 5].astype(np.int64)
        valid = (x >= self.x_crop[0]) & (x < self.x_crop[1])
        valid &= (x >= 0) & (x < self.width)
        x_span = self.x_crop[1] - self.x_crop[0]
        fraction = (x - self.x_crop[0]) / float(x_span)
        lower_y = self.y_crop[0] + fraction * (
            self.y_crop[2] - self.y_crop[0])
        upper_y = self.y_crop[1] + fraction * (
            self.y_crop[3] - self.y_crop[1])
        valid &= (y >= lower_y) & (y < upper_y)
        valid &= (y >= 0) & (y < self.height)
        return x, y, valid

    def build_activity_map(self, events):
        """Accumulate raw event multiplicity without morphology."""
        activity = np.zeros((self.height, self.width), dtype=np.uint16)
        if len(events):
            events = np.asarray(events)
            x, y, valid = self._valid_event_coordinates(events)
            np.add.at(activity, (y[valid], x[valid]), 1)
        return activity

    def _crop_mask(self):
        """Return the exact pixel-center crop used for event acceptance."""
        x = np.arange(self.width, dtype=np.float64)
        x_span = self.x_crop[1] - self.x_crop[0]
        fraction = (x - self.x_crop[0]) / float(x_span)
        lower_y = self.y_crop[0] + fraction * (
            self.y_crop[2] - self.y_crop[0])
        upper_y = self.y_crop[1] + fraction * (
            self.y_crop[3] - self.y_crop[1])
        y = np.arange(self.height, dtype=np.float64)[:, None]
        return ((x >= self.x_crop[0]) & (x < self.x_crop[1]) &
                (y >= lower_y) & (y < upper_y))

    def _grouping_mask(self, activity):
        self._last_removed_component_contours = ()
        crop_mask = self._crop_mask().astype(np.uint8)
        foreground = (activity >= self.activity_threshold).astype(np.uint8)
        foreground *= crop_mask
        threshold_count = int(cv2.countNonZero(foreground))
        self.counters["threshold_foreground_pixels"] += threshold_count

        if self.spatial_filter_enabled:
            neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
            neighbor_kernel[1, 1] = 0
            neighbor_count = cv2.filter2D(
                foreground, cv2.CV_16U, neighbor_kernel,
                borderType=cv2.BORDER_CONSTANT)
            foreground[neighbor_count <
                       self.spatial_filter_min_neighbors] = 0
            self.counters["spatial_filter_removed_pixels"] += (
                threshold_count - int(cv2.countNonZero(foreground)))

            component_count, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    foreground, connectivity=8))
            removed_labels = [
                label for label in range(1, component_count)
                if stats[label, cv2.CC_STAT_AREA] <
                self.spatial_filter_min_component_area_px
            ]
            if removed_labels:
                removed_mask = np.isin(labels, removed_labels).astype(
                    np.uint8)
                removed_pixels = int(cv2.countNonZero(removed_mask))
                foreground[removed_mask != 0] = 0
                removed_contours, _ = cv2.findContours(
                    removed_mask * 255, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                self._last_removed_component_contours = tuple(
                    contour.copy() for contour in removed_contours)
                self.counters["spatial_filter_removed_components"] += len(
                    removed_labels)
                self.counters[
                    "spatial_filter_removed_component_pixels"] += (
                        removed_pixels)

        mask = foreground * 255
        if (self.morphology_operation == "none" or
                self.morphology_kernel <= 0 or
                self.morphology_iterations <= 0):
            return mask
        kernel = np.ones(
            (self.morphology_kernel, self.morphology_kernel), np.uint8)
        if self.morphology_operation == "close":
            grouped = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel,
                iterations=self.morphology_iterations)
        else:
            # Explicit dilation is the only other allowed grouping operation.
            grouped = cv2.dilate(
                mask, kernel, iterations=self.morphology_iterations)
        return grouped * crop_mask

    def _candidate_rejection_reason(self, candidate):
        if candidate["area"] < self.min_blob_area_px:
            return "area_below_min"
        if candidate["area"] > self.max_blob_area_px:
            return "area_above_max"
        if candidate["width"] < self.min_blob_width_px:
            return "width_below_min"
        if candidate["width"] > self.max_blob_width_px:
            return "width_above_max"
        if candidate["height"] < self.min_blob_height_px:
            return "height_below_min"
        if candidate["height"] > self.max_blob_height_px:
            return "height_above_max"
        if candidate["raw_event_count"] < self.min_event_count:
            return "raw_event_count_below_min"
        if candidate["raw_event_count"] <= 0:
            return "no_raw_events"
        if (self.use_circularity and
                candidate["circularity"] < self.min_circularity):
            return "circularity_below_min"
        return ""

    def _candidates(self, activity, predicted_position):
        grouping_mask = self._grouping_mask(activity)
        contours, _ = cv2.findContours(
            grouping_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour_index, contour in enumerate(contours):
            filled = np.zeros_like(grouping_mask)
            cv2.drawContours(filled, [contour], -1, 1, thickness=-1)
            raw_weights = activity.astype(np.float64) * filled
            raw_event_count = int(raw_weights.sum())
            raw_y, raw_x = np.nonzero(raw_weights)
            if raw_event_count > 0:
                raw_values = raw_weights[raw_y, raw_x]
                x_com = float((raw_values * raw_x).sum() / raw_event_count)
                y_com = float((raw_values * raw_y).sum() / raw_event_count)
            else:
                x_com = y_com = 0.0
            area = int(cv2.countNonZero(filled))
            x, y, width, height = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            contour_area = float(cv2.contourArea(contour))
            circularity = (
                4.0 * math.pi * contour_area / (perimeter ** 2)
                if perimeter > 0 else 0.0)
            distance = (
                math.hypot(x_com - predicted_position[0],
                           y_com - predicted_position[1])
                if predicted_position is not None and raw_event_count > 0
                else 0.0)
            candidate = {
                "index": contour_index, "contour": contour,
                "x": x_com, "y": y_com, "area": area,
                "raw_event_count": raw_event_count,
                "count": raw_event_count, "bbox": (x, y, width, height),
                "width": width, "height": height, "perimeter": perimeter,
                "circularity": circularity,
                "distance_from_prediction_px": float(distance),
                "rejection_reason": "",
            }
            candidate["rejection_reason"] = (
                self._candidate_rejection_reason(candidate))
            candidates.append(candidate)
        return candidates, contours, grouping_mask

    def _predicted_position(self, timestamp_us):
        target = self.previous_position
        if self.detections:
            t0, x0, y0 = self.detections[-1]
            dt = (timestamp_us - t0) / 1e6
            target = (x0 + self._last_velocity[0] * dt,
                      y0 + self._last_velocity[1] * dt)
        return target

    @staticmethod
    def _initial_candidate_key(candidate):
        return (-candidate["raw_event_count"], -candidate["area"],
                candidate["y"], candidate["x"], candidate["index"])

    def _select(self, candidates, predicted_position):
        valid = [item for item in candidates if not item["rejection_reason"]]
        if not valid:
            return None, "no_valid_candidates"
        if predicted_position is None:
            return min(valid, key=self._initial_candidate_key), ""
        near = []
        for candidate in valid:
            if candidate["distance_from_prediction_px"] <= self.max_jump_px:
                near.append(candidate)
            else:
                candidate["rejection_reason"] = "beyond_max_jump"
        if near:
            selected = min(
                near,
                key=lambda item: (
                    item["distance_from_prediction_px"],
                    -item["raw_event_count"], -item["area"],
                    item["y"], item["x"], item["index"]))
            return selected, ""
        if self.missed_updates + 1 >= self.reacquire_after_misses:
            reacquired = min(valid, key=self._initial_candidate_key)
            reacquired["rejection_reason"] = ""
            return reacquired, "reacquired_after_misses"
        return None, "all_candidates_beyond_max_jump"

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

    def _complete_bins(self):
        completed = 0
        while (self.next_bin_start_us is not None and
               self.next_bin_start_us + self.bin_us <=
               self.max_seen_timestamp_us):
            start = self.next_bin_start_us
            rows = self.pending.pop(start, [])
            timestamps = self.pending_timestamps.pop(start, [])
            packet_ids = self.pending_packet_ids.pop(start, [])
            source_packet_ids = self.pending_source_packet_ids.pop(start, [])
            source_packet_valid = self.pending_source_packet_valid.pop(start, [])
            events_array = np.asarray(rows, dtype=np.uint16).reshape(-1, 6)
            timestamps_array = np.asarray(timestamps, dtype=np.int64)
            self.history_bins.append((
                start, events_array, timestamps_array,
                np.asarray(packet_ids, dtype=np.int64),
                np.asarray(source_packet_ids, dtype=np.uint64),
                np.asarray(source_packet_valid, dtype=bool)))
            self.next_bin_start_us += self.bin_us
            completed += 1
            self.counters["processed_1ms_bins"] += 1
            self.counters["empty_bins"] += int(events_array.size == 0)
        return completed

    def _window_events(self, window_start_us, window_end_us):
        event_chunks = []
        timestamp_chunks = []
        packet_id_chunks = []
        source_id_chunks = []
        source_valid_chunks = []
        for (_, events, timestamps, packet_ids, source_ids,
             source_valid) in self.history_bins:
            if not timestamps.size:
                continue
            selected = ((timestamps >= window_start_us) &
                        (timestamps < window_end_us))
            if np.any(selected):
                event_chunks.append(events[selected])
                timestamp_chunks.append(timestamps[selected])
                packet_id_chunks.append(packet_ids[selected])
                source_id_chunks.append(source_ids[selected])
                source_valid_chunks.append(source_valid[selected])
        if not event_chunks:
            return (np.empty((0, 6), dtype=np.uint16),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.uint64),
                    np.empty((0,), dtype=bool))
        return (np.concatenate(event_chunks), np.concatenate(timestamp_chunks),
                np.concatenate(packet_id_chunks), np.concatenate(source_id_chunks),
                np.concatenate(source_valid_chunks))

    def _trim_history(self, window_start_us):
        retention_start_us = window_start_us
        if self.debug_event_frame_window_us is not None:
            retention_start_us = min(
                retention_start_us,
                self.next_bin_start_us - self.debug_event_frame_window_us)
        history_cutoff = max(
            retention_start_us,
            self.max_seen_timestamp_us - self.history_limit_us)
        while (self.history_bins and
               self.history_bins[0][0] + self.bin_us <= history_cutoff):
            self.history_bins.popleft()

    def _process_window(self, window_start_us, window_end_us, events,
                        parent_id, source_packet_id=0,
                        source_packet_id_valid=False):
        total_start = time.monotonic_ns()
        t = time.monotonic_ns()
        activity = self.build_activity_map(events)
        self.timings["map_build"].append((time.monotonic_ns() - t) / 1e6)
        predicted_position = self._predicted_position(window_end_us)
        t = time.monotonic_ns()
        candidates, raw_contours, grouping_mask = self._candidates(
            activity, predicted_position)
        self.timings["blob_detection"].append(
            (time.monotonic_ns() - t) / 1e6)
        self.counters["candidate_blob_count"] += len(candidates)
        selected, selection_reason = self._select(
            candidates, predicted_position)
        window_event_count = int(activity.sum())
        self._tracker_update_id += 1
        detection = TrackerDetection(
            bin_start_us=window_start_us, bin_end_us=window_end_us,
            parent_packet_id=parent_id, event_count=window_event_count,
            tracker_update_id=self._tracker_update_id,
            source_packet_id=source_packet_id,
            source_packet_id_valid=source_packet_id_valid,
            window_start_us=window_start_us, window_end_us=window_end_us,
            window_event_count=window_event_count,
            candidate_count=len(candidates), start_steady_ns=total_start,
            rejection_reason=selection_reason)
        if selected is not None and selected["raw_event_count"] > 0:
            detection.valid = True
            detection.x_px, detection.y_px = selected["x"], selected["y"]
            detection.blob_area_px = selected["area"]
            detection.blob_event_count = selected["raw_event_count"]
            detection.blob_width_px = selected["width"]
            detection.blob_height_px = selected["height"]
            detection.blob_perimeter_px = selected["perimeter"]
            detection.circularity = selected["circularity"]
            denominator = max(1.0, self.min_event_count * 2.0)
            detection.confidence = min(
                1.0, selected["raw_event_count"] / denominator)
            detection.rejection_reason = selection_reason
            self.previous_position = (detection.x_px, detection.y_px)
            self.detections.append(
                (window_end_us, detection.x_px, detection.y_px))
            t = time.monotonic_ns()
            vx, vy, ready = self._velocity()
            self.timings["velocity_fit"].append(
                (time.monotonic_ns() - t) / 1e6)
            detection.vx_px_s, detection.vy_px_s = vx, vy
            detection.velocity_valid = ready
            detection.speed_px_s = math.hypot(vx, vy)
            self.missed_updates = 0
            self.counters["valid_detections"] += 1
            self.counters["velocity_ready_count"] += int(ready)
        else:
            self.missed_updates += 1
            self.counters["invalid_detections"] += 1
            self.counters[f"rejection_{detection.rejection_reason}"] += 1
        detection.end_steady_ns = time.monotonic_ns()
        elapsed = (detection.end_steady_ns - total_start) / 1e6
        self.timings["computation"].append(elapsed)
        self.counters["window_updates"] += 1

        accepted = tuple(
            item["contour"].copy() for item in candidates
            if not item["rejection_reason"] or item is selected)
        details = tuple({
            "area": item["area"], "width": item["width"],
            "height": item["height"],
            "raw_event_count": item["raw_event_count"],
            "circularity": item["circularity"],
            "distance_from_prediction_px":
                item["distance_from_prediction_px"],
            "rejection_reason": item["rejection_reason"],
        } for item in candidates)
        debug_events = None
        if self.debug_event_frame_window_us is not None:
            debug_start_us = window_end_us - self.debug_event_frame_window_us
            debug_events, _, _, _, _ = self._window_events(
                debug_start_us, window_end_us)
            debug_events = debug_events.copy()
            debug_events.setflags(write=False)
        snapshot = TrackerDebugSnapshot(
            activity=activity.copy(), threshold_mask=grouping_mask.copy(),
            detection=detection,
            candidate_contours=tuple(
                contour.copy() for contour in raw_contours),
            accepted_candidate_contours=accepted,
            selected_contour=(
                selected["contour"].copy() if selected is not None else None),
            predicted_position=predicted_position,
            trajectory=tuple(
                (float(x), float(y)) for _, x, y in self.detections),
            x_crop=self.x_crop, y_crop=self.y_crop,
            candidate_details=details,
            removed_component_contours=tuple(
                contour.copy() for contour in
                self._last_removed_component_contours),
            debug_events=debug_events)
        with self._debug_lock:
            self._latest_debug_snapshot = snapshot
        return detection

    def latest_debug_snapshot(self):
        """Return the latest packet-level window snapshot, if available."""
        with self._debug_lock:
            return self._latest_debug_snapshot

    def update(self, packet: EventPacket):
        """Integrate all complete bins and run at most one window detection."""
        update_start = time.monotonic_ns()
        self.counters["packets_received"] += 1
        self.counters["packets_with_events"] += int(packet.event_count > 0)
        self.counters["events_received"] += packet.event_count
        if packet.wire_format == "raw_evt20" and packet.wire_sequence >= 0:
            gap = sequence_gap(self._last_source_packet_id, packet.wire_sequence)
            self.counters["source_packet_gaps"] += gap
            self._last_source_packet_id = packet.wire_sequence
        if packet.event_count == 0:
            elapsed_ms = (time.monotonic_ns() - update_start) / 1e6
            self.timings["total_tracker_update"].append(elapsed_ms)
            return []

        if self.next_bin_start_us is None:
            first_bin = packet.first_event_timestamp_us // self.bin_us
            self.next_bin_start_us = int(first_bin * self.bin_us)
        for event, timestamp in zip(packet.events, packet.timestamps_us):
            bin_start = int(timestamp // self.bin_us * self.bin_us)
            if bin_start < self.next_bin_start_us:
                self.counters["late_events_or_bins"] += 1
                continue
            self.pending[bin_start].append(event)
            self.pending_timestamps[bin_start].append(int(timestamp))
            self.pending_packet_ids[bin_start].append(int(packet.packet_id))
            source_valid = (
                packet.wire_format == "raw_evt20" and packet.wire_sequence >= 0)
            self.pending_source_packet_ids[bin_start].append(
                int(packet.wire_sequence) if source_valid else 0)
            self.pending_source_packet_valid[bin_start].append(source_valid)
        packet_max = packet.last_event_timestamp_us
        self.max_seen_timestamp_us = max(
            packet_max, self.max_seen_timestamp_us or packet_max)
        completed = self._complete_bins()
        output = []
        if completed > 0:
            window_end_us = self.next_bin_start_us
            window_start_us = window_end_us - self.accumulation_window_us
            self._trim_history(window_start_us)
            (window_events, _, packet_ids, source_ids,
             source_valid) = self._window_events(
                window_start_us, window_end_us)
            parent_id = int(packet_ids[-1]) if packet_ids.size else packet.packet_id
            newest = int(np.argmax(packet_ids)) if packet_ids.size else -1
            provenance_valid = bool(source_valid[newest]) if newest >= 0 else False
            output.append(self._process_window(
                window_start_us, window_end_us, window_events,
                parent_id,
                int(source_ids[newest]) if provenance_valid else 0,
                provenance_valid))
        elapsed_ms = (time.monotonic_ns() - update_start) / 1e6
        self.timings["total_tracker_update"].append(elapsed_ms)
        return output

    def statistics(self):
        elapsed = max((time.monotonic_ns() - self.started_ns) / 1e9, 1e-9)
        result = dict(self.counters)
        result["event_rate_hz"] = result.get("events_received", 0) / elapsed
        result["processed_bin_rate_hz"] = (
            result.get("processed_1ms_bins", 0) / elapsed)
        result["window_update_rate_hz"] = (
            result.get("window_updates", 0) / elapsed)
        result["tracker_update_rate_hz"] = result["window_update_rate_hz"]
        result["valid_detection_rate_hz"] = (
            result.get("valid_detections", 0) / elapsed)
        result["valid_update_rate_hz"] = result["valid_detection_rate_hz"]
        result["invalid_update_rate_hz"] = (
            result.get("invalid_detections", 0) / elapsed)
        result["rejection_counts"] = {
            key.removeprefix("rejection_"): value
            for key, value in result.items() if key.startswith("rejection_")}
        result["position_output_rate_hz"] = result["valid_detection_rate_hz"]
        for name, values in self.timings.items():
            array = np.asarray(values, dtype=float)
            result[name] = (
                float(np.percentile(array, 50)),
                float(np.percentile(array, 95)), float(array.max())
            ) if array.size else (0.0, 0.0, 0.0)
        return result


def trace_detail_json(detection: TrackerDetection,
                      availability_timestamp_ns=None) -> str:
    """Build finite JSON with sensor timestamps outside ROS stamp fields."""
    detail = {
        "tracker_update_id": detection.tracker_update_id,
        "source_packet_id": detection.source_packet_id,
        "source_packet_id_valid": detection.source_packet_id_valid,
        "sensor_timestamp_domain": "genx320_microseconds",
        "sensor_window_start_us": detection.window_start_us,
        "sensor_window_end_us": detection.window_end_us,
        # Compatibility aliases retained for existing trace consumers.
        "window_start_us": detection.window_start_us,
        "window_end_us": detection.window_end_us,
        "window_event_count": detection.window_event_count,
        "bin_start_us": detection.bin_start_us,
        "bin_end_us": detection.bin_end_us,
        "event_count": detection.event_count,
        "candidate_count": detection.candidate_count,
        "confidence": detection.confidence,
        "selected_raw_event_count": detection.blob_event_count,
        "x_px": detection.x_px, "y_px": detection.y_px,
        "vx_px_s": detection.vx_px_s, "vy_px_s": detection.vy_px_s,
        "speed_px_s": detection.speed_px_s,
        "blob_area_px": detection.blob_area_px,
        "blob_width_px": detection.blob_width_px,
        "blob_height_px": detection.blob_height_px,
        "blob_perimeter_px": detection.blob_perimeter_px,
        "circularity": detection.circularity,
        "velocity_valid": detection.velocity_valid,
        "valid": detection.valid,
        "rejection_reason": detection.rejection_reason,
    }
    if availability_timestamp_ns is not None:
        detail["availability_timestamp_ns"] = int(availability_timestamp_ns)
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
    """Add upright stage and sliding-window metadata."""
    detection = snapshot.detection
    line1 = (
        f"{stage} win [{detection.window_start_us},"
        f"{detection.window_end_us}) us events={detection.window_event_count} "
        f"cand={detection.candidate_count}")
    velocity = (
        f"v=({detection.vx_px_s:.1f},{detection.vy_px_s:.1f}) px/s"
        if detection.velocity_valid else "v=not ready")
    line2 = (
        f"valid={str(detection.valid).lower()} {velocity} "
        f"x=[{snapshot.x_crop[0]},{snapshot.x_crop[1]}) "
        f"y={snapshot.y_crop or 'full'}")
    cv2.putText(image, line1, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, line2, (4, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if stage in ("contours", "tracking"):
        cv2.putText(
            image,
            "cyan=component removed orange=not selected green=selected",
            (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
            (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_debug_images(snapshot: TrackerDebugSnapshot, clip_count=16,
                        velocity_scale_s=0.02,
                        rotation_degrees=90):
    """Render synchronized BGR images for each sliding-window stage."""
    clip_count = max(1, int(clip_count))
    gray = np.rint(
        np.minimum(snapshot.activity, clip_count) * (255.0 / clip_count)
    ).astype(np.uint8)
    activity = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    threshold = cv2.cvtColor(snapshot.threshold_mask, cv2.COLOR_GRAY2BGR)
    contours = activity.copy()
    tracking = activity.copy()

    for removed in snapshot.removed_component_contours:
        cv2.drawContours(contours, [removed], -1, (255, 255, 0), 1)
        cv2.drawContours(tracking, [removed], -1, (255, 255, 0), 1)

    accepted_ids = {
        contour.tobytes() for contour in snapshot.accepted_candidate_contours}
    for contour in snapshot.candidate_contours:
        color = ((0, 0, 255) if contour.tobytes() in accepted_ids
                 else (0, 128, 255))
        cv2.drawContours(contours, [contour], -1, color, 1)

    contour_details = list(zip(
        snapshot.candidate_contours, snapshot.candidate_details))
    if contour_details:
        largest_contour, largest_detail = max(
            contour_details, key=lambda item: item[1]["area"])
        x, y, width, _ = cv2.boundingRect(largest_contour)
        label = f"area={largest_detail['area']}"
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_layer = np.zeros(
            (text_size[1] + 6, text_size[0] + 4, 3), dtype=np.uint8)
        cv2.putText(
            label_layer, label, (2, text_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            cv2.LINE_AA)
        label_layer = cv2.rotate(label_layer, cv2.ROTATE_90_CLOCKWISE)
        label_height, label_width = label_layer.shape[:2]
        label_x = x + width + 3
        if label_x + label_width >= contours.shape[1]:
            label_x = max(0, x - label_width - 3)
        label_y = min(
            max(0, y), max(0, contours.shape[0] - label_height))
        target = contours[
            label_y:label_y + label_height,
            label_x:label_x + label_width]
        label_mask = np.any(label_layer != 0, axis=2)
        target[label_mask] = label_layer[label_mask]

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
        cv2.polylines(tracking, [points], False, (255, 0, 0), 1,
                      cv2.LINE_AA)
    if detection.valid:
        center = (int(round(detection.x_px)), int(round(detection.y_px)))
        cv2.circle(tracking, center, 3, (0, 255, 0), thickness=-1)
        if detection.velocity_valid:
            tip = (
                int(round(detection.x_px +
                          detection.vx_px_s * velocity_scale_s)),
                int(round(detection.y_px +
                          detection.vy_px_s * velocity_scale_s)))
            cv2.arrowedLine(tracking, center, tip, (255, 255, 0), 1,
                            cv2.LINE_AA, tipLength=0.25)

    images = {"activity": activity, "threshold": threshold,
              "contours": contours, "tracking": tracking}
    lower_x, upper_x = snapshot.x_crop
    y_crop = snapshot.y_crop or (
        0, snapshot.activity.shape[0], 0, snapshot.activity.shape[0])
    left_lower, left_upper, right_lower, right_upper = y_crop
    crop_polygon = np.asarray([
        (lower_x, left_lower), (upper_x - 1, right_lower),
        (upper_x - 1, right_upper - 1), (lower_x, left_upper - 1),
    ], dtype=np.int32).reshape(-1, 1, 2)
    for image in images.values():
        cv2.polylines(image, [crop_polygon], True, (255, 0, 255), 1)
    return {
        stage: _debug_text(
            _rotate_debug_layer(image, rotation_degrees), snapshot, stage)
        for stage, image in images.items()}


def render_debug_image(snapshot: TrackerDebugSnapshot, clip_count=16,
                       velocity_scale_s=0.02,
                       rotation_degrees=90) -> np.ndarray:
    """Render the compatibility alias of the final tracking debug image."""
    return render_debug_images(
        snapshot, clip_count, velocity_scale_s, rotation_degrees)["tracking"]


def render_debug_event_frame(snapshot: TrackerDebugSnapshot, *, width=320,
                             height=320, contrast=4.0, step=1.0,
                             rotation_degrees=90) -> np.ndarray:
    """Render copied sensor-time events with the latest valid COM in red."""
    events = snapshot.debug_events
    if events is None:
        raise ValueError("snapshot does not contain debug event-frame events")
    mono = render_event_frame_from_arrays(
        event_type=events[:, 0], event_x=events[:, 4], event_y=events[:, 5],
        width=width, height=height, scaling_mode="legacy_per_frame_max",
        contrast=contrast, step=step)
    image = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    detection = snapshot.detection
    if detection.valid:
        center = (int(round(detection.x_px)), int(round(detection.y_px)))
        cv2.circle(image, center, 5, (0, 0, 255), 1, cv2.LINE_8)
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS,
                       11, 1, cv2.LINE_8)
    return _rotate_debug_layer(image, rotation_degrees)
