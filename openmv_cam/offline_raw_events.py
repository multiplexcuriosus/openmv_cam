"""Streaming, ROS-free reader for packetized raw-event HDF5 files."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class RecordedEventPacket:
    """One packet slice read from a raw-event recording."""

    event_type: np.ndarray
    event_x: np.ndarray
    event_y: np.ndarray
    event_t_us: np.ndarray
    packet_id: int
    packet_ros_t_ns: int
    packet_monotonic_t_ns: int


class RawEventHDF5Reader:
    """Validate raw-event metadata and stream one packet slice at a time."""

    EVENT_PATHS = ("events/type", "events/x", "events/y", "events/t_us")
    PACKET_PATHS = (
        "packets/ros_t_ns", "packets/start_event_idx",
        "packets/end_event_idx")

    def __init__(self, path, *, width=None, height=None):
        """Open and validate packet metadata without reading event columns."""
        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        try:
            self._validate(width, height)
        except Exception:
            self._file.close()
            raise

    def _dataset(self, path):
        if path not in self._file:
            raise ValueError(f"malformed raw-event HDF5: missing /{path}")
        dataset = self._file[path]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"malformed /{path}: expected a dataset")
        if dataset.ndim != 1:
            raise ValueError(
                f"malformed /{path}: expected one dimension, got {dataset.shape}")
        return dataset

    def _dimension_attr(self, name):
        candidates = (self._file.attrs.get(name),
                      self._file.get("events", {}).attrs.get(name)
                      if "events" in self._file else None,
                      self._file.attrs.get(f"sensor_{name}"))
        values = [int(value) for value in candidates if value is not None]
        if len(set(values)) > 1:
            raise ValueError(f"inconsistent HDF5 {name} attributes: {values}")
        return values[0] if values else None

    def _validate(self, configured_width, configured_height):
        self._events = [self._dataset(path) for path in self.EVENT_PATHS]
        event_lengths = [len(dataset) for dataset in self._events]
        if len(set(event_lengths)) != 1:
            raise ValueError(
                "malformed /events: event arrays have different lengths "
                f"{dict(zip(self.EVENT_PATHS, event_lengths))}")
        self.number_of_events = event_lengths[0]

        packet_datasets = [self._dataset(path) for path in self.PACKET_PATHS]
        packet_lengths = [len(dataset) for dataset in packet_datasets]
        optional = {}
        for name in ("monotonic_t_ns", "packet_id"):
            path = f"packets/{name}"
            if path in self._file:
                optional[name] = self._dataset(path)
                packet_lengths.append(len(optional[name]))
        if len(set(packet_lengths)) != 1:
            raise ValueError(
                "malformed /packets: packet arrays have different lengths "
                f"{packet_lengths}")
        self.number_of_packets = packet_lengths[0]
        self._ros = np.asarray(packet_datasets[0][:], dtype=np.int64)
        self._starts = np.asarray(packet_datasets[1][:], dtype=np.int64)
        self._ends = np.asarray(packet_datasets[2][:], dtype=np.int64)
        self._monotonic = (
            np.asarray(optional["monotonic_t_ns"][:], dtype=np.int64)
            if "monotonic_t_ns" in optional else None)
        self._packet_ids = (
            np.asarray(optional["packet_id"][:], dtype=np.int64)
            if "packet_id" in optional else None)

        if np.any(self._ros < 0):
            raise ValueError("malformed /packets/ros_t_ns: timestamps are negative")
        if np.any(np.diff(self._ros) < 0):
            raise ValueError(
                "malformed /packets/ros_t_ns: timestamps are not non-decreasing")
        if self._monotonic is not None:
            if np.any(self._monotonic < 0):
                raise ValueError(
                    "malformed /packets/monotonic_t_ns: timestamps are negative")
            if np.any(np.diff(self._monotonic) < 0):
                raise ValueError(
                    "malformed /packets/monotonic_t_ns: timestamps are not "
                    "non-decreasing")
        if (np.any(self._starts < 0) or np.any(self._ends < 0) or
                np.any(self._starts > self._ends) or
                np.any(self._ends > self.number_of_events)):
            raise ValueError(
                "malformed packet indices: require 0 <= start_event_idx <= "
                f"end_event_idx <= {self.number_of_events}")
        if (np.any(np.diff(self._starts) < 0) or
                np.any(np.diff(self._ends) < 0)):
            raise ValueError(
                "malformed packet indices: packet boundaries are not non-decreasing")

        stored_width = self._dimension_attr("width")
        stored_height = self._dimension_attr("height")
        self.width = int(configured_width or stored_width or 320)
        self.height = int(configured_height or stored_height or 320)
        if stored_width is not None and stored_width != self.width:
            raise ValueError(
                f"configured width {self.width} conflicts with HDF5 width {stored_width}")
        if stored_height is not None and stored_height != self.height:
            raise ValueError(
                f"configured height {self.height} conflicts with HDF5 height {stored_height}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sensor width and height must be positive")

    def __enter__(self):
        """Return this open reader."""
        return self

    def __exit__(self, *_):
        """Close the HDF5 file on context exit."""
        self.close()

    def close(self):
        """Close the underlying HDF5 file."""
        self._file.close()

    def __len__(self):
        """Return the number of recorded packets."""
        return self.number_of_packets

    def _packet(self, row):
        start, end = int(self._starts[row]), int(self._ends[row])
        event_type, event_x, event_y, event_t_us = (
            np.asarray(dataset[start:end]) for dataset in self._events)
        if np.any(event_t_us < 0):
            raise ValueError(
                f"malformed /events/t_us: packet row {row} has negative timestamps")
        if (np.any(event_x < 0) or np.any(event_x >= self.width)):
            raise ValueError(
                f"malformed /events/x: packet row {row} has coordinates outside "
                f"[0, {self.width})")
        if (np.any(event_y < 0) or np.any(event_y >= self.height)):
            raise ValueError(
                f"malformed /events/y: packet row {row} has coordinates outside "
                f"[0, {self.height})")
        monotonic = (int(self._monotonic[row]) if self._monotonic is not None
                     else -1)
        packet_id = (int(self._packet_ids[row]) if self._packet_ids is not None
                     else row)
        return RecordedEventPacket(
            event_type, event_x, event_y, event_t_us, packet_id,
            int(self._ros[row]), monotonic)

    def iter_packets(self, *, start_ros_t_ns=None, end_ros_t_ns=None):
        """Yield packets with ROS time in ``[start, end]`` inclusively."""
        first = (0 if start_ros_t_ns is None else int(np.searchsorted(
            self._ros, int(start_ros_t_ns), side="left")))
        last = (self.number_of_packets if end_ros_t_ns is None else
                int(np.searchsorted(self._ros, int(end_ros_t_ns), side="right")))
        for row in range(first, last):
            yield self._packet(row)
