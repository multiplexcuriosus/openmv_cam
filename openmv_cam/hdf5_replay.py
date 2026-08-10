"""Packet-wise replay of the raw-event HDF5 recording contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import h5py
import numpy as np

from .evt1_protocol import EventPacket


EVENT_DATASETS = {
    "type": np.dtype("uint8"),
    "x": np.dtype("uint16"),
    "y": np.dtype("uint16"),
    "t_us": np.dtype("int64"),
    "packet_id": np.dtype("int64"),
}
PACKET_DATASETS = {
    "ros_t_ns": np.dtype("int64"),
    "monotonic_t_ns": np.dtype("int64"),
    "start_event_idx": np.dtype("int64"),
    "end_event_idx": np.dtype("int64"),
    "event_count": np.dtype("int64"),
}


class ReplaySchemaError(ValueError):
    """The input is not an OpenMV raw-event recording."""


@dataclass
class ReplayDiagnostics:
    replay_packets_read: int = 0
    replay_packets_processed: int = 0
    replay_events_processed: int = 0
    replay_packets_skipped: int = 0
    replay_loops_completed: int = 0
    replay_timing_fallbacks: int = 0


def scaled_packet_delay_seconds(previous: int, current: int, rate: float,
                                *, units_per_second: float) -> float:
    """Return the scaled inter-packet delay or reject invalid timing."""
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("event_replay_rate must be positive and finite")
    previous = int(previous)
    current = int(current)
    if previous <= 0 or current <= 0 or current < previous:
        raise ValueError("packet timing timestamps are missing or non-monotonic")
    return (current - previous) / float(units_per_second) / float(rate)


class HDF5ReplayReader:
    """Validate and stream one recorded packet at a time."""

    def __init__(self, path: str, *, start_packet: int = 0,
                 end_packet: int = -1):
        replay_path = Path(path).expanduser()
        if not replay_path.is_file():
            raise FileNotFoundError(f"HDF5 replay file does not exist: {replay_path}")
        self.path = replay_path.resolve()
        self._file = h5py.File(str(self.path), "r")
        try:
            self.packet_count = self._validate_schema()
            self.start_packet, self.end_packet = self._validate_range(
                start_packet, end_packet)
        except Exception:
            self._file.close()
            raise

    def _validate_schema(self) -> int:
        for group_name, expected in (("events", EVENT_DATASETS),
                                     ("packets", PACKET_DATASETS)):
            if group_name not in self._file or not isinstance(
                    self._file[group_name], h5py.Group):
                raise ReplaySchemaError(f"missing HDF5 group /{group_name}")
            group = self._file[group_name]
            for name, dtype in expected.items():
                if name not in group or not isinstance(group[name], h5py.Dataset):
                    raise ReplaySchemaError(
                        f"missing HDF5 dataset /{group_name}/{name}")
                dataset = group[name]
                if dataset.ndim != 1:
                    raise ReplaySchemaError(
                        f"/{group_name}/{name} must be one-dimensional")
                if dataset.dtype != dtype:
                    raise ReplaySchemaError(
                        f"/{group_name}/{name} has dtype {dataset.dtype}, expected {dtype}")

        event_lengths = {len(self._file["events"][name]) for name in EVENT_DATASETS}
        packet_lengths = {len(self._file["packets"][name]) for name in PACKET_DATASETS}
        if len(event_lengths) != 1:
            raise ReplaySchemaError("event datasets have different lengths")
        if len(packet_lengths) != 1:
            raise ReplaySchemaError("packet datasets have different lengths")
        event_count = event_lengths.pop()
        packet_count = packet_lengths.pop()
        previous_end = 0
        packets = self._file["packets"]
        events_packet_id = self._file["events/packet_id"]
        for packet_idx in range(packet_count):
            start = int(packets["start_event_idx"][packet_idx])
            end = int(packets["end_event_idx"][packet_idx])
            count = int(packets["event_count"][packet_idx])
            if start != previous_end or end < start or end > event_count:
                raise ReplaySchemaError(
                    f"invalid event boundary for packet {packet_idx}: [{start}, {end})")
            if end - start != count:
                raise ReplaySchemaError(
                    f"packet {packet_idx} event_count={count}, boundary size={end - start}")
            if count and not np.all(events_packet_id[start:end] == packet_idx):
                raise ReplaySchemaError(
                    f"/events/packet_id disagrees with packet {packet_idx} boundaries")
            previous_end = end
        if previous_end != event_count:
            raise ReplaySchemaError("packet boundaries do not cover all recorded events")
        return packet_count

    def _validate_range(self, start: int, end: int):
        start = int(start)
        end = self.packet_count - 1 if int(end) == -1 else int(end)
        if start < 0 or start >= self.packet_count:
            raise ValueError(
                f"event_replay_start_packet {start} outside [0, {self.packet_count - 1}]")
        if end < start or end >= self.packet_count:
            raise ValueError(
                f"invalid replay packet range [{start}, {end}] for {self.packet_count} packets")
        return start, end

    def packet_timing_value(self, packet_idx: int, timing: str) -> int:
        if timing == "recorded":
            return int(self._file["packets/monotonic_t_ns"][packet_idx])
        if timing == "sensor":
            start = int(self._file["packets/start_event_idx"][packet_idx])
            end = int(self._file["packets/end_event_idx"][packet_idx])
            if start == end:
                return -1
            return int(np.min(self._file["events/t_us"][start:end]))
        raise ValueError(f"unsupported replay timing mode: {timing}")

    def read_packet(self, packet_idx: int, *, ros_now_ns: int,
                    monotonic_now_ns: int) -> EventPacket:
        packets = self._file["packets"]
        start = int(packets["start_event_idx"][packet_idx])
        end = int(packets["end_event_idx"][packet_idx])
        events = self._file["events"]
        return EventPacket.from_recorded_arrays(
            event_type=events["type"][start:end], event_x=events["x"][start:end],
            event_y=events["y"][start:end], timestamps_us=events["t_us"][start:end],
            packet_id=packet_idx, packet_ros_stamp_ns=ros_now_ns,
            packet_monotonic_stamp_ns=monotonic_now_ns,
            original_ros_stamp_ns=int(packets["ros_t_ns"][packet_idx]),
            original_monotonic_stamp_ns=int(packets["monotonic_t_ns"][packet_idx]))

    def indices(self) -> Iterator[int]:
        return iter(range(self.start_packet, self.end_packet + 1))

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def replay_packets(
    reader: HDF5ReplayReader, *, timing: str, rate: float, loop: bool,
    stop_event, process_packet: Callable[[EventPacket], None],
    ros_now_ns: Callable[[], int], monotonic_now_ns: Callable[[], int],
    diagnostics: Optional[ReplayDiagnostics] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> ReplayDiagnostics:
    """Run interruptible replay; ``stop_event.wait`` is the only sleep path."""
    if timing not in ("recorded", "sensor", "fast"):
        raise ValueError(f"invalid event_replay_timing: {timing!r}")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("event_replay_rate must be positive and finite")
    diagnostics = diagnostics or ReplayDiagnostics()
    diagnostics.replay_packets_skipped = (
        reader.packet_count - (reader.end_packet - reader.start_packet + 1))
    warn = warn or (lambda message: None)
    active_timing = timing
    units = 1_000_000_000.0 if timing == "recorded" else 1_000_000.0

    while not stop_event.is_set():
        previous_value = None
        deadline_ns = monotonic_now_ns()
        for packet_idx in reader.indices():
            if stop_event.is_set():
                return diagnostics
            diagnostics.replay_packets_read += 1
            if active_timing != "fast":
                value = reader.packet_timing_value(packet_idx, active_timing)
                if previous_value is not None:
                    try:
                        delay = scaled_packet_delay_seconds(
                            previous_value, value, rate, units_per_second=units)
                    except ValueError as error:
                        diagnostics.replay_timing_fallbacks += 1
                        warn(f"{active_timing} replay timing invalid at packet "
                             f"{packet_idx} ({error}); falling back to fast mode")
                        active_timing = "fast"
                    else:
                        deadline_ns += int(delay * 1_000_000_000)
                        remaining = (deadline_ns - monotonic_now_ns()) / 1e9
                        if remaining > 0.0 and stop_event.wait(remaining):
                            return diagnostics
                previous_value = value
            packet = reader.read_packet(
                packet_idx, ros_now_ns=ros_now_ns(),
                monotonic_now_ns=monotonic_now_ns())
            process_packet(packet)
            diagnostics.replay_packets_processed += 1
            diagnostics.replay_events_processed += packet.event_count
        diagnostics.replay_loops_completed += 1
        if not loop:
            return diagnostics
    return diagnostics
