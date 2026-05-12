#!/usr/bin/env python3

import struct
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import h5py
import numpy as np
import serial

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class OpenMVEventCamNode(Node):
    MAGIC = b"EVT1"
    HEADER_FMT = "<LL"   # event_count, payload_len
    HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

    print_log = False

    W = 320
    H = 320

    def __init__(self):
        super().__init__("openmv_event_cam")

        # Serial / ROS params
        self.declare_parameter("port", "/dev/openmvcam")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("timeout", 3.0)
        self.declare_parameter("topic", "/openmv_cam/image")
        self.declare_parameter("frame_id", "openmv_cam")
        self.declare_parameter("publish_fps", 30.0)

        # Preview/render params
        self.declare_parameter("window_ms", 100.0)
        self.declare_parameter("max_preview_packets", 10)
        self.declare_parameter("contrast", 4.0)
        self.declare_parameter("step", 1.0)
        self.declare_parameter("blur_kernel", 2)
        self.declare_parameter("sort_by_timestamp", False)

        # Raw event recording params
        self.declare_parameter("raw_event_output_path", "")
        self.declare_parameter("flush_every_packets", 50)
        self.declare_parameter("hdf5_chunk_size", 100000)
        self.declare_parameter("hdf5_compression", "gzip")
        self.declare_parameter("hdf5_compression_level", 4)

        self.port_name = self.get_parameter("port").get_parameter_value().string_value
        self.baud = self.get_parameter("baud").get_parameter_value().integer_value
        self.timeout = self.get_parameter("timeout").get_parameter_value().double_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.publish_fps = self.get_parameter("publish_fps").get_parameter_value().double_value

        self.window_ms = self.get_parameter("window_ms").get_parameter_value().double_value
        self.max_preview_packets = self.get_parameter("max_preview_packets").get_parameter_value().integer_value
        self.contrast = self.get_parameter("contrast").get_parameter_value().double_value
        self.step = self.get_parameter("step").get_parameter_value().double_value
        self.blur_kernel = self.get_parameter("blur_kernel").get_parameter_value().integer_value
        self.sort_by_timestamp = self.get_parameter("sort_by_timestamp").get_parameter_value().bool_value

        self.raw_event_output_path = self.get_parameter("raw_event_output_path").get_parameter_value().string_value
        self.flush_every_packets = self.get_parameter("flush_every_packets").get_parameter_value().integer_value
        self.hdf5_chunk_size = self.get_parameter("hdf5_chunk_size").get_parameter_value().integer_value
        self.hdf5_compression = self.get_parameter("hdf5_compression").get_parameter_value().string_value
        self.hdf5_compression_level = self.get_parameter(
            "hdf5_compression_level"
        ).get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.pub_raw = None
        self.publish_timer = None
        self._publishing_enabled = False

        self.serial_port = None
        self._open_serial()

        self._stop_event = threading.Event()
        self._preview_lock = threading.Lock()
        self._h5_lock = threading.Lock()

        # stores tuples: (host_arrival_time_monotonic, events_ndarray)
        self.preview_buffer = deque()

        self._recording_enabled = False
        self._h5_file = None
        self._h5_file_path = ""
        self._h5_events_type = None
        self._h5_events_x = None
        self._h5_events_y = None
        self._h5_events_t_us = None
        self._h5_events_packet_id = None
        self._h5_packets_ros_t_ns = None
        self._h5_packets_monotonic_t_ns = None
        self._h5_packets_start_event_idx = None
        self._h5_packets_end_event_idx = None
        self._h5_packets_event_count = None
        self._h5_packets_first_event_t_us = None
        self._h5_packets_last_event_t_us = None
        self._h5_event_count = 0
        self._h5_packet_count = 0
        self._h5_packets_since_flush = 0

        self.total_packets = 0
        self.total_events = 0
        self.total_payload_bytes = 0
        self.total_protocol_bytes = 0
        self.t0 = time.monotonic()
        self.last_stats_print = self.t0

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._start_pub_srv = self.create_service(
            Trigger,
            "/openmv_cam/start_event_frame_publishing",
            self._handle_start_event_frame_publishing,
        )
        self._stop_pub_srv = self.create_service(
            Trigger,
            "/openmv_cam/stop_event_frame_publishing",
            self._handle_stop_event_frame_publishing,
        )
        self._start_rec_srv = self.create_service(
            Trigger,
            "/openmv_cam/start_raw_event_recording",
            self._handle_start_raw_event_recording,
        )
        self._stop_rec_srv = self.create_service(
            Trigger,
            "/openmv_cam/stop_raw_event_recording",
            self._handle_stop_raw_event_recording,
        )

        self.get_logger().info(f"Serial port opened: {self.port_name} @ {self.baud}")
        self.get_logger().info("Event frame publishing initially disabled")
        self.get_logger().info("Raw event recording initially disabled")
        self.get_logger().info(
            "Available services: "
            "/openmv_cam/start_event_frame_publishing, "
            "/openmv_cam/stop_event_frame_publishing, "
            "/openmv_cam/start_raw_event_recording, "
            "/openmv_cam/stop_raw_event_recording"
        )

    def _open_serial(self):
        try:
            self.serial_port = serial.Serial(
                self.port_name,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=self.timeout,
            )
            self.serial_port.reset_input_buffer()
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.port_name}: {e}")
            raise

    def _normalize_hdf5_compression(self):
        compression = (self.hdf5_compression or "").strip().lower()
        if compression in ("", "none"):
            return None, None
        if compression == "gzip":
            return "gzip", int(self.hdf5_compression_level)
        if compression == "lzf":
            return "lzf", None
        self.get_logger().warn(
            f"Unknown hdf5_compression '{self.hdf5_compression}', using 'gzip'."
        )
        return "gzip", int(self.hdf5_compression_level)

    def _resolve_output_h5_path(self) -> Path:
        configured = (
            self.get_parameter("raw_event_output_path")
            .get_parameter_value()
            .string_value
            .strip()
        )

        if configured:
            return Path(configured).expanduser().resolve()

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return (Path.cwd() / f"openmv_raw_events_{ts}.h5").resolve()

    def _create_appendable_1d_dataset(self, group, name, dtype):
        compression, compression_level = self._normalize_hdf5_compression()
        kwargs = {
            "shape": (0,),
            "maxshape": (None,),
            "dtype": dtype,
            "chunks": (max(1, int(self.hdf5_chunk_size)),),
        }
        if compression is not None:
            kwargs["compression"] = compression
        if compression_level is not None:
            kwargs["compression_opts"] = compression_level
        return group.create_dataset(name, **kwargs)

    def _open_h5_for_recording_locked(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        h5f = h5py.File(str(path), mode="x")

        events_group = h5f.create_group("events")
        packets_group = h5f.create_group("packets")

        self._h5_events_type = self._create_appendable_1d_dataset(events_group, "type", np.uint8)
        self._h5_events_x = self._create_appendable_1d_dataset(events_group, "x", np.uint16)
        self._h5_events_y = self._create_appendable_1d_dataset(events_group, "y", np.uint16)
        self._h5_events_t_us = self._create_appendable_1d_dataset(events_group, "t_us", np.int64)
        self._h5_events_packet_id = self._create_appendable_1d_dataset(
            events_group, "packet_id", np.int64
        )

        self._h5_packets_ros_t_ns = self._create_appendable_1d_dataset(
            packets_group, "ros_t_ns", np.int64
        )
        self._h5_packets_monotonic_t_ns = self._create_appendable_1d_dataset(
            packets_group, "monotonic_t_ns", np.int64
        )
        self._h5_packets_start_event_idx = self._create_appendable_1d_dataset(
            packets_group, "start_event_idx", np.int64
        )
        self._h5_packets_end_event_idx = self._create_appendable_1d_dataset(
            packets_group, "end_event_idx", np.int64
        )
        self._h5_packets_event_count = self._create_appendable_1d_dataset(
            packets_group, "event_count", np.int64
        )
        self._h5_packets_first_event_t_us = self._create_appendable_1d_dataset(
            packets_group, "first_event_t_us", np.int64
        )
        self._h5_packets_last_event_t_us = self._create_appendable_1d_dataset(
            packets_group, "last_event_t_us", np.int64
        )

        h5f.attrs["source"] = "openmv_event_cam"
        h5f.attrs["width"] = int(self.W)
        h5f.attrs["height"] = int(self.H)
        h5f.attrs["event_column_order"] = "type,sec,ms,us,x,y"
        h5f.attrs["created_wall_time"] = float(time.time())
        h5f.attrs[
            "timestamp_note"
        ] = (
            "event t_us is reconstructed from OpenMV sec/ms/us fields; "
            "packet ros_t_ns is host ROS receive time for the packet."
        )
        h5f.attrs[
            "packet_first_last_timestamp_note"
        ] = (
            "first_event_t_us and last_event_t_us are reconstructed from the "
            "first/last event in each received packet using OpenMV sec/ms/us fields."
        )

        self._h5_file = h5f
        self._h5_file_path = str(path)
        self._h5_event_count = 0
        self._h5_packet_count = 0
        self._h5_packets_since_flush = 0

    def _close_h5_locked(self):
        closed_path = self._h5_file_path
        try:
            if self._h5_file is not None:
                self._h5_file.flush()
                self._h5_file.close()
        finally:
            self._h5_file = None
            self._h5_file_path = ""
            self._h5_events_type = None
            self._h5_events_x = None
            self._h5_events_y = None
            self._h5_events_t_us = None
            self._h5_events_packet_id = None
            self._h5_packets_ros_t_ns = None
            self._h5_packets_monotonic_t_ns = None
            self._h5_packets_start_event_idx = None
            self._h5_packets_end_event_idx = None
            self._h5_packets_event_count = None
            self._h5_packets_first_event_t_us = None
            self._h5_packets_last_event_t_us = None
            self._h5_event_count = 0
            self._h5_packet_count = 0
            self._h5_packets_since_flush = 0
            self._recording_enabled = False
        return closed_path

    def _append_packet_to_h5(
        self,
        events: np.ndarray,
        packet_id: int,  # no longer used; can remove later
        packet_ros_t_ns: int,
        packet_mono_t_ns: int,
    ):
        event_count = int(events.shape[0])
        event_t_us = self.event_timestamps_us(events) if event_count > 0 else None

        if event_count > 0:
            packet_first_event_t_us = int(event_t_us[0])
            packet_last_event_t_us = int(event_t_us[-1])
        else:
            packet_first_event_t_us = -1
            packet_last_event_t_us = -1

        if event_count > 0:
            event_type = events[:, 0].astype(np.uint8, copy=False)
            event_x = events[:, 4].astype(np.uint16, copy=False)
            event_y = events[:, 5].astype(np.uint16, copy=False)
        else:
            event_type = None
            event_x = None
            event_y = None
            event_t_us = None

        with self._h5_lock:
            if not self._recording_enabled or self._h5_file is None:
                return

            # Local packet index inside this HDF5 recording.
            packet_idx = self._h5_packet_count

            if event_count > 0:
                event_packet_id = np.full((event_count,), packet_idx, dtype=np.int64)
            else:
                event_packet_id = None

            old_n = self._h5_event_count
            new_n = old_n + event_count


            self._h5_events_type.resize((new_n,))
            self._h5_events_x.resize((new_n,))
            self._h5_events_y.resize((new_n,))
            self._h5_events_t_us.resize((new_n,))
            self._h5_events_packet_id.resize((new_n,))

            if event_count > 0:
                self._h5_events_type[old_n:new_n] = event_type
                self._h5_events_x[old_n:new_n] = event_x
                self._h5_events_y[old_n:new_n] = event_y
                self._h5_events_t_us[old_n:new_n] = event_t_us
                self._h5_events_packet_id[old_n:new_n] = event_packet_id

            packet_idx = self._h5_packet_count
            packet_new_idx = packet_idx + 1
            self._h5_packets_ros_t_ns.resize((packet_new_idx,))
            self._h5_packets_monotonic_t_ns.resize((packet_new_idx,))
            self._h5_packets_start_event_idx.resize((packet_new_idx,))
            self._h5_packets_end_event_idx.resize((packet_new_idx,))
            self._h5_packets_event_count.resize((packet_new_idx,))
            self._h5_packets_first_event_t_us.resize((packet_new_idx,))
            self._h5_packets_last_event_t_us.resize((packet_new_idx,))

            self._h5_packets_ros_t_ns[packet_idx] = np.int64(packet_ros_t_ns)
            self._h5_packets_monotonic_t_ns[packet_idx] = np.int64(packet_mono_t_ns)
            self._h5_packets_start_event_idx[packet_idx] = np.int64(old_n)
            self._h5_packets_end_event_idx[packet_idx] = np.int64(new_n)
            self._h5_packets_event_count[packet_idx] = np.int64(event_count)
            self._h5_packets_first_event_t_us[packet_idx] = np.int64(packet_first_event_t_us)
            self._h5_packets_last_event_t_us[packet_idx] = np.int64(packet_last_event_t_us)

            self._h5_event_count = new_n
            self._h5_packet_count = packet_new_idx
            self._h5_packets_since_flush += 1

            if self._h5_packets_since_flush >= max(1, int(self.flush_every_packets)):
                self._h5_file.flush()
                self._h5_packets_since_flush = 0

    def _ensure_publisher_and_timer(self):
        if self.pub_raw is None:
            self.pub_raw = self.create_publisher(Image, self.topic, 10)
        if self.publish_timer is None:
            self.publish_timer = self.create_timer(1.0 / self.publish_fps, self._publish_timer_cb)

    def _stop_publishing_internal(self):
        self._publishing_enabled = False

        if self.publish_timer is not None:
            self.publish_timer.cancel()
            self.destroy_timer(self.publish_timer)
            self.publish_timer = None

        with self._preview_lock:
            self.preview_buffer.clear()

    def _handle_start_event_frame_publishing(self, request, response):
        del request
        if self._publishing_enabled:
            response.success = False
            response.message = ""
            return response

        self._ensure_publisher_and_timer()
        self._publishing_enabled = True
        self.get_logger().info(
            f"Event frame publishing started on {self.topic} at {self.publish_fps:.1f} Hz"
        )
        response.success = True
        response.message = ""
        return response

    def _handle_stop_event_frame_publishing(self, request, response):
        del request
        if not self._publishing_enabled:
            response.success = False
            response.message = ""
            return response

        self._stop_publishing_internal()
        self.get_logger().info("Event frame publishing stopped")
        response.success = True
        response.message = ""
        return response     

    def _handle_start_raw_event_recording(self, request, response):
        del request
        with self._h5_lock:
            if self._recording_enabled:
                response.success = False
                response.message = ""
                return response

            target_path = self._resolve_output_h5_path()
            if target_path.exists():
                response.success = False
                response.message = ""   

                return response

            try:
                self._open_h5_for_recording_locked(target_path)
                self._recording_enabled = True
            except Exception as e:
                if self._h5_file is not None:
                    try:
                        self._h5_file.close()
                    except Exception:
                        pass
                self._h5_file = None
                self._recording_enabled = False
                response.success = False
                response.message = ""
                return response

        self.get_logger().info(f"Raw event recording started: {target_path}")
        response.success = True
        response.message = ("")
        return response

    def _handle_stop_raw_event_recording(self, request, response):
        del request
        with self._h5_lock:
            if not self._recording_enabled:
                response.success = False
                response.message = ""
                return response

            try:
                closed_path = self._close_h5_locked()
            except Exception as e:
                self.get_logger().warn(f"Failed to flush and close HDF5: {e}")
                response.success = False
                response.message = ""
                return response

        self.get_logger().info(f"Raw event recording stopped: {closed_path}")
        response.success = True
        response.message = ""
        return response

    def _read_exactly(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            if self._stop_event.is_set():
                raise RuntimeError("Stop requested.")
            chunk = self.serial_port.read(n - len(data))
            if not chunk:
                raise RuntimeError("Serial read timeout.")
            data.extend(chunk)
        return bytes(data)

    def _read_until_magic(self):
        window = bytearray()
        while not self._stop_event.is_set():
            b = self.serial_port.read(1)
            if not b:
                raise RuntimeError("Timeout while waiting for magic.")
            window += b
            if len(window) > len(self.MAGIC):
                window = window[-len(self.MAGIC):]
            if bytes(window) == self.MAGIC:
                return
        raise RuntimeError("Stop requested.")

    @staticmethod
    def event_timestamps_us(events: np.ndarray) -> np.ndarray:
        """
        Reconstruct absolute timestamps in microseconds from columns:
          1: sec
          2: ms
          3: us
        """
        return (
            events[:, 1].astype(np.int64) * 1_000_000
            + events[:, 2].astype(np.int64) * 1_000
            + events[:, 3].astype(np.int64)
        )

    def sort_events_by_timestamp_fn(self, events: np.ndarray) -> np.ndarray:
        if events.size == 0:
            return events
        ts = self.event_timestamps_us(events)
        order = np.argsort(ts, kind="stable")
        return events[order]

    @staticmethod
    def events_to_preview_frame(
        events: np.ndarray,
        width: int,
        height: int,
        contrast: float = 4.0,
        step: float = 1.0,
    ) -> np.ndarray:
        frame = np.full((height, width), 128.0, dtype=np.float32)

        if events.size == 0:
            return frame.astype(np.uint8)

        xs = events[:, 4].astype(np.int32)
        ys = events[:, 5].astype(np.int32)
        tp = events[:, 0].astype(np.int32)

        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = xs[valid]
        ys = ys[valid]
        tp = tp[valid]

        if xs.size == 0:
            return frame.astype(np.uint8)

        pos = (tp == 1)
        neg = ~pos

        acc = np.zeros((height, width), dtype=np.float32)
        np.add.at(acc, (ys[pos], xs[pos]), +step)
        np.add.at(acc, (ys[neg], xs[neg]), -step)

        m = np.max(np.abs(acc))
        if m > 0:
            acc /= m

        frame = 128.0 + acc * (contrast * 127.0)
        np.clip(frame, 0, 255, out=frame)
        return frame.astype(np.uint8)

    def _trim_preview_buffer_locked(self, now: float):
        cutoff = now - (self.window_ms / 1000.0)

        while self.preview_buffer and self.preview_buffer[0][0] < cutoff:
            self.preview_buffer.popleft()

        while len(self.preview_buffer) > self.max_preview_packets:
            self.preview_buffer.popleft()

    def _publish_image(self, img: np.ndarray):
        if self.pub_raw is None:
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_raw.publish(msg)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                self._read_until_magic()
                header_rest = self._read_exactly(struct.calcsize(self.HEADER_FMT))
                event_count, payload_len = struct.unpack(self.HEADER_FMT, header_rest)

                expected_len = event_count * 6 * 2
                if payload_len != expected_len:
                    raise RuntimeError(
                        f"Invalid payload length: got {payload_len}, expected {expected_len}"
                    )

                payload = self._read_exactly(payload_len)
                events = np.frombuffer(payload, dtype=np.uint16).reshape((event_count, 6)).copy()

                packet_ros_t_ns = int(self.get_clock().now().nanoseconds)
                packet_mono_t_ns = int(time.monotonic_ns())

                now = time.monotonic()
                packet_id = int(self.total_packets)
                self.total_packets += 1
                self.total_events += event_count
                self.total_payload_bytes += payload_len
                self.total_protocol_bytes += payload_len + self.HEADER_SIZE

                if self.total_packets <= 3 and event_count > 0:
                    self.get_logger().info(
                        f"packet {self.total_packets}: "
                        f"type={np.unique(events[:, 0])[:10]}, "
                        f"x=[{int(events[:, 4].min())},{int(events[:, 4].max())}], "
                        f"y=[{int(events[:, 5].min())},{int(events[:, 5].max())}]"
                    )



                if self._recording_enabled:
                    self._append_packet_to_h5(
                        events,
                        packet_id=packet_id,
                        packet_ros_t_ns=packet_ros_t_ns,
                        packet_mono_t_ns=packet_mono_t_ns,
                    )

                if self._publishing_enabled:
                    with self._preview_lock:
                        self.preview_buffer.append((now, events))
                        self._trim_preview_buffer_locked(now)

                if now - self.last_stats_print >= 2.0:
                    elapsed = now - self.t0
                    packets_per_s = self.total_packets / elapsed if elapsed > 0 else 0.0
                    events_per_s = self.total_events / elapsed if elapsed > 0 else 0.0
                    payload_MBps = self.total_payload_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
                    protocol_MBps = self.total_protocol_bytes / elapsed / 1e6 if elapsed > 0 else 0.0

                    if self.print_log:
                        self.get_logger().info(
                            "EVENT STREAM STATS | "
                            f"packets={self.total_packets}, "
                            f"events={self.total_events}, "
                            f"elapsed={elapsed:.2f}s, "
                            f"packets/s={packets_per_s:.1f}, "
                            f"events/s={events_per_s:.1f}, "
                            f"payload_MBps={payload_MBps:.3f}, "
                            f"protocol_MBps={protocol_MBps:.3f}"
                        )
                    self.last_stats_print = now

            except RuntimeError as e:
                if "Stop requested" in str(e):
                    break
                self.get_logger().warn(f"Event read failed: {e}")
                time.sleep(0.05)

            except Exception as e:
                self.get_logger().warn(f"Unexpected error in reader loop: {e}")
                time.sleep(0.05)

    def _publish_timer_cb(self):
        if not self._publishing_enabled:
            return

        with self._preview_lock:
            now = time.monotonic()
            self._trim_preview_buffer_locked(now)

            if not self.preview_buffer:
                frame = np.full((self.H, self.W), 128, dtype=np.uint8)
            else:
                chunks = [ev for (_, ev) in self.preview_buffer]
                chunk = np.concatenate(chunks, axis=0)

                if self.sort_by_timestamp:
                    chunk = self.sort_events_by_timestamp_fn(chunk)

                frame = self.events_to_preview_frame(
                    chunk,
                    self.W,
                    self.H,
                    contrast=self.contrast,
                    step=self.step,
                )

        if self.blur_kernel and self.blur_kernel > 1:
            frame = cv2.blur(frame, (self.blur_kernel, self.blur_kernel))

        self._publish_image(frame)

    def destroy_node(self):
        self._stop_event.set()

        if self.publish_timer is not None:
            self.publish_timer.cancel()
            self.destroy_timer(self.publish_timer)
            self.publish_timer = None

        with self._preview_lock:
            self.preview_buffer.clear()

        if hasattr(self, "_reader_thread") and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        with self._h5_lock:
            if self._recording_enabled:
                try:
                    closed_path = self._close_h5_locked()
                    self.get_logger().info(f"Closed raw event recording on shutdown: {closed_path}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to close HDF5 on shutdown: {e}")

        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass

        self.get_logger().info("Serial port closed")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenMVEventCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()