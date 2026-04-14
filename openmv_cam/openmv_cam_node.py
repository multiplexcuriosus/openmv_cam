#!/usr/bin/env python3

import struct
import threading
import time
from collections import deque

import cv2
import numpy as np
import serial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class OpenMVEventCamNode(Node):
    MAGIC = b"EVT1"
    HEADER_FMT = "<LL"   # event_count, payload_len
    HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

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

        self.bridge = CvBridge()
        self.pub_raw = self.create_publisher(Image, self.topic, 10)

        self.serial_port = None
        self._open_serial()

        self._stop_event = threading.Event()
        self._buffer_lock = threading.Lock()

        # stores tuples: (host_arrival_time_monotonic, events_ndarray)
        self.preview_buffer = deque()

        self.total_packets = 0
        self.total_events = 0
        self.total_payload_bytes = 0
        self.total_protocol_bytes = 0
        self.t0 = time.monotonic()
        self.last_stats_print = self.t0

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self.publish_timer = self.create_timer(1.0 / self.publish_fps, self._publish_timer_cb)

        self.get_logger().info(
            f"OpenMV event serial on {self.port_name} @ {self.baud}, "
            f"publishing {self.topic} at {self.publish_fps:.1f} Hz"
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

                now = time.monotonic()

                self.total_packets += 1
                self.total_events += event_count
                self.total_payload_bytes += payload_len
                self.total_protocol_bytes += payload_len + self.HEADER_SIZE

                if self.total_packets <= 3:
                    self.get_logger().info(
                        f"packet {self.total_packets}: "
                        f"type={np.unique(events[:, 0])[:10]}, "
                        f"x=[{int(events[:, 4].min())},{int(events[:, 4].max())}], "
                        f"y=[{int(events[:, 5].min())},{int(events[:, 5].max())}]"
                    )

                with self._buffer_lock:
                    self.preview_buffer.append((now, events))
                    self._trim_preview_buffer_locked(now)

                if now - self.last_stats_print >= 2.0:
                    elapsed = now - self.t0
                    packets_per_s = self.total_packets / elapsed if elapsed > 0 else 0.0
                    events_per_s = self.total_events / elapsed if elapsed > 0 else 0.0
                    payload_MBps = self.total_payload_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
                    protocol_MBps = self.total_protocol_bytes / elapsed / 1e6 if elapsed > 0 else 0.0

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
        with self._buffer_lock:
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

        if hasattr(self, "_reader_thread") and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass

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