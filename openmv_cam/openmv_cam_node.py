#!/usr/bin/env python3

import struct
import threading
import time
import serial
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class OpenMVCamNode(Node):
    MAGIC = b'OMV1'
    HEADER_FMT = '<LLLL'   # width, height, payload_len, format
    HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)
    STATS_PRINT_PERIOD_S = 2.0

    def __init__(self):
        super().__init__('openmv_cam')

        self.declare_parameter('port', '/dev/openmvcam')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('topic', '/openmv_cam/image')
        self.declare_parameter('frame_id', 'openmv_cam')

        self.port_name = self.get_parameter('port').get_parameter_value().string_value
        self.baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.pub_raw = self.create_publisher(Image, self.topic, 10)

        self.serial_port = None
        self._open_serial()

        self._stop_event = threading.Event()
        self._stream_active = False
        self._stream_start_time = None
        self._reset_stream_stats()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self.get_logger().info(
            f"OpenMV serial on {self.port_name} @ {self.baud}, publishing {self.topic}"
        )

    def _reset_stream_stats(self):
        self._frames_received = 0
        self._payload_bytes_total = 0
        self._protocol_bytes_total = 0
        self._stats_start_time = None
        self._last_stats_print_time = None

    def _open_serial(self):
        try:
            self.serial_port = serial.Serial(
                self.port_name,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                xonxoff=False,
                rtscts=False,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
                dsrdtr=False,   # keep this simple unless you know you need it
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
        """
        Consume bytes until MAGIC is found.
        """
        window = bytearray()
        while not self._stop_event.is_set():
            b = self.serial_port.read(1)
            if not b:
                raise RuntimeError("Serial read timeout while searching for frame magic.")
            window += b
            if len(window) > len(self.MAGIC):
                window = window[-len(self.MAGIC):]
            if bytes(window) == self.MAGIC:
                return
        raise RuntimeError("Stop requested.")

    def unpack_4bit_to_mono8(self,buf, width, height):

        packed = np.frombuffer(buf, dtype=np.uint8)

        hi = packed >> 4
        lo = packed & 0x0F

        out = np.empty(packed.size * 2, dtype=np.uint8)
        out[0::2] = hi * 17
        out[1::2] = lo * 17

        return out.reshape((height, width))

    def _read_frame(self) -> np.ndarray:
        self._read_until_magic()

        header_rest = self._read_exactly(struct.calcsize(self.HEADER_FMT))
        width, height, payload_len, fmt = struct.unpack(self.HEADER_FMT, header_rest)

        buf = self._read_exactly(payload_len)

        if fmt == 0:  # mono8
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width))

        elif fmt == 1:  # 4-bit packed
            img = self.unpack_4bit_to_mono8(buf, width, height)

        else:
            raise RuntimeError(f"Unknown format: {fmt}")

        return img

    def _publish_image(self, img: np.ndarray):
        if img.ndim == 3:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        else:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_raw.publish(msg)

    def _maybe_print_stats(self):
        if self._stats_start_time is None or self._last_stats_print_time is None:
            return

        now = time.monotonic()
        if (now - self._last_stats_print_time) < self.STATS_PRINT_PERIOD_S:
            return

        elapsed_time = now - self._stats_start_time
        if elapsed_time <= 0.0:
            return

        payload_MBps = (self._payload_bytes_total / elapsed_time) / (1024.0 * 1024.0)
        protocol_MBps = (self._protocol_bytes_total / elapsed_time) / (1024.0 * 1024.0)
        effective_fps = self._frames_received / elapsed_time

        self.get_logger().info(
            "\n"
            "===== STREAM STATS =====\n"
            f"frames_received     : {self._frames_received}\n"
            f"elapsed_time        : {elapsed_time:.2f} s\n"
            f"effective_fps       : {effective_fps:.1f}\n"
            f"payload_MBps        : {payload_MBps:.3f}\n"
            f"protocol_MBps       : {protocol_MBps:.3f}\n"
            "========================"
        )
        self._last_stats_print_time = now

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                img = self._read_frame()
                self._publish_image(img)

                if not self._stream_active:
                    self._stream_active = True
                    self._stream_start_time = time.monotonic()
                    self._reset_stream_stats()
                    self._stats_start_time = self._stream_start_time
                    self._last_stats_print_time = self._stats_start_time
                    self.get_logger().info("Frame stream started")

                frame_payload_bytes = int(img.nbytes)
                self._frames_received += 1
                self._payload_bytes_total += frame_payload_bytes
                self._protocol_bytes_total += frame_payload_bytes + self.HEADER_SIZE
                self._maybe_print_stats()

            except RuntimeError as e:
                if "Stop requested" in str(e):
                    break

                if self._stream_active:
                    duration_s = time.monotonic() - self._stream_start_time
                    self.get_logger().info(
                        f"Frame stream stopped. Duration: {duration_s:.3f} s"
                    )
                    self._stream_active = False
                    self._stream_start_time = None
                    self._reset_stream_stats()

                self.get_logger().warn(f"Frame read failed: {e}")
                time.sleep(0.05)

            except Exception as e:
                self.get_logger().warn(f"Unexpected error in reader loop: {e}")
                time.sleep(0.05)

    def destroy_node(self):
        self._stop_event.set()

        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass

        super().destroy_node()


def main():
    rclpy.init()
    node = OpenMVCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()