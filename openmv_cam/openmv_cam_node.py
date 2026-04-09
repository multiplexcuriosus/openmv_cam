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
    HEADER_FMT = '<LLL'   # width, height, payload_len
    HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

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

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self.get_logger().info(
            f"OpenMV serial on {self.port_name} @ {self.baud}, publishing {self.topic}"
        )

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

    def _read_frame(self) -> np.ndarray:
        self._read_until_magic()

        header_rest = self._read_exactly(struct.calcsize(self.HEADER_FMT))
        width, height, payload_len = struct.unpack(self.HEADER_FMT, header_rest)

        if width <= 0 or height <= 0 or width > 10000 or height > 10000:
            raise RuntimeError(f"Invalid image size: {width}x{height}")

        expected_len = width * height
        if payload_len != expected_len:
            raise RuntimeError(
                f"Invalid payload length: got {payload_len}, expected {expected_len}"
            )

        buf = self._read_exactly(payload_len)
        img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width))
        return img

    def _publish_image(self, img: np.ndarray):
        if img.ndim == 3:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        else:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_raw.publish(msg)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                img = self._read_frame()
                self._publish_image(img)

                if not self._stream_active:
                    self._stream_active = True
                    self._stream_start_time = time.monotonic()
                    self.get_logger().info("Frame stream started")

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