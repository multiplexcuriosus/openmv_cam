#!/usr/bin/env python3

import struct
import serial
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class OpenMVCamNode(Node):
    def __init__(self):
        super().__init__('openmv_cam')

        # Parameters (so you don't hardcode /dev/openmvcam etc.)
        self.declare_parameter('port', '/dev/openmvcam')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('topic', '/openmv_cam/image/raw')
        self.declare_parameter('frame_id', 'openmv_cam')

        self.port_name = self.get_parameter('port').get_parameter_value().string_value
        self.baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.pub_raw = self.create_publisher(Image, self.topic, 10)

        self.serial_port = None
        self._open_serial()

        period = 1.0 / self.fps if self.fps > 0 else 0.1
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"OpenMV serial on {self.port_name} @ {self.baud}, publishing {self.topic} at {self.fps} Hz"
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
                timeout=None,
                dsrdtr=True,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.port_name}: {e}")
            raise

    def _read_exactly(self, n: int) -> bytes:
        """Read exactly n bytes or raise."""
        data = b''
        while len(data) < n:
            chunk = self.serial_port.read(n - len(data))
            if not chunk:
                raise RuntimeError("Serial read returned empty bytes.")
            data += chunk
        return data

    def read_image(self):
        # request
        self.serial_port.write(b"snap")
        self.serial_port.flush()

        # read size (uint32 little-endian)
        size_bytes = self._read_exactly(4)
        size = struct.unpack('<L', size_bytes)[0]

        # read payload
        buf = self._read_exactly(size)

        # decode jpeg bytes
        x = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode returned None (corrupt frame?)")
        return img

    def publish_image(self, img):
        # Convert to ROS Image
        if img.ndim == 3:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        else:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_raw.publish(msg)

    def _tick(self):
        try:
            img = self.read_image()
            self.publish_image(img)
        except Exception as e:
            # Don’t crash the node; log and keep trying
            self.get_logger().warn(f"Frame read/publish failed: {e}")


def main():
    rclpy.init()
    node = OpenMVCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial_port is not None:
            try:
                node.serial_port.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()