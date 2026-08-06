from builtin_interfaces.msg import Time
import numpy as np

from openmv_cam.openmv_cam_node import build_activity_image_message


def test_single_bin_activity_message_layout():
    image = np.zeros((320, 320), dtype=np.uint8)
    stamp = Time(sec=12, nanosec=34)
    msg = build_activity_image_message(image, stamp=stamp)

    assert msg.header.stamp == stamp
    assert msg.header.frame_id == "openmv_cam"
    assert (msg.height, msg.width) == (320, 320)
    assert msg.encoding == "8UC1"
    assert msg.step == 320
    assert len(msg.data) == 102400


def test_multi_bin_activity_message_layout_and_order():
    image = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    msg = build_activity_image_message(image, stamp=Time())

    assert (msg.height, msg.width) == (2, 3)
    assert msg.encoding == "8UC4"
    assert msg.step == 12
    assert bytes(msg.data) == image.tobytes(order="C")
