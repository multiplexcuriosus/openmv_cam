from builtin_interfaces.msg import Time
import numpy as np
import pytest

from openmv_cam.openmv_cam_node import build_hwc9_image_message


def test_hwc9_message_has_explicit_layout_and_bytes():
    image = np.arange(2 * 3 * 9, dtype=np.uint8).reshape(2, 3, 9)
    stamp = Time(sec=12, nanosec=34)

    msg = build_hwc9_image_message(
        image,
        encoding="8UC9",
        stamp=stamp,
        frame_id="openmv_cam",
    )

    assert msg.header.stamp == stamp
    assert msg.header.frame_id == "openmv_cam"
    assert msg.height == 2
    assert msg.width == 3
    assert msg.encoding == "8UC9"
    assert msg.is_bigendian == 0
    assert msg.step == 3 * 9
    assert len(msg.data) == 2 * 3 * 9
    assert bytes(msg.data) == image.tobytes(order="C")


@pytest.mark.parametrize(
    "image, message",
    [
        (np.zeros((2, 3, 9), dtype=np.float32), "dtype uint8"),
        (np.zeros((2, 3, 8), dtype=np.uint8), "shape"),
        (np.zeros((2, 3), dtype=np.uint8), "shape"),
        (np.zeros((2, 3, 9), dtype=np.uint8)[:, ::-1, :], "C-contiguous"),
    ],
)
def test_hwc9_message_rejects_invalid_arrays(image, message):
    with pytest.raises(ValueError, match=message):
        build_hwc9_image_message(
            image,
            encoding="8UC9",
            stamp=Time(),
            frame_id="openmv_cam",
        )


def test_hwc9_message_rejects_wrong_encoding():
    with pytest.raises(ValueError, match="encoding"):
        build_hwc9_image_message(
            np.zeros((2, 3, 9), dtype=np.uint8),
            encoding="mono8",
            stamp=Time(),
            frame_id="openmv_cam",
        )
