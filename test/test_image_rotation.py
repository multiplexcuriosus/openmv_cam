import numpy as np
import pytest

from openmv_cam.image_rotation import rotate_event_frame


@pytest.fixture
def image():
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)


def test_zero_degrees_leaves_image_unchanged(image):
    assert rotate_event_frame(image, 0) is image


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        (90, [[3, 6], [2, 5], [1, 4]]),
        (180, [[6, 5, 4], [3, 2, 1]]),
        (-90, [[4, 1], [5, 2], [6, 3]]),
    ],
)
def test_supported_right_angle_rotations(image, degrees, expected):
    np.testing.assert_array_equal(
        rotate_event_frame(image, degrees),
        np.asarray(expected, dtype=np.uint8),
    )


def test_rotation_preserves_trailing_channels():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated = rotate_event_frame(image, 90)

    assert rotated.shape == (3, 2, 3)
    np.testing.assert_array_equal(rotated[:, :, 0], np.rot90(image[:, :, 0]))


def test_rotation_preserves_hwc9_temporal_channel_order():
    image = np.arange(2 * 3 * 9, dtype=np.uint8).reshape(2, 3, 9)
    rotated = rotate_event_frame(image, -90)

    assert rotated.shape == (3, 2, 9)
    assert rotated.flags.c_contiguous
    for channel in range(9):
        np.testing.assert_array_equal(
            rotated[:, :, channel],
            np.rot90(image[:, :, channel], k=-1),
        )


def test_unsupported_rotation_is_rejected(image):
    with pytest.raises(ValueError, match="must be one of"):
        rotate_event_frame(image, 45)
