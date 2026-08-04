"""Right-angle rotation helpers for published event-frame images."""

import numpy as np


SUPPORTED_ROTATIONS_DEG = (-90, 0, 90, 180)


def rotate_event_frame(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Rotate an image counterclockwise by a supported absolute angle."""
    if rotation_degrees not in SUPPORTED_ROTATIONS_DEG:
        raise ValueError(
            f"rotation_degrees must be one of {SUPPORTED_ROTATIONS_DEG}, "
            f"got {rotation_degrees}"
        )

    quarter_turns = rotation_degrees // 90
    if quarter_turns == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=quarter_turns, axes=(0, 1)))
