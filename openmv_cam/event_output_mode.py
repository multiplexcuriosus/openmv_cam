"""Backward-compatible event image output selection."""

from dataclasses import dataclass


SUPPORTED_EVENT_OUTPUT_MODES = (
    "legacy_flags",
    "event_voxel_1ms",
    "all",
    "none",
)


@dataclass(frozen=True)
class EventOutputs:
    """Resolved event image publisher enablement."""

    mono: bool
    event_frame_3ch: bool
    legacy_voxel: bool
    event_voxel_1ms: bool


def resolve_event_outputs(
    mode: str,
    *,
    publish_3_channel_img: bool,
    publish_xyt_voxel: bool,
    publish_event_voxel_1ms: bool,
) -> EventOutputs:
    """Resolve an output mode while preserving legacy flag precedence."""
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_EVENT_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported event_output_mode {mode!r}; expected one of "
            f"{SUPPORTED_EVENT_OUTPUT_MODES}"
        )
    if normalized == "legacy_flags":
        return EventOutputs(
            mono=True,
            event_frame_3ch=bool(publish_3_channel_img),
            legacy_voxel=bool(publish_xyt_voxel),
            event_voxel_1ms=bool(publish_event_voxel_1ms),
        )
    if normalized == "event_voxel_1ms":
        return EventOutputs(False, False, False, True)
    if normalized == "all":
        return EventOutputs(True, True, True, True)
    return EventOutputs(False, False, False, False)
