# OpenMV event camera ROS node

## Hardware wire modes

The hardware serial reader supports two custom USB VCP wire formats. The
default remains `processed_evt1` for compatibility:

```bash
ros2 launch openmv_cam openmv.launch.py event_wire_mode:=processed_evt1
ros2 launch openmv_cam openmv.launch.py event_wire_mode:=raw_evt20
```

`processed_evt1` requires the existing H7 script
`openmv_cam/helpers/main.py`. It sends `EVT1`, an `<LL>` event-count/payload
header, and decoded six-`uint16` rows.

`raw_evt20` requires `openmv_cam/helpers/main_raw_evt20.py` and OpenMV firmware
that provides `csi.IOCTL_GENX320_READ_EVENTS_RAW`. The script sends the chip's
native EVT2.0 bytes over `USB_VCP` using this custom framing:

```text
magic   b"EVR1"
header  struct.pack("<LL", sequence, payload_length)
payload native little-endian EVT2.0 words (payload_length % 4 == 0)
```

The sequence is a wrapping unsigned 32-bit packet counter. The ROS node warns
about gaps and decodes CD words while preserving TIME_HIGH state across EVR1
packets. TIME_HIGH, TRIGGER, and other control words are not recorded as CD
events.

This EVR1/USB_VCP stream is intentionally separate from the official OpenMV
Protocol V2 benchmark/streaming script. The ROS node uses `pyserial` and a
custom framing protocol; Protocol V2 cannot be selected here unless a complete
Protocol V2 bridge is implemented. The H7 script and `event_wire_mode` must
always match: `openmv_cam/helpers/main.py` for `processed_evt1`, and
`openmv_cam/helpers/main_raw_evt20.py` for `raw_evt20`.

## Offline sparse-tracking datasets (no ROS required)

`openmv_cam.offline_dataset` streams packet slices from a raw-event HDF5,
reconstructs the existing `EventPacket` representation, and invokes the
unchanged `EventBallTracker`. The two-stage design makes the tracker sidecar
reusable for more than one enrichment pass. Tracker state is reset at every
episode boundary, and packets are assigned from their ROS timestamps and the
episode intervals, never from file ordering. `--pre-roll-ms` optionally warms
up a new tracker; detections made before the episode start are discarded.

For a 100-episode recording, run:

```bash
python3 -m openmv_cam.offline_dataset build \
  --raw-events /data/run/run_raw_events.h5 \
  --episodes /data/run/hdf5_dataset \
  --output /data/run/hdf5_dataset_sparse \
  --tracker-config config/offline_tracker_example.json \
  --tracker-output /data/run/tracker_outputs.h5 \
  --overwrite
```

The raw input schema is `/events/{type,x,y,t_us}` and
`/packets/{ros_t_ns,start_event_idx,end_event_idx}`, with optional
`monotonic_t_ns` and `packet_id`. Packet event ranges are half-open. Sensor
dimensions come from `width`/`height` root or `/events` attributes and default
to 320x320. Only one event packet is read at a time.

The sidecar stores metadata in `/metadata` attributes and one extendable row
per tracker update beneath `/episodes/episode_N`. In particular,
`available_ros_t_ns` is packet ROS time, while `sensor_window_start_us` and
`sensor_window_end_us` remain GENX320 sensor time. These domains must not be
interchanged.

Enriched copies preserve the complete source file and add
`/observations/sparse_tracking`. Every dataset has the same leading dimension
as `/observations/timestamps`. Sampling selects the latest update available at
or before each observation, so it is causal. The group contains event position,
velocity, validity/update flags, source time and age, sensor window bounds,
event/blob diagnostics, confidence, and rejection reason. Missing initial
updates use zero coordinates, false flags, and NaN source time/age. Original
files are never changed; each output is copied to a temporary sibling and
atomically renamed.

RGB 2D data is deliberately optional. `--rgb-tracks-dir` accepts one HDF5 per
episode containing `timestamps`, `rgb_2d_px`, and `valid`, sampled with the same
causal rule. Without it, event-only output is written with a warning. Use
`--write-empty-rgb-track` to explicitly add zero-filled RGB fields, or
`--require-rgb-2d` to fail when an episode track is absent.

## Hardware and raw-event HDF5 replay

`openmv_cam` accepts interchangeable raw packet sources. The default remains
the physical GenX320 EVT1 serial stream:

```bash
ros2 launch openmv_cam openmv.launch.py event_input_mode:=hardware
```

Recorded sessions can instead enter the identical `EventPacket` processing
path. Replay does not open `/dev/openmvcam`, and hardware mode does not open an
HDF5 input file:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_input_mode:=hdf5_replay \
  event_replay_path:=/home/jau/data/bags/example/example_raw_events.h5 \
  event_replay_timing:=recorded \
  event_replay_rate:=1.0 \
  event_replay_loop:=false \
  event_tracker_enabled:=true \
  event_tracker_debug_enabled:=true \
  event_tracker_debug_clip_count:=2 \
  event_tracker_activity_threshold:=1 \
  event_tracker_min_event_count:=1 \
  event_tracker_min_blob_area_px:=500 \
  event_tracker_max_blob_area_px:=1500 \
  event_tracker_max_jump_px:=100.0 \
  event_tracker_velocity_history_size:=4 \
  event_tracker_velocity_min_span_ms:=1.0 \
  event_tracker_bin_ms:=10.0 \
  event_tracker_morphology_operation:=dilate \
  event_tracker_morphology_kernel:=7 \
  event_tracker_morphology_iterations:=3
```

Replay parameters and defaults are:

- `event_input_mode=hardware`: `hardware` or `hdf5_replay`.
- `event_replay_path=""`: input file required for replay.
- `event_replay_timing=recorded`: `recorded`, `sensor`, or `fast`.
- `event_replay_rate=1.0`: positive finite timing multiplier; `2.0` is twice
  real time and `0.5` is half speed.
- `event_replay_start_packet=0`: first packet index, inclusive.
- `event_replay_end_packet=-1`: final packet index, inclusive; `-1` selects the
  end of the recording.
- `event_replay_loop=false`: restart at the selected first packet after EOF.

The raw recorder and replay reader use this schema (all datasets are 1-D):

```text
/events/type       uint8
/events/x          uint16
/events/y          uint16
/events/t_us       int64
/events/packet_id  int64
/packets/ros_t_ns             int64
/packets/monotonic_t_ns       int64
/packets/start_event_idx      int64
/packets/end_event_idx        int64
/packets/event_count          int64
/packets/first_event_t_us     int64
/packets/last_event_t_us      int64
```

The packet boundary is the half-open event range
`[start_event_idx, end_event_idx)`. The replay validator requires the five
event datasets and the first five packet datasets shown above, checks their
dtypes, lengths, contiguous boundaries, counts, and `packet_id` membership.
The final two packet timestamp-bound datasets are recorder metadata; replay
derives sensor pacing directly from `/events/t_us`, allowing compatible older
files without those two columns. Events are read packet-wise rather than
loading the stream into RAM.

Timing modes have the following semantics:

- `recorded` schedules packet arrivals from recorded monotonic host timestamps.
  ROS epoch timestamps are never used for sleeps.
- `sensor` schedules from the minimum GenX320 timestamp in each packet. Tracker
  timing and velocity continue to use the unchanged sensor timestamps.
- `fast` performs no intentional sleep and is suitable for parameter sweeps.

Recorded and sensor delays are divided by `event_replay_rate` and scheduled
against the current host monotonic clock. A missing, zero, reset, or
non-monotonic pacing timestamp produces one clear warning and switches the
remainder of replay to `fast`; an empty packet therefore causes sensor timing
to fall back. Loop timing is re-anchored on each selected range. All waits are
interruptible during node shutdown.

New ROS messages always use current replay-time ROS timestamps, and host
measurements use current monotonic time. Original ROS/monotonic timestamps are
retained only in replay `EventPacket` metadata and finite latency-trace JSON.
GenX320 timestamps, event order, polarity, coordinates, packet boundaries, and
recorded packet indices are preserved. Tracker debug images and all event image
and voxel publishers consume the same shared packet buffer as hardware input.
Latency tracing emits the existing input/completion contract with
`source=hdf5_replay`, packet index, and original recorded timestamps in
`detail_json`.

Raw recording services remain unchanged. They may record replayed packets into
the same schema while preserving event timestamps and packet boundaries; the
new file records the current replay arrival clocks in its packet metadata.

Useful variants are:

```bash
ros2 launch openmv_cam openmv.launch.py event_input_mode:=hdf5_replay \
  event_replay_path:=/data/events.h5 event_replay_timing:=fast

ros2 launch openmv_cam openmv.launch.py event_input_mode:=hdf5_replay \
  event_replay_path:=/data/events.h5 event_replay_timing:=sensor

ros2 launch openmv_cam openmv.launch.py event_input_mode:=hdf5_replay \
  event_replay_path:=/data/events.h5 event_replay_timing:=recorded \
  event_replay_rate:=2.0
```

## Raw-packet event ball tracker MVP

The optional event tracker detects a high-activity ball-like blob directly in
validated EVT1 packets. Its path is `EVT1 serial -> EventPacket -> sensor-time
1 ms history bins -> sliding raw-event window -> activity map -> blob/velocity
estimate`. It runs in the serial reader path: it does not subscribe to
`/openmv_cam/event_voxel_1ms` and does not wait for the 30 Hz image timer. The
same validated `EventPacket` is
shared with preview buffering and HDF5 recording, so EVT1 timestamps are
reconstructed once. Wire order is retained for compatibility; packet-local
timestamp disorder is handled explicitly by sensor-time bin assignment.

Completed 1 ms bins are retained in bounded sensor-time history. Once per
non-empty packet, at most, the tracker forms the newest exact half-open 10 ms
window `[window_start_us, window_end_us)`. Both polarities and repeated events
increment a dense 320x320 raw count map. The expected output rate is therefore
the EVT1 packet rate (typically 50-70 Hz), not the approximately 1 kHz internal
bin rate.

The grouping mask supports only `none`, `close`, and `dilate`; morphological
opening is never used. When enabled, spatial filtering applies the crop, the
optional 8-neighbor cutoff, and an 8-connected component-area cutoff before
morphology. Closing or dilation may connect sparse elongated trails,
but generated pixels are used only for grouping. Candidate event count and COM
always come from original raw activity inside the filled grouping contour.
Candidates are constrained by raw event count, fill area, width, and height.
The first acquisition uses raw event count with deterministic tie-breaking;
tracked selection prioritizes distance from the sensor-time prediction and
enforces the maximum jump. Circularity is diagnostic and optionally filterable,
but disabled by default so elongated arcs are accepted. After configured missed
updates, deterministic highest-event-count reacquisition is allowed. Velocity
remains a bounded least-squares fit over sensor-time COM detections. There is no
Kalman filter.

Outputs (only for valid detections) are:

- `/openmv_cam/event_tracker/ball_2d_px` (`geometry_msgs/msg/PointStamped`)
- `/openmv_cam/event_tracker/ball_velocity_px_s`
  (`geometry_msgs/msg/Vector3Stamped`; z is scalar speed)
- `/openmv_cam/event_tracker/valid` (`std_msgs/msg/Bool`)

Coordinates are native OpenMV/GENX320 pixels: 320x320, top-left origin, x right,
y down, and `frame_id=openmv_cam`. Tracker coordinates are never rotated.
Message header stamps are the host ROS publication time. Sensor microseconds
are used for binning and velocity only; they are not ROS epoch timestamps.

The tracker defaults to enabled with the tuned ball-tracking configuration.
Important parameters and defaults are:

- `event_tracker_enabled=true`
- `event_tracker_position_topic=/openmv_cam/event_tracker/ball_2d_px`
- `event_tracker_velocity_topic=/openmv_cam/event_tracker/ball_velocity_px_s`
- `event_tracker_valid_topic=/openmv_cam/event_tracker/valid`
- `event_tracker_bin_ms=1.0`
- `event_tracker_accumulation_window_ms=3.0`
- `event_tracker_history_limit_ms=100.0`
- `event_tracker_activity_threshold=1`
- `event_tracker_spatial_filter_enabled=false`
- `event_tracker_spatial_filter_min_neighbors=1`
- `event_tracker_spatial_filter_min_component_area_px=1`
- `event_tracker_min_event_count=1`
- `event_tracker_min_blob_area_px=250`, `event_tracker_max_blob_area_px=1500`
- `event_tracker_min_blob_width_px=1`, `event_tracker_max_blob_width_px=320`
- `event_tracker_min_blob_height_px=1`, `event_tracker_max_blob_height_px=320`
- `event_tracker_morphology_operation=dilate`
- `event_tracker_morphology_kernel=3`,
  `event_tracker_morphology_iterations=3`
- `event_tracker_use_circularity=false`, `event_tracker_min_circularity=0.1`
- `event_tracker_max_jump_px=100.0`
- `event_tracker_reacquire_after_misses=3`
- `event_tracker_x_crop=[100, 210]`
- `event_tracker_y_crop=[35, 275, 85, 235]`
- `event_tracker_velocity_history_size=4`
- `event_tracker_velocity_min_span_ms=1.0`
- `event_tracker_stats_period_sec=5.0`
- `event_tracker_debug_enabled=false`
- `event_tracker_debug_topic=/openmv_cam/event_tracker/debug_image`

For example, remove components smaller than four foreground pixels before one
iteration of 3x3 dilation:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_tracker_spatial_filter_enabled:=true \
  event_tracker_spatial_filter_min_neighbors:=1 \
  event_tracker_spatial_filter_min_component_area_px:=4 \
  event_tracker_morphology_operation:=dilate \
  event_tracker_morphology_kernel:=3 \
  event_tracker_morphology_iterations:=1
```
- `event_tracker_debug_fps=10.0`
- `event_tracker_debug_clip_count=16`
- `event_tracker_debug_rotation_degrees=90`
- `event_tracker_debug_event_frame_enabled=true`
- `event_tracker_debug_event_frame_topic=/openmv_cam/event_tracker/debug/event_frame_33ms`
- `event_tracker_debug_event_frame_window_ms=33.0`
- `event_tracker_debug_activity_topic=/openmv_cam/event_tracker/debug/activity`
- `event_tracker_debug_threshold_topic=/openmv_cam/event_tracker/debug/threshold`
- `event_tracker_debug_contours_topic=/openmv_cam/event_tracker/debug/contours`
- `event_tracker_debug_tracking_topic=/openmv_cam/event_tracker/debug/tracking`

When enabled, the debug topic is a native-coordinate 320x320 `bgr8`
`sensor_msgs/msg/Image`. A low-rate timer renders the latest cached window
snapshot; annotation is never rendered in the serial reader. Activity uses the
fixed mapping `round(255 * min(count, clip_count) / clip_count)` rather than
per-image normalization. Non-selected candidates are orange, the selected
contour and weighted COM are green, prediction is a yellow cross, recent COM
history is a blue polyline, and valid velocity is a cyan arrow. Text identifies
the sensor-time window, raw event/candidate counts, validity, and velocity.
The graphical layer defaults to 90 degrees counterclockwise while the text is
drawn afterward and remains upright. This affects only the debug image; tracker
coordinates and numeric topics remain native and unrotated.

Each new cached window is rendered into synchronized, upright-text `bgr8` stage
images at no more than `event_tracker_debug_fps`:

- `debug/activity`: fixed-scale raw sliding-window activity map.
- `debug/threshold`: grouping mask after configured close/dilation.
- `debug/contours`: filtered-in contours in red and rejected contours in orange.
- `debug/tracking`: selection, COM, prediction, trajectory, and velocity.
- `debug/event_frame_33ms`: native mono-polarity event rendering for the exact
  newest 33 ms of completed tracker sensor time, with a valid newest COM marked
  red before applying the configured debug rotation.

The original `/openmv_cam/event_tracker/debug_image` remains as a compatibility
alias of `debug/tracking`.

`event_tracker_x_crop` and `event_tracker_y_crop=[a,b,c,d]` restrict detection
to a native-coordinate trapezoid. At the left x bound its half-open y range is
`[a,b)`; at the right x bound it is `[c,d)`, with both y boundaries linearly
interpolated between them. Events outside this polygon never enter
the activity map, contours, candidate selection, or velocity history. Both
crop boundaries are drawn as a magenta polygon before debug-image rotation.
Numeric outputs remain in native coordinates. Example:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_tracker_enabled:=true \
  event_tracker_debug_enabled:=true \
  event_tracker_x_crop:="[80, 240]" \
  event_tracker_y_crop:="[40, 260, 80, 220]"
```

Example:

```bash
ros2 launch openmv_cam openmv.launch.py event_tracker_enabled:=true
```

Debug-view example:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_tracker_enabled:=true \
  event_tracker_debug_enabled:=true \
  event_tracker_debug_fps:=10.0 \
  event_tracker_debug_clip_count:=16
```

The sensor-time event-frame diagnostic does not depend on preview publishing:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_tracker_enabled:=true \
  event_tracker_debug_enabled:=true \
  event_tracker_debug_event_frame_enabled:=true \
  event_tracker_debug_event_frame_window_ms:=33.0
```

Optional latency traces use best-effort QoS on
`/intercept_trace/event_2d_ball_detection` when
`publish_latency_traces=true`. `input` traces correspond one-to-one with
validated packets; at most one `complete` trace is emitted for the packet-level
sliding-window update and carries packet lineage in `parent_sequence`. Both use stage
`event_2d_ball_detection` and modality `event`. ROS timestamp fields contain
host-clock values. GENX320 window timestamps and finite raw-event detection/blob
values are kept in `detail_json`. Consequently source-to-output latency begins when a
complete EVT1 packet is available on the PC, not at physical sensor exposure.
If `intercept_latency_monitor` is not installed, tracing logs an error and is
disabled without stopping the reader. Configure with:

- `publish_latency_traces=false`
- `latency_trace_topic=/intercept_trace/event_2d_ball_detection`
- `latency_trace_run_id=""`

## Native 1 ms activity voxel

`/openmv_cam/event_voxel_1ms` is a `sensor_msgs/msg/Image` intended as the
frontend for a downstream classical event blob detector. It is disabled by
default. This is a rolling voxel anchored to the newest sensor event available
at a publication tick; it is not a lossless event-time-binned stream. Multiple
sensor-time bins can arrive between publication ticks and are represented only
if they remain inside the configured rolling depth. Raw HDF5 recording remains
the lossless representation.

Every channel is exactly `event_voxel_bin_ms` wide (1.0 ms by default). The HWC
channel order is oldest to newest, with right-closed intervals. For anchor `A`,
depth `N`, and bin width `B`, the represented sensor-time interval is
`(A - N*B, A]`. Events are selected and binned by the reconstructed EVT1 sensor
timestamp, never by serial packet boundaries, and future events are excluded.

With one temporal bin the message is 320x320 `8UC1` (`step=320`). With multiple
bins it is 320x320 `8UCN`, where channel N is temporal, and `step=320*N`.
Absolute activity is the default: both event types increment the same pixel
count, the background is zero, and the deterministic mapping is
`round(255 * min(count, clip_count) / clip_count)`. `signed_activity` is an
optional cancellation/debug mode; its absolute signed accumulation is scaled
with the same fixed rule. There is no per-frame normalization.

The output uses native OpenMV coordinates: origin at the image's top-left,
positive x to the right, positive y downward, 0 <= x,y < 320. It is never
rotated and always has `frame_id="openmv_cam"`; the existing rotation services
continue to affect the legacy preview/3-channel/9-bin outputs only. Consumers
should use these native coordinates, not RGB-calibrated or projected ones.

The header stamp is the host ROS receive timestamp of the serial packet that
contains the anchor event, because EVT1 does not currently provide a reliable
sensor-clock-to-ROS-clock mapping. A timer tick publishes only after a new
non-empty packet has entered the shared buffer. With no new events, no stale
non-empty voxel or falsely fresh timestamp is published. Small timestamp
disorder can amend and republish a rolling voxel with the same anchor/stamp.

Launch example:

```bash
ros2 launch openmv_cam openmv.launch.py \
  publish_event_voxel_1ms:=true \
  event_voxel_bin_ms:=1.0 \
  event_voxel_temporal_bins:=1 \
  event_voxel_publish_fps:=30.0
```

The equivalent detector-only selector is:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_output_mode:=event_voxel_1ms \
  event_voxel_bin_ms:=1.0 \
  event_voxel_temporal_bins:=1 \
  event_voxel_publish_fps:=30.0 \
  event_diagnostics_enabled:=true
```

`fr3_teleop/launch/vision.launch.py` forwards these detector settings to the
same included OpenMV node. `event_output_mode=legacy_flags` is the default and
preserves historical precedence: the existing publish flags retain their exact
meaning. Explicit `event_voxel_1ms`, `all`, or `none` modes override event-image
flags only. Detector-only mode publishes only the native activity voxel; `all`
enables every event image; `none` disables event images. Services, serial input,
and raw HDF5 recording are unaffected.

Parameters and defaults:

- `publish_event_voxel_1ms=false`
- `topic_event_voxel_1ms=/openmv_cam/event_voxel_1ms`
- `event_voxel_bin_ms=1.0`
- `event_voxel_temporal_bins=9` (set to 1 for a single detector slice; the
  legacy `/openmv_cam/event_voxel` remains fixed at its compatible 9 channels)
- `event_voxel_activity_mode=absolute_activity`
- `event_voxel_clip_count=16.0`
- `event_voxel_publish_fps=30.0`
- `event_output_mode=legacy_flags`
- `event_diagnostics_enabled=false`
- `event_diagnostics_period_sec=5.0`

The output bandwidth is approximately `320*320*N*fps` bytes/s before ROS
overhead (3.1 MB/s for one bin at 30 Hz). The shared packet buffer has a safety
cap (`max_event_frame_packets`), so an undersized cap or host-arrival retention
can make a live rolling voxel incomplete; the node warns when the cap truncates
history. HDF5 recording is unaffected and should be used when lossless capture
is required.

When diagnostics are enabled, one throttled summary reports serial packet/event
rates, requested and actual timer intervals, timer/publish/skip counts,
safety-cap drops, anchor advancement, and rolling p50/p95/max timings for
buffer processing, voxel construction, message construction, and the ROS
publish call.

Headless topic validation:

```bash
python3 scripts/validate_event_voxel.py \
  --topic /openmv_cam/event_voxel_1ms \
  --duration 20 \
  --output-dir /tmp/event_voxel_validation \
  --save-every 10 \
  --save-npz
```

This writes `summary.json`, `frames.csv`, selected PNGs, and optionally
`frames.npz`. It reports arrival/header rates, header-to-arrival delay, layout,
activity and bounding boxes, empty frames, repeated payloads, stale non-empty
frames, and repeated or non-monotonic timestamps.
