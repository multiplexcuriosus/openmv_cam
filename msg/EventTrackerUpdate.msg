# Provenance and result for exactly one event-tracker sliding-window update.
# header.stamp is the host ROS clock time at which this result became available.
# header.frame_id is "openmv_cam"; coordinates are native GenX320 pixels.
std_msgs/Header header

# Monotonic, process-local update sequence. It increments once for every
# processed tracker window, including invalid windows.
uint64 tracker_update_id
bool tracker_update_id_valid

# EVR1 uint32 sequence of the newest raw EVT2.0 packet contributing an event
# to this window, widened to uint64. This is unavailable for processed EVT1
# input and current HDF5 recordings.
uint64 source_packet_id
bool source_packet_id_valid

# Same instant as header.stamp, represented as nanoseconds in the ROS clock
# domain for consumers that need an integer timestamp.
int64 availability_timestamp_ns

# Half-open [start,end) tracker window in the GenX320 sensor microsecond
# domain. These values are not Unix/ROS epoch timestamps.
int64 sensor_window_start_us
int64 sensor_window_end_us

# Native sensor pixel coordinates and sensor-time-derived pixel velocities.
float64 x_px
float64 y_px
float64 vx_px_s
float64 vy_px_s

bool valid
# Stable machine-readable reason; empty for ordinary valid updates. A valid
# reacquisition may use "reacquired_after_misses" diagnostically.
string rejection_reason

uint32 candidate_count
uint32 window_event_count
float64 confidence
bool velocity_valid
