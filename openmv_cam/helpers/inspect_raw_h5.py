import sys
import h5py
import numpy as np


path = sys.argv[1]


with h5py.File(path, "r") as f:
    counts = np.asarray(f["packets/event_count"], dtype=np.int64)
    event_times_us = np.asarray(f["events/t_us"], dtype=np.int64)
    packet_times_ns = np.asarray(f["packets/monotonic_t_ns"], dtype=np.int64)


print("packets:", len(counts))
print("total events:", int(counts.sum()))

if len(counts) > 0:
    print(
        "packet events min/mean/max:",
        int(counts.min()),
        float(counts.mean()),
        int(counts.max()),
    )

if len(event_times_us) > 0:
    sensor_duration_s = (
        int(event_times_us.max()) - int(event_times_us.min())
    ) / 1e6
    print(f"duration from sensor timestamps: {sensor_duration_s:.3f} s")

if len(packet_times_ns) > 1:
    host_duration_s = (
        int(packet_times_ns[-1]) - int(packet_times_ns[0])
    ) / 1e9
    print(f"recorded host-time duration: {host_duration_s:.3f} s")


for limit in (1024, 2048, 4096, 8192):
    if len(counts) > 0:
        print(
            f">= {limit:4d}:",
            int(np.sum(counts >= limit)),
            f"({100 * np.mean(counts >= limit):.2f}%)",
        )