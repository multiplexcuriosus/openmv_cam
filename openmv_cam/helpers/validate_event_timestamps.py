#!/usr/bin/env python3
import argparse
import struct
import time

import numpy as np
import serial

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len


def read_exactly(ser, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise RuntimeError("Serial read timeout.")
        data.extend(chunk)
    return bytes(data)


def read_until_magic(ser):
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise RuntimeError("Timeout while waiting for magic.")
        window += b
        if len(window) > len(MAGIC):
            window = window[-len(MAGIC):]
        if bytes(window) == MAGIC:
            return


def ts_us(events: np.ndarray) -> np.ndarray:
    # columns:
    # 0: type
    # 1: sec
    # 2: ms
    # 3: us
    # 4: x
    # 5: y
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def main():
    ap = argparse.ArgumentParser(description="Validate GenX320 event timestamps from EVT1 stream")
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--print-first-packets", type=int, default=3)
    args = ap.parse_args()

    ser = serial.Serial(
        args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
        timeout=args.timeout,
    )
    ser.reset_input_buffer()

    total_packets = 0
    total_events = 0

    intra_nonmono = 0
    inter_nonmono = 0
    zero_dt = 0
    negative_dt = 0

    global_min_dt = None
    global_max_dt = None

    sec_min = None
    sec_max = None
    ms_min = None
    ms_max = None
    us_min = None
    us_max = None

    prev_last_ts = None
    start = time.monotonic()

    print(f"[INFO] listening on {args.port} @ {args.baud} for {args.duration:.1f}s")

    try:
        while time.monotonic() - start < args.duration:
            read_until_magic(ser)
            header_rest = read_exactly(ser, struct.calcsize(HEADER_FMT))
            event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

            expected_len = event_count * 6 * 2
            if payload_len != expected_len:
                raise RuntimeError(
                    f"Invalid payload length: got {payload_len}, expected {expected_len}"
                )

            payload = read_exactly(ser, payload_len)
            events = np.frombuffer(payload, dtype=np.uint16).reshape((event_count, 6))

            total_packets += 1
            total_events += event_count

            if event_count == 0:
                continue

            secs = events[:, 1].astype(np.int64)
            mss = events[:, 2].astype(np.int64)
            uss = events[:, 3].astype(np.int64)
            ts = ts_us(events)

            # field ranges
            pkt_sec_min = int(secs.min())
            pkt_sec_max = int(secs.max())
            pkt_ms_min = int(mss.min())
            pkt_ms_max = int(mss.max())
            pkt_us_min = int(uss.min())
            pkt_us_max = int(uss.max())

            sec_min = pkt_sec_min if sec_min is None else min(sec_min, pkt_sec_min)
            sec_max = pkt_sec_max if sec_max is None else max(sec_max, pkt_sec_max)
            ms_min = pkt_ms_min if ms_min is None else min(ms_min, pkt_ms_min)
            ms_max = pkt_ms_max if ms_max is None else max(ms_max, pkt_ms_max)
            us_min = pkt_us_min if us_min is None else min(us_min, pkt_us_min)
            us_max = pkt_us_max if us_max is None else max(us_max, pkt_us_max)

            # within-packet monotonicity
            dts = np.diff(ts)
            if dts.size > 0:
                intra_nonmono += int(np.sum(dts < 0))
                zero_dt += int(np.sum(dts == 0))
                negative_dt += int(np.sum(dts < 0))

                pkt_min_dt = int(dts.min())
                pkt_max_dt = int(dts.max())
                global_min_dt = pkt_min_dt if global_min_dt is None else min(global_min_dt, pkt_min_dt)
                global_max_dt = pkt_max_dt if global_max_dt is None else max(global_max_dt, pkt_max_dt)

            # cross-packet monotonicity
            first_ts = int(ts[0])
            last_ts = int(ts[-1])
            if prev_last_ts is not None and first_ts < prev_last_ts:
                inter_nonmono += 1
            prev_last_ts = last_ts

            if total_packets <= args.print_first_packets:
                print(f"\n[DEBUG] packet {total_packets}")
                print(f"  events         : {event_count}")
                print(f"  sec range      : {pkt_sec_min} .. {pkt_sec_max}")
                print(f"  ms range       : {pkt_ms_min} .. {pkt_ms_max}")
                print(f"  us range       : {pkt_us_min} .. {pkt_us_max}")
                print(f"  ts first/last  : {first_ts} .. {last_ts}")
                if dts.size > 0:
                    print(f"  dt min/max     : {int(dts.min())} .. {int(dts.max())}")
                    print(f"  dt<0 count     : {int(np.sum(dts < 0))}")
                    print(f"  dt==0 count    : {int(np.sum(dts == 0))}")

    finally:
        ser.close()

    elapsed = time.monotonic() - start
    eps = total_events / elapsed if elapsed > 0 else 0.0

    print("\n===== TIMESTAMP VALIDATION SUMMARY =====")
    print(f"packets_received          : {total_packets}")
    print(f"events_received           : {total_events}")
    print(f"elapsed_time              : {elapsed:.2f} s")
    print(f"events_per_s              : {eps:.1f}")
    print(f"intra_packet_nonmonotonic : {intra_nonmono}")
    print(f"inter_packet_nonmonotonic : {inter_nonmono}")
    print(f"zero_dt_count             : {zero_dt}")
    print(f"negative_dt_count         : {negative_dt}")
    print(f"global_min_dt_us          : {global_min_dt}")
    print(f"global_max_dt_us          : {global_max_dt}")
    print(f"sec_range                 : {sec_min} .. {sec_max}")
    print(f"ms_range                  : {ms_min} .. {ms_max}")
    print(f"us_range                  : {us_min} .. {us_max}")
    print("========================================")

    print("\nInterpretation:")
    print("- intra_packet_nonmonotonic should ideally be 0")
    print("- inter_packet_nonmonotonic should ideally be 0")
    print("- ms_range should usually stay within 0..999")
    print("- us_range should usually stay within 0..999")
    print("- some zero_dt_count can be normal if multiple events share a timestamp")
    print("- negative_dt_count should ideally be 0")


if __name__ == "__main__":
    main()