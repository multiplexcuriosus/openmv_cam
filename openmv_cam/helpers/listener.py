#!/usr/bin/env python3
import struct
import time
import serial
import numpy as np

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len
HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

ser = serial.Serial("/dev/openmvcam", baudrate=115200, timeout=1.0)

def read_exactly(n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise RuntimeError("Serial read timeout.")
        data.extend(chunk)
    return bytes(data)

def read_until_magic():
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise RuntimeError("Timeout while searching for magic.")
        window += b
        if len(window) > len(MAGIC):
            window = window[-len(MAGIC):]
        if bytes(window) == MAGIC:
            return

while True:
    read_until_magic()

    header_rest = read_exactly(struct.calcsize(HEADER_FMT))
    event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

    if payload_len != event_count * 12:
        raise RuntimeError(
            f"Invalid payload_len={payload_len}, expected={event_count*12}"
        )

    payload = read_exactly(payload_len)
    events = np.frombuffer(payload, dtype=np.uint16).reshape(event_count, 6)

    print("count =", event_count)