#!/usr/bin/env python3
import argparse
import struct
import time
from dataclasses import dataclass

import cv2
import numpy as np
import serial

SYNC_WORD = 0xD5AA
HEADER_SIZE = 10
MAGIC_BAUD = 921600

FLAG_ACK = 1 << 0
FLAG_NAK = 1 << 1
FLAG_RTX = 1 << 2
FLAG_ACK_REQ = 1 << 3
FLAG_FRAGMENT = 1 << 4
FLAG_EVENT = 1 << 5

OP_PROTO_SYNC = 0x00
OP_PROTO_GET_CAPS = 0x01
OP_PROTO_SET_CAPS = 0x02
OP_CH_LOCK = 0x22
OP_CH_UNLOCK = 0x23
OP_CH_SIZE = 0x25
OP_CH_READ = 0x26
OP_CH_IOCTL = 0x28

CH_STREAM = 3

IOCTL_STREAM_CTRL = 0x00
IOCTL_STREAM_RAW_CTRL = 0x01
IOCTL_STREAM_RAW_CFG = 0x02
IOCTL_STREAM_SOURCE = 0x03

CRC16_INIT = 0xFFFF
CRC16_POLY = 0xF94F
CRC32_INIT = 0xFFFFFFFF
CRC32_POLY = 0xFA567D89


@dataclass
class Packet:
    sync: int
    seq: int
    channel: int
    flags: int
    opcode: int
    length: int
    crc16: int
    payload: bytes
    crc32: bytes


def _build_crc16_table():
    table = [0] * 256
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ CRC16_POLY
            else:
                crc = (crc << 1) & 0xFFFF
        table[i] = crc
    return table


def _build_crc32_table():
    table = [0] * 256
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ CRC32_POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table[i] = crc
    return table


CRC16_TABLE = _build_crc16_table()
CRC32_TABLE = _build_crc32_table()


def crc16(data: bytes) -> int:
    crc = CRC16_INIT
    for b in data:
        idx = ((crc >> 8) ^ b) & 0xFF
        crc = ((crc << 8) & 0xFFFF) ^ CRC16_TABLE[idx]
    return crc


def crc32_custom(data: bytes) -> int:
    crc = CRC32_INIT
    for b in data:
        idx = ((crc >> 24) ^ b) & 0xFF
        crc = ((crc << 8) & 0xFFFFFFFF) ^ CRC32_TABLE[idx]
    return crc


class OmvProtocolReceiver:
    def __init__(self, port: str, timeout: float, max_payload: int):
        self.ser = serial.Serial(
            port,
            baudrate=MAGIC_BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            timeout=timeout,
        )
        self.ser.setDTR(True)
        self.ser.setRTS(True)
        self.ser.reset_input_buffer()
        self.seq = 0
        self.max_payload = max_payload
        self._last_sync_error = ""

    def close(self):
        self.ser.close()

    def _read_exactly(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.ser.read(n - len(out))
            if not chunk:
                raise RuntimeError("Serial read timeout")
            out.extend(chunk)
        return bytes(out)

    def _read_packet_with_timeout(self, timeout_s: float) -> Packet:
        old_timeout = self.ser.timeout
        try:
            # Use short chunked reads so sync can retry quickly.
            self.ser.timeout = max(0.02, min(timeout_s, 0.1))
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    return self._read_packet()
                except RuntimeError as e:
                    # Keep scanning until deadline; remember last reason.
                    self._last_sync_error = str(e)
            raise RuntimeError("Timed out waiting for protocol response")
        finally:
            self.ser.timeout = old_timeout

    def _read_packet(self) -> Packet:
        # Sync scan.
        w = bytearray()
        while True:
            b = self._read_exactly(1)
            w += b
            if len(w) > 2:
                w = w[-2:]
            if len(w) == 2 and struct.unpack("<H", w)[0] == SYNC_WORD:
                break

        header_rest = self._read_exactly(8)
        seq, ch, flags, opcode, length, header_crc = struct.unpack("<BBBBHH", header_rest)

        header_wo_crc = struct.pack("<HBBBBH", SYNC_WORD, seq, ch, flags, opcode, length)
        calc_hcrc = crc16(header_wo_crc)
        if calc_hcrc != header_crc:
            raise RuntimeError("Header CRC mismatch")

        payload = b""
        payload_crc = b""
        if length:
            payload = self._read_exactly(length)
            payload_crc = self._read_exactly(4)
            recv_crc32 = struct.unpack("<I", payload_crc)[0]
            calc_crc32 = crc32_custom(payload)
            if recv_crc32 != calc_crc32:
                raise RuntimeError("Payload CRC mismatch")

        return Packet(
            sync=SYNC_WORD,
            seq=seq,
            channel=ch,
            flags=flags,
            opcode=opcode,
            length=length,
            crc16=header_crc,
            payload=payload,
            crc32=payload_crc,
        )

    def _send_packet(self, channel: int, opcode: int, payload: bytes = b"", flags: int = 0):
        header_wo_crc = struct.pack("<HBBBBH", SYNC_WORD, self.seq, channel, flags, opcode, len(payload))
        hcrc = crc16(header_wo_crc)
        header = header_wo_crc + struct.pack("<H", hcrc)

        self.ser.write(header)
        if payload:
            self.ser.write(payload)
            self.ser.write(struct.pack("<I", crc32_custom(payload)))

    def _request(self, channel: int, opcode: int, payload: bytes = b"") -> Packet:
        req_seq = self.seq
        self._send_packet(channel, opcode, payload, flags=0)

        # In lockstep mode, next sequence is expected after one command/response turn.
        self.seq = (self.seq + 1) & 0xFF

        while True:
            pkt = self._read_packet()

            # Ignore unrelated events while waiting for response.
            if pkt.flags & FLAG_EVENT:
                continue

            if pkt.seq != req_seq:
                continue

            # Accept ACK/NAK status or normal response packet.
            if pkt.opcode == opcode:
                if pkt.flags & FLAG_NAK:
                    code = struct.unpack("<H", pkt.payload[:2])[0] if pkt.payload else -1
                    raise RuntimeError(f"NAK for opcode=0x{opcode:02X}, status={code}")
                return pkt

    def sync(self):
        # PROTO_SYNC is always accepted even with sequence mismatch in firmware.
        # Retry to handle CDC mode transitions on some hosts/boards.
        self.seq = 0
        self._last_sync_error = ""

        # Give the device a moment after opening port and applying line coding.
        time.sleep(0.05)

        for _ in range(20):
            try:
                self.ser.setDTR(True)
                self.ser.setRTS(True)
                self.ser.reset_input_buffer()

                self._send_packet(channel=0, opcode=OP_PROTO_SYNC, payload=b"", flags=0)
                self.ser.flush()

                pkt = self._read_packet_with_timeout(0.2)
                if pkt.opcode == OP_PROTO_SYNC and (pkt.flags & FLAG_ACK):
                    # Device increments sequence after sending non-event packets.
                    self.seq = (pkt.seq + 1) & 0xFF
                    return

                self._last_sync_error = (
                    f"unexpected sync reply opcode=0x{pkt.opcode:02X}, flags=0x{pkt.flags:02X}"
                )
            except RuntimeError as e:
                self._last_sync_error = str(e)

            # Small backoff while USB CDC control state settles.
            time.sleep(0.05)

        raise RuntimeError(
            "Failed to sync protocol after retries. "
            "Check that firmware protocol mode is enabled and that the port is correct. "
            f"Last error: {self._last_sync_error or 'none'}"
        )

    def set_caps(self, crc_enabled=True, seq_enabled=False, ack_enabled=False, events_enabled=True):
        flags = (
            (1 if crc_enabled else 0)
            | ((1 if seq_enabled else 0) << 1)
            | ((1 if ack_enabled else 0) << 2)
            | ((1 if events_enabled else 0) << 3)
        )
        payload = struct.pack("<IH10s", flags, self.max_payload, b"\x00" * 10)
        self._request(channel=0, opcode=OP_PROTO_SET_CAPS, payload=payload)

    def stream_ioctl(self, cmd: int, arg: bytes):
        payload = struct.pack("<I", cmd) + arg
        self._request(channel=CH_STREAM, opcode=OP_CH_IOCTL, payload=payload)

    def lock_stream(self) -> bool:
        try:
            self._request(channel=CH_STREAM, opcode=OP_CH_LOCK, payload=b"")
            return True
        except RuntimeError:
            return False

    def unlock_stream(self):
        try:
            self._request(channel=CH_STREAM, opcode=OP_CH_UNLOCK, payload=b"")
        except RuntimeError:
            pass

    def stream_size(self) -> int:
        pkt = self._request(channel=CH_STREAM, opcode=OP_CH_SIZE, payload=b"")
        if len(pkt.payload) < 4:
            return 0
        return struct.unpack("<I", pkt.payload[:4])[0]

    def stream_read(self, total_len: int) -> bytes:
        req = struct.pack("<II", 0, total_len)
        pkt = self._request(channel=CH_STREAM, opcode=OP_CH_READ, payload=req)
        chunks = [pkt.payload]

        while pkt.flags & FLAG_FRAGMENT:
            pkt = self._read_packet()
            if pkt.flags & FLAG_EVENT:
                continue
            if pkt.opcode != OP_CH_READ:
                continue
            chunks.append(pkt.payload)

        data = b"".join(chunks)
        return data[:total_len]


def decode_stream_frame(blob: bytes):
    if len(blob) < 20:
        raise RuntimeError("Stream blob too small for framebuffer header")

    width, height, pixfmt, size_field, offset = struct.unpack("<IIIII", blob[:20])
    if offset < 20 or offset > len(blob):
        raise RuntimeError("Invalid framebuffer header offset")

    flags = (pixfmt >> 24) & 0x1F
    bpp = pixfmt & 0xFF
    is_compressed = (flags >> 1) & 0x1
    is_color = (flags >> 2) & 0x1

    payload = blob[offset:]

    if is_compressed:
        encoded = payload[:size_field]
        img = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("Failed to decode compressed frame")
        return img, width, height, pixfmt, True

    expected = width * height * bpp
    raw = payload[:expected]
    if len(raw) < expected:
        raise RuntimeError("Raw frame shorter than expected")

    if bpp == 1:
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        return img, width, height, pixfmt, False

    if bpp == 2 and is_color:
        rgb565 = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
        r = ((rgb565 >> 11) & 0x1F).astype(np.uint8)
        g = ((rgb565 >> 5) & 0x3F).astype(np.uint8)
        b = (rgb565 & 0x1F).astype(np.uint8)
        bgr = np.dstack((
            (b * 255 // 31),
            (g * 255 // 63),
            (r * 255 // 31),
        )).astype(np.uint8)
        return bgr, width, height, pixfmt, False

    # Fallback: visualize first byte of each pixel for unsupported formats.
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, bpp))
    return arr[:, :, 0], width, height, pixfmt, False


def main():
    ap = argparse.ArgumentParser(description="IDE-like OpenMV stream receiver (protocol path)")
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--stats-period", type=float, default=2.0)
    ap.add_argument("--max-payload", type=int, default=4082)
    ap.add_argument("--raw", action="store_true", help="Enable raw stream mode")
    ap.add_argument("--raw-width", type=int, default=320)
    ap.add_argument("--raw-height", type=int, default=320)
    ap.add_argument("--stream-source", type=lambda x: int(x, 0), default=None)
    args = ap.parse_args()

    client = OmvProtocolReceiver(args.port, args.timeout, args.max_payload)

    frames = 0
    bytes_payload = 0
    t0 = time.monotonic()
    t_last = t0

    try:
        print("SYNC...")
        client.sync()
        print("SYNC OK")

        # Reduce protocol complexity to match a lightweight host receiver.
        print("SET_CAPS...")
        client.set_caps(crc_enabled=True, seq_enabled=False, ack_enabled=False, events_enabled=True)
        print("SET_CAPS OK")


        print("STREAM_CTRL ON...")
        client.stream_ioctl(IOCTL_STREAM_CTRL, struct.pack("<I", 1))
        print("STREAM_CTRL OK")
        print("RAW_CTRL...")
        client.stream_ioctl(IOCTL_STREAM_RAW_CTRL, struct.pack("<I", 0))
        print("RAW_CTRL OK")

        if args.raw:
            client.stream_ioctl(IOCTL_STREAM_RAW_CFG, struct.pack("<II", args.raw_width, args.raw_height))

        if args.stream_source is not None:
            client.stream_ioctl(IOCTL_STREAM_SOURCE, struct.pack("<I", args.stream_source))

        while True:
            if not client.lock_stream():
                time.sleep(0.001)
                continue

            size = client.stream_size()
            if size <= 0:
                client.unlock_stream()
                time.sleep(0.001)
                continue

            blob = client.stream_read(size)
            client.unlock_stream()

            img, w, h, pixfmt, compressed = decode_stream_frame(blob)

            frames += 1
            bytes_payload += len(blob)

            now = time.monotonic()
            if now - t_last >= args.stats_period:
                elapsed = now - t0
                fps = frames / elapsed if elapsed > 0 else 0.0
                mbps = bytes_payload / elapsed / 1e6 if elapsed > 0 else 0.0
                print(
                    f"frames={frames} fps={fps:.2f} payload_MBps={mbps:.3f} "
                    f"last=({w}x{h}) pixfmt=0x{pixfmt:08X} compressed={compressed}"
                )
                t_last = now

            if args.show:
                cv2.imshow("OpenMV IDE-like stream", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        try:
            client.stream_ioctl(IOCTL_STREAM_CTRL, struct.pack("<I", 0))
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
