"""OpenMV H7 GenX320 raw EVT2.0 streamer using custom EVR1 framing."""

import csi
import time
import ustruct
from pyb import USB_VCP


MAGIC = b"EVR1"
HEADER_FORMAT = "<LL"
EVENT_BUFFER_SIZE = 2048
CSI_FIFO_DEPTH = 8


def send_all(usb, data, timeout=5000):
    """Send the complete bytes-like object or raise on timeout."""
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        count = usb.send(view[sent:], timeout=timeout)
        if count is None or count <= 0:
            raise OSError("usb.send failed or timed out")
        sent += count


usb = USB_VCP()
usb.setinterrupt(-1)

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(
    csi.IOCTL_GENX320_SET_MODE,
    csi.GENX320_MODE_EVENT,
    EVENT_BUFFER_SIZE,
)
csi0.__write_reg(0x7044, 0)
csi0.framebuffers(CSI_FIFO_DEPTH)
csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 238, 242)
csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_LOW_NOISE)
time.sleep_ms(200)

sequence = 0
while True:
    try:
        raw_frame = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS_RAW)
        payload = memoryview(raw_frame.bytearray())
        if not payload:
            continue
        if len(payload) % 4:
            raise ValueError("raw EVT2.0 frame length is not divisible by four")
        send_all(usb, MAGIC + ustruct.pack(HEADER_FORMAT, sequence, len(payload)))
        send_all(usb, payload)
        sequence = (sequence + 1) & 0xFFFFFFFF
    except Exception:
        time.sleep_ms(10)
