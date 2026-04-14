import csi
import time
import ustruct
from pyb import USB_VCP
from ulab import numpy as np

# ============================================================
# OpenMV GenX320 raw event streamer
#
# Packet format:
#   MAGIC (4B): b'EVT1'
#   HEADER    : <LL   (event_count, payload_len)
#   PAYLOAD   : raw bytes of events[:event_count]
#
# Event layout per row (uint16 x 6):
#   0: event type / polarity-like field
#   1: seconds timestamp
#   2: milliseconds timestamp
#   3: microseconds timestamp
#   4: x coordinate
#   5: y coordinate
# ============================================================

MAGIC = b'EVT1'
HEADER_FMT = '<LL'

BUF_SIZE = 8192

usb = USB_VCP()
usb.setinterrupt(-1)

events = np.zeros((BUF_SIZE, 6), dtype=np.uint16)

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])
#csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 130, 160)
csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_LOW_NOISE)

time.sleep_ms(200)

def send_all(data, timeout=5000):
    mv = memoryview(data)
    sent_total = 0
    while sent_total < len(data):
        n = usb.send(mv[sent_total:], timeout=timeout)
        if n is None or n <= 0:
            raise OSError("usb.send failed or timed out")
        sent_total += n

while True:
    try:
        event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

        if event_count < 0:
            continue

        if event_count == 0:
            continue

        payload = events[:event_count]
        payload_len = event_count * 6 * 2  # 6 uint16 fields per event

        header = MAGIC + ustruct.pack(HEADER_FMT, event_count, payload_len)

        send_all(header)
        send_all(payload)

    except Exception:
        time.sleep_ms(10)
