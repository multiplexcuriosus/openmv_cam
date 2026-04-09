import csi, time, ustruct
from pyb import USB_VCP

usb = USB_VCP()
usb.setinterrupt(-1)

MAGIC = b'OMV1'

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.pixformat(csi.GRAYSCALE)
csi0.framesize((320, 320))
csi0.brightness(128)
csi0.contrast(16)
csi0.framerate(100)
csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 130, 160)

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
        img = csi0.snapshot()
        buf = img.bytearray()
        width = img.width()
        height = img.height()
        payload_len = len(buf)

        header = MAGIC + ustruct.pack("<LLL", width, height, payload_len)

        send_all(header)
        send_all(buf)

    except Exception:
        time.sleep_ms(50)