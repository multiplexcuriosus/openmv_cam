import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE) # Must always be grayscale.
sensor.set_framesize(sensor.B320X320) # Must always be 320x320.
sensor.set_framerate(400)

sensor.ioctl(sensor.IOCTL_GENX320_SET_BIAS, sensor.GENX320_BIAS_DIFF_OFF, 28)
sensor.ioctl(sensor.IOCTL_GENX320_SET_BIAS, sensor.GENX320_BIAS_DIFF_ON, 25)
sensor.ioctl(sensor.IOCTL_GENX320_SET_BIAS, sensor.GENX320_BIAS_FO, 25)
sensor.ioctl(sensor.IOCTL_GENX320_SET_BIAS, sensor.GENX320_BIAS_HPF, 50)
sensor.ioctl(sensor.IOCTL_GENX320_SET_BIAS, sensor.GENX320_BIAS_REFR, 10)

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    print(clock.fps())
