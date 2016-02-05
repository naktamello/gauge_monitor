#!/usr/bin/env python
import time
import cv2
import os, platform, subprocess, re
import temp_sensor
import defines as DEF
from datetime import datetime

# rasbperry modules
try:
    from picamera import PiCamera
except ImportError:
    "picamera not found"
try:
    import picamera.array
except ImportError:
    "picamera.array not found"


def test_time():
    f=open('gauge_data.txt','r')
    f=list(f)
    for line in f[:1]:
        t1 = datetime.strptime(line.rstrip(), "%Y-%m-%d %H:%M:%S.%f")
        diff = datetime.now() - t1
        if diff.total_seconds() < 3600:
            return True
    return False


def return_circle():
    with open('gauge_data.txt') as searchfile:
        for line in searchfile:
            _, sep, right = line.partition('circle_x')
            if sep: # True iff 'Figure' in line
                circle_x = float(right[1:].rstrip())
            _, sep, right = line.partition('circle_y')
            if sep: # True iff 'Figure' in line
                circle_y = float(right[1:].rstrip())
            _, sep, right = line.partition('radius')
            if sep: # True iff 'Figure' in line
                radius = int(right[1:].rstrip())
    return circle_x, circle_y, radius

class OS:
    def __init__(self):
        self.processor = self.get_processor_name(self)
        if "ARMv7" in self.processor:
            print "running on Raspberry Pi 2"
            #subprocess.call("~/rpi-fbcp/build/fbcp &", shell=True)
            subprocess.call("gpio mode 4 out", shell=True)
            self.camera = PiCamera()
            self.camera.resolution = (1000, 1000)
            temp_sensor.initialize()
            self.temperature = temp_sensor.read_temp()
            self.running_on_pi = True
        else:
            print "running on Desktop"
            self.temperature = None
            self.running_on_pi = False

    def read_temp(self):
        self.temperature = temp_sensor.read_temp()
        return self.temperature

    def get_image(self):
        if self.running_on_pi:
            subprocess.call("gpio write 4 1", shell=True)
            time.sleep(0.1)
            #camera.start_preview()
            with picamera.array.PiRGBArray(self.camera) as stream:
                self.camera.capture(stream, format='bgr')
                gauge_image = stream.array
            subprocess.call("gpio write 4 0", shell=True)
            return gauge_image
        else:
            image_path = DEF.IMG_PATH
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            im = cv2.imread(image_path)
            return im

    def show_image(self, img):
        if self.running_on_pi:
            if os.path.exists("./canvas.jpg"):
                subprocess.call("sudo fbi -T 1 canvas.jpg", shell=True)
            pass
        else:
            pass
            # cv2.imshow(window_name, img)
            # cv2.waitKey(0)

    def write_image(self, *img_tuple):
        imgs = list(img_tuple)
        for img in imgs:
            cv2.imwrite(img[0], img[1])

    def write_to_file(self, mode, *input_text):
        lines = list(input_text)
        f = open('./gauge_data.txt', mode)
        if mode is 'w':
            f.write(str(datetime.now())+"\n")
        for line in lines:
            f.write(line+"\n")
        f.close()

    @staticmethod
    def get_processor_name(self):
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            import os
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1)
        return ""
