#!/usr/bin/env python
import os
import glob
import time

device_file = None
initialized = False
def initialize():
    global device_file
    global initialized
    os.system('modprobe w1-gpio')
    os.system('modprobe w1-therm')
    base_dir = '/sys/bus/w1/devices/'
    device_folder = glob.glob(base_dir + '28*')[0]
    device_file = device_folder + '/w1_slave'
    initialized = True


def read_temp_raw():
    if device_file:
        f = open(device_file, 'r')
        lines = f.readlines()
        f.close()
        return lines
    else:
        return None


def read_temp():
    if initialized:
        lines = read_temp_raw()
        while lines[0].strip()[-3:] != 'YES':
            time.sleep(0.2)
            lines = read_temp_raw()
        equals_pos = lines[1].find('t=')
        if equals_pos != -1:
            temp_string = lines[1][equals_pos+2:]
            temp_c = float(temp_string) / 1000.0
            return temp_c
    else:
        return "OFF"

if __name__ == "__main__":
    while True:
        print(read_temp())
        time.sleep(1)
