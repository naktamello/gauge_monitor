#!/usr/bin/env python

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (52, 255, 255)
AREA = 1000

# (pressure in Mpa, angle in degrees)
GAUGE_MIN = {'pressure':0.0, 'angle':42.5}
GAUGE_MAX = {'pressure':3.5, 'angle':312.0}
NEEDLE_SHAPE = 4.0

THICKNESS = 4
WIDTH = 1000
HEIGHT = 1000
KERNEL = (15,15)
MAX_DISTANCE = 150
CIRCLE_SCALE_FACTOR = 1.15

SAMPLE_PATH = "./img_samples/file.jpg"
OUTPUT_PATH = "./img_output/file.jpg"
THRESH_PATH = "./img_output/thresh_output/file.jpg"