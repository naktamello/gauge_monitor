#!/usr/bin/python2.7

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (52, 255, 255)
AREA = 1000

#(pressure in Mpa, angle in degrees)
GAUGE_MIN = {'pressure':0.0, 'angle':45.0}
GAUGE_MAX = {'pressure':3.5, 'angle':315.0}
NEEDLE_SHAPE = 6.0
THICKNESS = 4
WIDTH = 720
MAX_DISTANCE = 200