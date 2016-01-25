#!/usr/bin/env python

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (52, 255, 255)
AREA = 1500

# (pressure in Mpa, angle in degrees)
GAUGE_MIN = {'pressure':0.0, 'angle':45.0}
GAUGE_MAX = {'pressure':3.5, 'angle':315.0}
NEEDLE_SHAPE = 4.0

THICKNESS = 4
WIDTH = 1000
HEIGHT = 1000
MAX_DISTANCE = 200