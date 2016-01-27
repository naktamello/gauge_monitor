#!/usr/bin/env python
import sys
import numpy as np
import cv2
import defines as DEF
import exception_handler as EX

width = DEF.WIDTH
height = DEF.HEIGHT
error_screen = np.ndarray(shape=(width,height))

def hough_c(gray_im):
        # HoughCircles returns output vector [(x,y,radius)]
    # one element list so [0] is appended at the end
    circles = \
        cv2.HoughCircles(gray_im, cv2.HOUGH_GRADIENT, 2, 10, None, param1=50, param2=30, minRadius=120,
                         maxRadius=width-100, )[0]
    if circles is None:
        EX.error_output(error_screen, err_msg="no circles found")
        sys.exit("no circles found")
    # use only gauge image that is inside the first circle found
    return circles


def hough_l(prep):
    lines = cv2.HoughLines(prep, 4, np.pi / 360, 600)
    if lines is None:
        EX.error_output(error_screen, err_msg="no lines found")
        sys.exit("there are no lines inside the circle")  # if no lines are found terminate the program here
    return lines


def hough_pl(prep):
    plines = cv2.HoughLinesP(prep, 1, np.pi / 180, 20, None, minLineLength=height / 10, maxLineGap=1)
    return plines