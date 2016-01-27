#!/usr/bin/env python
import sys
import numpy as np
import cv2
import defines as DEF
import exception_handler as EX


class Gauge:
    x = 0  # these will hold the coordinates of the object "centers". see moments of inertia
    y = 0
    angles_in_radians = []
    angles_in_degrees = []

    def __init__(self, name="null"):
        self.type = "Object"
        self.color = DEF.RED
        self.position = (0,0)
        self.area = 0
        self.lines = []


def erode_n_dilate(image):
    erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.erode(image, erode_element, 2)
    image = cv2.dilate(image, dilate_element, 1)
    return image


def blur_n_threshold(prep,do_blur=0, do_threshold=1):
    # preprocessing image:
    if do_blur:
        prep = cv2.GaussianBlur(prep, (3, 3), 2, 2)

    if do_threshold:
        # create structuring element that will be used to "dilate" and "erode" image.
        # the element chosen here is a 3px by 3px rectangle
        # prep = cv2.equalizeHist(prep)
        # prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        prep = cv2.threshold(prep, 65, 255, cv2.THRESH_BINARY_INV)[1]
        #prep = erode_n_dilate(prep)
        #prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 3)
        # prep = cv2.Canny(prep, 20, 60)
        #prep = erode_n_dilate(prep)

    return prep


def find_contour(this_object, img, draw=False):
    objects = []
    copy = np.zeros_like(img)  # output image
    np.copyto(copy, img)

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None and len(contours) > 0:
        num_objects = len(contours)
        # print("objects = %d" % numObjects )
        if num_objects < 1024:
            for i in range(0, len(contours)):
                moment = cv2.moments(contours[i])
                area = moment['m00']
                if area > DEF.AREA:
                    # print("area = %d" % area)
                    object = Gauge()
                    object.area = area
                    object.x = int(moment['m10'] / area)
                    object.y = int(moment['m01'] / area)
                    object.position = i
                    objects.append(object)
                    # print len(objects)
                    # print "area is greater than .."
                    objectFound = True
                    # else: objectFound = False
    if draw:
        assert (len(objects) <= len(contours)), \
        "objects: %d, contours: %d" % (len(objects), len(contours))
        while (len(objects) > 0):
            this_contour = objects.pop()
            # contours[thisObject.position] = cv2.convexHull(contours[thisObject.position], returnPoints = True)
            cv2.drawContours(img, contours, this_contour.position, DEF.WHITE, 5, 2)
    return img