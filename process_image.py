#!/usr/bin/env python
import sys, glob, re, os
import numpy as np
import cv2
import defines as DEF
import exception_handler as EX


class Gauge:
    @staticmethod
    def import_needle():
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            directory = str(here + "/img_samples")
            os.chdir(directory)
            needle_png = str(glob.glob("*.png")[0])
        except ImportError:
            "needle image not found"
        assert (needle_png)
        pattern = re.compile(r"""\(\((?P<n1>.*?)\,(?P<n2>.*?)\)\,\((?P<n3>.*?)\,(?P<n4>.*?)\)\)\.*""")
        match = pattern.match(needle_png)
        size_x = int(match.group("n1"))
        size_y = int(match.group("n2"))
        size = (size_x, 2 * size_y)  # image is only half of the gauge
        orig_length = int(match.group("n3"))
        # center_x = int(match.group("n3"))
        # center_y = size_y  # int(match.group("n4"))
        # center = (center_x, center_y)

        color_bottom_half = cv2.imread(directory + "/" + needle_png)
        gray_bottom_half = cv2.cvtColor(color_bottom_half, cv2.COLOR_BGR2GRAY)
        gray_upper_half = cv2.flip(gray_bottom_half, 0)
        gray_needle = np.concatenate((gray_upper_half, gray_bottom_half), axis=0)
        needle_height, needle_width = gray_needle.shape
        longer_side = needle_width - orig_length
        shorter_side = orig_length
        assert (longer_side > shorter_side)
        gray_needle = cv2.bitwise_not(gray_needle)
        horizontal_pad = np.zeros((2 * size_y, longer_side - shorter_side), dtype=np.uint8)
        needle_im = np.concatenate((horizontal_pad, gray_needle), axis=1)
        needle_height, needle_width = needle_im.shape
        vertical_pad = np.zeros(((2 * longer_side - 2 * size_y) / 2, needle_width), dtype=np.uint8)
        needle_im = np.concatenate((vertical_pad, needle_im), axis=0)
        needle_im = np.concatenate((needle_im, vertical_pad), axis=0)
        # expected format:(int, int),(int, int).png
        needle_height, needle_width = needle_im.shape
        center_x = needle_width / 2
        center_y = needle_height / 2
        center = (center_x, center_y)

        return needle_im, size, center

    def update_needle(self, cv2_angle):
        rotation = 270 - cv2_angle
        rot = cv2.getRotationMatrix2D((self.needle_center[1], self.needle_center[0]), rotation, 1)
        self.rotated = cv2.warpAffine(self.needle_im, rot, self.needle_im.shape)

    def get_needle(self, center):
        needle_height, needle_width = self.needle_im.shape
        needle_top_left_x = 0
        needle_top_left_y = 0
        needle_bottom_right_x = needle_width
        needle_bottom_right_y = needle_height
        top_left_x = center[0] - self.needle_center[0]
        if top_left_x < 0:
            needle_top_left_x = abs(top_left_x)
            top_left_x = 0
        top_left_y = center[1] - self.needle_center[1]
        if top_left_y < 0:
            needle_top_left_y = abs(top_left_y)
            top_left_y = 0
        bottom_right_x = top_left_x + needle_width
        if bottom_right_x > DEF.WIDTH:
            needle_bottom_right_x = needle_bottom_right_x - (bottom_right_x - DEF.WIDTH)
            bottom_right_x = DEF.WIDTH
        bottom_right_y = top_left_y + needle_height
        if bottom_right_y > DEF.HEIGHT:
            needle_bottom_right_y = needle_bottom_right_y - (bottom_right_y - DEF.HEIGHT)
            bottom_right_y = DEF.HEIGHT
        self.needle_canvas[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = self.rotated[
                                                                                   needle_top_left_y:needle_bottom_right_y,
                                                                                   needle_top_left_x:needle_bottom_right_x]

    def __init__(self, name="null"):
        self.needle_im, self.needle_size, self.needle_center = self.import_needle()
        self.needle_canvas = np.zeros((DEF.HEIGHT, DEF.WIDTH), dtype=np.uint8)
        self.rotated = np.zeros_like(self.needle_im)
        self.needle_angle = 0
        self.type = "Object"
        self.color = DEF.RED
        self.position = (0, 0)
        self.area = 0
        self.lines = []
        self.x = 0  # these will hold the coordinates of the object "centers". see moments of inertia
        self.y = 0
        self.angles_in_radians = []
        self.angles_in_degrees = []


def erode_n_dilate(image):
    erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DEF.KERNEL)
    dilate_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DEF.KERNEL)
    image = cv2.erode(image, erode_element, 2)
    image = cv2.dilate(image, dilate_element, 1)
    return image


def blur_n_threshold(prep, do_blur=0, do_threshold=1):
    # preprocessing image:
    if do_blur:
        prep = cv2.GaussianBlur(prep, (15, 15), 0)

    if do_threshold:
        # create structuring element that will be used to "dilate" and "erode" image.
        # the element chosen here is a 3px by 3px rectangle
        # prep = cv2.equalizeHist(prep)
        # prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        prep = cv2.threshold(prep, 65, 255, cv2.THRESH_BINARY_INV)[1]
        # prep = erode_n_dilate(prep)
        # prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 3)
        # prep = cv2.Canny(prep, 20, 60)
        # prep = erode_n_dilate(prep)

    return prep


def find_contour(img, draw=False):
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
            # epsilon = 0.001*cv2.arcLength(contours[this_contour.position],False)
            # contours[this_contour.position] = cv2.approxPolyDP(contours[this_contour.position],epsilon,False)
            # contours[this_contour.position] = cv2.convexHull(contours[this_contour.position], returnPoints = True)
            cv2.drawContours(img, contours, this_contour.position, DEF.WHITE, 3, 2)
    return img
