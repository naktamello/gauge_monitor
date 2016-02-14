#!/usr/bin/env python
import sys, re, os, math
import numpy as np
import cv2
# user modules
import defines as DEF
import exception_handler as EX
import histogram_analysis as hist_an
import hough_transforms
import geometry
import process_image
import housekeeping


def main():
    # parameters
    files = []
    for i in range(18, 36):
        files.append(str(i).zfill(3))
    # instance of OS class that contains all interactions with host machine

    first = True
    # files = ['010']
    for file_name in files:
        image_path = re.sub('file', str(file_name), DEF.OUTPUT_PATH)
        im = cv2.imread(image_path)
        if first:
            canvas = np.zeros_like(im)
            np.copyto(canvas, im)
            first = False
        else:
            canvas = np.concatenate((canvas, im), axis=0)

    # cv2.namedWindow("stitched", cv2.WINDOW_NORMAL)
    # cv2.imshow("stitched", canvas)
    # cv2.waitKey(0)

    cv2.imwrite('./stitched.jpg',canvas)

    return True


if __name__ == "__main__":
    main()