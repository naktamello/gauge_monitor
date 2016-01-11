#!/usr/bin/env python
import sys
import math
import numpy as np
import os, platform, subprocess, re
import cv2
#from matplotlib import pyplot as plt
# user modules
import defines as DEF
# import temp_sensor

import os, platform, subprocess, re


def get_processor_name():
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


def make_line(point_tuple):
    p1 = point_tuple[0]
    p2 = point_tuple[1]
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def erode_n_dilate(image):
    erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.erode(image, erode_element, 2)
    image = cv2.dilate(image, dilate_element, 1)
    return image


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def rad_to_degrees(radians):
    degrees = (radians * 360 / (2 * np.pi))
    return degrees


# def graph_histogram(in_image):
    # hist = cv2.calcHist([in_image], [0], None, [256], [0, 256])
    # plt.figure()
    # plt.title("hist")
    # plt.xlabel("bins")
    # plt.ylabel("pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()
    # cv2.waitKey(0)


class Gauge:
    x = 0  # these will hold the coordinates of the object "centers". see moments of inertia
    y = 0

    def __init__(self, name="null"):
        self.type = "Object"
        self.color = DEF.RED
        self.position = 0
        self.area = 0
        self.lines = []


def main():
    # checks to see if we are running on raspberry pi2
    processor = get_processor_name()
    print processor
    if "ARMv7" in processor:
        print "running on Raspberry Pi 2"
        import temp_sensor
        subprocess.call("gpio mode 1 pwm", shell=True)
        from picamera import PiCamera
        import picamera.array
        camera = PiCamera()
        camera.resolution = (720, 480)
        running_on_pi = True
    else:
        print "running on Desktop"
        running_on_pi = False

    # parameters
    draw = True
    hough_circle = True
    hough_line = True
    hough_pline = False
    do_blur = True
    do_threshold = True
    do_contours = True
    window_name = 'gauge'
    line_thickness = 2
    gauge_needle = Gauge()

    # setup image
    if running_on_pi:
        subprocess.call("gpio pwm 1 20", shell=True)
        camera.start_preview()
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            gauge_image = stream.array
        subprocess.call("gpio pwm 1 0", shell=True)
        im = gauge_image
    else:
        gauge_image = "./test.jpg"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        im = cv2.imread(gauge_image)

    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    prep = gray_im  # copy of grayscale image to be preprocessed for angle detection
    mask = np.zeros_like(prep)
    canvas = np.zeros_like(im)  # output image
    np.copyto(canvas, im)
    blank = np.zeros_like(im)
    height, width = gray_im.shape
    assert height == 480
    assert width == 720
    scale_factor = 1.15

    # preprocessing image:
    if do_blur:
        prep = cv2.GaussianBlur(prep, (3, 3), 2, 2)

    if do_threshold:
        # create structuring element that will be used to "dilate" and "erode" image.
        # the element chosen here is a 3px by 3px rectangle
        # prep = cv2.equalizeHist(prep)
        # prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        prep = cv2.threshold(prep, 30, 255, cv2.THRESH_BINARY_INV)[1]
        prep = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 3)
        # prep = cv2.Canny(prep, 20, 60)

    if not draw:
        cv2.imshow(window_name, prep)
        cv2.waitKey(0)
        sys.exit("terminating early")

    # Hough transforms (circles, lines, plines)
    if hough_circle:
        # For some reason HoughCircles returns output vector (x,y,radius)
        # inside one element list so [0] is appended at the end
        circles = \
            cv2.HoughCircles(gray_im, cv2.HOUGH_GRADIENT, 2, 10, None, param1=50, param2=30, minRadius=height / 10,
                             maxRadius=0, )[0]
        assert(circles is not None)
        # use only gauge image that is inside the first circle found
        for c in circles[:1]:
            # scale factor makes applicable area slightly larger just to be safe we are not missing anything
            cv2.circle(mask, (c[0], c[1]), int(c[2] * scale_factor), (255, 255, 255), -1)
            # mask=cv2.threshold(mask, 128, 255,cv2.THRESH_BINARY)
            prep = cv2.bitwise_and(prep, mask)

    if do_contours:
        objects = []
        copy = np.zeros_like(prep)  # output image
        np.copyto(copy, prep)
        _, contours, hierarchy = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None and len(contours) > 0:
            numObjects = len(contours)
            # print("objects = %d" % numObjects )
            if numObjects < 1024:
                for i in range(0, len(contours)):
                    moment = cv2.moments(contours[i])
                    area = moment['m00']
                    if area > DEF.AREA:
                        # print("area = %d" % area)
                        object = gauge_needle
                        object.area = area
                        object.x = int(moment['m10'] / area)
                        object.y = int(moment['m01'] / area)
                        object.position = i
                        objects.append(object)
                        # print len(objects)
                        # print "area is greater than .."
                        objectFound = True
                        # else: objectFound = False

        assert (len(objects) <= len(contours)), \
            "objects: %d, contours: %d" % (len(objects), len(contours))
        while (len(objects) > 0):
            thisObject = objects.pop()
            # contours[thisObject.position] = cv2.convexHull(contours[thisObject.position], returnPoints = True)
            cv2.drawContours(blank, contours, thisObject.position, DEF.WHITE, 5, 2)

    if hough_line:
        lines = cv2.HoughLines(prep, 3, np.pi / 90, 100)

    if hough_pline:
        plines = cv2.HoughLinesP(prep, 1, np.pi / 180, 20, None, minLineLength=height / 10, maxLineGap=1)

    first = True
    for c in circles[:1]:
        # green for circles (only draw the 1 strongest)
        cv2.circle(canvas, (c[0], c[1]), c[2], (0, 255, 0), line_thickness)
        # cv2.circle()
        # res = cv2.bitwise_and(frame,frame, mask= mask)
        if first:
            first = False
            circle_x = c[0]
            circle_y = c[1]
            radius = int(c[2])

    angles_in_radians = []
    angles_in_degrees = []
    if lines is None:
        sys.exit("there are no lines inside the cicle")  # if no lines are found terminate the program here
    else:
        for line in lines[:5]:
            for (rho, theta) in line:
                # blue for infinite lines (only draw the 2 strongest)
                x0 = np.cos(theta) * rho
                y0 = np.sin(theta) * rho
                pt1 = (int(x0 + (height + width) * (-np.sin(theta))), int(y0 + (height + width) * np.cos(theta)))
                pt2 = (int(x0 - (height + width) * (-np.sin(theta))), int(y0 - (height + width) * np.cos(theta)))
                gauge_needle.lines.append((pt1, pt2))
                angles_in_radians.append(theta)
                angles_in_degrees.append(rad_to_degrees(theta))

    number_of_angles = len(angles_in_degrees)
    for i in range(number_of_angles):
        cur_angle = angles_in_degrees[i]
        for j in range(number_of_angles):
            if j == i:
                continue
            angle_diff = abs(cur_angle - angles_in_degrees[j])
            print i, j, angle_diff
            if (angle_diff > (DEF.NEEDLE_SHAPE - 0.5)) and (angle_diff < (DEF.NEEDLE_SHAPE + 0.5)):
                angle_index = (i, j)
                break
        else:
            pass
        break

    assert (len(gauge_needle.lines) >= 2)
    assert (angle_index is not None)

    print "angle1: ", angles_in_degrees[angle_index[0]]
    print "angle2: ", angles_in_degrees[angle_index[1]]
    print "angle difference", angle_diff

    avg_theta = (angles_in_radians[angle_index[0]] + angles_in_radians[angle_index[1]]) / 2
    intersecting_pt = intersection(make_line(gauge_needle.lines[angle_index[0]]), make_line(gauge_needle.lines[angle_index[1]]))
    if intersecting_pt is not None:
        print "Intersection detected:", intersecting_pt
        cv2.circle(canvas, intersecting_pt, 10, DEF.RED, 3)
    else:
        print "No single intersection point detected"

    # Although we found a line coincident with the gauge needle,
    # we need to find which of the two direction it is pointing
    # guess1 and guess2 are 180 degrees apart
    guess1 = [avg_theta, (0, 0)]
    guess2 = [avg_theta + np.pi, (0, 0)]
    if guess2[0] > 2 * np.pi:   # in case adding pi made it greater than 2pi
        guess2[0] -= 2 * np.pi
    guess1[1] = (int(circle_x + radius * np.sin(guess1[0])), int(circle_y - radius * np.cos(guess1[0])))
    guess2[1] = (int(circle_x + radius * np.sin(guess2[0])), int(circle_y - radius * np.cos(guess2[0])))
    # find the distance between our guess and intersection of gauge needle lines
    # the guess that is closer to the intersection is the correct one
    dist1 = math.hypot(intersecting_pt[0] - guess1[1][0], intersecting_pt[1] - guess1[1][1])
    dist2 = math.hypot(intersecting_pt[0] - guess2[1][0], intersecting_pt[1] - guess2[1][1])
    if dist1 > dist2:
        needle_angle = rad_to_degrees(guess1[0])
    else:
        needle_angle = rad_to_degrees(guess2[0])
    if needle_angle >= 360:
        needle_angle -= 360

    pressure = np.interp(needle_angle, [DEF.GAUGE_MIN['angle'], DEF.GAUGE_MAX['angle']],
                         [DEF.GAUGE_MIN['pressure'], DEF.GAUGE_MAX['pressure']])
    print "pressure = ", pressure
    prep = cv2.bitwise_and(prep, mask)

    cv2.line(canvas, gauge_needle.lines[angle_index[0]][0], gauge_needle.lines[angle_index[0]][1], (255, 0, 0),
             line_thickness)
    cv2.line(canvas, gauge_needle.lines[angle_index[1]][0], gauge_needle.lines[angle_index[1]][1], (255, 0, 0),
             line_thickness)
    cv2.putText(canvas, "pressure:" + str(round(pressure, 3)) + " MPa", (int(height * 0.4), int(width * 0.6)),
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=(height / 200), color=(200, 50, 0), thickness=3)

    if hough_pline:
        for pline in plines[:10]:
            for l in pline:
                # red for line segments
                cv2.line(canvas, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), line_thickness)

    if draw:
        canvas = np.concatenate((im, canvas), axis=1)
        image_to_show = canvas
    else:
        image_to_show = prep

    if not running_on_pi:
        cv2.imshow(window_name, image_to_show)
        cv2.waitKey(0)

    # save the resulting image
    if draw:
        cv2.imwrite("res.jpg", canvas)

    f = open('./gauge_angle.txt', 'w')
    f.write("angle: " + str(needle_angle) + "degrees")
    f.write("pressure: " + str(pressure) + "MPa")
    f.close()


if __name__ == "__main__":
    main()
