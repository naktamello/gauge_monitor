#!/usr/bin/env python
import sys
import math
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

try:
    from matplotlib import pyplot as plt
except ImportError:
    "matplotlib_pyplot not found"

window_name = 'gauge'

def display_values(canvas, display_value):
    display_unit = "MPa"
    font_size = (DEF.HEIGHT / 80)
    hor_offset = font_size * 12 * sum(c.isdigit() for c in display_value)
    ver_offset = 0
    value_position = (int(DEF.WIDTH * 0.5), int(DEF.HEIGHT * 0.2))
    unit_position = (value_position[0] + hor_offset, value_position[1])
    cv2.putText(canvas, display_value, value_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size, color=DEF.RED, thickness=10)
    cv2.putText(canvas, display_unit, unit_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size / 2, color=DEF.RED, thickness=4)


def main():
    # parameters
    gauge = process_image.Gauge()

    # instance of OS class that contains all interactions with host machine
    host = housekeeping.OS()

    # setup image
    reuse_circle = housekeeping.test_time()
    im = host.get_image()
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    prep = gray_im  # copy of grayscale image to be preprocessed for angle detection
    mask = np.zeros_like(prep)
    canvas = np.zeros_like(im)  # output image
    np.copyto(canvas, im)
    error_screen = np.zeros_like(im)
    height, width = gray_im.shape
    assert height == DEF.HEIGHT
    assert width == DEF.WIDTH

    # mask_bandpass is a mask of the gauge obtained from histogram peaks
    hist0, mask_bandpass = hist_an.bandpass(gray_im)
    # prep is binary image ready for line detection
    prep = process_image.blur_n_threshold(prep)
    host.write_image(("mask_bandpass.jpg", mask_bandpass))
    # Hough transforms
    if reuse_circle:
        print "reusing previous circle"
        circle_x, circle_y, radius = housekeeping.return_circle()
    else:
        circles = hough_transforms.hough_c(gray_im)
        for c in circles[:1]:
            # scale factor makes applicable area slightly larger just to be safe we are not missing anything
            cv2.circle(mask, (c[0], c[1]), int(c[2] * DEF.CIRCLE_SCALE_FACTOR), (255, 255, 255), -1)
            prep = cv2.bitwise_and(prep, mask)
            cv2.circle(canvas, (c[0], c[1]), c[2], (0, 255, 0), DEF.THICKNESS)
            circle_x = c[0]
            circle_y = c[1]
            radius = int(c[2])

    cv2.circle(mask, (int(circle_x), int(circle_y)), int(radius * DEF.CIRCLE_SCALE_FACTOR), (255, 255, 255), -1)
    cv2.circle(canvas, (int(circle_x), int(circle_y)), radius, (0, 255, 0), DEF.THICKNESS)
    prep = cv2.bitwise_and(prep, mask)

    lines = hough_transforms.hough_l(prep)

    prep = cv2.bitwise_and(mask_bandpass, prep)

    for line in lines[:10]:
        for (rho, theta) in line:
            # blue for infinite lines (only draw the 2 strongest)
            x0 = np.cos(theta) * rho
            y0 = np.sin(theta) * rho
            pt1 = (int(x0 + (height + width) * (-np.sin(theta))), int(y0 + (height + width) * np.cos(theta)))
            pt2 = (int(x0 - (height + width) * (-np.sin(theta))), int(y0 - (height + width) * np.cos(theta)))
            gauge.lines.append((pt1, pt2))
            gauge.angles_in_radians.append(theta)
            gauge.angles_in_degrees.append(geometry.rad_to_degrees(theta))
            # cv2.line(prep_rgb, pt1, pt2, (255, 0, 0), DEF.THICKNESS / 2)
    angle_index = geometry.get_angles(gauge.angles_in_degrees)

    if len(gauge.lines) < 2:
        EX.error_output(error_screen, err_msg="unable to find needle from lines")
        sys.exit("unable to find needle form lines")

    line_found = False
    for this_pair in angle_index:
        # print "angle1: ", gauge.angles_in_degrees[this_pair[0]]
        # print "angle2: ", gauge.angles_in_degrees[this_pair[1]]

        line1 = geometry.make_line(gauge.lines[this_pair[0]])
        line2 = geometry.make_line(gauge.lines[this_pair[1]])
        intersecting_pt = geometry.intersection(line1, line2)
        if intersecting_pt is not None:
            print "Intersection detected:", intersecting_pt
            cv2.circle(canvas, intersecting_pt, 10, DEF.RED, 3)
        else:
            print "No single intersection point detected"

        # Although we found a line coincident with the gauge needle,
        # we need to find which of the two direction it is pointing
        # guess1 and guess2 are 180 degrees apart
        avg_theta = (gauge.angles_in_radians[this_pair[0]] + gauge.angles_in_radians[this_pair[1]]) / 2
        guess1 = [avg_theta, (0, 0)]
        guess2 = [avg_theta + np.pi, (0, 0)]
        if guess2[0] > 2 * np.pi:  # in case adding pi made it greater than 2pi
            guess2[0] -= 2 * np.pi
        print "guess1: ", guess1[0]
        print "guess2: ", guess2[0]
        guess1[1] = (int(circle_x + radius * np.sin(guess1[0])), int(circle_y - radius * np.cos(guess1[0])))
        guess2[1] = (int(circle_x + radius * np.sin(guess2[0])), int(circle_y - radius * np.cos(guess2[0])))
        # find the distance between our guess and intersection of gauge needle lines
        # the guess that is closer to the intersection is the correct one
        dist1 = math.hypot(intersecting_pt[0] - guess1[1][0], intersecting_pt[1] - guess1[1][1])
        dist2 = math.hypot(intersecting_pt[0] - guess2[1][0], intersecting_pt[1] - guess2[1][1])
        print "dist1: ", dist1
        print "dist2: ", dist2
        if dist1 < DEF.MAX_DISTANCE or dist2 < DEF.MAX_DISTANCE:
            line_found = True
            if dist1 < dist2:
                correct_guess = guess1
            else:
                correct_guess = guess2

        if line_found is True:
            cv2.circle(canvas, guess1[1], 10, DEF.GREEN, 3)
            cv2.circle(canvas, guess2[1], 10, DEF.BLUE, 3)
            cv2.line(canvas, gauge.lines[this_pair[0]][0], gauge.lines[this_pair[0]][1], (255, 0, 0),
                     DEF.THICKNESS)
            cv2.line(canvas, gauge.lines[this_pair[1]][0], gauge.lines[this_pair[1]][1], (255, 0, 0),
                     DEF.THICKNESS)
            break

    if line_found is False:
        EX.error_output(error_screen, err_msg="no positive pair")
        needle_angle = 0.0
    else:
        ref_offset = np.pi
        needle_angle = geometry.rad_to_degrees(correct_guess[0] - ref_offset)
    print "correct angle(cv2)= ", needle_angle
    if needle_angle < 0.0:
        needle_angle += 360
    print "needle_angle: ", needle_angle
    pressure = np.interp(needle_angle, [DEF.GAUGE_MIN['angle'], DEF.GAUGE_MAX['angle']],
                         [DEF.GAUGE_MIN['pressure'], DEF.GAUGE_MAX['pressure']])
    pressure_str = str(round(pressure, 2))
    print "pressure = ", pressure
    display_values(canvas, pressure_str)

    prep = cv2.bitwise_and(prep, mask)
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.subplot(232)
    plt.imshow(255 - mask_bandpass, 'gray')
    plt.subplot(233)
    plt.imshow(prep, 'gray')
    plt.subplot(234)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.subplot(235)
    plt.plot(hist0)
    plt.xlim([0, 256])
    plt.ylim([0, 50000])
    plt.show()
    host.write_to_file('w', "Pressure: " + pressure_str + " MPa", "Temperature: " + str(host.read_temp()))
    host.write_to_file('a', "circle_x:" + str(circle_x), "circle_y:" + str(circle_y), "radius:" + str(radius))
    # pass in tuples ("filename.ext", img_to_write)
    # host.write_image(("prep.jpg", prep), ("canvas.jpg", canvas), ("orig_im.jpg", im))
    prep = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate((prep, canvas), axis=1)
    # host.show_image(canvas)
    # cv2.waitKey(0)

    return True


if __name__ == "__main__":
    main()
