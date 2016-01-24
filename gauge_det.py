#!/usr/bin/env python
import sys
import math
import numpy as np
import time
import os, platform, subprocess, re
import cv2
# user modules
import defines as DEF
import temp_sensor
# rasbperry modules
try:
    from picamera import PiCamera
except ImportError:
    "picamera not found"
try:
    import picamera.array
except ImportError:
    "picamera.array not found"
try:
    from matplotlib import pyplot as plt
except ImportError:
    "matplotlib_pyplot not found"


def error_output(img,err_msg="ERROR", height=480, width=720):
    font_size = (height / 100)
    value_position = (int(width*0.1), int(height*0.5))
    cv2.putText(img, err_msg, value_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size, color=DEF.RED, thickness=10)
    cv2.imwrite("canvas.jpg", img)

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


def get_angles(angle_vals):
    possible_angles = []
    for i in range(len(angle_vals)):
        cur_angle = angle_vals[i]
        for j in range(len(angle_vals)):
            if j == i:
                continue
            angle_diff = abs(cur_angle - angle_vals[j])
            print i, j, angle_diff
            if (angle_diff > (DEF.NEEDLE_SHAPE - 0.5)) and (angle_diff < (DEF.NEEDLE_SHAPE + 0.5)):
                possible_angles.append((i, j))
    return possible_angles

def preprocess_img(prep,do_blur=0, do_threshold=1):
    # preprocessing image:
    if do_blur:
        prep = cv2.GaussianBlur(prep, (3, 3), 2, 2)

    if do_threshold:
        # create structuring element that will be used to "dilate" and "erode" image.
        # the element chosen here is a 3px by 3px rectangle
        prep = cv2.equalizeHist(prep)
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


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise (ValueError,
                'Input vectors y_axis and x_axis must have same length')

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


# https://gist.github.com/sixtenbe/1178136
def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def bandpass_filter(image, peak, height=480,width=720, peak_width = 10):
    mask = np.zeros_like(image)
    for x, y in np.ndindex((height, width)):
        if (image[x,y] < peak - peak_width/2) or (image[x,y] >  peak + peak_width/2):
            mask[x,y] = 255
    return mask


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


class OS:
    def __init__(self):
        self.processor = self.get_processor_name(self)
        if "ARMv7" in self.processor:
            print "running on Raspberry Pi 2"
            #subprocess.call("~/rpi-fbcp/build/fbcp &", shell=True)
            subprocess.call("gpio mode 4 out", shell=True)
            self.camera = PiCamera()
            self.camera.resolution = (720, 480)
            temp_sensor.initialize()
            self.temperature = temp_sensor.read_temp()
            self.running_on_pi = True
        else:
            print "running on Desktop"
            self.temperature = None
            self.running_on_pi = False

    def read_temp(self):
        self.temperature = temp_sensor.read_temp()
        return self.temperature

    def get_image(self):
        if self.running_on_pi:
            subprocess.call("gpio write 4 1", shell=True)
            time.sleep(0.1)
            #camera.start_preview()
            with picamera.array.PiRGBArray(self.camera) as stream:
                self.camera.capture(stream, format='bgr')
                gauge_image = stream.array
            subprocess.call("gpio write 4 0", shell=True)
            return gauge_image
        else:
            image_path = "./test.jpg"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            im = cv2.imread(image_path)
            return im

    def show_image(self, img):
        if self.running_on_pi:
            #if os.path.exists("./canvas.jpg"):
                #subprocess.call("sudo fbi -T 1 canvas.jpg", shell=True)
            pass
        else:
            cv2.imshow(window_name, img)
            cv2.waitKey(0)

    def write_image(self, *img_tuple):
        imgs = list(img_tuple)
        for img in imgs:
            cv2.imwrite(img[0], img[1])

    def write_to_file(self, *input_text):
        lines = list(input_text)
        f = open('./gauge_angle.txt', 'w')
        for line in lines:
            f.write(line+"\n")
        f.close()

    @staticmethod
    def get_processor_name(self):
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

window_name = 'gauge'

def main():
    # parameters
    draw = True
    hough_circle = True
    hough_line = True
    hough_pline = False
    gauge = Gauge()

    host = OS()
    # setup image
    im = host.get_image()
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    prep = gray_im  # copy of grayscale image to be preprocessed for angle detection
    prep_history = []
    mask = np.zeros_like(prep)
    mask_bandpass = np.zeros_like(prep)
    canvas = np.zeros_like(im)  # output image
    np.copyto(canvas, im)
    blank = np.zeros_like(im)
    error_screen = np.zeros_like(im)
    height, width = gray_im.shape
    assert height == 480
    assert width == 720
    scale_factor = 1.15

    # prepare image for circle and line detection
    prep = preprocess_img(prep)
    hist = cv2.calcHist([gray_im],[0],None,[256],[0,256])
    # hist = plt.hist(gray_im.ravel(),256,[0,256])
    # plt.show()
    hist_list = list(hist)
    [max, min] = peakdetect(hist, lookahead=10, delta=150)
    # first element of max is the gauge needle (darkest peak excluding 0)
    assert (max is not None and (max[0][0] > 0 and max[0][0] < 50))
    print "max: ", max
    #plt.savefig("hist.png")
    # for x, y in np.ndindex((height,width)):
    #     if (gray_im[x,y] > 45 - 20) and (gray_im[x,y] <  45 + 20):
    #         mask_debug[x,y] = 255
    # for x, y in np.ndindex((height,width)):
    #     if (gray_im[x,y] > 120 - 20) and (gray_im[x,y] <  120 + 20):
    #         mask_debug[x,y] = 255
    mask_bandpass = bandpass_filter(gray_im,max[0][0], peak_width=15)

    #cv2.imshow(window_name, mask_debug)
    #cv2.waitKey(0)

    if not draw:
        cv2.imshow(window_name, prep)
        cv2.waitKey(0)
        sys.exit("terminating early")

    #error_output(blank)

    # Hough transforms (circles, lines, plines)
    if hough_circle:
        # HoughCircles returns output vector [(x,y,radius)]
        # one element list so [0] is appended at the end
        circles = \
            cv2.HoughCircles(gray_im, cv2.HOUGH_GRADIENT, 2, 10, None, param1=50, param2=30, minRadius=120,
                             maxRadius=180, )[0]
        if circles is None:
            error_output(error_screen, err_msg="no circles found")
            sys.exit("no circles found")
        # use only gauge image that is inside the first circle found
        for c in circles[:1]:
            # scale factor makes applicable area slightly larger just to be safe we are not missing anything
            cv2.circle(mask, (c[0], c[1]), int(c[2] * scale_factor), (255, 255, 255), -1)
            # mask=cv2.threshold(mask, 128, 255,cv2.THRESH_BINARY)
            prep = cv2.bitwise_and(prep, mask)

    #prep = find_contour(gauge, prep, draw=True)
    if hough_line:
        lines = cv2.HoughLines(prep, 4, np.pi / 360, 600)

    if hough_pline:
        plines = cv2.HoughLinesP(prep, 1, np.pi / 180, 20, None, minLineLength=height / 10, maxLineGap=1)

    first = True
    for c in circles[:1]:
        # green for circles (only draw the 1 strongest)
        cv2.circle(canvas, (c[0], c[1]), c[2], (0, 255, 0), DEF.THICKNESS)
        # cv2.circle()
        # res = cv2.bitwise_and(frame,frame, mask= mask)
        if first:
            first = False
            circle_x = c[0]
            circle_y = c[1]
            radius = int(c[2])

    print "radius: ", radius
    print "width: ", width

    #prep=cv2.Canny(prep,25,155)
    mask_bandpass = (255 - mask_bandpass)
    prep=cv2.bitwise_and(mask_bandpass, prep)
    prep_rgb=cv2.cvtColor(prep,cv2.COLOR_GRAY2BGR)
    if lines is None:
        error_output(error_screen, err_msg="no lines found")
        sys.exit("there are no lines inside the cicle")  # if no lines are found terminate the program here
    else:
        for line in lines[:10]:
            for (rho, theta) in line:
                # blue for infinite lines (only draw the 2 strongest)
                x0 = np.cos(theta) * rho
                y0 = np.sin(theta) * rho
                pt1 = (int(x0 + (height + width) * (-np.sin(theta))), int(y0 + (height + width) * np.cos(theta)))
                pt2 = (int(x0 - (height + width) * (-np.sin(theta))), int(y0 - (height + width) * np.cos(theta)))
                gauge.lines.append((pt1, pt2))
                gauge.angles_in_radians.append(theta)
                gauge.angles_in_degrees.append(rad_to_degrees(theta))
                cv2.line(prep_rgb,pt1,pt2, (255, 0, 0), DEF.THICKNESS/2)
    number_of_angles = len(gauge.angles_in_degrees)
    angle_index = get_angles(gauge.angles_in_degrees)
    angle_set = set(tuple(sorted(l)) for l in angle_index)
    print "lengh: ", len(angle_index)
    print
    if len(gauge.lines) < 2:
        error_output(error_screen, err_msg="unable to find needle from lines")
        sys.exit("unable to find needle form lines")

    line_found = False
    for this_pair in angle_index:
        print "angle1: ", gauge.angles_in_degrees[this_pair[0]]
        print "angle2: ", gauge.angles_in_degrees[this_pair[1]]

        line1 = make_line(gauge.lines[this_pair[0]])
        line2 = make_line(gauge.lines[this_pair[1]])
        intersecting_pt = intersection(line1, line2)
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
        if guess2[0] > 2 * np.pi:   # in case adding pi made it greater than 2pi
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
        error_output(error_screen, err_msg="no positive pair")
        needle_angle = 0.0
    else:
        ref_offset = np.pi
        needle_angle = rad_to_degrees(correct_guess[0]-ref_offset)
    print "correct angle(cv2)= ", needle_angle
    if needle_angle < 0.0:
        needle_angle += 360
    print "needle_angle: ", needle_angle
    pressure = np.interp(needle_angle, [DEF.GAUGE_MIN['angle'], DEF.GAUGE_MAX['angle']],
                         [DEF.GAUGE_MIN['pressure'], DEF.GAUGE_MAX['pressure']])
    print "pressure = ", pressure
    prep = cv2.bitwise_and(prep, mask)

    display_value = str(round(pressure, 2))
    display_unit = "MPa"
    font_size = (height / 80)
    hor_offset = font_size * 12 * sum(c.isdigit() for c in display_value)
    ver_offset = 0
    value_position = (int(width*0.5), int(height*0.2))
    unit_position = (value_position[0]+hor_offset, value_position[1])
    cv2.putText(canvas, display_value, value_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size, color=DEF.RED, thickness=10)
    cv2.putText(canvas, display_unit, unit_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size/2, color=DEF.RED, thickness=4)
    if hough_pline:
        for pline in plines[:10]:
            for l in pline:
                # red for line segments
                cv2.line(canvas, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), DEF.THICKNESS)

    plt.subplot(221)
    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.imshow(prep, 'gray')
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    plt.subplot(224)
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.ylim([0,7500])
    plt.show()
    host.write_to_file("Pressure: " + display_value + " MPa", "Temperature: " + str(host.read_temp()))
    # pass in tuples ("filename.ext", img_to_write)
    #host.write_image(("prep.jpg", prep), ("canvas.jpg", canvas), ("orig_im.jpg", im))
    prep = cv2.cvtColor(prep,cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate((prep, canvas), axis=1)
    canvas = np.concatenate((prep_rgb, canvas), axis=1)
    #host.show_image(canvas)
    #time.sleep(3)

    return True
if __name__ == "__main__":
    main()
