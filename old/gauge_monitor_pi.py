#!/usr/bin/python2.7


#downloaded modules
import numpy as np
import cv2
import cv2.cv as cv
import time
#user module
import email_from_monitor as email
import temp_sensor
def main():
    #parameters
    file = "./test.jpg"
    draw = True
    hough_circle = True
    hough_line = True
    hough_pline = True
    do_canny = False
    do_blur = True
    do_threshold = True
    window_name = 'gauge'
    line_thickness = 2

    #setup
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    im = cv2.imread(file)
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    canvas = np.zeros_like(im)
    canvas = np.array(im)
    prep = gray_im
    m,n = gray_im.shape
    mask = np.zeros_like(prep)
    scale_factor = 1.15
    #preprocessing image
    if do_blur:
        prep = cv2.GaussianBlur(prep, (3,3), 2,2)
    if do_canny:
        prep = cv2.Canny(prep, 20, 60)
    if do_threshold:
        #create structuring element that will be used to "dilate" and "erode" image.
        #the element chosen here is a 3px by 3px rectangle
        prep = cv2.threshold(prep, 50, 255, cv2.THRESH_BINARY_INV)[1]
        erodeElement = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(3,3))
        #dilate with larger element so make sure object is nicely visible
        dilateElement = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(4,4))
        prep = cv2.erode(prep,erodeElement,1)
        prep = cv2.dilate(prep,dilateElement,1)

    #Hough transforms (circles, lines, plines)
    if hough_circle:
        circles = cv2.HoughCircles(gray_im, cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 20, 60, m/10)[0]
        for c in circles[:1]:
            cv2.circle(mask, (c[0],c[1]), int(c[2]*scale_factor), (255,255,255), -1)
            #mask=cv2.threshold(mask, 128, 255,cv2.THRESH_BINARY)
            prep = cv2.bitwise_and(prep,mask)
    if hough_line:
        lines = cv2.HoughLines(prep, 2, np.pi/90, 80)
    if hough_pline:
        plines = cv2.HoughLinesP(prep, 1, np.pi/180, 20,None, 10, 1)

    first = True
    for c in circles[:1]:
        #green for circles (only draw the 3 strongest)
        cv2.circle(canvas, (c[0],c[1]), c[2], (0,255,0), line_thickness)
        #cv2.circle()
        #res = cv2.bitwise_and(frame,frame, mask= mask)
        if first:
            first = False
            circle_x = c[0]
            circle_y = c[1]
            radius = int(c[2])
    first = True
    lines = np.swapaxes(lines,0,1)
    for line in lines[:3]:
        for (rho, theta) in line:
            # blue for infinite lines (only draw the 5 strongest)
            x0 = np.cos(theta)*rho
            y0 = np.sin(theta)*rho
            pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
            pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
            cv2.line(canvas, pt1, pt2, (255,0,0), line_thickness)
            if first:
                first = False
                line_x = x0
                line_y = y0
                angle = 90 - (theta *360 / (2 * np.pi))
                line_mask = np.zeros_like(mask)
                rect_height = int(radius / 4)
                rect_width = int(radius * 2)
                cv2.line(line_mask, pt1, pt2, (255,255,255), rect_height) #mask width is quarter of circle radius found above
                mask = cv2.bitwise_and(mask, line_mask)
                M = cv2.getRotationMatrix2D((x0,y0),(theta *360 / (2 * np.pi))-90,1)

    prep = cv2.bitwise_and(prep, mask)
    density = cv2.warpAffine(prep,M,(m,n))
    density_rect_width = radius*2
    if density_rect_width % 2 is not 0:
        density_rect_width -= 1
    density_rect_height = rect_height/3
    if density_rect_height % 2 is not 0:
        density_rect_height -= 1
    density_rect_origin_x = int(line_x - density_rect_width/2)
    density_rect_origin_y = int(line_y - density_rect_height/2)
    density_rect = np.zeros(shape=(density_rect_height,density_rect_width)).astype(np.uint8)
    for i, row in enumerate(density_rect):
        for j, pixel in enumerate(row):
            density_rect[i,j] = density[density_rect_origin_y+i, density_rect_origin_x+j]
    left, right = np.hsplit(density_rect, 2)
    left_density = 0
    right_density = 0
    for row in left:
        for pixel in row:
            if pixel:
                left_density += 1
    for row in right:
        for pixel in row:
            if pixel:
                right_density += 1
    if left_density > right_density:
        angle += 180
    cv2.putText(canvas,"angle:"+str(angle),(int(m*0.2),int(n*0.9)),cv2.FONT_HERSHEY_PLAIN,2,(200,50,0),3)
    plines=np.swapaxes(plines,0,1)
    for pline in plines[:10]:
        for l in pline:
             # red for line segments
             cv2.line(canvas, (l[0],l[1]), (l[2],l[3]), (0,0,255), line_thickness)


    if draw:
        canvas = np.concatenate((im, canvas), axis = 1)
        #cv2.imshow(window_name,canvas)
    else:
        pass
        #cv2.imshow(window_name,prep)
    #cv2.waitKey(0)

    # save the resulting image
    if draw:
        cv2.imwrite("res.jpg",canvas)
        time.sleep(1)

    temperature = temp_sensor.read_temp()
    f = open('gauge_angle.txt', 'w')
    f.write("angle:"+str(angle)+"\n")
    f.write("temperature:"+str(temperature))
    f.close()
    email.sendMail( ["joejsy@gmail.com"],
            "this is the subject",
            "this is the body text of the email",
            ["res.jpg","gauge_angle.txt"] )


if __name__ == "__main__":
    main()