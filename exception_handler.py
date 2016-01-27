#!/usr/bin/env python
import cv2
import defines as DEF


def error_output(img, err_msg="ERROR", height=1000, width=1000):
    font_size = (height / 100)
    value_position = (int(width*0.1), int(height*0.5))
    cv2.putText(img, err_msg, value_position,
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_size, color=DEF.RED, thickness=10)
    cv2.imwrite("canvas.jpg", img)
