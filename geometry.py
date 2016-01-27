#!/usr/bin/env python
import cv2
import numpy as np
import defines as DEF


def make_line(point_tuple):
    p1 = point_tuple[0]
    p2 = point_tuple[1]
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


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
    avg_angle = sum(angle_vals)/len(angle_vals)
    possible_angles = []
    for i in range(len(angle_vals)):
        cur_angle = angle_vals[i]
        for j in range(len(angle_vals)):
            if j == i:
                continue
            next_angle = angle_vals[j]
            angle_diff = abs(cur_angle - next_angle)
            avg_this_pair = (cur_angle + next_angle)/2
            per_diff = float(avg_this_pair/avg_angle)
            if (per_diff < 1.1) and (per_diff > 0.9):
                if (angle_diff > (DEF.NEEDLE_SHAPE - 0.5)) and (angle_diff < (DEF.NEEDLE_SHAPE + 0.5)):
                    possible_angles.append((i, j))
    return possible_angles
