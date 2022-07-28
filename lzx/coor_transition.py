import numpy as np


def xyxy2tlwh(xyxy):
    xyxy = xyxy.copy()
    xyxy[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xyxy[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xyxy


def tlwh2xyxy(tlwh):
    tlwh = tlwh.copy()
    tlwh[:, 2] = tlwh[:, 2] + tlwh[:, 0]
    tlwh[:, 3] = tlwh[:, 3] + tlwh[:, 1]
    return tlwh


def normlize01_xyxy(len_xy, xyxy):
    xyxy[:, [0, 2]] /= len_xy[0]
    xyxy[:, [1, 3]] /= len_xy[1]
    return xyxy


def unnormlize01_xyxy(len_xy, xyxy):
    xyxy[:, [0, 2]] *= len_xy[0]
    xyxy[:, [1, 3]] *= len_xy[1]
    return xyxy