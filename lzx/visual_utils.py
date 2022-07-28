import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


def scatter(points, image=None, size=200, color=None, scale=2, thickness_fact=1.0):
    """
    points: np array - Nx2
    """
    if isinstance(size, int):
        size = [size, size]
    if image is None:
        image = np.zeros([*size, 3]).astype(np.uint8)
    points = points.copy()
    shape_min = np.min(image.shape[:2])
    thickness = int(np.ceil(shape_min / 200))
    if scale:
        if scale == 1:
            for i in range(2):
                points_min = np.min(points[:,i])
                points_max = np.max(points[:,i])
                points[:,i] = (points[:,i] - points_min) / (points_max - points_min) * image.shape[i]
        else:
            points_min = np.min(points)
            points_max = np.max(points)
            points = (points - points_min) / (points_max - points_min) * shape_min
    points = points.astype(np.int)
    points = np.clip(points, thickness, np.max(image.shape[:2]))
    for i, p in enumerate(points):
        if color is None:
            n = 223
            ps = 25
            gap = n ** 3 / len(points)
            curr = int(gap * i + gap)
            clr = (
                (curr // (n ** 2)) + ps,
                ((curr // n) % n) + ps,
                curr % n + ps,
            )
        else:
            clr = color
        cv2.circle(image, (int(p[0]), int(p[1])), math.ceil(thickness*2 * thickness_fact), clr,
                   math.ceil(thickness*4 * thickness_fact))
    return image


def show_and_wait(im, name="1", w=True):
    cv2.imshow(name, im)
    if w:
        cv2.waitKey()


def _test_scatter():
    # points = np.random.rand(10,2)
    points = np.array([[ 2.0187, -1.1409],
        [ 2.3116, -0.8749],
        [ 2.4567, -0.5878],
        [ 2.8274, -1.2642],
        [ 2.8274, -0.9425],
        [ 2.8274, -0.6207],
        [ 3.6362, -1.1409],
        [ 3.3433, -0.8749],
        [ 3.1982, -0.5878]])
    points = np.array([[     2.8274,     -0.9425],
        [     2.8274,      0.6183],
        [     2.8274,      0.6233],
        [     4.3924,     -0.0081],
        [     3.7145,      0.4223],
        [     3.3794,      0.5493],
        [     4.3953,     -0.0040],
        [     4.0113,      0.2622],
        [     3.7162,      0.4255]])
    show_and_wait(scatter(points))


if __name__ == '__main__':
    _test_scatter()
