import functools
from PIL import Image
import os
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy, xyxy2xywh


from queue import Queue
from utils.metrics import bbox_iou


def edge_erase(img, labels, eps=1e-4):
    """
    :param img: h-w-3. for example, 512-1024-3
    :param labels:   [[          0     0.68643     0.65773     0.10444     0.49967]
                     [          0     0.82467     0.65197     0.18684     0.50855]
                     [          6     0.13553     0.49227    0.029276    0.044408]
                     [          3     0.04926     0.49046    0.072533     0.14145]]
    """

    labels = np.array(labels)
    shape = img.shape
    xywh_all = labels[:, -4:]
    xyxy_all = xywh2xyxy(xywh_all)
    blurred = cv2.GaussianBlur(img, (29,29), 15)
    marker = np.zeros(shape[:2], dtype=np.bool)
    noedge_boxes = set(range(len(xyxy_all)))
    # print(shape, "shape")
    def set_marker(xyxy_, value):
        marker[
            int(xyxy_[1] * shape[0]): int(xyxy_[3] * shape[0]),
            int(xyxy_[0] * shape[1]): int(xyxy_[2] * shape[1]),
        ] = value

    for i, xyxy in enumerate(xyxy_all):
        if xyxy[0] < eps or xyxy[2] > 1 - eps:
            set_marker(xyxy, 1)
            noedge_boxes.remove(i)

    for i in noedge_boxes:
        # print(i, len(xyxy_all))
        set_marker(xyxy_all[i], 0)

    marker = marker[:,:,None]
    img = marker * blurred + (~ marker) * img
    return img, labels[list(noedge_boxes)]



from extensions.merge_bbs import read_txt_single, rec_img, xyxy_mult_imgshape


def _test2():
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\labels\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.txt"

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

    img = cv2.resize(img, (640, 320))

    confidence, xyhw, cls = read_txt_single(label_path)

    img1 = rec_img(img.copy(), xyxy_mult_imgshape(xywh2xyxy(xyhw), img.shape))
    cv2.imshow("1", cv2.resize(img1, (640, 640)))

    print(xyhw)
    enhanced, enhanced_xyhw = edge_erase(img, xyhw)

    print(enhanced_xyhw)
    enhanced = rec_img(enhanced, xyxy_mult_imgshape(xywh2xyxy(enhanced_xyhw), img.shape))
    cv2.imshow("2", cv2.resize(enhanced, (640, 640)))

    cv2.waitKeyEx()


if __name__=='__main__':
    _test2()