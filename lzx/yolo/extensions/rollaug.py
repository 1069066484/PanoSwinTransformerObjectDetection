import functools
from PIL import Image
import os
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from lzx.yolo.utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy, xyxy2xywh
from lzx.yolo.extensions.padding2 import merge_adjbox


def roll_aug_raw(img, labels, roll_dist=None, merge=True, is_xyxy=False, clip01=True):
    """
    :param img: h-w-3. for example, 512-1024-3
    :param labels:   [[          0     0.68643     0.65773     0.10444     0.49967]
                     [          0     0.82467     0.65197     0.18684     0.50855]
                     [          6     0.13553     0.49227    0.029276    0.044408]
                     [          3     0.04926     0.49046    0.072533     0.14145]]
    :param roll_dist: [0.0, 1.0]
    :return: [img, labels]
    """
    h, w, _ = img.shape
    # assert 0 <= roll_dist <= 1.0, "It's required that 0 <= roll_dist <= 1.0, get {}".format(roll_dist)
    if roll_dist is None:
        roll_dist = np.random.rand()
    # print("roll_distroll_dist", roll_dist)
    roll_dist = (int(roll_dist * 100000) % 100000) / 100000
    shift = int(roll_dist * w)
    h, w, _ = img.shape
    img = np.roll(img, shift=shift, axis=1)
    labels = np.array(labels)

    xywh_all = labels[:, -4:]

    if not is_xyxy:
        xyxy_all = xywh2xyxy(xywh_all)
    else:
        xyxy_all = xywh_all

    xyxy_all[:, 0] += roll_dist
    xyxy_all[:, 2] += roll_dist
    for i in range(len(labels)):
        xyxy = xyxy_all[i]
        if xyxy[2] > 1.0:
            if (xyxy[2] + xyxy[0]) / 2 > 1.0:
                xyxy[2] -= 1.0
                if clip01: xyxy[0] = max(xyxy[0] - 1.0, 0)
                else: xyxy[0] -= 1.0
            else:
                if clip01: xyxy[2] = 1.0
        xyxy_all[i] = xyxy

    if not is_xyxy:
        labels[:, -4:] = xyxy2xywh(xyxy_all)
    else:
        labels[:, -4:] = xyxy_all
    # labels = labels[labels[:, -2] > 0.008]
    return img, merge_adjbox(labels, roll_dist, is_xyxy=is_xyxy), shift


def roll_aug(img, labels, roll_dist=None, merge=True, is_xyxy=False, clip01=True):
    img, boxes, shift = roll_aug_raw(img=img, labels=labels, roll_dist=roll_dist, merge=merge, is_xyxy=is_xyxy, clip01=clip01)
    return img, boxes



from lzx.yolo.extensions.merge_bbs import read_txt_single, rec_img, xyxy_mult_imgshape


def _test2():
    np.random.seed(0)
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\labels\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.txt"

    folder = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_trainable0119\images\train"
    for name in os.listdir(folder):
        print(name)
        split_name = os.path.splitext(name)
        name = split_name[0]
        if not split_name[1].lower() == '.jpg':
            continue
        img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_trainable0119\images\train\{}.jpg".format(name)
        label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_trainable0119\labels\train\{}.txt".format(name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

        img = cv2.resize(img, (640, 320))

        confidence, xyhw, cls = read_txt_single(label_path)

        img1 = rec_img(img.copy(), xyxy_mult_imgshape(xywh2xyxy(xyhw), img.shape))
        cv2.imshow("1", cv2.resize(img1, (640, 640)))

        print(xyhw)
        enhanced, enhanced_xyhw = roll_aug(img, xyhw)

        print(enhanced_xyhw)
        enhanced = rec_img(enhanced, xyxy_mult_imgshape(xywh2xyxy(enhanced_xyhw), img.shape))
        cv2.imshow("2", cv2.resize(enhanced, (640, 640)))

        cv2.waitKeyEx()


if __name__=='__main__':
    _test2()