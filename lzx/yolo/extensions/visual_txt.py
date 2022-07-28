import functools
from PIL import Image
import os
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from extensions.merge_bbs import read_txt_single, rec_img, xyxy_mult_imgshape
from utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy


img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\labels\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.txt"

names = ['ACdist', 'batteries', 'wirelesscabinet', 'heat-tubeac', 'ladderbatteries', 'generalcabinet', 'powercabinet', 'FSU', 'groundwire', 'ac', 'othercabinet', 'fansys', 'hangingac', 'groundbox', 'transformer', 'powerbox', 'Libatteries', 'unifiedcabinet', 'monitorbox', 'acexternal', 'cabinet', 'DCdistribution', 'ladderbattery']
# img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
# img = cv2.resize(img, (640, 320))
# confidence, xyhw, cls = read_txt_single(label_path)
# xyxy = xywh2xyxy(xyhw)
# xyxy_mult_imgshape(xyxy, img.shape)
# img1 = rec_img(img.copy(), xyxy, txts=[names[cls[i]] + " " + str(confidence[i])[:3] for i in range(len(cls))], scale=1, thick=2)
# cv2.imshow("1", cv2.resize(img1, (640, 640)))
# cv2.waitKeyEx()


names = ['ACdist', 'batteries', 'wirelesscabinet', 'heat-tubeac', 'ladderbatteries', 'generalcabinet',
         'powercabinet', 'FSU', 'groundwire', 'ac', 'othercabinet', 'fansys', 'hangingac', 'groundbox',
         'transformer', 'powerbox', 'Libatteries', 'unifiedcabinet', 'monitorbox', 'acexternal', 'cabinet',
         'DCdistribution', 'ladderbattery']


np.random.seed(0)
colors = np.random.randint(0,255, (100,3))

def visual(img_path, label_path, dst_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    confidence, xyhw, cls = read_txt_single(label_path)
    xyxy = xywh2xyxy(xyhw)
    xyxy_mult_imgshape(xyxy, img.shape)
    img1 = rec_img(img.copy(),
                   np.array([xyxy[i] for i in range(len(cls)) if confidence[i] > 0.1]),
                   txts=[names[cls[i]] + " " + str(confidence[i])[:3] for i in range(len(cls)) if confidence[i] > 0.1],
                   # scale=1,
                   # thick=2,
                   color=[colors[cls[i]] for i in range(len(cls)) if confidence[i] > 0.1])
    cv2.imencode('.jpg', img1)[1].tofile(dst_path)


def visual_mult(img_fd, label_fd, dst_fd, pref=""):
    img_fd_files = sorted(os.listdir(img_fd))
    label_fd_files = sorted(os.listdir(label_fd))
    if not os.path.exists(dst_fd):
        os.mkdir(dst_fd)
    for img_f in img_fd_files:
        txt_f = os.path.splitext(img_f)[0] + ".txt"
        if txt_f not in label_fd_files:
            continue
        dst_file = os.path.join(dst_fd, img_f.split('.')[0] + "_" + pref + ".jpg")
        visual(os.path.join(img_fd, img_f),
               os.path.join(label_fd, txt_f),
               dst_file)

if 0:visual_mult(r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\hw_yolo\valid_images",
            r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\hw_yolo\v1215_12222\labels",
            r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\hw_yolo\1220",
            "auged"
            )


visual_mult(r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\hw_yolo\valid_images",
            r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\                                                              hw_yolo\valid_labels",
            r"E:\ori_disks\D\fduStudy\labZXD\weekly_report\hw_yolo\1220",
            "ori"
            )

