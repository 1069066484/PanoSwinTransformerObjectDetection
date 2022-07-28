import json
import os
import cv2
import numpy as np
from lzx.yolo.utils.general import xywh2xyxy
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox

folder_top = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360"

def get_img_and_all_bb(imgi):
    root = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images"
    annFile = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\train.json"
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    image = images[imgi]['file_name']
    image = os.path.join(root, image)

    annIds = coco.getAnnIds(imgIds=imgIds[imgi])
    print(annIds)
    ann = coco.loadAnns(annIds)
    print(ann, len(ann))
    exit()
    image = cv2.imread(image)
    boxes = np.array([a['bbox'] for a in ann])
    scale = 2
    image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
    # boxes = boxes // scale
    return image, boxes


def _test_bb():
    for i in range(100):
        print("indices-----------",i)
        # 4 8 13 14 17
        im, bb = get_img_and_all_bb(i)
        print(bb)
        # bb = np.array([[100,200,30,80]])
        # im1 = rec_img(im, bb)
        cv2.imshow("1", im)
        cv2.waitKey()


if __name__=='__main__':
    _test_bb()




