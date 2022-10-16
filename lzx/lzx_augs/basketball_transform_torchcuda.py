import cv2
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from lzx.utils import *
import torch.nn.functional as F
import math
import einops
from lzx.coor_transition import *


def preprocess(shape, patch_size, ratio_v=(0.0, 1.0)):
    # print(77777777, shape, patch_size, ratio_v)
    # print(shape);exit()
    pi = math.pi
    X_LEN = shape[1]
    Y_LEN = shape[0]
    # print(patch_num_y, patch_size, shape);exit() # 20 100 (500, 1000, 3)
    # x: [0, X_LEN], u: [0,2pi]
    # y: [0, Y_LEN], u: [-0.5pi,0.5pi]
    # print(Y_LEN, ratio_v)
    gap_uv = patch_size / Y_LEN * pi * (ratio_v[1] - ratio_v[0])
    gap_xy = patch_size

    U_LEN = shape[1] / gap_xy * gap_uv
    V_LEN = pi * (ratio_v[1] - ratio_v[0])

    us = [gap_uv * i for i in range( round(U_LEN / gap_uv) )]
    vs = [gap_uv * i + (ratio_v[0] - 0.5) * pi for i in range( round(V_LEN / gap_uv) )]
    # print(X_LEN, Y_LEN, U_LEN, V_LEN, pi, gap_uv, gap_xy, us, vs);exit() # 10 5
    Y_LEN0 = round(Y_LEN / (ratio_v[1] - ratio_v[0]) * ratio_v[0])
    Y_LEN1 = round(Y_LEN / (ratio_v[1] - ratio_v[0]) * 1)
    return X_LEN, Y_LEN, U_LEN, V_LEN, pi, gap_uv, gap_xy, us, vs, Y_LEN0, Y_LEN1


def stb_adj_info(shape_wh, patch_size, ratio_v=(0.0, 1.0), length=None):
    """
    return adjcent_information-[side, top, bottom]
    """
    if length is None:
        length = shape_wh[0]
    X_LEN, Y_LEN, U_LEN, V_LEN, pi, gap_uv, gap_xy, us, vs, Y_LEN0, Y_LEN1 = preprocess(
        [length, shape_wh[1]] , patch_size, ratio_v=ratio_v)
    stb_adj = [
        U_LEN + 0.5 * gap_uv > pi * 2,
        ratio_v[0] < 1e-5,
        ratio_v[1] + 1e-5 > 1.0
    ]
    return stb_adj


def get_v_all_patches(tc_im, patch_size, ratio_v=(0.0, 1.0), length=None, force_div=True):
    """
    @param tc_im: torch.Tensor image
    @param patch_size: patch size
    @param ratio_v: min/max of v coordinate
    @param length: length of u in image scale
    @param force_div:
    @return:
    """
    if isinstance(ratio_v, list) and len(ratio_v) == 1:
        ratio_v = ratio_v[0]
    if isinstance(ratio_v, torch.Tensor):
        ratio_v = ratio_v.cpu().numpy()

    if length is None:
        length = tc_im.shape[1]
    X_LEN, Y_LEN, U_LEN, V_LEN, pi, gap_uv, gap_xy, us, vs, Y_LEN0, Y_LEN1 = preprocess(
        [length, tc_im.shape[2]] , patch_size, ratio_v=ratio_v)

    shape = list(tc_im.shape[1:])
    # print(list(tc_im.shape[1:]), "list(tc_im.shape[1:])")
    if force_div:
        assert tc_im.shape[1] % gap_xy == 0 and tc_im.shape[2] % gap_xy == 0
    else:
        # print(math.ceil(shape[0] / gap_xy), shape[0] / gap_xy)
        shape[0] = math.ceil(shape[0] / gap_xy) * gap_xy
        shape[1] = math.ceil(shape[1] / gap_xy) * gap_xy

    NUM_PATCH_Y = shape[0] // gap_xy
    NUM_PATCH_X = shape[1] // gap_xy

    # print("shape", shape, gap_xy, NUM_PATCH_Y, NUM_PATCH_X)

    device = tc_im.device
    v_single_column = (torch.arange(NUM_PATCH_Y).to(device) + 0.5) * gap_uv + (ratio_v[0] - 0.5) * pi
    v_all_patches = v_single_column[:, None].repeat(1, NUM_PATCH_X)
    return v_all_patches, gap_xy, NUM_PATCH_Y


def correct_center(tc_im, patch_size, ratio_v=(0.0, 1.0), length=None, padding_value=0):
    v_all_patches, gap_xy, NUM_PATCH_Y = get_v_all_patches(tc_im, patch_size, ratio_v, length)
    device = tc_im.device
    target_sizes = torch.round(torch.cos(v_all_patches) * gap_xy).int()
    tc_im = einops.rearrange(tc_im, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=gap_xy, p2=gap_xy)
    target_sizes = einops.rearrange(target_sizes, 'h w -> (h w)')
    ret = torch.zeros_like(tc_im).to(device).float() + padding_value
    for size in torch.unique(target_sizes):
        if size >= 1:
            indices = (size == target_sizes)
            interpolated = F.interpolate(tc_im[indices], size=(gap_xy, size))
            start = (gap_xy - size) // 2
            ret[indices, :, :,start: start + size] = interpolated
    ret = einops.rearrange(ret, '(h w) c p1 p2 -> c (h p1) (w p2)', h=NUM_PATCH_Y)
    return ret, v_all_patches


def rec_img(img, xyxy, txts=None, color=None):
    # print(img.shape, img.dtype, np.max(img), np.min(img))
    # img = img.astype(np.uint)
    for i,  xyxy_i in enumerate(xyxy.astype(np.int)):
        clr = (0,255,0) if color is None else \
                            (int(color[i][0]), int(color[i][1]), int(color[i][2]),)
        img = cv2.rectangle(img, (xyxy_i[0], xyxy_i[1]), (xyxy_i[0] + xyxy_i[2], xyxy_i[1] + xyxy_i[3]),
                            color=clr, thickness=max(img.shape[0] // 500, 1))
        if txts is not None:
            txt = txts[i]
            cv2.putText(img, txt, (xyxy_i[0], xyxy_i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        clr, 3)
    return img


from lzx.yolo.utils.general import xywh2xyxy
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox

import numpy as np
import pylab


def get_img_and_all_bb(imgi):
    root = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\JPEGImages"
    annFile = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\all.json"
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    image = images[imgi]['file_name']
    image = os.path.join(root, image)

    annIds = coco.getAnnIds(imgIds=imgIds[imgi])
    ann = coco.loadAnns(
        annIds)  # [{'area': 6000, 'iscrowd': 0, 'image_id': 3, 'bbox': [481, 533, 75, 80], 'category_id': 0, 'id': 17, 'ignore': 0, 'segmentation': []}, {'area': 7047,
    image = cv2.imread(image)
    boxes = np.array([a['bbox'] for a in ann])
    scale = 2
    image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
    boxes = boxes // scale
    return image, boxes


def _test_bb():
    # print(basketball_uvmap_foreground([960,1920], 20).shape);exit()
    # print(ann) # {'area': 4928, 'iscrowd': 0, 'image_id': 1, 'bbox': [1112, 535, 77, 64], 'category_id': 0, 'id': 1, 'ignore': 0, 'segmentation': []}
    im, bb = get_img_and_all_bb(17)
    # 4 [torch.Size([2, 96, 125, 250]), torch.Size([2, 192, 63, 125]), torch.Size([2, 384, 32, 63]), torch.Size([2, 768, 16, 32])]
    patch = 20 # 500 // 7
    poses = ['center']
    im = cv2.resize(im, (800, 400))
    # ------544-528-3     20   4    'center'     0.052,0.896
    # 99 554 (600, 1200, 3) [0.165, 0.9233333333333333]
    # im = im[250:, ]

    im = correct_center(torch.from_numpy(im).permute(2,0,1).cuda().float(), patch_size=patch, ratio_v=(0.5, 0.6))
    cv_show1(im, name="12")
    cv2.waitKey()



if __name__=='__main__':
    _test_bb()




