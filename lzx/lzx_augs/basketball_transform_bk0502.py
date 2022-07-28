import cv2
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from lzx.utils import *
import torch.nn.functional as F
import math
from lzx.coor_transition import *


def preprocess(shape, patch_num_y):
    X_LEN = shape[1]
    Y_LEN = shape[0]
    pi = math.pi
    # x: [0, X_LEN], u: [0,2pi]
    # y: [0, Y_LEN], u: [-0.5pi,0.5pi]
    gap = pi / patch_num_y
    gap2 = gap / pi * Y_LEN
    us = [gap * i for i in range( round(2*pi / gap) )]
    vs = [gap * i - 0.5 * pi for i in range( round(pi / gap) )]
    return X_LEN, Y_LEN, pi, gap, gap2, us, vs


def pose_init(poses):
    for p in poses:
        assert p in {"center", "center2", "left", "right"}
    poses = set(poses)
    assert ('center2' in poses and len(poses) == 1) or 'center2' not in poses
    if poses is None:
        poses = ['center']
    return poses


def correct(tc_im, patch_num_y=20, poses=None):
    poses = pose_init(poses)

    X_LEN, Y_LEN, pi, gap, gap2, us, vs = preprocess(tc_im.shape[1:], patch_num_y)

    if 'center2' in poses:
        tc_im = tc_im.roll(round(gap2 * 0.5), 2)

    transed = dict((k, []) for k in poses)
    # print(list(transed.keys())); exit()
    base_plus = 100
    x2 = 0
    for ui, u in enumerate(us):
        x = x2
        x2 = round((u + gap) / 2 / pi * X_LEN)
        if ui == len(us) - 1:
            x2 = min(round(x2 * 1.1), X_LEN)
        for k in transed:
            transed[k].append([])
        for vi, v in enumerate(vs):
            y = (v / pi + 0.5) * Y_LEN
            y2 = round(y + gap2)
            if vi == len(us) - 1:
                y2 = min(round(y2 * 1.1), Y_LEN)
            y = round(y)

            curr_ori = tc_im[:, y:y2, x:x2]
            curr_trans = dict((k, torch.zeros_like(curr_ori)) for k in transed)

            for it in range(y, y2):
                v_it_coord = it / Y_LEN * pi - 0.5 * pi
                x_len_it = round(math.cos(v_it_coord) * curr_ori.shape[1])
                if x_len_it:
                    it -= y
                    interpolated = F.interpolate(
                            curr_ori[None, :, it:it + 1, :], size=(1, x_len_it))[0]
                    if 'center' in curr_trans:
                        start = max(round((gap2 - x_len_it)//2), 0)
                        curr_trans['center'][:,it:it+1, start:start+x_len_it] = \
                            interpolated[...,:min(start+x_len_it, curr_trans['center'].shape[-1]) - start]
                    if 'center2' in curr_trans:
                        start = max(round((gap2 - x_len_it)//2), 0)
                        curr_trans['center2'][:,it:it+1,start:start+x_len_it] =\
                            interpolated[...,:min(start+x_len_it, curr_trans['center2'].shape[-1]) - start]
                    if 'left' in curr_trans:
                        curr_trans['left'][:,it:it+1,0:x_len_it] = interpolated[...,:min(x_len_it,curr_trans['left'].shape[-1])]
                    if 'right' in curr_trans:
                        curr_trans['right'][:,it:it+1,-x_len_it:] = interpolated[...,min(x_len_it,-curr_trans['right'].shape[-1]):]
            for k in transed:
                transed[k][-1].append(curr_trans[k])
        for k in transed:
            transed[k][-1] = torch.cat(transed[k][-1], 1)
    for k in transed:
        transed[k] = torch.cat(transed[k], 2)
    if 'center2' in poses:
        transed['center2'] = transed['center2'].roll(-round(gap2 * 0.5), 2)
    return transed


def basketball_uvmap_foreground(shape, patch_num_y=20):
    # torch.Size([960, 1920])
    poses = ['center', 'center2', 'left', 'right']
    us = torch.arange(shape[1]) / shape[-1] * math.pi * 2
    vs = torch.arange(shape[0]) / shape[-1] * math.pi - math.pi * 0.5
    foreground = torch.ones(shape[:2])
    uvmap = torch.stack([
        us[None].repeat([shape[0], 1]),
        vs[:,None].repeat([1, shape[1]]),
        foreground
    ])
    ret = basketball_transition(uvmap, patch_num_y, poses)
    return ret


def basketball_transition(im, patch_num_y=20, poses=None):
    if isinstance(im, np.ndarray):
        tc_im = torch.from_numpy(im.copy()).float()
    else:
        tc_im = im

    if im.shape[-1] == 3:
        # 224, 224, 3
        tc_im = tc_im.permute(2,0,1)

    if poses is None:
        poses = ['center']
    transed = correct(tc_im, patch_num_y, [p for p in poses if p != 'center2'])
    if 'center2' in poses:
        transed['center2'] = correct(tc_im, patch_num_y, ['center2'])['center2']

    for k in transed:
        if im.shape[-1] == 3:
            # 3, 224, 224
            transed[k] = transed[k].permute(1,2,0)
        if isinstance(im, np.ndarray):
            transed[k] = transed[k].numpy().astype(im.dtype)
    return transed


def basketball_transition_xy(shape, xys, patch_num_y=20, poses=None):
    # xys: [n, 2]
    poses = pose_init(poses)
    # y does not change
    X_LEN, Y_LEN, pi, gap_uv, gap_xy, us, vs = preprocess(shape, patch_num_y)
    if 'center2' in poses:
        xys[:, 0] = (xys[:, 0] + gap_xy // 2) % X_LEN
    transed = {}
    us.append(pi * 2)
    center_lines = np.array([(us[i] + us[i + 1]) * 0.5 for i in range(len(us) - 1)]) * X_LEN / 2 / pi
    interval_index = np.argmin(np.abs(xys[:,0:1] - center_lines[None, :]), -1)
    if 'center' in poses:
        transed['center'] = center_lines[interval_index]
    if 'center2' in poses:
        transed['center2'] = center_lines[interval_index]
    if 'left' in poses:
        transed['left'] = np.array(us)[interval_index] * X_LEN / 2 / pi
    if 'right' in poses:
        transed['right'] = np.array(us)[1:][interval_index] * X_LEN / 2 / pi
    for k in transed:
        x_bias = xys[:, 0] - transed[k]
        x_bias = x_bias * np.cos((xys[:,1] / Y_LEN - 0.5) * pi)
        xys2 = xys.copy()
        xys2[:,0] = (np.round(x_bias).astype(np.int) + transed[k] + xys[:,0]) * 0.5
        transed[k] = xys2
    if 'center2' in transed:
        transed['center2'][:, 0] = (xys[:, 0] - gap_xy // 2) % X_LEN
    return transed


def basketball_transition_xy2uv_ladder(shape, tlwh):
    print(shape, tlwh)
    # we assume the pole is longer side
    tp = tlwh[:,:2]
    bt = tp + tlwh[:,2:]





def basketball_transition_bb(shape, tlwh, patch_num_y=20, poses=None):
    xyxy = tlwh2xyxy(tlwh)
    if poses is None:
        poses = ['center']
    transed_xyxy = basketball_transition_xy(shape, xyxy.reshape([-1, 2]), patch_num_y=patch_num_y,
                                            poses=[p for p in poses if p != 'center2'])
    if 'center2' in poses:
        transed_xyxy['center2'] = basketball_transition_xy(shape, xyxy.reshape([-1, 2]), patch_num_y=patch_num_y,
                                            poses=['center2'])['center2']
    for k in transed_xyxy:
        transed_xyxy[k] = xyxy2tlwh(transed_xyxy[k].reshape(xyxy.shape))
    return transed_xyxy


def _test_backup():
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    # img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\Omnidirectional Street-view Dataset\equirectangular\JPEGImages\000002.jpg"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.resize(img, (1334, 667))

    # print(img.shape, '222222222222222222222222222222222222222')
    # print(img.shape)  # (960, 1920, 3)
    # tc_im = transform(img)
    # print(tc_im.shape)  # torch.Size([3, 960, 1920])

    patch_num_y = 8
    transed = basketball_transition(img, patch_num_y, ['center', 'center2', 'left', 'right'])
    print([transed[t].shape for t in transed], img.shape)
    sz = (400, 200)
    print(len(transed))
    # sz = (600,300)
    for k in transed:
        cv_show1(transed[k][..., :3], w=False, name=k, sz=sz)
    cv_show1(img, w=True, sz=sz)


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
    if 0:
        for i in range(100):
            print("indices-----------",i)
            # 4 8 13 14 17
            im, bb = get_img_and_all_bb(i)
            # bb = np.array([[100,200,30,80]])
            im1 = rec_img(im, bb)
            cv2.imshow("1", im1)
            cv2.waitKey()
            continue
    else:
        im, bb = get_img_and_all_bb(17)
        im1 = rec_img(im, bb)
        print(im1.shape)
        cv2.imshow("1", im1)
    # cv2.waitKey()
    # 500*100
    # 4 [torch.Size([2, 96, 125, 250]), torch.Size([2, 192, 63, 125]), torch.Size([2, 384, 32, 63]), torch.Size([2, 768, 16, 32])]
    patch = 10 # 500 // 7
    poses = ['center']
    bb2 = basketball_transition_bb(im.shape, bb, patch, poses=poses)
    im = basketball_transition(im, patch, poses)
    for k in poses:
        im2 = im[k].copy()
        # print(im.shape, im.dtype, np.max(im), np.min(im));exit()
        # im2 = (im2 * 0).astype(np.uint8)
        im2 = rec_img(im2, bb2[k])
        cv2.imshow(k, im2)

    uvf = basketball_uvmap_foreground(im[poses[0]].shape, patch)

    for k in poses:
        print(uvf[k].shape)
        for i in range(3):
            arr = torch.from_numpy(np.stack([uvf[k][i]] * 3).astype(np.float))
            # print(arr.shape)
            cv_show1(arr, name='uvf' + str(i), w=False)

    cv2.waitKey()
    print(33333)


def _test_basketball_transition_xy2uv_ladder():
    im, bb = get_img_and_all_bb(0)
    basketball_transition_xy2uv_ladder(im.shape, bb)
    exit()
    im1 = rec_img(im, bb)
    cv2.imshow("1", im1)
    cv2.waitKey()


if __name__=='__main__':
    _test_basketball_transition_xy2uv_ladder()




