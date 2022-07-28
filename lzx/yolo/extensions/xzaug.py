import functools
from PIL import Image
import os
import numpy as np
from scipy.ndimage import map_coordinates
import cv2


def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv


@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v

def label2point(width, height,label_path):
    points = []
    catgory = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            item = line.split(' ')
            catgory.append(item[0])
            zhongxin_x = width * float(item[1])
            zhongxin_y = height * float(item[2])
            img_width = width * float(item[3])
            img_height = height * float(item[4])
            x0 = zhongxin_x - (img_width / 2)
            x1 = zhongxin_x + (img_width / 2)
            y0 = zhongxin_y - (img_height / 2)
            y1 = zhongxin_y + (img_height / 2)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            tmp = []
            tmp.append(x0)
            tmp.append(y0)
            points.append(tmp)
            tmp2 = []
            tmp2.append(x1)
            tmp2.append(y1)
            points.append(tmp2)
    return catgory,points

def point2label(width, height, points, category):
    len = points.shape[0]
    label = []
    for i in range(len // 2):
        cate = int(category[i])
        x0 = int(points[i * 2][0])
        y0 = int(points[i * 2][1])
        x1 = int(points[i * 2 + 1][0])
        y1 = int(points[i * 2 + 1][1])
        item1 = ((x0 + x1) / 2) / width
        item2 = ((y0 + y1) / 2) / height
        item3 = (x1 - x0) / width
        item4 = (y1 - y0) /height
        tmp = [cate,item1,item2,item3,item4]
        label.append(tmp)
    return label


def getAug(img, kx, ky, oripoints):
    # img = Image.open(img_path)
    # width, height = img.size[0], img.size[1]
    # img = np.array(img, np.float32)[..., :3]

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=1, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    # catgory, points = label2point(width,height,label_path)
    # points = np.array(points)
    # corners_u0 = coorx2u(points[:, 0], img.shape[1])
    # corners_v0 = coory2v(points[:, 1], img.shape[0])
    # corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    # corners_v = np.arctan(np.tan(corners_v0) * np.sin(corners_u) / np.sin(corners_u0) / ky)
    # cornersX = u2coorx(corners_u, img.shape[1])
    # cornersY = v2coory(corners_v, img.shape[0])
    # stretched_corners = np.stack([cornersX, cornersY], axis=-1)

    im = Image.fromarray(np.uint8(stretched_img))  # numpy 转 image类
    img2 = np.asarray(im) #cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


    corners_u0 = coorx2u(oripoints[:, 0], img.shape[1])
    corners_v0 = coory2v(oripoints[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    corners_v = np.arctan(np.tan(corners_v0) * np.sin(corners_u) / np.sin(corners_u0) / ky)
    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    new_points = np.stack([cornersX, cornersY], axis=-1)

    return img2, new_points


def xzaug_xywh(img, lxywh, is_xyxy=False, kxy=(4.0,4.0)):
    # return img, lxyxy
    # return [augmented image, transformed label_xyxy]
    label = lxywh[:, :1]

    if is_xyxy:
        xyxy = lxywh[:, 1:]
    else:
        xyxy = xywhn2xyxy(lxywh[:, 1:], img.shape[1], img.shape[0])
    # print(img.shape)
    # print(lxywh[:, 1:])
    # print(xyxy)
    # print(111, img.shape)
    img, xyxy = _xzaug(img, xyxy.copy(), kxy=kxy)
    # print(222, img.shape)
    # print(xyxy)
    lxywh = np.concatenate([label, xyxy.copy() if is_xyxy else xyxy2xywhn(xyxy.copy(), img.shape[1], img.shape[0])
                            ], 1)
    # print("aaaaaaa", lxywh)
    return img, lxywh


def xzaug(img, lxyxy):
    # return img, lxyxy
    # return [augmented image, transformed label_xyxy]
    label = lxyxy[:, :1]
    xyxy = lxyxy[:, 1:]
    img, xyxy = _xzaug(img, xyxy)
    lxyxy = np.concatenate([label, xyxy], 1)
    return img, lxyxy


def _xzaug(img, xyxy, kxy=(4.0,4.0)):
    '''
    xyxy1111 [[     400.21         166      475.68      285.47]
     [     538.95      206.74      635.47      293.05]
     [     3.0526      142.63      116.11      178.63]
     [     123.05      147.05      160.21      214.63]
     [     245.05      154.84      267.16      169.47]
     [     494.32      151.26      503.26      153.05]]
    xyxy2222 [[     412.23      162.77      476.49       252.9]
     [     528.36      183.57      634.39      275.39]
     [     4.0091      149.75      124.53      168.54]
     [     130.32       153.7      160.06      186.57]
     [     232.93      157.13      256.17      164.86]
     [      491.2      155.75      498.35      156.55]]
    '''
    # return [augmented image, transformed xyxy]
    kx = np.random.uniform(1.0, kxy[0])
    ky = np.random.uniform(1.0, kxy[1])
    if np.random.rand() < 0.5: kx = 1.0 / kx
    if np.random.rand() < 0.5: ky = 1.0 / ky

    # print("xyxy1111", xyxy)
    new_img, new_points = getAug(img, kx, ky, xyxy.reshape([-1, 2]))
    # print("xyxy2222", new_points.reshape([-1, 4]))
    return new_img, new_points.reshape([-1, 4])


from lzx.yolo.extensions.merge_bbs import read_txt_single, rec_img, xyxy_mult_imgshape
from lzx.yolo.utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy
"""
1.06  ,0.53,  3.21,0.88 , 0.65 , 3.05,0.08 , 1.47  ,0.88
 
 0.06,  0.04,  0.06, 0.22 , 0.15 0.01, 0.27 , 0.09 , 0.10
"""

def _test():
    # xzaug_xywh
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\labels\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.txt"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.resize(img, (640,320))
    confidence, xyhw, cls = read_txt_single(label_path)
    xyxy = xywh2xyxy(xyhw)
    xyxy_mult_imgshape(xyxy, img.shape)
    img1 = rec_img(img.copy(), xyxy)
    cv2.imshow("1", cv2.resize(img1, (640, 640)))

    enhanced, enhanced_xywh = xzaug_xywh(img, np.concatenate([cls.reshape([-1, 1]), xyhw], 1))
    # print("enhanced_xywh:\n", enhanced_xywh)
    enhanced_xywh[:,1:] = xywh2xyxy(enhanced_xywh[:,1:])
    xyxy_mult_imgshape(enhanced_xywh[:,1:], img.shape)
    enhanced = rec_img(enhanced.copy(), enhanced_xywh[:,1:])
    cv2.imshow("2", cv2.resize(enhanced, (640,640)))

    cv2.waitKeyEx()


def _test2():
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    label_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\labels\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.txt"

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

    img = cv2.resize(img, (640, 320))

    confidence, xyhw, cls = read_txt_single(label_path)
    xyxy = xywh2xyxy(xyhw)

    xyxy_mult_imgshape(xyxy, img.shape)
    img1 = rec_img(img.copy(), xyxy)
    cv2.imshow("1", cv2.resize(img1, (640, 320)))
    # print(xyxy)
    enhanced, enhanced_xyxy = _xzaug(img, xyxy)
    # print(enhanced_xyxy)
    enhanced = rec_img(enhanced, enhanced_xyxy)
    cv2.imshow("2", cv2.resize(enhanced, (640, 320)))

    cv2.waitKeyEx()


if __name__=='__main__':
    _test()