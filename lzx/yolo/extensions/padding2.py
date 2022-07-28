import functools
from PIL import Image
import os
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from lzx.yolo.utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy, xyxy2xywh


from queue import Queue

from lzx.yolo.utils.metrics import bbox_iou


def padding2(img, labels, eps=1e-9):
    """
    :param img: h-w-3. for example, 512-1024-3
    :param labels:   [[          0     0.68643     0.65773     0.10444     0.49967]
                     [          0     0.82467     0.65197     0.18684     0.50855]
                     [          6     0.13553     0.49227    0.029276    0.044408]
                     [          3     0.04926     0.49046    0.072533     0.14145]]
    """
    labels = np.array(labels)
    labels[:, [-4, -2]] *= 0.5
    labels2 = labels.copy()
    labels2[:,-4] += 0.5
    labels = np.concatenate([labels, labels2], 0)
    xywh_all = labels[:, -4:]
    xyxy_all = xywh2xyxy(xywh_all)
    xyxy_all_m_idx1 = np.abs(xyxy_all[:,-4] - 0.5) < eps
    xyxy_all_m_idx2 = np.abs(xyxy_all[:,-2] - 0.5) < eps
    indices_remove = []
    labels_add = []
    for i1 in np.where(xyxy_all_m_idx1)[0]:
        for i2 in np.where(xyxy_all_m_idx2)[0]:
            if np.abs(xyxy_all[i1, -1] - xyxy_all[i2, -1]) < eps and np.abs(xyxy_all[i1, -3] - xyxy_all[i2, -3]) < eps:
                indices_remove += [i1, i2]
                labels_add.append(np.concatenate([labels[i2][:-4], xyxy_all[i2]],0))
                labels_add[-1][-2] = xyxy_all[i1][-2]
    indices_remove = set(indices_remove)
    for i in range(len(xyxy_all)):
        if i not in indices_remove:
            labels_add.append(np.concatenate([labels[i][:-4], xyxy_all[i]], 0))
    labels = np.array(labels_add)
    labels[:, -4:] = xyxy2xywh(labels[:, -4:])
    return np.concatenate([img]*2, -2), labels


def merge_adjbox(labels, x_pos, eps=1e-9, is_xyxy=False):
    xywh_all = labels
    xyxy_all = xywh_all if is_xyxy else xywh2xyxy(xywh_all)
    xyxy_all_m_idx1 = set(np.where(np.abs(xyxy_all[:, -4] - x_pos) < eps)[0])
    xyxy_all_m_idx2 = set(np.where(np.abs(xyxy_all[:, -2] - x_pos) < eps)[0])
    indices_remove = []
    labels_add = []
    for i1 in xyxy_all_m_idx1:
        for i2 in xyxy_all_m_idx2:
            indices_remove += [i1, i2]
            labels_add.append(xyxy_all[i2])
            labels_add[-1][-2] = xyxy_all[i1][-2]
    indices_remove = set(indices_remove)
    for i in range(len(xyxy_all)):
        if i not in indices_remove:
            labels_add.append(xyxy_all[i])
    labels = np.array(labels_add)
    if not is_xyxy:
        labels[:, -4:] = xyxy2xywh(labels[:, -4:])
    return labels


def py_nms(
        dets, iou_thresh=0.75
        # dets, thresh
    ):
    # print(dets)
    # print(111111321444)
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    if dets.shape[1] < 5:
        scores = np.array([1] * dets.shape[0])
    else:
        scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = [] # list(range(len(dets[0])))
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= iou_thresh)[0]
        # print("inds", inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    keep = sorted(list(set(keep)))
    return keep


def padding2_reverse(img, labels):
    return padding2_reverse_img(img), padding2_reverse_labels(labels)


def padding2_reverse_img(img):
    img = img[:, :img.shape[1] // 2]
    return img


def padding2_reverse_labels_xyxy(labels, shape_yx=None, nms=True):
    if shape_yx is None:
        shape_yx = [1,1]
    labels[:, [-2, -4]] /= shape_yx[1]
    labels[:, [-1, -3]] /= shape_yx[0]
    labels[:, -4:] = xyxy2xywh(labels[:, -4:])
    labels = padding2_reverse_labels(labels, nms=nms)
    labels[:, -4:] = xywh2xyxy(labels[:, -4:])
    labels[:, [-2, -4]] *= shape_yx[1] * 0.5
    labels[:, [-1, -3]] *= shape_yx[0]
    return labels

def padding2_reverse_labels(labels, nms=True):
    labels = np.array(labels)
    xywh_all = labels[:, -4:]
    xyxy_all = xywh2xyxy(xywh_all)
    xyxy_all[:,[-2,-4]] *= 2
    xyxy_all_m_idx = (xyxy_all[:,-2] > 1) & (xyxy_all[:,-4] < 1)
    for i in np.where(xyxy_all_m_idx)[0]:
        xyxy_all = np.concatenate([xyxy_all, [xyxy_all[i].copy()]], 0)
        labels = np.concatenate([labels, [labels[i].copy()]], 0)
        xyxy_all[i][-2] = 1
        xyxy_all[-1][-4] = 1
    xyxy_all[:, -4][xyxy_all[:, -4] >= 1] -= 1
    xyxy_all[:, -2][xyxy_all[:, -2] > 1] -= 1
    labels[:, -4:] = xyxy2xywh(xyxy_all)
    if nms:
        keep = py_nms(xyxy_all[:,-4:], 0.95)
        return labels[keep]
    else:
        return labels


from lzx.yolo.extensions.merge_bbs import read_txt_single, rec_img, xyxy_mult_imgshape


def _test2():
    name = "tuRSTUVyaW_230129908000000018_kbcl_mnopqqrs_1630459525895____VID_20210706_144611_00_007_001881"
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
        # print("xyhw11111", xyhw)
        img1 = rec_img(img.copy(), xyxy_mult_imgshape(xywh2xyxy(xyhw), img.shape))
        cv2.imshow("1", cv2.resize(img1, (640, 640)))


        # print("xyhw", xyhw)
        enhanced, enhanced_xyhw = padding2(img, xyhw)

        # print(enhanced_xyhw)
        enhanced = rec_img(enhanced, xyxy_mult_imgshape(xywh2xyxy(enhanced_xyhw), enhanced.shape))
        cv2.imshow("2", cv2.resize(enhanced, (640, 640)))


        # print("xyhw", xyhw)
        enhanced, enhanced_xyhw = padding2_reverse(enhanced, enhanced_xyhw)
        enhanced = rec_img(enhanced, xyxy_mult_imgshape(xywh2xyxy(enhanced_xyhw), enhanced.shape))
        cv2.imshow("3", cv2.resize(enhanced, (640, 640)))

        cv2.waitKeyEx()


if __name__=='__main__':
    _test2()

