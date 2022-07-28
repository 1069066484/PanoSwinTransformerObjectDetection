import os
import shutil

import cv2
import numpy as np
# hw0805/第二批10-7/哈道里二场村南(1.jpg


def read_label_xc_yc_xr_yr_from_txt(file):
    labels = []
    xc_yc_xr_yr = []
    encoding = "utf-8"
    # print(file)
    with open(file, 'r', encoding=encoding) as f:
        lines = f.read()
        # print(lines)
        # lines = str(lines.encode(encoding), encoding=encoding)
        lines = lines.strip()
    if lines == "":
        return [], np.array([[]])
    for line in lines.split('\n'):
        content = line.strip().split(' ')
        labels.append(int(content[0]))
        content = [float(c) for c in content[1:]]
        xc_yc_xr_yr.append(content)
    return labels, np.array(xc_yc_xr_yr)


def write_label_xc_yc_xr_yr_to_txt(file, labels, xc_yc_xr_yr):
    with open(file, 'w') as f:
        for label, xc_yc_xr_yr_ in zip(labels, xc_yc_xr_yr):
            f.write("{} {}\n".format(label, " ".join([str(x) for x in xc_yc_xr_yr_])))


def clean_xy(xr, x_thresh=0.6):
    # return good_indices
    return np.argwhere(xr < x_thresh).reshape([-1]).tolist()


# from utils.general import xywh2xyxy


def xc_yc_xr_yr2xminymin_xmaxymax(xc_yc_xr_yr):
    return [xc_yc_xr_yr[0] - xc_yc_xr_yr[2] / 2, xc_yc_xr_yr[1] - xc_yc_xr_yr[3] / 2,
            xc_yc_xr_yr[0] + xc_yc_xr_yr[2] / 2, xc_yc_xr_yr[1] + xc_yc_xr_yr[3] / 2]



def add_xy(labels, xc_yc_xr_yr, thresh=0.3):
    labels_new = []
    xc_yc_xr_yr_new = []

    for label, xc_yc_xr_yr_ in zip(labels, xc_yc_xr_yr):
        # print(111, xminymin_xmaxymax_, xc_yc_xr_yr_)
        xminymin_xmaxymax_ = xc_yc_xr_yr2xminymin_xmaxymax(xc_yc_xr_yr_)
        range_x = xminymin_xmaxymax_[0] + 1.0 - xminymin_xmaxymax_[2]
        # print(xc_yc_xr_yr_, xminymin_xmaxymax_, range_x)
        # input()

        if xminymin_xmaxymax_[0] > thresh * range_x: # left
            labels_new.append(label)
            xc_yc_xr_yr_new.append([xminymin_xmaxymax_[0] / 2, xc_yc_xr_yr_[1], xminymin_xmaxymax_[0], xc_yc_xr_yr_[3]])
            # print(xc_yc_xr_yr_, xminymin_xmaxymax_, xc_yc_xr_yr_new[-1]); input()
        if (1.0 - xminymin_xmaxymax_[2]) > thresh * range_x: # right
            labels_new.append(label)
            xc_yc_xr_yr_new.append([1.0 - (1.0 - xminymin_xmaxymax_[2]) / 2, xc_yc_xr_yr_[1], 1.0 - xminymin_xmaxymax_[2], xc_yc_xr_yr_[3]])
    # print(111, xc_yc_xr_yr, xc_yc_xr_yr_new)
    return labels_new, xc_yc_xr_yr_new


def area_filter(labels, xc_yc_xr_yr, thresh=0.):
    labels_n = []
    xc_yc_xr_yr_n = []
    for l, xy in zip(labels, xc_yc_xr_yr):
        # area = xy[-2] * xy[-1]
        if xy[-2] > thresh and xy[-2] < 0.99:
            labels_n.append(l)
            xc_yc_xr_yr_n.append(xy)
    xc_yc_xr_yr_n = np.array(xc_yc_xr_yr_n)
    if len(xc_yc_xr_yr_n) == 0:
        xc_yc_xr_yr_n = np.random.rand(0,4)
    # print("11111")
    return labels_n, xc_yc_xr_yr_n


def update_single(labels, xc_yc_xr_yr, x_thresh_clean=0.6, add_xy_thresh=0.3):
    if xc_yc_xr_yr.shape[1] * xc_yc_xr_yr.shape[0] == 0:
        print("warning! xc_yc_xr_yr shape: {}, return None, None".format(xc_yc_xr_yr.shape))
        return None, None

    good_indices = clean_xy(xc_yc_xr_yr[:, 2], x_thresh_clean)
    # print("123", xc_yc_xr_yr, xc_yc_xr_yr[:, 2], good_indices, len(labels))
    if len(good_indices) == len(labels):
        return area_filter(labels, xc_yc_xr_yr)
    # print('ori', labels, xc_yc_xr_yr)
    bad_indices = np.array([i for i in range(len(labels)) if i not in good_indices])
    labels_new, xc_yc_xr_yr_new = add_xy(np.array(labels)[bad_indices], np.array(xc_yc_xr_yr)[bad_indices], add_xy_thresh)
    labels = list(np.array(labels)[good_indices].tolist()) + labels_new
    xc_yc_xr_yr = np.concatenate([np.array(xc_yc_xr_yr)[good_indices], np.array(xc_yc_xr_yr_new)], 0)
    # print('later', labels, xc_yc_xr_yr)
    # print("xc_yc_xr_yr_new", labels_new, xc_yc_xr_yr_new)
    return area_filter(labels, xc_yc_xr_yr)


def clean(folder):
    file_total = 0
    file_refresh = 0
    item_total = 0
    item_remove = 0
    print("Cleaning {}".format(folder))
    for root, dirs, files in os.walk(folder):
        for name in files:
            txt = os.path.join(root, name)
            if os.path.splitext(txt)[-1] != '.txt': continue
            labels, xc_yc_xr_yr = read_label_xc_yc_xr_yr_from_txt(txt)
            if len(labels) == 0:
                file_total += 1
                continue
            good_indices = clean_xy(xc_yc_xr_yr[:, 2])
            file_total += 1
            item_total += len(labels)
            if len(good_indices) != len(labels) or True:
                item_remove += len(labels) - len(good_indices)
                file_refresh += 1
                print("{}: remove {} and refresh".format(txt, [i for i in range(len(labels)) if i not in good_indices]))
                # print(good_indices, labels, xc_yc_xr_yr)
                labels, xc_yc_xr_yr = [np.array(x)[good_indices] for x in [labels, xc_yc_xr_yr]]
                write_label_xc_yc_xr_yr_to_txt(txt, labels, xc_yc_xr_yr)
    print("file_total: {}, file_refresh: {}, item_total: {}, item_remove: {}".format(
        file_total, file_refresh, item_total, item_remove
    ))


def update(folder, add_xy_thresh=0.3):
    files_cnt = 0
    items_0 = 0
    items_n = 0
    print("Updating {}".format(folder))
    for root, dirs, files in os.walk(folder):
        for name in files:
            txt = os.path.join(root, name)
            if os.path.splitext(txt)[-1] != '.txt':
                print("updating {} error, abort".format(txt))
                continue

            files_cnt += 1
            labels, xc_yc_xr_yr = read_label_xc_yc_xr_yr_from_txt(txt)
            items_0 += len(labels)
            labels, xc_yc_xr_yr = update_single(labels, xc_yc_xr_yr, add_xy_thresh=add_xy_thresh)

            if labels is None:
                print("In {}, no data".format(txt))
                labels = []
                xc_yc_xr_yr = np.random.rand(0,4)
                # continue
            items_n += len(labels)
            write_label_xc_yc_xr_yr_to_txt(txt, labels, xc_yc_xr_yr)

    print("files: {}, items_0: {}, items_n: {}".format(files_cnt, items_0, items_n))


# update(r"data_trainable0927/labels")
