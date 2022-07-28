import cv2
import os
import numpy as np
import torch

from lzx.yolo.utils.metrics import bbox_iou

from lzx.yolo.utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy
import copy
EMPTY = [np.random.rand(0), np.random.rand(0,4), np.random.rand(0).astype(np.int)]


def py_nms(
        conf_bb1, conf_bb2, iou_thresh=0.75, conf_thresh=0.5,
        # dets, thresh
    ):
    # conf_bb1 is the ground truth
    # print(conf_bb1);exit()
    """Pure Python NMS baseline."""
    # print(conf_bb1)
    # exit()
    dets = np.concatenate(
        [np.concatenate([conf_bb1[1], conf_bb1[0][:,None], conf_bb1[2][:,None]], -1),
         np.concatenate([conf_bb2[1], conf_bb2[0][:,None], conf_bb2[2][:,None]], -1),
         ], 0
    )
    dets = dets[dets[:,4] > conf_thresh]
    dets_ori = dets.copy()
    dets[:,:4] = xywh2xyxy(dets[:,:4])
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = list(range(len(conf_bb1[0])))
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
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    keep = sorted(list(set(keep)))
    dets_ret = dets_ori[keep].copy()
    # print(dets_ret, dets_ret.shape)
    # print(dets_ret,  dets_ret[:, 5], dets_ret.shape);exit()
    ret = [dets_ret[:, 4],
            dets_ret[:, :4],
            dets_ret[:, 5]]
    # print(ret)
    return ret


def merge(conf_bb1, conf_bb2, iou_thresh=0.75, conf_thresh=0.5):
    """
    n = 10
    conf_bb1 = [np.random.rand(n), np.random.rand(n, 4)]
    conf_bb2 = [np.random.rand(n), np.random.rand(n, 4)]
    merged = merge(conf_bb1, conf_bb2)
    print(merged[0].shape, merged[1].shape)

    :param conf_bb1: [np.array: confidence, np.array: xyhw]
    :param conf_bb2: [np.array: confidence, np.array: xyhw]
    :param iou_thresh:
    :param conf_thresh:
    :return: [np.array: confidence, np.array: xyhw]
    """
    len_cb = len(conf_bb1)
    conf_bb_r = [[] for _ in range(len_cb)]
    conf_bbs = [conf_bb1, conf_bb2]
    for conf_bb in conf_bbs:
        for i in range(1, len_cb): conf_bb[i] = conf_bb[i][conf_bb[0] >= conf_thresh]
        conf_bb[0] = conf_bb[0][conf_bb[0] > conf_thresh]
        if isinstance(conf_bb[0], np.ndarray):
            for i in range(0, len_cb):  conf_bb[i] = torch.from_numpy(conf_bb[i])
    conf_bb1, conf_bb2 = conf_bbs
    l1 = len(conf_bb1[0])
    l2 = len(conf_bb2[0])
    # ind1s, ind2s = torch.where(bb_mat > iou_thresh)
    ab2 = set()
    for i1 in range(l1):
        ious = bbox_iou(conf_bb1[1][i1], conf_bb2[1], False)
        index_intersect = torch.where(ious >= iou_thresh)[0]
        if len(index_intersect) == 0:
            for i in range(len_cb):

                conf_bb_r[i].append(conf_bb1[i][i1])
            continue
        # we assume index_intersect == 1
        i2 = index_intersect[0].item()
        if conf_bb1[0][i1] > conf_bb2[0][i2]:
            for i in range(len_cb):
                conf_bb_r[i].append(conf_bb1[i][i1])
            ab2.add(i2)
            continue
        if i2 in ab2: continue
        for i in range(len_cb):
            conf_bb_r[i].append(conf_bb2[i][i2])
        ab2.add(i2)
    for i2 in range(l2):
        if i2 not in ab2:
            for i in range(len_cb):
                # print("11113414",conf_bb2[i][i2]) # ;exit()
                conf_bb_r[i].append(conf_bb2[i][i2])
    if len(conf_bb_r[0]) == 0:
        return copy.deepcopy(EMPTY)
    for i in range(len_cb):
        # print(i, conf_bb_r[i])
        conf_bb_r[i] = torch.stack(conf_bb_r[i]).numpy()
    # print(conf_bb_r) #;exit()
    return conf_bb_r


def read_txt_single(txt_fn):
    # return [confidence, xyhw, class]
    if not os.path.exists(txt_fn):
        # print("read_txt_single - warning: {} does not exist, return empty arrays".format(txt_fn))
        return copy.deepcopy(EMPTY)
    with open(txt_fn, 'r', encoding='utf8') as f:
        # with open(txt_fn, 'r', encoding='unicode') as f:
        txt = f.read()
    lines = [s for s in txt.split("\n") if len(s) != 0]
    ret = [[],[],[]]
    for l in lines:
        # cls, *xywh, conf
        l = l.split(' ')
        if len(l) == 5: l.append("1.0")
        ret[0].append(int(eval(l[0])))
        ret[1].append(np.array([float(i) for i in l[1:1+4]]))
        ret[2].append(eval(l[-1]))
    if len(ret[0]) == 0: return copy.deepcopy(EMPTY)
    for i in range(3): ret[i] = np.stack(ret[i])
    ret[0], ret[-1] = ret[-1], ret[0]
    return ret


def write_txt_single(txt_fn, xyhw, cls):
    """
    write to a txt file:
        cls1 xyhw1
        cls2 xyhw2
        ...
    :param txt_fn: target file
    :param xyhw: float nx4 array
    :param cls: int array
    :return: None
    """
    with open(txt_fn, 'w', encoding='utf8') as f:
        for xyhw_i, cls_i in zip(xyhw, cls):
            f.write(("{} " * 4 + "{}\n").format(*([cls_i] + xyhw_i.tolist())))
    # print("write {}, {} to {} OK".format(xyhw.shape, cls.shape, txt_fn))


def run(ground_truth_fd, reference_fd, target_fd, iou_thresh=0.75, conf_thresh=0.5, fun=py_nms):
    if not os.path.exists(target_fd):
        os.makedirs(target_fd)
    ground_truth = set(os.listdir(ground_truth_fd))
    references = set(os.listdir(reference_fd))
    files = ground_truth # | references
    objects_truth = 0
    objects_reference = 0
    objects_refined = 0
    for i, f in enumerate(files):
        try:
            f_target = os.path.join(target_fd, f)
            f_truth = os.path.join(ground_truth_fd, f)
            f_reference = os.path.join(reference_fd, f)
            truth_labels = read_txt_single(f_truth)
            objects_truth += len(truth_labels[0])
            reference = read_txt_single(f_reference)
            objects_reference += len(reference[0])
            merged = fun(truth_labels, reference, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
            objects_refined += len(merged[0])
            write_txt_single(f_target, merged[1], merged[2])
        except:
            print("Error processing {} and {}".format(f_truth, f_target))
            continue
        if i % 1000 == 0:
            print("processing {}, objects_truth:{}, objects_reference:{}, objects_refined:{}".format(
                i, objects_truth, objects_reference, objects_refined))
    print("processing {}, objects_truth:{}, objects_reference:{}, objects_refined:{}".format(
        i, objects_truth, objects_reference, objects_refined))


from lzx.yolo.utils.general import xywh2xyxy


def rec_img(img, xyxy, txts=None, color=None):
    for i,  xyxy_i in enumerate(xyxy.astype(np.int)):
        # (color[i][0], color[i][1], color[i][2],)
        #
        clr = (0,255,0) if color is None else \
                            (int(color[i][0]), int(color[i][1]), int(color[i][2]),)
        img = cv2.rectangle(img, (xyxy_i[0], xyxy_i[1]), (xyxy_i[2], xyxy_i[3]),
                            color=clr , thickness=max(img.shape[0] // 500, 1))
        if txts is not None:
            txt = txts[i]
            cv2.putText(img, txt, (xyxy_i[0], xyxy_i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        clr, 3)
    return img




def xyxy_mult_imgshape(xyxy, imgshape):
    xyxy[:,0] *= imgshape[1]
    xyxy[:,2] *= imgshape[1]
    xyxy[:,1] *= imgshape[0]
    xyxy[:,3] *= imgshape[0]
    return xyxy


def _test():
    np.random.seed(0)
    ground_truth_fd = r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\train_ground_truth"
    reference_fd = r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\labels_exp_ori_640_conf08"
    target_fd = r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\target"
    img_fd = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train"
    iou_thresh = 0.75
    conf_thresh = 0.5
    if not os.path.exists(target_fd):
        os.makedirs(target_fd)
    files = set(os.listdir(ground_truth_fd)) | set(os.listdir(reference_fd))
    for i, f in enumerate(files):
        img_fn = os.path.join(img_fd, f.replace(".txt", ".jpg")).replace("____", "__")
        if not os.path.exists(img_fn): continue
        img = cv2.imdecode(np.fromfile(img_fn, dtype=np.uint8), -1)

        # print(f)
        f_truth = os.path.join(ground_truth_fd, f)
        f_reference = os.path.join(reference_fd, f)
        truth_labels = read_txt_single(f_truth)

        reference = read_txt_single(f_reference)
        # print(type(truth_labels), [type(x) for x in truth_labels])
        # merged = merge(truth_labels, reference, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
        # for m in merged: print(m.shape)
        merged = py_nms(truth_labels, reference, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
        # for m in merged: print(m.shape)
        # print(type(truth_labels), [type(x) for x in truth_labels])

        # <class 'list'> [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>]
        # print(merged)
        truth_labels_xyxy = xyxy_mult_imgshape(xywh2xyxy(truth_labels[1]), img.shape)
        reference_xyxy = xyxy_mult_imgshape(xywh2xyxy(reference[1]), img.shape)
        merged_xyxy = xyxy_mult_imgshape(xywh2xyxy(merged[1]), img.shape)

        if truth_labels_xyxy is None or reference_xyxy is None: continue

        truth_labels_img = rec_img(img.copy(), truth_labels_xyxy)
        reference_img = rec_img(img.copy(), reference_xyxy)
        merged_img = rec_img(img.copy(), merged_xyxy)

        cv2.imshow("1", np.concatenate([cv2.resize(im, (480,480)) for im in [
            truth_labels_img, reference_img, merged_img]], 1))
        cv2.waitKey()


import argparse
# ~/lzx/yolo5v0823/runs/val/exp_all1104/labels
# processing 8000, objects_truth:34300, objects_reference:56256, objects_refined:54957

def main():
    if 0:
        run("/home/xz/dataset/hw0805/data_trainable1104/labels/valid/",
            "/home/xz/lzx/yolo5v0823/runs/val/exp_all1104/labels",
            "/home/xz/dataset/hw0805/data_trainable1104/labels/valid/",
            conf_thresh=0.3)
    conf_thresh = 0.01
    iou_thresh = 0.6
    if 0:
        run("/home/xz/dataset/hw0805/data_trainable1228/labels/train/",
        "/home/xz/lzx/yolo5v0823/runs/detect/good_detect1227_train/labels",
        "/home/xz/dataset/hw0805/data_trainable1228/labels/train/",
        conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        run("/home/xz/dataset/hw0805/data_trainable1228/labels/valid/",
       "/home/xz/lzx/yolo5v0823/runs/detect/good_detect1227_train/labels",
        "/home/xz/dataset/hw0805/data_trainable1228/labels/valid/",
        conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # exit()
    _test()
    exit()
    run(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\train_ground_truth",
        r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\labels_exp_ori_640_conf04",
        r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp640_train\target_exp_ori_640_conf04",
        conf_thresh=0.4)


if __name__=="__main__":
    # main();exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fun', type=str, default='nms', help='[nms/mg]')
    parser.add_argument('--ground_truth', type=str)
    parser.add_argument('--reference', type=str)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--iou', type=float, default=0.6)
    args = parser.parse_args()
    if args.target is None:
        args.target = args.ground_truth
    fun = py_nms if args.fun == 'nms' else merge
    run(args.ground_truth,
        args.reference,
        args.target,
        conf_thresh=args.conf, iou_thresh=args.iou, fun=fun)





# conf08: processing 8715, objects_truth:37377, objects_reference:35016, objects_refined:42504
# conf04: processing 8715, objects_truth:37377, objects_reference:45937, objects_refined:50838
