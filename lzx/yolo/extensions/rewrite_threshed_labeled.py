import cv2
import os
import numpy as np
import torch

from utils.metrics import bbox_iou

from utils.general import xywh2xyxy, xyxy2xywhn, xywhn2xyxy
import copy
EMPTY = [np.random.rand(0), np.random.rand(0,4), np.random.rand(0).astype(np.int)]


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


def run(folder, conf_thresh=0.1):
    count = 0
    total_obj = 0
    saved_obj = 0
    files = list(os.listdir(folder))
    for i, f in enumerate(files):
        if not f.endswith(".txt"): continue
        if count % 1000 == 0:
            print("processed valid files {}, file count {}, total file count {}, total_obj: {}, saved_obj: {}".format(
                count, i, len(files), total_obj, saved_obj))
        try:
            count += 1
            f = os.path.join(folder, f)
            truth_labels = read_txt_single(f)
            total_obj += len(truth_labels[0])
            good_conf = truth_labels[0] > conf_thresh
            saved_obj += np.sum(good_conf)
            write_txt_single(f, truth_labels[1][good_conf], truth_labels[2][good_conf])
        except:
            print("Error processing {}".format(f))
            continue


if __name__=="__main__":
    run(".", 0.1)


'''
Namespace(atts=3, aug=1, auto_stop=50, batch_size=60, bb='gn', dataset='sketchy', decay=0.999, epochs=25, folder_top='/home/hadoop/project/ZhixinLing/datasets/Sketchy/256x256', gpu=0, imagenet_seresnext=None, individual_trp=0, lr=0.0001, max_step=200000, negwhat='im', opt='adam', print_every=50, save=1, save_dir='save_adram/adram_trpw110_negim', save_every=200, savet=40, search=0, seed=0, skip=1, start_from=None, syn_step=1000, syn_type=0, trp=0.3, type=15, weights="{'trp_individual':1.0,'trp_total':10.0,'loss_sk_att':1.0,'loss_im_att':1.0,'trp_individual_negs':0.0, '_thresh':0.9}", with_free_ovlp=0, workers=5)


Namespace(atts=3, aug=1, auto_stop=50, batch_size=60, bb='gn', dataset='sketchy', decay=0.999, epochs=25, folder_top='/home/hadoop/project/ZhixinLing/datasets/Sketchy/256x256', gpu=0, imagenet_seresnext=None, individual_trp=0, lr=0.0001, max_step=200000, negwhat='im', opt='adam', print_every=50, save=1, save_dir='save_adram/adram_trpw110_negim_idvneg1', save_every=200, savet=40, search=0, seed=0, skip=1, start_from=None, syn_step=1000, syn_type=0, trp=0.3, type=15, weights="{'trp_individual':1.0,'trp_total':10.0,'loss_sk_att':1.0,'loss_im_att':1.0,'trp_individual_negs':1.0, '_thresh':0.9}", with_free_ovlp=0, workers=5)


'''