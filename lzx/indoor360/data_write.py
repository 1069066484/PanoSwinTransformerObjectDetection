import json
import os
import cv2


def id_update(file):
    with open(file, 'rb') as f:
        d = json.load(f)
    for i in range(len(d['annotations'])):
        d['annotations'][i]['id'] = i + 1
    with open(file, 'w') as f:
        json.dump(d, f)


def _test_id_update():
    f = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\train.json"
    id_update(f)


if __name__=='__main__':
    _test_id_update()


