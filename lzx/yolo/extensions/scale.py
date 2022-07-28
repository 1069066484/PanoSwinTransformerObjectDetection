#!/usr/bin/python
# -*- coding: utf-8 -*-
# usage: image scaling processor
# author: Zhixin Ling
# email: 1069066484@qq.com
# example: python scale.py --folder "."  --sz 32 > scale.out 2>&1 &

import os
import cv2
import numpy as np
import argparse


def run(folder, sz_min=960):
    print("Scaling folder {}, sz_min: {}".format(folder, sz_min))
    i = 0
    i_f = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[-1].lower() not in [".jpg",".jpeg",".png"]: continue
            try:
                i += 1
                file = os.path.join(root, file)
                im = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
                shape = im.shape
                print("File {} sz: {}".format(file, shape))
                if shape[0] <= sz_min or shape[1] <= sz_min: continue
                if shape[0] > shape[1]:
                    im = cv2.resize(im, (sz_min, int(shape[0] / shape[1] * sz_min)))
                else:
                    im = cv2.resize(im, (int(shape[1] / shape[0] * sz_min), sz_min))
                cv2.imencode('.jpg', im)[1].tofile(file)
                print("File {} sz: {} -> {}, write to {}".format(
                    file, shape, im.shape, file
                ))
            except:
                i_f += 1
                print("{} failed".format(file))
            print("num: {}, failed num: {}".format(i, i_f))


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='scale.py')
    parser.add_argument('--folder', type=str, default='.', help='The target folder to process. '
                                                                'All images in the folder and its subfolders would be processed.'
                                                                'The original images would not be maintained.')
    parser.add_argument('--sz', type=int, default=960, help='the minimum height or width after scaling')
    opt = parser.parse_args()
    run(opt.folder, opt.sz)