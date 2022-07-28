import os
import shutil
import pickle as pkl
import cv2
import numpy as np
import copy
# hw0805/第二批10-7/哈道里二场村南(1.jpg


label_folders = {  # '230123600000000355_变压器_全景照片_1625214057757',
    '230129908000000108_延寿寿山开道林场_机房',
    # '230108700000236438_铁塔_全景照片_1625208784657',
    '230129908000000155_机房_全景照片_1625213689683',
    '230111908000000597_机房_全景照片_1625812710353',
    # '230110908000000011哈香坊孙家村_机房001_全景照片_1625810968649',
    '230109908000000223_机房001_全景照片_1625229621229',
    '230124908000000191_方正先锋村_机房001_全景照片_1625638746782',
    '23010301000188南岗王岗镇蚕业研究所东_机房_全景照片_1625654669551',
    '230109500000000160_哈松北普宁医院东南__机房__002__全景照片_1625792803984',
    '230111700000153115哈呼兰利民雪花啤酒_机房_全景照片_1625812621524',
    '哈阿城中都大街_230112500000000101_机房2',
    '230112908000000249_阿城平山北川_机房001',
    '230109908000000416利民黑建职A食堂_机房001_全景照片_1625793037819',
    # '230112908000000327_变压器_全景照片_1625216047745'
}


def split(img_folder=r"processed_lxy1112/new_img", txt_folder=r"processed_lxy1112/new_label", target_folder="data_lxy1112"):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    images_folder = os.path.join(target_folder, "images")
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)
    label_folder = os.path.join(target_folder, "labels")
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)
    images_folder_train = os.path.join(images_folder, "train")
    images_folder_test = os.path.join(images_folder, "valid")
    label_folder_train = os.path.join(label_folder, "train")
    label_folder_test = os.path.join(label_folder, "valid")
    for p in [images_folder_train, images_folder_test, label_folder_train, label_folder_test]:
        if not os.path.exists(p):
            os.mkdir(p)

    cnts = [0,0,0,0]
    for im in os.listdir(img_folder):
        if im.split("____")[0] in label_folders:
            shutil.copy(os.path.join(img_folder, im), images_folder_test)
            cnts[0] += 1
        else:
            shutil.copy(os.path.join(img_folder, im), images_folder_train)
            cnts[1] += 1
    for txt in os.listdir(txt_folder):
        if txt.split("____")[0] in label_folders:
            shutil.copy(os.path.join(txt_folder, txt), label_folder_test)
            cnts[2] += 1
        else:
            shutil.copy(os.path.join(txt_folder, txt), label_folder_train)
            cnts[3] += 1
    print("split done, image-val: {}, image-train: {}, txt-val: {}, txt-train: {}"
          .format(cnts[0], cnts[1], cnts[2], cnts[3]))


if __name__=="__main__":
    split()