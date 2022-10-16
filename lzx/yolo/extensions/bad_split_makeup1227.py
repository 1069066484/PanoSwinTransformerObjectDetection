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


for file in os.listdir('.'):
    if file.split('____')[0] not in label_folders:
        shutil.move(file, 'tmp')