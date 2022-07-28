import os
import split0728
import tt_split
import data_clean


dir_name = "data_trainable1215"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
if not os.path.exists(os.path.join(dir_name, "train")):
    os.mkdir(os.path.join(dir_name, "train"))
if not os.path.exists(os.path.join(dir_name, "valid")):
    os.mkdir(os.path.join(dir_name, "valid"))


if 1:
    split0728.xml2txt_hw(r"hw0805/")
    tt_split.run("hw0805", "{}/images".format(dir_name), "{}/labels".format(dir_name), all_rd=True)
    tt_split.txt_label_trans(r"{}/labels".format(dir_name))
    data_clean.update(r"{}/labels".format(dir_name))
else:
    tt_split.run(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\ori",
                 r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/images",
                 r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/labels")
    tt_split.run(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\ori",
                 r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/images",
                 r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/labels2")
    tt_split.txt_label_trans(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/labels")
    tt_split.txt_label_trans(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/labels2")
    data_clean.update(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data/labels2")


def _visual_test():
    folder = r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\23012301000095_机房002_全景照片_1625193461325"
    for jpg in os.listdir(folder):
        if not jpg.endswith(".jpg"): continue
        jpg = os.path.join(folder, jpg)
        xml = jpg.replace(".jpg", "_mask.xml")


