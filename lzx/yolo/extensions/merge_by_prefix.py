import os
import shutil


def run(top):
    save_dirs = os.path.join(top, "save")
    if not os.path.exists(save_dirs):
        os.mkdir(save_dirs)
    for file in os.listdir(top):
        if not os.path.isdir(file):
            pref = file.split('__')[0]
            save_fd = os.path.join(save_dirs, pref)
            if not os.path.exists(save_fd):
                os.mkdir(save_fd)
            shutil.copy(os.path.join(top, file), save_fd)


run(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5v0823\runs\detect\exp643")