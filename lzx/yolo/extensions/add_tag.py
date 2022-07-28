import os
import shutil


def run(folder, tag):
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            if os.path.splitext(path)[-1] in [".jpg", ".txt"]:
                shutil.move(path, "{}____{}_{}".format(path.split("____")[0], tag, path.split("____")[1]))


run("data_lxy1112_", "lxy")