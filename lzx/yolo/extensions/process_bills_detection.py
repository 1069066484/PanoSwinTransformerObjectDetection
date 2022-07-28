import os


top_folder = r"/home/jqzh/datasets/bills_detection/marked"
with open(os.path.join(top_folder, 'data', 'predefined_classes.txt'), 'r') as f:
    yaml_names = f.read().split('\n')
    yaml_names = list(set(yaml_names))
print(yaml_names)

def mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return d


target_folder = mkdir(os.path.join(top_folder, 'yolo5_data'))
images_folder = mkdir(os.path.join(target_folder, 'images'))
labels_folder = mkdir(os.path.join(target_folder, 'labels'))
xml_folder = os.path.join(top_folder, 'xml')

images_folder_train = mkdir(os.path.join(images_folder, 'train'))
images_folder_valid = mkdir(os.path.join(images_folder, 'valid'))

labels_folder_train = mkdir(os.path.join(labels_folder, 'train'))
labels_folder_valid = mkdir(os.path.join(labels_folder, 'valid'))

name2idx = dict([(n, i) for i, n in enumerate(yaml_names)])

import shutil
from xml.dom.minidom import parse


def cls_pos_xml(xml_f):
    l = []
    domTree = parse(xml_f)
    xm = int(domTree.getElementsByTagName("size")[0].getElementsByTagName("width")[0].childNodes[0].data)
    ym = int(domTree.getElementsByTagName("size")[0].getElementsByTagName("height")[0].childNodes[0].data)
    for o in domTree.getElementsByTagName("object"):
        bndbox = o.getElementsByTagName("bndbox")[0]
        xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
        k = o.getElementsByTagName("name")[0].childNodes[0].data
        if k not in name2idx:
            print("error key: {}, xml_f: {}, skip".format(k, xml_f))
            continue
        cls_name = name2idx[k]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        l.append([cls_name, x_center / xm, y_center / ym, w / xm, h / ym])
    return l


def write_label(f, l):
    with open(f, 'w') as f:
        for item in l:
            f.write(' '.join(str(i) for i in item) + '\n')


icnt = 0
total_ims = 0
for d, _, f_oris in os.walk(os.path.join(top_folder, "file")):
    for f_ori in f_oris:
        f = os.path.join(d, f_ori)
        if f.endswith(".jpg"):
            total_ims += 1
            xml_f = os.path.join(xml_folder, f_ori.split('.')[-2] + '.xml')
            if not os.path.exists(xml_f):
                print("error, image {} without corresponding label {}, skip".format(f, xml_f))
                continue
            label = cls_pos_xml(xml_f)
            icnt += 1
            if icnt % 10:
                shutil.copy(f, images_folder_train)
                write_label(os.path.join(labels_folder_train, f_ori.split('.')[-2] + '.txt'), label)
            else:
                shutil.copy(f, images_folder_valid)
                write_label(os.path.join(labels_folder_valid, f_ori.split('.')[-2] + '.txt'), label)
            if total_ims % 50 == 0:
                print("total images {}, disposed count {}".format(total_ims, icnt))
print("total images {}, disposed count {}".format(total_ims, icnt))


