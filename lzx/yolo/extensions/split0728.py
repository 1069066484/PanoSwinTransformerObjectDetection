import os
import cv2
import numpy as np
from merge_bbs import *
from utils.general import *


def run_txt(
        source_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\imgaes',
        labels_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\labels',
        out_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\out'):
    if not os.path.exists(out_fd):
        os.mkdir(out_fd)
    for name in os.listdir(source_fd):
        image_name = name
        label_name = name.split('.')[0] + ".txt"
        image_path = os.path.join(source_fd, image_name)
        label_path = os.path.join(labels_fd, label_name)
        if not os.path.exists(label_path):
            print("{} missing, skip".format(label_path))
            continue
        # im = cv2.imread(image_path)cv2.imencode('.jpg',im)[1].tofile('C:\\测试\\你好.jpg')
        im = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        with open(label_path) as f:
            labels = f.readlines()

        labels = [l.strip().split(' ') for l in labels]
        # center_x, center_y, range_x, range_y
        for label in labels:
            max_y, max_x, _ = im.shape
            class_folder = os.path.join(out_fd, label[0])
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)
            center_x = int(max_x * float(label[1]))
            center_y = int(max_y * float(label[2]))
            range_x = int(max_x * float(label[3]))
            range_y = int(max_y * float(label[4]))
            label_x = [center_x - range_x // 2, center_x + range_x // 2]
            label_y = [center_y - range_y // 2, center_y + range_y // 2]
            im_cut = im[label_y[0]:label_y[1], label_x[0]:label_x[1]]
            cv2.imencode('.jpg', im_cut)[1].tofile(os.path.join(class_folder, "{}__{}.jpg".format(name.split('.')[0],len(os.listdir(class_folder)))))
            # cv2.imwrite(os.path.join(class_folder, "{}__{}.jpg".format(name.split('.')[0],len(os.listdir(class_folder)))), im_cut)
# E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\labels\PK2A001424N000000000PAN.txt missing, skip

if 0: run_txt(source_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\imgaes',
        labels_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\labels',
        out_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0703\out')
if 0: run_txt(source_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0718\images',
        labels_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0718\txt',
        out_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0718\out')


import xml
from xml.dom import minidom


def xml2txt(
        xml_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\labels',
        txt_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\txt'):
    if not os.path.exists(txt_fd):
        os.mkdir(txt_fd)
    for xml_f in os.listdir(xml_fd):
        if not xml_f.endswith(".xml"): continue
        path_xml = os.path.join(xml_fd, xml_f)
        xml_data = xml.dom.minidom.parse(path_xml)
        range_x = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("width")[0].childNodes[0].data)
        range_y = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("height")[0].childNodes[0].data)
        objects = xml_data.documentElement.getElementsByTagName('object')
        with open(os.path.join(txt_fd, xml_data.documentElement.getElementsByTagName('filename')[0].childNodes[0].data.split('.')[0] + '.txt'), 'w') as f:
            for object in objects:
                class_name = object.getElementsByTagName("name")[0].childNodes[0].data
                x_min = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")[0].childNodes[0].data)
                y_min = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")[0].childNodes[0].data)
                x_max = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")[0].childNodes[0].data)
                y_max = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")[0].childNodes[0].data)
                # print('min_max', x_min, y_min, x_max, y_max)
                f.write("{} {} {} {} {}\n".format(class_name, (x_max + x_min) / 2 / range_x, (y_max + y_min) / 2 / range_y,
                                                (x_max - x_min) / range_x, (y_max - y_min) / range_y))


if 0:
    xml2txt(
            xml_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\labels',
            txt_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\txt')

    run_txt(
            source_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\0707',
            labels_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\txt',
            out_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\out')


if 0:
    xml2txt(
            xml_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\zyhou0809\书法标注实验',
            txt_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\zyhou0809\书法标注实验')


def format_image(folder, postfix='.jpg'):
    files = [f for f in os.listdir(folder) if not f.endswith(postfix)]
    for f in files:

        f = os.path.join(folder, f)
        im = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1)
        # print(im.min(), im.max(), im.dtype)
        # print("Find file {}: {}".format(f, im.shape))
        im = (im / im.max() * 255).astype(np.uint8)
        cv2.imshow("1", im)
        cv2.waitKeyEx()
        cv2.imencode('.jpg', im)[1].tofile(f.split('.')[0] + postfix)


# format_image(r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\zyhou0809\jpgs", '.jpg')


def read_xml_ch(filePath):
    # ["gbk", "utf-8", "unicode"]
    encoding = "gbk"
    with open(filePath, "r", encoding=encoding) as f:
        r = f.read()
        text = str(r.encode(encoding), encoding=encoding)
        DOMTree = xml.dom.minidom.parseString(text)
        # print("Process {}, {} OK".format(filePath, encoding))
        return DOMTree


def xml2txt_hw1(
        xml_path=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\dataset_motorRoom_0806\230104908000000335_机房001_全景照片_1625229142489\VID_20210621_151612_00_008_000001_mask.xml',
        txt_path=None, write=True, forcererun=False):
    if txt_path is None:
        txt_path = os.path.splitext(xml_path)[0] + ".txt"
    if os.path.exists(txt_path) and write and not forcererun:
        # print("{} exists, skip".format(txt_path))
        return txt_path, None, None, None
    xml_data = read_xml_ch(xml_path)
    range_x = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("width")[0].childNodes[0].data)
    range_y = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("height")[0].childNodes[0].data)
    objects = xml_data.documentElement.getElementsByTagName('object')
    pts = []
    labels = []
    pts_rec = []
    with open(txt_path, 'w', encoding="utf-8") as f:
        for object in objects:
            class_name = object.getElementsByTagName("name")[0].childNodes[0].data
            labels.append(class_name)
            x_min = 1e20
            x_max = -1
            y_min = 1e20
            y_max = -1
            pts.append([])
            pts_rec.append([])
            for p in object.getElementsByTagName("polygon")[0].getElementsByTagName("point"):
                posx = int(p.getAttribute("posx"))
                posy = int(p.getAttribute("posy"))
                pts[-1].append((posx,posy))
                x_min = min(x_min, posx)
                x_max = max(x_max, posx)
                y_min = min(y_min, posy)
                y_max = max(y_max, posy)
            pts_rec[-1].append((x_min, y_min))
            pts_rec[-1].append((x_max, y_max))
            xc = (x_max + x_min) / 2 / range_x
            yc = (y_max + y_min) / 2 / range_y
            xr = (x_max - x_min) / range_x
            yr = (y_max - y_min) / range_y
            print("pts_rec", pts_rec)
            print("pts", pts)
            if write: f.write("{} {} {} {} {}\n".format(class_name, xc, yc, xr, yr))
    return txt_path, pts, pts_rec, labels


def xml2txt_hw2(
        xml_path=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\dataset_motorRoom_0806\230104908000000335_机房001_全景照片_1625229142489\VID_20210621_151612_00_008_000001_mask.xml',
        txt_path=None, write=True, forcererun=False):
    if txt_path is None:
        txt_path = os.path.splitext(xml_path)[0] + ".txt"
    if os.path.exists(txt_path) and write and not forcererun:
        # print("{} exists, skip".format(txt_path))
        return txt_path, None, None, None
    xml_data = read_xml_ch(xml_path)
    range_x = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("width")[0].childNodes[0].data)
    range_y = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("height")[0].childNodes[0].data)
    objects = xml_data.documentElement.getElementsByTagName('object')
    pts = []
    labels = []
    pts_rec = []
    with open(txt_path, 'w', encoding="utf-8") as f:
        for object in objects:
            class_name = object.getElementsByTagName("name")[0].childNodes[0].data
            labels.append(class_name)
            pts.append([])
            pts_rec.append([])
            for p in object.getElementsByTagName("polygon")[0].getElementsByTagName("point"):
                posx = int(p.getAttribute("posx"))
                posy = int(p.getAttribute("posy"))
                pts[-1].append((posx,posy))

            # print("pts_rec", pts_rec, range_x)
            # print("pts", pts[-1])
            pts[-1] = np.asarray(pts[-1])
            ptsm1_meanx = np.mean(pts[-1][:, 0])
            ptsm1_dists2meanx = pts[-1][:, 0] - ptsm1_meanx
            pts[-1] = pts[-1][np.abs(ptsm1_dists2meanx) < 0.5 * range_x]

            x_max = np.max(pts[-1][:, 0])
            y_max = np.max(pts[-1][:, 1])
            x_min = np.min(pts[-1][:, 0])
            y_min = np.min(pts[-1][:, 1])

            xc = (x_max + x_min) / 2 / range_x
            yc = (y_max + y_min) / 2 / range_y
            xr = (x_max - x_min) / range_x
            yr = (y_max - y_min) / range_y

            pts_rec[-1].append((x_min, y_min))
            pts_rec[-1].append((x_max, y_max))

            if write: f.write("{} {} {} {} {}\n".format(class_name, xc, yc, xr, yr))
    return txt_path, pts, pts_rec, labels


def xml2txt_hw3(
        xml_path=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\dataset_motorRoom_0806\230104908000000335_机房001_全景照片_1625229142489\VID_20210621_151612_00_008_000001_mask.xml',
        txt_path=None, write=True, forcererun=True):
    if txt_path is None:
        txt_path = os.path.splitext(xml_path)[0] + ".txt"
    if os.path.exists(txt_path) and write and not forcererun:
        # print("{} exists, skip".format(txt_path))
        return txt_path, None, None, None
    xml_data = read_xml_ch(xml_path)
    range_x = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("width")[0].childNodes[0].data)
    range_y = int(xml_data.documentElement.getElementsByTagName('size')[0].getElementsByTagName("height")[0].childNodes[0].data)
    objects = xml_data.documentElement.getElementsByTagName('object')
    pts = []
    labels = []
    pts_rec = []
    with open(txt_path, 'w', encoding="utf-8") as f:
        for object in objects:
            class_name = object.getElementsByTagName("name")[0].childNodes[0].data
            pts.append([])
            def write_item(_x_max, _x_min, _y_max, _y_min):
                # print("write_item", _x_max, _x_min, _y_max, _y_min, class_name)
                labels.append(class_name)
                pts_rec.append([])
                pts_rec[-1].append(np.array([_x_min, _y_min]))
                pts_rec[-1].append(np.array([_x_max, _y_max]))
                _xc = (_x_max + _x_min) / 2 / range_x
                _yc = (_y_max + _y_min) / 2 / range_y
                _xr = (_x_max - _x_min) / range_x
                _yr = (_y_max - _y_min) / range_y
                if write: f.write("{} {} {} {} {}\n".format(class_name, _xc, _yc, _xr, _yr))
                if _xr > 0.75:
                    print("bad_item", _x_max, _x_min, _y_max, _y_min, class_name)
                    print(xml_path, txt_path)
                    exit()

            for p in object.getElementsByTagName("polygon")[0].getElementsByTagName("point"):
                posx = int(p.getAttribute("posx"))
                posy = int(p.getAttribute("posy"))
                pts[-1].append((posx,posy))

            # print(1111, pts[-1])
            pts[-1] = np.asarray(pts[-1])
            pts_curr = pts[-1].copy()
            if np.max(pts_curr[:, 0]) - np.min(pts_curr[:, 0]) > 0.6 * range_x:
                selected_left = pts_curr[:,0] < 0.5 * range_x
                pts_curr[:,0][selected_left] += range_x

            x_max = np.max(pts_curr[:, 0])
            y_max = np.max(pts_curr[:, 1])
            x_min = np.min(pts_curr[:, 0])
            y_min = np.min(pts_curr[:, 1])

            if x_max >= range_x:
                write_item(range_x, x_min, y_max, y_min)
                write_item(x_max - range_x, 0, y_max, y_min)
            else:
                write_item(x_max, x_min, y_max, y_min)
    return txt_path, pts, pts_rec, labels


def xml2txt_hw(
        xml_fd=r'E:\ori_disks\D\fduStudy\labZXD\repos\datasets\data0728\0728\0707\labels'):
    for root, dirs, files in os.walk(xml_fd):
        for name in files:
            xml = os.path.join(root, name)
            if not xml.endswith(".xml"): continue
            # print("\nFind {}".format(xml))
            try:
                txt = xml2txt_hw3(xml, write=True)
            except:
                print("Process {}, skip".format(xml))
                continue


# xml2txt_hw(r"../t")

# xml2txt_hw(r".")

from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/home/jqzh/ZhixinLing/yolov5hw/fonts/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    # print("111111111111111")
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


from data_clean import clean_xy, update_single, xc_yc_xr_yr2xminymin_xmaxymax


def xml_visual(folder):
    np.random.seed(0)
    for f in os.listdir(folder):
        # if not f.count("VID_20210531_152230_00_011_000841") and \
        # if not f.count("VID_20210614_131609_00_019_00156"): continue
        if not os.path.splitext(f)[-1] == ".jpg": continue
        # if "VID_20210531_152230_00_011_000281" not in f: continue
        f = os.path.join(folder, f)
        img_path = os.path.join(folder, f)
        xml_path = os.path.splitext(f)[0] + "_mask.xml"
        # print(xml_path, img_path)
        txt_path, pts, pts_rec, labels = xml2txt_hw3(xml_path, write=True, forcererun=True)
        # txt_path, pts, pts_rec, labels = xml2txt_hw1(xml_path, write=False)
        # print("labels", labels, pts, pts_rec)
        im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        im = cv2.resize(im, (6080, 3040))
        im2 = np.array(im.copy())
        im3 = np.array(im.copy())

        pts_rec_arr = np.array(pts_rec)
        print("pts_rec_arr", pts_rec_arr)
        shape0 = pts_rec_arr.shape

        xywh = xyxy2xywhn(pts_rec_arr.reshape([-1, 4]).astype(np.float), im.shape[1],
                                                                         im.shape[0]) #.reshape(shape0)
        # print(pts_rec_arr.reshape([-1, 4]))
        # print(xywh)


        # print(im.shape, pts_rec); exit()

        # good_indices = np.array(clean_xy((pts_rec_arr[:, 1, 0] - pts_rec_arr[:, 0, 0]) / im.shape[1] ))

        # contents = [np.array(x)[good_indices]  for x in [pts, pts_rec, labels]]
        # pts, pts_rec, labels = contents
        # print(111,'\n', "\n".join([str(p) for p in pts]),111,'\n',"\n".join([str(p) for p in pts_rec]) , labels)
        thickness = 15

        for i, (pt, label) in enumerate(zip(pts, labels)):
            # print(i, pt, pt_rec, label)
            color = tuple(np.random.randint(0,255, 3, dtype=np.int32))
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.polylines(im, np.int32([pt]), isClosed=True, color=color, thickness=thickness)
            # im = cv2ImgAddText(im, label, pt_rec[0][0], pt_rec[0][1], (225, 255, 255), 150)

        for i, (pt_rec, label) in enumerate(zip(pts_rec, labels)):
            # print(i, pt, pt_rec, label)
            color = tuple(np.random.randint(0,255, 3, dtype=np.int32))
            color = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(im2, pt_rec[0], pt_rec[1], color=color, thickness=thickness)
            im2 = cv2ImgAddText(im2, label, pt_rec[0][0], pt_rec[0][1], (225, 255, 255), 150)
            # print(pt_rec[0][0], pt_rec[0][1])
            sz0 = 1000
            # print(pt_rec, im.shape) # [[3333 1609], [3719 2390]] (3040, 6080, 3)
            # cv2.imshow("22", cv2.resize(im2, (sz0, im.shape[0] * sz0 // im.shape[1])))
            # cv2.waitKeyEx()

        if 1:
            sz0 = 500
            im = cv2.resize(im, (sz0, im.shape[0] * sz0// im.shape[1]))
            im2 = cv2.resize(im2, (sz0, im.shape[0] * sz0 // im.shape[1]))
            ims = np.concatenate([im, im2], 0)
            cv2.imshow("1", ims)
            cv2.waitKeyEx()
            continue

        labels_u, xc_yc_xr_yr_u = update_single(list(labels), xywh)
        for xc_yc_xr_yr_u_i, label in zip(xc_yc_xr_yr_u, labels_u):
            # print(xc_yc_xr_yr_u_i, label)
            color = tuple(np.random.randint(0,255, 3, dtype=np.int32))
            color = (int(color[0]), int(color[1]), int(color[2])//2+128)
            xminymin_xmaxymax = xc_yc_xr_yr2xminymin_xmaxymax(xc_yc_xr_yr_u_i)
            # print(xc_yc_xr_yr_u_i, xminymin_xmaxymax)
            cv2.rectangle(im3, (int(xminymin_xmaxymax[0] * im.shape[1]), int(xminymin_xmaxymax[1] * im.shape[0])),
                                (int(xminymin_xmaxymax[2] * im.shape[1]), int(xminymin_xmaxymax[3] * im.shape[0])), color=color, thickness=thickness)
            topleft = (int((xc_yc_xr_yr_u_i[0] - xc_yc_xr_yr_u_i[2] * 0.5) * im.shape[1]), int((xc_yc_xr_yr_u_i[1] - xc_yc_xr_yr_u_i[3] * 0.5) * im.shape[0]))
            # bottomright = (int((xc_yc_xr_yr_u_i[0] + xc_yc_xr_yr_u_i[2] * 0.5) * im.shape[1]), int((xc_yc_xr_yr_u_i[1] + xc_yc_xr_yr_u_i[3] * 0.5) * im.shape[0]))
            # print(list(topleft), list(bottomright))
            im3 = cv2ImgAddText(im3, label,
                                topleft[0],
                                topleft[1],
                                (225, 255, 255), 150)

        sz0 = 500
        im = cv2.resize(im, (sz0, im.shape[0] * sz0// im.shape[1]))
        im2 = cv2.resize(im2, (sz0, im.shape[0] * sz0 // im.shape[1]))
        im3 = cv2.resize(im3, (sz0, im.shape[0] * sz0 // im.shape[1]))
        ims = np.concatenate([im, im2, im3], 0)
        cv2.imshow("1", ims)
        cv2.waitKeyEx()


# xml_visual(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\23012301000095_机房002_全景照片_1625193461325")
# xml2txt_hw(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\23012301000095_机房002_全景照片_1625193461325")
# xml2txt_hw()




def update_visual(folder_im=r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_trainable0927\images\valid", folder_lb=r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_trainable0927\labels\valid"):

    for f_im in os.listdir(folder_im):
        # if "业研究所东_机房_全景照片_1625654669551____VID_20210614_131609_00_019_00156" not in f_im: continue
        #
        if "230129908000000108_" not in f_im: continue
        f_lb = os.path.splitext(f_im)[0] + ".txt"
        f_im = os.path.join(folder_im, f_im)
        f_lb = os.path.join(folder_lb, f_lb)
        print(f_im, f_lb)
        im = cv2.imdecode(np.fromfile(f_im, dtype=np.uint8), -1)
        confidence, xyhw, cls = read_txt_single(f_lb)

        im1 = im.copy()
        rec_img(im1, xywhn2xyxy(xyhw, im.shape[1], im.shape[0]), [str(i) for i in cls])
        im2 = im.copy()

        cls_u, xyhw_u = update_single(cls.tolist(), xyhw)

        rec_img(im2, xywhn2xyxy(xyhw_u, im.shape[1], im.shape[0]), [str(i) for i in cls_u])
        cv2.imshow("1", cv2.resize(im1, (640, 640)))
        cv2.imshow("2", cv2.resize(im2, (640, 640)))
        cv2.waitKeyEx()


# update_visual()
# tuRSTUVyaW_230129908000000018_kbcl_mnopqqrs_1630459525895
# 23012301000095_机房002_全景照片_1625193461325
xml_visual(r"E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\tuRSTUVyaW_230129908000000018_kbcl_mnopqqrs_1630459525895")

