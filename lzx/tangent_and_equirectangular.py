import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from lzx.utils import torch_stat_dict
from lzx.visual_utils import *
import torch


torch.set_printoptions(precision=4, sci_mode=False)


def tangent_xy2equirectangular_uv(xy, uv0):
    """
    xy - torch.Size([289, 2])
    uv0 - torch.Size([2]), u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    return uv - torch.Size([289, 2]),  u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    """
    rho = torch.clip(torch.norm(xy, p=2, dim=-1), min=1e-8)
    tanT_rho = torch.arctan(rho)
    u = uv0[0] + torch.atan2(xy[:,0] * torch.sin(tanT_rho), (rho * torch.cos(uv0[1]) * torch.cos(tanT_rho) -
                                  xy[:,1] * torch.sin(uv0[1]) * torch.sin(tanT_rho)))
    v = torch.arcsin(torch.cos(tanT_rho) * torch.sin(uv0[1]) +
                       xy[:,1] * torch.sin(tanT_rho) * torch.cos(uv0[1]) / rho
                       )
    # u[(u < 0) * v ] += math.pi
    uv = torch.stack([u, v], -1)
    return uv


def equirectangular_uv2tangent_xy(uv, uv0):
    """
    uv - torch.Size([289, 2]),  u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    uv0 - torch.Size([2]), u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    return xy - torch.Size([289, 2])
    """
    numerator = torch.sin(uv0[1]) * torch.sin(uv[:,1]) + \
                torch.cos(uv0[1]) * torch.cos(uv[:,1]) * torch.cos(uv[:,0] - uv0[0])
    x = (torch.cos(uv[:,1]) * torch.sin(uv[:,0] - uv0[0])) / numerator
    y = (torch.cos(uv0[1]) * torch.sin(uv[:,1]) -
        torch.sin(uv0[1]) * torch.cos(uv[:,1]) * torch.cos(uv[:,0] - uv0[0])) / numerator
    xy = torch.stack([x, y], -1)
    return xy


def _test_tangent_xy2equirectangular_uv():
    n = 11
    xy = torch.meshgrid([torch.arange(n)]*2)
    xy = torch.stack(xy, -1).float() - n // 2

    xy[1:-1,1:-1] = 1000
    # xy[1:-1] = 1000
    xy = xy[xy<900]

    xy = xy.reshape([-1, 2])
    xy = xy / n
    pi = math.pi
    print("xy", xy)
    # xy[:,0] *= 0.2
    # xy[:, 1] *= 0.3
    uv0 = torch.tensor([0., -0.3 * pi]).float()
    print(torch_stat_dict(xy))
    show_and_wait(scatter(xy.numpy()), name="0")
    uv = tangent_xy2equirectangular_uv(xy=xy, uv0=uv0)
    torch.set_printoptions(precision=4, sci_mode=False)
    print(torch_stat_dict(uv))
    print("uv", xy)
    print(uv)
    show_and_wait(scatter(uv.numpy()), name="1")
    xy2 = equirectangular_uv2tangent_xy(uv=uv, uv0=uv0)
    print(torch_stat_dict(xy2))
    show_and_wait(scatter(xy2.numpy()), name="2")
    # print(xy2, xy)


def _test_equirectangular_uv2tangent_xy():
    torch.set_printoptions(precision=4, sci_mode=False)
    n = 21
    u = torch.arange(n).float() - n // 2
    v = torch.arange(n).float() - n // 2
    print(v)
    pi = math.pi
    u = u / n * pi
    v = v / n * pi
    print(torch_stat_dict(u))
    print(torch_stat_dict(v))

    uv = torch.meshgrid([u, v])
    uv = torch.stack(uv, -1).reshape([-1, 2]) * 0.4
    print(uv.max(), uv.min())
    uv0 = torch.tensor([pi, 0.2 * pi]).float()

    show_and_wait(scatter(uv.numpy()), name="0")
    xy = equirectangular_uv2tangent_xy(uv=uv, uv0=uv0)
    print(torch_stat_dict(xy))
    show_and_wait(scatter(xy.numpy()), name="1")
    # uv2 = tangent_xy2equirectangular_uv(xy=xy, uv0=uv0)
    # show_and_wait(scatter(uv2.numpy()), name="2")


def _test_equirectangular_xywh2tangent_xy():
    # min_xy = [-pi, -0.5pi], max_xy = [pi, 0.5pi]
    pi = math.pi
    xywh = [0, 0, 0.2 * pi, 0.2 * pi]


def _test_build_equirectangular_uvwh2tangent_xy_dict():
    # min_xy = [-pi, -0.5pi], max_xy = [pi, 0.5pi]
    pi = math.pi
    resolution = 10
    vwh = torch.zeros(resolution, resolution, resolution, dtype=torch.float16)
    for v in torch.arange(-0.5 * pi, 0.5 * pi, step=pi / resolution):
        for w in torch.arange(-pi, pi, step=2 * pi / resolution):
            for h in torch.arange(-0.5 * pi, 0.5 * pi, step=pi / resolution):
                center_uv = [0, v]
                topleft_uv = [-0.5 * w, v-0.5 * h]
                topright_uv = [-0.5 * w, v+0.5 * h]
                bottomleft_uv = [0.5 * w, v-0.5 * h]
                bottomright_uv = [0.5 * w, v+0.5 * h]
                uv = torch.tensor([
                    center_uv,
                    topleft_uv,
                    topright_uv,
                    bottomleft_uv,
                    bottomright_uv
                ])
                show_and_wait(scatter(uv.numpy()), name="equirectangular", w=False)
                xy = equirectangular_uv2tangent_xy(uv=uv, uv0=uv[0])
                show_and_wait(scatter(xy.numpy()), name="tangent", w=True)


from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox
import os


def get_img_and_all_bb(
        imgi,
        root = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images",
        annFile = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\train.json",
        ):
    coco = COCO(annFile)
    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda cat: cat['id'])
    # print(categories);exit()
    if 1:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
        image = images[imgi]['file_name']
        image = os.path.join(root, image)
        annIds = coco.getAnnIds(imgIds=imgIds[imgi])
    else:
        # 2492, 2493
        imgIds = [2492]
        images = coco.loadImgs(imgIds)
        image = images[0]['file_name']
        image = os.path.join(root, image)
        annIds = coco.getAnnIds(imgIds=imgIds[0])
    ann = coco.loadAnns(
        annIds)  # [{'area': 6000, 'iscrowd': 0, 'image_id': 3, 'bbox': [481, 533, 75, 80], 'category_id': 0, 'id': 17, 'ignore': 0, 'segmentation': []}, {'area': 7047,
    print(image)
    image = cv2.imread(image)

    # print(list(ann[0].keys()));exit()
    boxes = np.array([a['bbox'] for a in ann])

    cat_names = [categories[a['category_id'] - 1]['name'] for a in ann]
    cat_ids = [a['category_id'] for a in ann]
    # print(cat_names);    exit()
    # scale = 2
    # image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
    # boxes = boxes // scale
    return image, boxes, cat_names, cat_ids


def make_xys(tan_wh2, n=10, gap=0.1):
    xys = []
    """
        [[ + tan_wh2[0],  + tan_wh2[1]],
    [ - tan_wh2[0],  + tan_wh2[1]],
    [ + tan_wh2[0], - tan_wh2[1]],
    [ - tan_wh2[0], - tan_wh2[1]],
    """
    # print(tan_wh2)
    if gap is not None:
        n = int((tan_wh2[1] * 2) / gap)
    for i in range(n+1):
        xys.append([tan_wh2[0], - tan_wh2[1] + tan_wh2[1] * 2 / n * i])
        xys.append([-tan_wh2[0], - tan_wh2[1] + tan_wh2[1] * 2 / n * i])
    if gap is not None:
        n = int(tan_wh2[0] * 2 / gap)
    for i in range(n + 1):
        xys.append([- tan_wh2[0] + tan_wh2[0] * 2 / n * i, tan_wh2[1]])
        xys.append([- tan_wh2[0] + tan_wh2[0] * 2 / n * i, -tan_wh2[1]])
    xys = torch.tensor(xys)
    # print(xys); exit()
    return xys


def uv_expand(uv_all, WH):
    print(uv_all, WH)
    size_h = WH[1]
    uv_all = uv_all / math.pi * size_h

    uv_all[:, 0] += size_h
    uv_all[:, 1] += size_h // 2
    print(uv_all, WH)
    uv_all[:, 0][uv_all[:, 0] < 0] += WH[0]
    uv_all[:, 0][uv_all[:, 0] > WH[0]] -= WH[0]
    # exit()
    return uv_all


def _test_true_tangent_xy2equirectangular_uvwh():
    # train -- 2024  good
    # 7l8KH.jpg
    # 7fQR9

    im_real, boxes, cat_names, cat_ids = get_img_and_all_bb(
        annFile = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\coco_format\test.json",
        imgi=613,
    )
    # 2492
    uv_all = []
    WH = np.array([im_real.shape[1], im_real.shape[0]]) # 1920 - 960
    np.random.seed(0)
    colors = [np.random.randint(100, 255, 3) for _ in range(50)]

    for i, (box, cat, cat_id) in enumerate(zip(boxes, cat_names, cat_ids)):
        # if cat != 'person': continue
        box[1] = -box[1]
        # if box[1] < 1 or i == 78: continue

        color = colors[cat_id]
        color = tuple(color.tolist())

        uv0 = torch.tensor(box[:2])
        true_pos_uv0 = ((uv0 + torch.Tensor([math.pi, math.pi * 0.5])) / math.pi * WH[1]).int().numpy().tolist()
        im_real = cv2.putText(im_real, text=cat + "{}".format(round(np.random.rand() * 0.4 + 0.6, 1)),
                              org=[true_pos_uv0[0]-60, true_pos_uv0[1]],
                              fontScale=1, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color)
        tan_wh = box[2:4] * 5.3
        # tan_wh = box[2:4] * 4

        pi = math.pi
        UV = np.array([2 * pi, pi])
        tan_wh = np.array(tan_wh) / WH * UV
        tan_wh2 = tan_wh * 0.5
        # print(uv0, tan_wh)

        # xy = make_xys(tan_wh2, n=20, gap=0.002)
        xy = make_xys(tan_wh2, n=20, gap=0.002)
        # print(xy.shape); exit()

        uv = tangent_xy2equirectangular_uv(xy=xy, uv0=uv0)
        # print(xy,'\n', uv0,'\n' , uv)
        # exit()

        # uv_all.append(uv)
        uv = uv_expand(uv, WH)
        im_real = scatter(uv.numpy(), image=im_real, scale=0, size=(WH[1], WH[0]), thickness_fact=0.1, color=color)

        # 最右边 (0.3142, 0.3299)
        # 次右边 (-0.3142,  0.5498]
        # print("uv0=", uv0) # uv0= tensor([0.1325, 1.1797], dtype=torch.float64)
        if 0:
            for xyi, uvi in zip(xy, uv):
                print(xyi, uvi)
                # 最右边 tensor([0.3142, 0.3299], dtype=torch.float64) tensor([1407.7789,  868.7537], dtype=torch.float64)
                # 次右边 tensor([-0.3142,  0.5498], dtype=torch.float64) tensor([1363.0220,  871.2645], dtype=torch.float64)
                uvi = [int(uvi[0].item()), int(uvi[1].item())]
                im_real2 = cv2.putText(im_real, text="{},{}".format(round(xyi[0].item(),2 ),round(xyi[1].item(),2 )), org=[uvi[0], uvi[1]],
                                      fontScale=0.7, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color)
                cv2.imshow("1", cv2.resize(im_real2, (1200, 600))); cv2.waitKey()
        # show_and_wait(cv2.resize(im_real, (1200, 600)), name="0")
    # uv_all = torch.cat(uv_all, 0)
    # uv_expand(uv_all, WH)
    # im_plot = np.clip(im_real + im, 0, 255)
    show_and_wait(cv2.resize(im_real, (1200, 600)), name="0")
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7frn3.jpg"
    # E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7fQR9.jpg


def _debug():
    xy = torch.tensor([
        # [0.3142, 0.3299],
        [0.3142, 0.41],
        [0.3142, 0.42],
        # [-0.3142,  0.5498]
    ])
    print(xy.shape)
    uv0 = torch.tensor([0.1325, 1.1797]).float()
    # show_and_wait(scatter(xy.numpy()), name="0")

    uv = tangent_xy2equirectangular_uv(xy=xy, uv0=uv0)
    print(uv)
    # exit()

    torch.set_printoptions(precision=4, sci_mode=False)
    # print(torch_stat_dict(uv))
    print(xy)
    print("uv")
    print(uv)

    size = 400
    uv[:, 0] = (uv[:, 0] + math.pi) / math.pi * size
    uv[:, 1] = (uv[:, 1] + math.pi * 0.5) / math.pi * size
    print(uv)
    show_and_wait(scatter(uv.numpy(), size=(size, size * 2), scale=False, thickness_fact=0.5), name="1")
    xy2 = equirectangular_uv2tangent_xy(uv=uv, uv0=uv0)
    print(torch_stat_dict(xy2))
    show_and_wait(scatter(xy2.numpy(), scale=False), name="2")


if __name__=='__main__':
    # _test_true_tangent_xy2equirectangular_uvwh()
    # _test_build_equirectangular_uvwh2tangent_xy_dict()
    # _test_tangent_xy2equirectangular_uv()
    _test_equirectangular_uv2tangent_xy()
    # _debug()

