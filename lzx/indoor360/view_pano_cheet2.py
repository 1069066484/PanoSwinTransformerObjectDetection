from lzx.tangent_and_equirectangular import \
    tangent_xy2equirectangular_uv, get_img_and_all_bb, make_xys, equirectangular_uv2tangent_xy
import math
import numpy as np
import torch
import cv2
from lzx.visual_utils import *


pi = math.pi
UV = torch.tensor([2 * pi, pi])
BOTTOM_RIGHT_UV = torch.Tensor([math.pi, math.pi * 0.5])
torch.set_printoptions(precision=4, sci_mode=False, threshold=np.inf)
POLES = torch.tensor([[0, -0.5 * math.pi], [0, 0.5 * math.pi]])

def standalize_360indoor_uvwh0_box(box, WH):
    if not isinstance(box, torch.Tensor):
        box = torch.tensor(box)
    box[..., -3] = -box[..., -3]
    box[...,-2:] *= 5.3
    box[...,-2:] = box[...,-2:] / WH * UV
    return box


def unbias2bias_uv(uv0, H=1):
    return ((uv0 + BOTTOM_RIGHT_UV) / math.pi * H)


def uv_expand(uv_all, WH):
    size_h = WH[1]
    uv_all = uv_all / math.pi * size_h
    uv_all[:, 0] += size_h
    uv_all[:, 1] += size_h // 2
    uv_all[:, 0][uv_all[:, 0] < 0] += WH[0]
    uv_all[:, 0][uv_all[:, 0] > WH[0]] -= WH[0]
    return uv_all


def equirectangular_bounding_xyxy(uvwh, WH):
    if isinstance(uvwh, np.ndarray):
        uvwh = torch.from_numpy(uvwh)
    xy = make_xys(uvwh[2:]/2, n=2, gap=None)
    uv = tangent_xy2equirectangular_uv(xy=xy, uv0=uvwh[:2])
    uvuv = torch.tensor([[uv[:,0].min(), uv[:,1].min()], [uv[:,0].max(), uv[:,1].max()]])
    poles = torch.abs(equirectangular_uv2tangent_xy(uv=POLES, uv0=uvwh[:2])) * 2
    if poles[0][0] <= uvwh[2] and poles[0][1] <= uvwh[3]:
        # north pole
        uvuv[0][0] = -math.pi
        uvuv[0][1] = -0.5 * math.pi
        uvuv[1][0] = math.pi
        xyxy = [uv_expand(uvuv, WH).reshape([-1])]
    elif poles[1][0] <= uvwh[2] and poles[1][1] <= uvwh[3]:
        # south pole
        uvuv[0][0] = -math.pi
        uvuv[1][1] = 0.5 * math.pi
        uvuv[1][0] = math.pi
        xyxy = [uv_expand(uvuv, WH).reshape([-1])]
    elif uvuv[0][0] < -math.pi or uvuv[1][0] > math.pi:
        uvuv2 = uvuv.clone()
        uvuv[0][0] = -math.pi
        uvuv2[0][0] += 2 * math.pi
        uvuv2[1][0] = math.pi
        xyxy = [uv_expand(uvuv_, WH).reshape([-1]) for uvuv_ in [uvuv, uvuv2]]
    else:
        xyxy = [uv_expand(uvuv, WH).reshape([-1])]
    return torch.stack(xyxy, 0)


def visual_uvbox(std_uvwh, WH, gap=0.005):
    xy = make_xys(std_uvwh[2:4] * 0.5, gap=gap)
    uv = tangent_xy2equirectangular_uv(xy=xy, uv0=std_uvwh[:2])
    # print(uv); input()
    uv = uv_expand(uv, WH)
    return uv


def get_visual_image(np_hwc,
                     box_ori_uvwh,
                     cat_names=None,
                     cat_ids=None,
                     colors=None,
                     plt_name=True,
                     is_boxstd=False,
                     plt_xyxy=True,
                     probs=None
                     ):
    WH = torch.tensor([np_hwc.shape[1], np_hwc.shape[0]])
    if cat_names is None:
        cat_names = [""] * len(box_ori_uvwh)
    if cat_ids is None:
        names = list(set(cat_names))
        cat_ids = [names.index(n) for n in names]
    if colors is None:
        colors = [np.random.randint(100, 255, 3) for _ in range(50)]
    for i, box_ori in enumerate(box_ori_uvwh):
        if not is_boxstd:
            box = standalize_360indoor_uvwh0_box(box_ori, WH)
        else:
            box = box_ori

        # if cat_names[i] == 'picture' and np.random.rand() < 0.5: continue
        if 0:
            box += box * (torch.rand(4) * 0.04 - 0.02)

            if cat_names[i] == 'sink':
                colors[cat_ids[cat_names.index('sink')]] = colors[cat_ids[cat_names.index('toilet')]]
                cat_names[i] = 'toilet'

            if cat_names[i] == 'window' and box[0] > 0:
                box[1] += 0.1
                box[3] *= 0.4

        # if cat_names[i] != 'cabinet' or box[0] > -0.6 * math.pi: continue
        # if cat_names[i] != 'light' or box[1] > -0.4 * math.pi: continue
        xyxys = equirectangular_bounding_xyxy(box, WH).int().numpy()
        color = tuple(colors[cat_ids[i]].tolist())
        if plt_xyxy:
            for xyxy in xyxys:
                np_hwc = cv2.rectangle(np_hwc,
                                        (xyxy[0], xyxy[1]),
                                        (xyxy[2], xyxy[3]),
                                        color=color, thickness=2)

        if plt_name:
            prob = round(probs[i], 2) if probs is not None else ""

            xyxy = xyxys[0]
            org = xyxy[:2]
            org[1] -=5
            org[0] = ((xyxy[0] + xyxy[2]) * 0.5 - 40 - 20 * (prob == ""))
            # else:
            #   true_pos_uv0 = unbias2bias_uv(box[:2], WH[1]).int().numpy().tolist()
            #   org = [true_pos_uv0[0] - 60, true_pos_uv0[1]]
            if org[1] < 0:
                org[1] = 30

            np_hwc = cv2.putText(np_hwc, text=cat_names[i] + " {}".format(prob),
                                  org=org,
                                  fontScale=1, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color)
        uv = visual_uvbox(box, WH, gap=0.002)
        # color = (255, 0, 0)  # red
        np_hwc = scatter(uv.numpy(), image=np_hwc, scale=0, size=(WH[1], WH[0]), thickness_fact=0.1, color=color)
    return np_hwc


def run():
    np.random.seed(0)
    colors = [np.random.randint(100, 255, 3) for _ in range(50)]
    # imgi= 12

    # E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\11454324954_1b858800d1_k.jpg
    # std: tensor([1.9619, 0.6300, 1.0407, 0.5550], dtype=torch.float64)

    """
    visual imgi=724
    
    """

    for imgi in range(20, 100):
        # imgi = 8
        # imgi= 17
        # imgi = 2

        probs = np.random.rand(100) * 0.4 + 0.6 + (np.random.rand(100) * 0.2 - 0.1)
        imgi = 1000

        print("imgi=", imgi)
        im_real, boxes, cat_names, cat_ids = get_img_and_all_bb(
            annFile=r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\coco_format\test.json",
            imgi=imgi)

        cat_names.append('light')
        cat_ids.append(13)
        boxes = np.concatenate([boxes, np.array([[-0.05, 0.9, 12, 12]])],0)

        cat_names.append('light')
        cat_ids.append(13)
        boxes = np.concatenate([boxes, np.array([[-1.9, 0.25, 16, 20]])],0)

        cat_names.append('light')
        cat_ids.append(13)
        boxes = np.concatenate([boxes, np.array([[-2.25, 0.25, 12, 13]])],0)


        cat_names.append('picture')
        cat_ids.append(27)
        boxes = np.concatenate([boxes, np.array([[-2.35, -0.1, 12, 30]])],0)


        cat_names.append('chair')
        cat_ids.append(26)
        boxes = np.concatenate([boxes, np.array([[3.05, -0.45, 12, 16]])],0)


        WH = torch.tensor([im_real.shape[1], im_real.shape[0]])  # 1920 - 960
        np.random.seed(0)
        im_real = get_visual_image(im_real, boxes, cat_names=cat_names, cat_ids=cat_ids, colors=colors, plt_xyxy=False, probs=probs)
        show_and_wait(cv2.resize(im_real, (380*2, 380)), name="0")


if __name__=='__main__':
    run()