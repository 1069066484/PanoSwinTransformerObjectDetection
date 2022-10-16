import cv2
import einops
import matplotlib.pyplot as plt
import math
import numpy as np
from lzx.utils import torch_stat_dict
from lzx.visual_utils import *
import torch
from lzx.tangent_and_equirectangular import equirectangular_uv2tangent_xy
import torch.nn.functional as F


NORTH_POLE = torch.tensor([0, -0.5]) * math.pi


def uv2xyz(uv, r=1):
    """
    u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    @param uv: tensor(b, u, v),
    @param r: radius
    @return:
    """
    x = torch.sin(uv[:, 0]) * torch.sin(uv[:, 1] + math.pi * 0.5)
    y = torch.cos(uv[:, 0]) * torch.sin(uv[:, 1] + math.pi * 0.5)
    z = torch.cos(uv[:, 1] + math.pi * 0.5)
    xyz = torch.stack([x, y, z], -1) * r
    return xyz


def rotate(np_uv, s_uv, eps=1e-15):
    """
    u ~ [-pi, pi), v ~ [-0.5pi, 0.5pi]
    @param np_uv: north pole, tensor(u, v)
    @param s_uv: source uv, tensor(b, u, v)
    @return: target: uv, tensor(b, u, v)
    """
    radius = 1
    s_uv = torch.cat([s_uv, torch.Tensor([[0, -0.5 * math.pi]]).to(s_uv.device)], 0)
    np_xyz = uv2xyz(np_uv[None,:], r=radius)
    s_xyz = uv2xyz(s_uv, r=radius)
    d_np2s = torch.norm(np_xyz - s_xyz, dim=1, p=2)
    v_new = 2 * torch.asin(d_np2s / (2 * radius)) * radius - 0.5 * math.pi
    directions = torch.cross(s_xyz, np_xyz.repeat(s_xyz.shape[0], 1))
    directions = F.normalize(directions, p=2, dim=-1)
    x_direction = directions[-1]
    y_direction = torch.cross(x_direction[None], np_xyz)[0]
    u_new = torch.arccos(torch.clip((x_direction[None] * directions).sum(-1), min=-1+eps,max=1-eps))
    u_new[(y_direction[None] * directions).sum(-1) < 0] *= -1
    u_new = u_new[:-1]
    # print(111, u_new.shape, u_new.max(), u_new.min())
    # u_new -= math.pi
    # u_new[u_new >= math.pi] += math.pi * 2
    v_new = v_new[:-1]
    uv_new = torch.stack([u_new, v_new], 1)
    return uv_new


def u_correct(us, inplace=True):
    if not inplace:
        us = us.clone()
    us[us > math.pi] -= math.pi * 2
    us[us <= -math.pi] += math.pi * 2
    return us


def pano_rotate(np_uv, s_uv, reverse=False, eps=1e-15):
    """
    rotate pano coordinates
    @param np_uv: north pole, tensor(u, v)
    @param s_uv: source uv, tensor(b, u, v)
    @param reverse:
        the reversed pano rotation
        note that:
            pano_rotate(np_uv, pano_rotate(np_uv, s_uv), reverse=True) == s_uv
    @return: target: uv, tensor(b, u, v)
    """
    if torch.abs(np_uv[1] + math.pi * 0.5) < eps:
        return s_uv
    np_uv = np_uv.clone()
    if not reverse:
        # s_uv[:, 0] -= np_uv[0]
        # s_uv[:, 0] = u_correct(s_uv[:, 0])
        # np_uv[0] = 0
        return rotate(np_uv, s_uv, eps)
    else:
        # s_uv[:, 0] -= np_uv[0]
        # s_uv[:, 0] = u_correct(s_uv[:, 0])
        # np_uv[0] = 0
        pole = rotate(np_uv, NORTH_POLE[None], eps)[0].to(s_uv.device)
        rotated = rotate(pole, s_uv, eps)
        rotated[:, 0] += np_uv[0]
        rotated[:, 0] = u_correct(rotated[:, 0])
        # rotated[:, 0][rotated[:, 0] >= math.pi] -= 2 * math.pi
        # rotated[:, 0][rotated[:, 0] < -math.pi] += 2 * math.pi
        return rotated


def _test_reverse():
    pi = math.pi
    np_uv = torch.tensor([-0.1, 0.5]) * pi
    s_uv = torch.tensor([[0.5, 0],
                         [0, 0.42],
                         [0, -0.4],
                         [0.7, -0.4],
                         ]) * pi
    # rotated = rotate(np_uv, s_uv)
    # print(rotated)
    # print(rotate(torch.tensor([0.0000,  0.3142]), rotated) / pi)
    # print(pano_rotate(np_uv, rotated, reverse=True) / pi)
    print(pano_rotate(np_uv, pano_rotate(np_uv, s_uv), reverse=True) / pi)
    s_uv = (torch.rand(1000, 2) - 0.5) * math.pi
    s_uv[:, 0] *= 2
    reversed_uv = pano_rotate(np_uv, pano_rotate(np_uv, s_uv), reverse=True)
    print((torch.abs(reversed_uv - s_uv) < 1e-4).sum())


def _test():
    pi = math.pi
    np_uv = torch.tensor([0.1, 0.5])
    s_uv = torch.tensor([[0.25*pi, -0.25*pi],
                         [0.5*pi, -0.5*pi],
                         [0.5*pi, -0.2*pi],
                         ])
    rotated = rotate(np_uv, s_uv)
    print(s_uv / pi)
    print(rotated / pi)
    print(torch.norm(uv2xyz(s_uv)[:1] - uv2xyz(s_uv)[1:2], dim=-1))
    print(torch.norm(uv2xyz(rotated)[:1] - uv2xyz(rotated)[1:2], dim=-1))
    print(torch.norm(uv2xyz(s_uv)[:1] - uv2xyz(s_uv)[2:3], dim=-1))
    print(torch.norm(uv2xyz(rotated)[:1] - uv2xyz(rotated)[2:3], dim=-1))


from lzx.indoor360.view_pano import make_xys, tangent_xy2equirectangular_uv, uv_expand


def pano_rotate_image_uvs(np_uv, uv):
    u0 = np_uv[0]
    np_uv = np_uv.clone()
    np_uv[0] = 0
    uv = pano_rotate(np_uv, uv, reverse=False)
    uv[:, 0] += u0
    uv[:, 0][uv[:, 0] > math.pi] -= 2 * math.pi
    uv[:, 0][uv[:, 0] < -math.pi] += 2 * math.pi
    return uv


def _pano_rotate_image_s_uvs(tuvwh2xyxy_boxes, WH, np_uv):
    if tuvwh2xyxy_boxes is None:
        return None
    else:
        np_uv = np_uv.clone()
        s_uvs_cat = torch.cat(tuvwh2xyxy_boxes, 0)
        t_uvs = s_uvs_cat.clone()
        for i, tuvwh in enumerate(s_uvs_cat):
            xy = make_xys(tuvwh[2:4] * 0.5, gap=None, n=5)
            uv = tangent_xy2equirectangular_uv(xy=xy, uv0=tuvwh[:2])  # good
            uv = pano_rotate_image_uvs(np_uv, uv)
            uv = uv_expand(uv, WH)
            xyxy = torch.tensor([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
            t_uvs[i] = xyxy
        s = 0
        ret = []
        for s_uv in tuvwh2xyxy_boxes:
            ret.append(t_uvs[s: s + len(s_uv)])
            s += len(t_uvs[-1])
    return ret


def pano_rotate_image(bcwh, np_uv, tuvwh2xyxy_boxes=None):
    """
    @param bcwh: [b x c x w x h]
    @param np_uv: north pole, tensor(u, v)
    @param tuvwh2xyxy_boxes: additional tangent uvwh boxes, list(tensor(n x 4)), or None
    @return: a rotated image batch, [b x c x w x h]
    """
    B, C, H, W = bcwh.shape
    mesh_v, mesh_u = torch.meshgrid(torch.arange(H) / H - 0.5, torch.arange(W) / H - 1)
    s_uv = torch.stack([mesh_u, mesh_v], -1) * math.pi
    s_uv = einops.rearrange(s_uv, "H W C -> (H W) C")
    rotated_uv = pano_rotate(np_uv, s_uv, reverse=False)  # rotate_f(s_uv)
    xyxy_boxes = _pano_rotate_image_s_uvs(tuvwh2xyxy_boxes, [W, H], np_uv)
    eps = 5e-4
    rotated_uv[..., 0] = torch.clip(rotated_uv[..., 0] / math.pi, min=eps-1, max=1-eps)
    rotated_uv[..., 1] = torch.clip(rotated_uv[..., 1] / math.pi * 2, min=eps-1, max=1-eps)
    rotated_uv = einops.rearrange(rotated_uv, "(H W) C -> H W C", W=W)
    out = F.grid_sample(bcwh, rotated_uv[None], mode='bilinear', padding_mode='border', align_corners=False)
    return out, xyxy_boxes


def _debug():
    uv = torch.tensor([[2.45363, 0.31975],
            [1.47017, 0.31975],
            [1.96190, 0.35931],
            [2.64109, 0.77626],
            [2.45363, 0.31975]])
    reverse = False
    np_uv = torch.tensor([0.2, -0.4]) * math.pi
    print(pano_rotate(np_uv, uv, reverse=reverse))
    np_uv = torch.tensor([0, -0.4]) * math.pi
    print(pano_rotate(np_uv, uv, reverse=reverse))


from torchvision import transforms
from lzx.utils import cv_show1
from lzx.indoor360.view_pano import get_visual_image
from lzx.yolo.utils.plots import plot_one_box
def _test_pano_rotate_image():
    # E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\11454324954_1b858800d1_k.jpg
    # std: tensor([1.9619, 0.6300, 1.0407, 0.5550], dtype=torch.float64)
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\11454324954_1b858800d1_k.jpg"
    im = cv2.imread(file)
    im = cv2.resize(im, (800,400))
    H, W, C = im.shape
    if 0:
        for i in range(0, H, 100):
            im = cv2.circle(im, (W // 2, i), radius=10, color=(0, 0, 255), thickness=10)
    # im = cv2.circle(im, (W // 2, H), radius=10, color=(0, 0, 255), thickness=10)

    uvwh = torch.tensor([1.9619, 0.6300, 1.0407, 0.5550])[None]
    # im2 = get_visual_image(im.copy(), uvwh, is_boxstd=True)
    im2 = im.copy()
    # im = cv2.circle(im, (W // 2, H-1), radius=20, color=(0, 0, 255), thickness=30)
    cv_show1(im2, name="0", w=False)
    # for t in range(5):
    #     im = cv2.circle(im, (W // 4 * t, H // 2), radius=20, color=(255, 0, 255), thickness=30)
    # im = get_visual_image(im.copy(), uvwh, is_boxstd=True, plt_xyxy=True)
    im = transforms.ToTensor()(im)
    np_uv = torch.tensor([0.3, -0.4]) * math.pi
    np_uv = torch.tensor([1, -0.0]) * math.pi
    im3, xyxy_boxes = pano_rotate_image(im[None].clone(), np_uv, [uvwh])
    xyxy_boxes = xyxy_boxes[0][0].numpy()
    im3 = np.ascontiguousarray((255 * im3[0]).permute(1,2,0).numpy().astype(np.uint8))
    # im3 = plot_one_box(xyxy_boxes, im3, label="", color=(0, 255, 0), line_width=2)
    # for uvi in uv:
    #     im3 = cv2.circle(im3, (uvi[0], uvi[1]), radius=2, color=(0, 0, 255), thickness=2)
    cv_show1(im3, name="1")


def _test_show2():
    pi = math.pi
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7frn3.jpg"
    H = 300
    W = H * 2
    im = cv2.imread(file)
    im = cv2.resize(im, (W, H))
    # cv2.imshow("1", im); cv2.waitKey()
    im_new = np.zeros(im.shape).astype(im.dtype)
    np_uv = torch.tensor([1, -0.4]) *pi
    mesh_v, mesh_u = torch.meshgrid(torch.arange(H) / H - 0.5,  torch.arange(W) / H - 1)
    s_uv = torch.stack([mesh_u, mesh_v], -1) * pi
    s_uv = einops.rearrange(s_uv, "H W C -> (H W) C")
    rotated_uv = pano_rotate(np_uv, s_uv, reverse=True)
    rotated_uv = einops.rearrange(rotated_uv, "(H W) C -> H W C", W=W)
    rotated_uv[..., 0] = torch.clip((rotated_uv[..., 0] + pi) / 2 / pi * W, min=0, max=W - 1).int()
    rotated_uv[..., 1] = torch.clip((rotated_uv[..., 1] + 0.5 * pi) / pi * H, min=0, max=H - 1).int()
    rotated_xy = rotated_uv.numpy().astype(int)
    for y in range(W):
        for x in range(H):
            # print("{} -> {}".format(rotated_xy[x][y], (x,y,)), im.shape, rotated_xy[x][y])
            im_new[x][y] = im[rotated_xy[x][y][1]][rotated_xy[x][y][0]]
    # exit()
    cv2.imshow("1", cv2.resize(im, (600, 300)))
    cv2.imshow("2", cv2.resize(im_new, (600, 300)))
    cv2.waitKey()


def _test_show():
    pi = math.pi
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7frn3.jpg"
    H = 200
    W = H * 2
    im = cv2.imread(file)
    im = cv2.resize(im, (W, H))
    # cv2.imshow("1", im); cv2.waitKey()
    im_new = np.zeros(im.shape).astype(im.dtype)
    np_uv = torch.tensor([0*pi, -0.4*pi])
    s_xy = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)), -1)

    for x in range(W):
        for y in range(H):
            s_uv = torch.tensor([[x / H - 1, y / H - 0.5]])  * pi
            rotated_uv = rotate(np_uv, s_uv)[0]
            x_ = int(((rotated_uv[0] + pi) / pi * H).item() + 0.5)
            y_ = int(((rotated_uv[1] + 0.5 * pi) / pi * H).item() + 0.5)

            # print("{} -> {}".format((x, y,), (x_, y_,)))
            # continue
            x_ = np.clip(x_, a_min=0, a_max=W-1)
            y_ = np.clip(y_, a_min=0, a_max=H-1)
            # print("{} -> {}".format((x_, y_,), (x, y,)))
            im_new[y][x] = im[y_][x_]
    # exit()
    cv2.imshow("1", im)
    cv2.imshow("2", im_new)
    cv2.waitKey()


if __name__=='__main__':
    # _debug()
    _test_pano_rotate_image()
    # _test_show2()
    # _test_reverse()
    # _test_show()