from lzx.pano_rotate import *


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


def _pano_rotate_image_s_uvs(tuvwh2xyxy_boxes, WH, np_uv):
    if tuvwh2xyxy_boxes is None:
        return None
    else:
        np_uv = np_uv.clone()
        u0 = np_uv[0].clone()
        s_uvs_cat = torch.cat(tuvwh2xyxy_boxes, 0)
        t_uvs = s_uvs_cat.clone()
        for i, tuvwh in enumerate(s_uvs_cat):
            xy = make_xys(tuvwh[2:4] * 0.5, gap=None, n=5)
            uv = tangent_xy2equirectangular_uv(xy=xy, uv0=tuvwh[:2])  # good
            np_uv[0] = 0
            uv = pano_rotate(np_uv, uv, reverse=False)  # rotate_f(uv) # bad
            uv[:, 0] += u0
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
from lzx.utils import ncolors
def _test_pano_rotate_image():
    # E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\11454324954_1b858800d1_k.jpg
    # std: tensor([1.9619, 0.6300, 1.0407, 0.5550], dtype=torch.float64)
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\11454324954_1b858800d1_k.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\13518741753_9bb025f904_o.jpg"
    im = cv2.imread(file)
    im = cv2.resize(im, (800,400))
    H, W, C = im.shape

    mesh_v, mesh_u = torch.meshgrid(torch.arange(H // 6, H, H // 3) / H - 0.5, torch.arange(W // 6, W, W // 3) / H - 1)
    s_uv = (torch.stack([mesh_u, mesh_v], -1) * math.pi).reshape([-1, 2])
    # print(s_uv[:,0].min(), s_uv[:,1].min(), s_uv[:,0].max(), s_uv[:,1].max())
    im1 = im.copy()
    colors = ncolors(s_uv.shape[0]).tolist()
    for i, uv in enumerate(s_uv):
        uv = (uv / math.pi * H).numpy().astype(int)
        # print(uv, (uv[1] + H // 2, uv[0] + W // 2))
        # cv2.circle(im1, (uv[0] + W // 2, uv[1] + H // 2), radius=5, color=colors[i], thickness=15)

    cv_show1(im1, name="0", w=False)
    im = transforms.ToTensor()(im)
    np_uv = torch.tensor([0.3, -0.2]) * math.pi
    im3, _ = pano_rotate_image(im[None].clone(), np_uv, None)
    t_uv = pano_rotate_image_uvs(np_uv, s_uv)
    # print(t_uv[:, 0].min(), t_uv[:, 1].min(), t_uv[:, 0].max(), t_uv[:, 1].max())
    im3 = np.ascontiguousarray((255 * im3[0]).permute(1, 2, 0).numpy().astype(np.uint8))
    for i, uv in enumerate(t_uv):
        uv = (uv / math.pi * H).numpy().astype(int)
        # print(uv, (uv[1] + H // 2, uv[0] + W // 2))
        # cv2.circle(im3, (uv[0] + W // 2, uv[1] + H // 2), radius=5, color=colors[i], thickness=15)
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


if __name__=='__main__':
    # _debug()
    _test_pano_rotate_image()
    # _test_show2()
    # _test_reverse()
    # _test_show()