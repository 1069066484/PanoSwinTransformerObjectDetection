import torch
import numpy as np
import cv2

"""
Given a pano1, at (u0, v0) of pano1, we set a new axis - pano2
we build a transition from (u1, v1) -> (u2, v2)

v2 - v1 = v0
"""


def _pano_test():
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\JPEGImages\000002.jpg"
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7fQTY.jpg"
    # im = cv2.imread(file)
    im = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    print(im.shape)
    ms = 300
    ms2 = ms // 2
    im = cv2.resize(im, (ms*2,ms))
    print(ms, im.shape)
    cv2.imshow("1", im)
    im2 = im.copy()
    pad = 60
    im2_1 = np.concatenate([np.flip(im2[:pad, ms:], 0), im2[:ms - pad, :ms]], 0)
    im2_2 = np.concatenate([im[ms - pad:, :ms], np.flip(im[pad:, ms:], 0)], 0)
    # cv2.imshow("2", im2_1)
    # cv2.imshow("3", im2_2)
    # cv2.waitKey()

    im3 = np.concatenate([im2_1, im2_2], 0)
    cv2.imshow("im3", im3)
    cv2.waitKey()


from lzx.utils import cv_show1



def pole_cat(im, pad=0, left_up=True, ceil_center=True):
    ms = im.shape[1]
    im1 = im[..., ms:]
    im2 = im[..., :ms]
    if not left_up:
        im1, im2 = im2, im1
    im2_1 = torch.cat([im1[..., :pad, :].flip(-2), im2[..., :ms - pad, :ms]], -2)
    im2_2 = torch.cat([im2[..., ms - pad:, :], im1[..., pad:, :].flip(-2)], -2)
    if not ceil_center:
        im2_1, im2_2 = im2_2, im2_1
    im3 = torch.cat([im2_2.flip(2), im2_1], -2)
    return im3


def pole_cat2(im, center='N'):
    """
    @param im: [b, c, w, h] or [c, w, h]
    @param center: 'N' or 'S'
    @return:
    """
    assert im.shape[-2] * 2 == im.shape[-1]
    ms = im.shape[-2]
    left = im[..., :ms]
    right = im[..., ms:]
    right = torch.flip(right, dims=[-1,-2])
    result = torch.cat([right, left], dim=-2)
    if center == 'S':
        result = torch.roll(result, dims=[-2], shifts=ms)
    return result


def _pano_tc_test():
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7fS1v.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\JPEGImages\000002.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\dataset_motorRoom_0806\230104908000000335_机房001_全景照片_1625229142489\VID_20210621_151612_00_008_000001.jpg"
    im = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    sz = 300
    im = cv2.resize(im, (sz * 2, sz))
    img_tc = torch.from_numpy(im).permute(2,0,1)
    roll_pixels = 50

    # cv_show1(torch.cat([img_tc[..., -roll_pixels:], img_tc], -1), name="-2", w=True)

    # cv_show1(img_tc, name="0", w=False)
    img_tc = torch.roll(img_tc, roll_pixels, -1)
    # cv_show1(img_tc, name="1", w=False)

    img_tc1 = pole_cat(img_tc, ceil_center=True)
    cv_show1(img_tc1, name="pole_cat1", w=False)
    cv_show1(pole_cat2(img_tc[None, ...])[0], name="pole_cat2", w=True)

    cv_show1(torch.cat([img_tc1[..., -roll_pixels:, :], img_tc1], -2), name="-1", w=True)

    # img_tc11 = pole_cat(img_tc, ceil_center=False)
    # cv_show1(img_tc11, name="3", w=True)

    img_tc2 = torch.roll(img_tc1, roll_pixels, -2)
    cv_show1(img_tc2, name="22")


def ew2ns(im):
    """
    Convert a east-west panoramic representation to a north-south one
    east -> north
    west -> south
    @return:
    """
    assert im.shape[-2] * 2 == im.shape[-1]
    ms = im.shape[-2]
    left = im[..., :ms]
    right = im[..., ms:]
    right = torch.flip(right, dims=[-1,-2])
    result = torch.cat([right, left], dim=-2)
    return result


def ns2we(im):
    """
    Convert a north-south panoramic representation to a west-east one
    @param im: [b, c, w, h] or [c, w, h]
    @return:
    """
    assert im.shape[-2] == im.shape[-1] * 2
    ms = im.shape[-1]
    top = im[..., :ms, :]
    bottom = im[..., ms:, :]
    top = torch.flip(top, dims=[-1, -2])
    result = torch.cat([bottom, top], dim=-1)
    return result


def trans_test():
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7fS1v.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\24169505635_0f76622e9b_o.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\JPEGImages\000002.jpg"
    # file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\dataset_motorRoom_0806\230104908000000335_机房001_全景照片_1625229142489\VID_20210621_151612_00_008_000001.jpg"
    im = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    sz = 300
    im = cv2.resize(im, (sz * 2, sz))
    img_tc = torch.from_numpy(im).permute(2,0,1)
    cv_show1(img_tc, name="1", w=False)
    img_ns = ew2ns(img_tc)
    cv_show1(img_ns, name="img_ns", w=False)
    img_we = ns2we(img_ns)
    cv_show1(img_we, name="img_we", w=True)


def swin_ori():
    file = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\indoor360\images\7fS1v.jpg"
    trunc = 50
    im = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    sz = 300
    im = cv2.resize(im, (sz * 2, sz))
    img_tc = torch.from_numpy(im).permute(2,0,1)
    cv_show1(img_tc, name="1", w=False)
    img_tc = torch.cat([img_tc[:, -trunc:, ...], img_tc[:, :-trunc, ...]], 1)
    cv_show1(img_tc, name="2", w=False)
    img_tc = torch.cat([img_tc[:, :, -trunc:], img_tc[:, :, :-trunc]], 2)
    cv_show1(img_tc, name="3", w=True)

    img_tc = torch.cat([img_tc[:, :, -trunc:], img_tc[:, :, :-trunc]], 2)
    cv_show1(img_tc, name="3", w=True)




if __name__=='__main__':
    # _pano_test()
    # _pano_tc_test()
    swin_ori()
    pass


"""
[
0 . .
. . 0
. . .
]
1-5,19


[
. . .
0 . .
. . 0
]
1-5,19
2-19,5


tensor([[12, 11, 10,  7,  6,  5,  2,  1,  0],
        [13, 12, 11,  8,  7,  6,  3,  2,  1],
        [14, 13, 12,  9,  8,  7,  4,  3,  2],
        [17, 16, 15, 12, 11, 10,  7,  6,  5],
        [18, 17, 16, 13, 12, 11,  8,  7,  6],
        [19, 18, 17, 14, 13, 12,  9,  8,  7],
        [22, 21, 20, 17, 16, 15, 12, 11, 10],
        [23, 22, 21, 18, 17, 16, 13, 12, 11],
        [24, 23, 22, 19, 18, 17, 14, 13, 12]])
        
tensor([[12, 13, 14, 17, 18, 19, 22, 23, 24],
        [11, 12, 13, 16, 17, 18, 21, 22, 23],
        [10, 11, 12, 15, 16, 17, 20, 21, 22],
        [ 7,  8,  9, 12, 13, 14, 17, 18, 19],
        [ 6,  7,  8, 11, 12, 13, 16, 17, 18],
        [ 5,  6,  7, 10, 11, 12, 15, 16, 17],
        [ 2,  3,  4,  7,  8,  9, 12, 13, 14],
        [ 1,  2,  3,  6,  7,  8, 11, 12, 13],
        [ 0,  1,  2,  5,  6,  7, 10, 11, 12]])
"""