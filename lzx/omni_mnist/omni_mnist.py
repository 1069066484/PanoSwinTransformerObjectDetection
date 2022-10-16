# Mathematical
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import cv2

# Pytorch
import torch
from torch.utils import data
from torchvision import datasets

# Misc

from functools import lru_cache
import os
import argparse
from PIL import Image


def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return np.stack([u, v], axis=-1)


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)


def uv2img_idx(uv, h, w, u_fov, v_fov, v_c=0):
    assert 0 < u_fov and u_fov < np.pi
    assert 0 < v_fov and v_fov < np.pi
    assert -np.pi < v_c and v_c < np.pi

    xyz = uv2xyz(uv.astype(np.float64))
    Ry = np.array([
        [np.cos(v_c), 0, -np.sin(v_c)],
        [0, 1, 0],
        [np.sin(v_c), 0, np.cos(v_c)],
    ])
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(v_c) * xyz[..., 0] - np.sin(v_c) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(v_c) * xyz[..., 0] + np.cos(v_c) * xyz[..., 2]
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2)
    x[invalid] = -100
    y[invalid] = -100

    return np.stack([y, x], axis=0)


class OmniDataset(data.Dataset):
    def __init__(self, dataset, fov=120, outshape=(60, 60),
                 flip=False, h_rotate=False, v_rotate=False,
                 img_mean=None, img_std=None, fix_aug=False,
                 w2=True, planer=True):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        self.dataset = dataset
        self.fov = fov
        self.outshape = outshape
        self.flip = flip
        self.h_rotate = h_rotate
        self.v_rotate = v_rotate
        self.img_mean = img_mean
        self.img_std = img_std
        self.w2 = w2
        self.planer = planer

        self.aug = None
        if fix_aug:
            self.aug = [
                {
                    'flip': np.random.randint(2) == 0,
                    'h_rotate': np.random.randint(outshape[1]),
                    'v_rotate': np.random.uniform(-np.pi/2, np.pi/2),
                }
                for _ in range(len(self.dataset))
            ]

    def __len__(self):
        return len(self.dataset)

    def make_sphere_data(self, idx):
        img = np.array(self.dataset[idx][0], np.float32)
        label = self.dataset[idx][1]
        h, w = img.shape[:2]
        uv = genuv(*self.outshape)
        fov = self.fov * np.pi / 180

        if self.v_rotate:
            if self.aug is not None:
                v_c = self.aug[idx]['v_rotate']
            else:
                v_c = np.random.uniform(-np.pi / 2, np.pi / 2)
            img_idx = uv2img_idx(uv, h, w * (1 + self.w2), fov, fov, v_c)
        else:
            img_idx = uv2img_idx(uv, h, w * (1 + self.w2), fov, fov, 0)
        x = map_coordinates(img, img_idx, order=1)

        # Random flip
        if self.aug is not None:
            if self.aug[idx]['flip']:
                x = np.flip(x, axis=1)
        elif self.flip and np.random.randint(2) == 0:
            x = np.flip(x, axis=1)

        # Random horizontal rotate
        if self.h_rotate:
            if self.aug is not None:
                dx = self.aug[idx]['h_rotate']
            else:
                dx = np.random.randint(x.shape[1])
            x = np.roll(x, dx, axis=1)

        # Normalize image
        if self.img_mean is not None:
            x = x - self.img_mean
        if self.img_std is not None:
            x = x / self.img_std
        shape = x.shape
        # print(type(x), x.shape)
        x = cv2.resize(x, (shape[1] * (1 + self.w2), shape[0]),
                       cv2.INTER_CUBIC)
        x = torch.FloatTensor(x.copy())
        return x, label

    def make_planar_data(self, idx):
        img = np.array(self.dataset[idx][0], np.float32)
        shape = img.shape
        label = self.dataset[idx][1]
        img = cv2.resize(img, (self.outshape[1], (1 + self.w2) * shape[0]),
                       cv2.INTER_CUBIC)
        x = torch.FloatTensor(img)
        return x, label

    def __getitem__(self, idx):
        if self.planer:
            x, label = self.make_planar_data(idx)
        else:
            x, label = self.make_sphere_data(idx)
        return x, label


class OmniMNIST(OmniDataset):
    def __init__(self, root='datas/MNIST', train=True,
                 download=True, *args, **kwargs):
        '''
        Omnidirectional MNIST
        @root (str)       root directory storing the dataset
        @train (bool)     train or test split
        @download (bool)  whether to download if data now exist
        '''
        self.MNIST = datasets.MNIST(root, train=train, download=download)
        super(OmniMNIST, self).__init__(self.MNIST, *args, **kwargs)


class OmniFashionMNIST(OmniDataset):
    def __init__(self, root='datas/FashionMNIST', train=True,
                 download=True, *args, **kwargs):
        '''
        Omnidirectional FashionMNIST
        @root (str)       root directory storing the dataset
        @train (bool)     train or test split
        @download (bool)  whether to download if data now exist
        '''
        self.FashionMNIST = datasets.FashionMNIST(root, train=train, download=download)
        super(OmniFashionMNIST, self).__init__(self.FashionMNIST, *args, **kwargs)


def make_arg():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--idx', nargs='+', default=[1,2,3,4], help='image indices to demo')
    parser.add_argument(
        '--out_dir', default='datas/demo', help='directory to output demo image')
    parser.add_argument(
        '--dataset', default='OmniMNIST', choices=['OmniMNIST', 'OmniFashionMNIST'], help='which dataset to use')
    parser.add_argument(
        '--fov', type=int, default=120, help='fov of the tangent plane')
    parser.add_argument(
        '--flip', action='store_true', help='whether to apply random flip')
    parser.add_argument(
        '--h_rotate', action='store_true', help='whether to apply random panorama horizontal rotation')
    parser.add_argument(
        '--v_rotate', action='store_true', help='whether to apply random panorama vertical rotation')
    parser.add_argument(
        '--fix_aug', action='store_true', help='whether to apply random panorama vertical rotation')
    args = parser.parse_args()
    return args


def _make_dataset(args):
    # os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == 'OmniMNIST':
        dataset = OmniMNIST(
            fov=args.fov, flip=args.flip, h_rotate=args.h_rotate,
            v_rotate=args.v_rotate, fix_aug=args.fix_aug, outshape=(args.sz, args.sz))
    elif args.dataset == 'OmniFashionMNIST':
        dataset = OmniFashionMNIST(
            fov=args.fov, flip=args.flip, h_rotate=args.h_rotate,
            v_rotate=args.v_rotate, fix_aug=args.fix_aug, outshape=(args.sz, args.sz))
    return dataset


from easydict import EasyDict
def make_dataset(split, dataset='OmniMNIST', sz=60):
    fov = 120
    train_dict = EasyDict({'fov': fov, 'flip':True, 'h_rotate': True, 'sz':sz,
                     'v_rotate': True, 'fix_aug': False, 'dataset': dataset})
    test_dict = EasyDict({'fov': fov, 'flip':False, 'h_rotate': False, 'sz':sz,
                     'v_rotate': False, 'fix_aug': True, 'dataset': dataset})
    assert split in ['train', 'test']
    assert dataset in ['OmniMNIST', 'OmniFashionMNIST']
    if split == 'train':
        return _make_dataset(train_dict)
    else:
        return _make_dataset(test_dict)

def save_dataset(args, dataset):
    # print(args.idx)
    # flip=True, h_rotate=True, v_rotate=True
    for idx in args.idx:
        idx = int(idx)
        path = os.path.join(args.out_dir, '%d.png' % idx)
        x, label = dataset[idx]
        # print(path, label)
        Image.fromarray(x.numpy().astype(np.uint8)).save(path)


def _test_omni_mnist():
    args = make_arg()
    # args.flip = True
    args.h_rotate = True
    args.v_rotate = True
    dataset = make_dataset(args)
    save_dataset(args, dataset)


if __name__ == '__main__':
    _test_omni_mnist()
