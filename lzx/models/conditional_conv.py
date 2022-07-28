import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
import torch.nn.functional as F


class NaiveConv(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size):
        super(NaiveConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

"""
上下有对称性
"""
class CondConv(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size, cond_c=1):
        super(CondConv, self).__init__()
        assert patch_size % 2 == 1
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))
        self.cond_conv = nn.Conv2d(cond_c, out_dim, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        self.cond_att = nn.Sequential(
            nn.Conv2d(cond_c, out_dim, kernel_size=patch_size, stride=1, padding=patch_size // 2),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid())
        # conv.weight.shape torch.Size([48, 3, 4, 4])
        # conv.bias.shape torch.Size([48])

    def forward(self, x, cond=None):
        x_shape = x.shape
        x = self.conv(x)
        cond = torch.ones([x_shape[0], 1, *x_shape[2:]]).float().to(x.device)
        # print(x.shape, cond.shape);exit() # torch.Size([16, 48, 2, 2]) torch.Size([16, 1, 2, 2])
        if cond is None:
            return x

        if cond is None:
            cond = torch.ones([x_shape[0], 1, *x_shape[2:]]).float().to(x.device)
        # print(cond.shape, self.patch_size) # torch.Size([16, 1, 2, 2]) [5, 5]
        cond_t = rearrange(cond, 'b c (w p1) (h p2) -> (b w h) c p1 p2', p1=self.patch_size[0], p2=self.patch_size[1])
        cond_att = self.cond_att(cond_t)
        # print(cond_att.shape, x_shape, x_shape[2] // self.patch_size[0])
        # torch.Size([64, 48, 5, 5]) torch.Size([16, 3, 10, 10]) 2
        cond_att = rearrange(cond_att, '(b w h) c p1 p2 -> b c (w p1) (h p2)',
                             w=x_shape[2] // self.patch_size[0],
                             h=x_shape[3] // self.patch_size[1])
        print(cond_att.shape, x.shape);exit()
        return cond_att * x


from lzx.lzx_augs.basketball_transform import basketball_transition, basketball_uvmap_foreground
from torchvision import transforms
import cv2
import numpy as np
from lzx.utils import *




import torchvision
def run_cifar():
    batch_size = 16
    transform = transforms.Compose([
        transforms.Resize(10),
        # transforms.CenterCrop([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\cifar"
    train_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        download=True,
        transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # 是否打乱数据，一般都打乱
        num_workers=0
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    dim = 3
    dim_out = 48
    net = CondConv(dim, dim_out, 5).cuda()

    # chceck_params_rec(net, 2)

    out_linear = nn.Sequential(
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(dim_out * 4, 10)
    ).cuda()
    optimizer = torch.optim.Adam(list(net.parameters()) + list(out_linear.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for i in range(100):
        train_loss = 0
        train_acc = 0
        net.train()  # 网络设置为训练模式

        for it_train, (data, label) in enumerate(train_loader):
            # data [64,3,32,32] label [64]
            # 前向传播
            label = label.cuda()
            data = data.cuda()
            output = out_linear(net(data))
            # 记录单批次一次batch的loss
            loss = criterion(output, label)
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化
            # 累计单批次误差
            train_loss = train_loss + loss.item()
            # 计算分类的准确率
            _, pred = output.max(1)  # 求出每行的最大值,值与序号pred
            num_correct = (pred == label).sum().item()
            acc = num_correct / label.shape[0]
            train_acc = train_acc + acc

        print('epoch: {}, trainloss: {:.4f},trainacc: {:.4f}'.format(i + 1, train_loss / len(train_loader),
                                                                     train_acc / len(train_loader)))

        # 测试集进行测试
        test_loss = 0
        eval_acc = 0
        net.eval()
        with torch.no_grad():
            for data, label in test_loader:
                label = label.cuda()
                data = data.cuda()
                output = out_linear(net(data))
                # 记录单批次一次batch的loss，并且测试集不需要反向传播更新网络
                loss = criterion(output, label)
                test_loss = test_loss + loss.item()
                _, pred = output.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / label.shape[0]
                eval_acc = eval_acc + acc

        print('epoch: {}, evalloss: {:.4f},evalacc: {:.4f}'.format(i + 1, test_loss / len(test_loader),
                                                                   eval_acc / len(test_loader)))


run_cifar()