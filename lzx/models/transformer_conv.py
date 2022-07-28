import einops
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy


class TransformerEncoder(nn.Module):
    """
    A common transformer encoder
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # fead forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x


import math
class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """
    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term) # take the odd (jump by 2)
        pe[:, 1::2] = torch.cos(position * exp_term) # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PixelTransformer(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size):
        super().__init__()
        self.in_chans = in_chans
        intermid_dim = out_dim
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        intermid_dim2 = intermid_dim
        self.preprocess = nn.Sequential(
            nn.Linear(in_chans + 4, intermid_dim)
        )
        self.preprocess_nouv = nn.Sequential(
            nn.Linear(in_chans, intermid_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermid_dim, nhead=8, dim_feedforward=intermid_dim2, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_size[0] * self.patch_size[1], intermid_dim))

    def process_x(self, x):
        x = einops.rearrange(x, 'b c (p1 w) (p2 h) -> (b w h) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[1])
        return x

    def forward(self, x, uv_mask=None):
        shape = x.shape
        x = self.process_x(x)
        if uv_mask is None:
            x = self.preprocess_nouv(x).transpose(0, 1)
            x = x + self.pos_embedding[:,:x.shape[1]].transpose(0,1)
            x = self.encoder(x).transpose(0, 1)
            x = x.sum(1)
        else:
            uv_mask = self.process_x(uv_mask)
            pos = torch.cat([torch.cos(uv_mask[..., :2]), torch.sin(uv_mask[..., :2])], 2)
            mask = ~uv_mask[..., -1].bool()
            x = torch.cat([x, pos], 2)
            x = self.preprocess(x).transpose(0,1)
            x = self.encoder(x, src_key_padding_mask=mask).transpose(0,1)
            x = (x * mask[..., None]).sum(1) / mask.sum(1, True)
        x = einops.rearrange(x, '(b w h) s -> b s w h', w=shape[2] // self.patch_size[0], h=shape[3] // self.patch_size[1])
        return x



from lzx.lzx_augs.basketball_transform import basketball_transition, basketball_uvmap_foreground
from torchvision import transforms
import cv2
import numpy as np
from lzx.utils import *


def _base_test():
    img_path = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\hw0805\data_scaled\images\train\23010301000156__南岗区赣水路德霖高尔夫__机房__全景照片_1625654534870__VID_20210612_090058_00_026_000001.jpg"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    sz2 = 20
    img = cv2.resize(img, (sz2 * 2, sz2))
    sz_small = 4
    patch_num_y = sz2 // sz_small
    k = 'center'
    transed = basketball_transition(img, patch_num_y, [k])
    uv_fore = basketball_uvmap_foreground(img.shape, patch_num_y)

    trans = PixelTransformer2(3, 96, sz_small)
    chceck_params_rec(trans, 2)
    print(num_params(nn.Conv2d(3, 96, kernel_size=7, stride=7)))
    with torch.no_grad():
        trans.eval()
        print(trans(transform(img)[None, ...].repeat(2,1,1,1),).shape)# transform(uv_fore[k])[None, ...].repeat(2,1,1,1))
        print(trans(transform(img)[None, ...].repeat(2,1,1,1), transform(uv_fore[k])[None, ...].repeat(2,1,1,1)).shape)


import torchvision
def run_cifar():
    batch_size = 16
    transform = transforms.Compose([
        transforms.Resize(8),
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
    dim_out = 96
    net = PixelTransformer(dim, dim_out, 4).cuda()

    # chceck_params_rec(net, 2)

    out_linear = nn.Sequential(
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(dim_out * 4, 10)
    ).cuda()
    optimizer = torch.optim.Adam(list(net.parameters()) + list(out_linear.parameters()), lr=0.00005)
    criterion = nn.CrossEntropyLoss()

    for i in range(50):
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