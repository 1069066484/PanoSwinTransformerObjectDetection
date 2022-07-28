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


class PixelTransformer(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size):
        super().__init__()
        self.in_chans = in_chans
        intermid_dim = 32
        intermid_dim2 = intermid_dim * 4
        self.preprocess = nn.Sequential(
            nn.Linear(in_chans, intermid_dim // 2),
            nn.ReLU(),
            nn.Linear(intermid_dim // 2, intermid_dim),
        )
        encoder_layer = TransformerEncoderLayer(intermid_dim, 8, intermid_dim2)
        self.encoder = TransformerEncoder(encoder_layer, 3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, intermid_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 8**2, intermid_dim))
        self.pos_fix = PositionalEncoding(intermid_dim, 8**2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(intermid_dim),
            nn.Linear(intermid_dim, out_dim)
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape([b, self.in_chans, -1]).transpose(1,2)
        x = self.preprocess(x)
        if 0:
            pos = self.pos_embedding[:,:x.shape[1]]
        else:
            pos = self.pos_fix(x)
        x = self.encoder(x + pos)
        x = x.mean(dim=1)
        return self.mlp_head(x)

if 0:
    class PixelTransformer(nn.Module):
        def __init__(self, in_chans, out_dim, patch_size):
            super().__init__()
            self.in_chans = in_chans
            self.feat = nn.Sequential(
                nn.Conv2d(3,16,3,2,1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16,64,3,2,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(64, out_dim)
            )

        def forward(self, x):
            return self.feat(x)


import torchvision.transforms as transforms
import torchvision

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
net = PixelTransformer(dim, 10, 16).cuda()



optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for i in range(50):
    train_loss = 0
    train_acc = 0
    net.train()   # 网络设置为训练模式

    for it_train, (data,label) in enumerate(train_loader):
        # data [64,3,32,32] label [64]
        # 前向传播
        label = label.cuda()
        data = data.cuda()
        output = net(data)
        # 记录单批次一次batch的loss
        loss = criterion(output, label)
        optimizer.zero_grad()   # 梯度归零
        loss.backward()    # 反向传播
        optimizer.step()   # 优化
        # 累计单批次误差
        train_loss = train_loss + loss.item()
        # 计算分类的准确率
        _, pred = output.max(1)   # 求出每行的最大值,值与序号pred
        num_correct = (pred == label).sum().item()
        acc = num_correct/label.shape[0]
        train_acc = train_acc + acc

    print('epoch: {}, trainloss: {:.4f},trainacc: {:.4f}'.format(i+1, train_loss/len(train_loader), train_acc/len(train_loader)))


    # 测试集进行测试
    test_loss = 0
    eval_acc=0
    net.eval()
    with torch.no_grad():
        for data,label in test_loader:
            label = label.cuda()
            data = data.cuda()
            output = net(data)
            # 记录单批次一次batch的loss，并且测试集不需要反向传播更新网络
            loss = criterion(output, label)
            test_loss = test_loss + loss.item()
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct/label.shape[0]
            eval_acc = eval_acc + acc

    print('epoch: {}, evalloss: {:.4f},evalacc: {:.4f}'.format(i+1, test_loss/len(test_loader), eval_acc/len(test_loader)))

