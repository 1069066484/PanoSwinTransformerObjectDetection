import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy


class PixelTransformer(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size):
        super().__init__()
        self.in_chans = in_chans
        intermid_dim = 32
        intermid_dim2 = intermid_dim * 2
        self.preprocess = nn.Sequential(
            nn.Linear(in_chans, intermid_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=intermid_dim, nhead=16, dim_feedforward=intermid_dim2, dropout=0.1)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, intermid_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 4**2 + 1, intermid_dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(intermid_dim),
            nn.Linear(intermid_dim, out_dim)
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape([b, self.in_chans, -1]).transpose(1,2)
        x = self.preprocess(x)
        if 1:
            pos = self.pos_embedding[:,:x.shape[1]]
        else:
            pos = self.pos_fix(x)
        x = self.encoder((x + pos).transpose(0,1)).transpose(0,1)
        x = x.mean(dim=1)
        return self.mlp_head(x)



import torchvision.transforms as transforms
import torchvision

batch_size = 16
transform = transforms.Compose([
    transforms.Resize(4),
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

