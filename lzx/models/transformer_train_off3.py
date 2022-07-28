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
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        intermid_dim2 = intermid_dim * 4
        self.preprocess = nn.Sequential(
            nn.Linear(in_chans + 4, intermid_dim),
        )
        self.preprocess_nouv = nn.Sequential(
            nn.Linear(in_chans, intermid_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermid_dim, nhead=4, dim_feedforward=intermid_dim2, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_size[0] * self.patch_size[1], 4))
        self.pos_fix = PositionalEncoding(intermid_dim, 8 ** 2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(intermid_dim),
            nn.Linear(intermid_dim, out_dim)
        )

    def process_x(self, x):
        x = einops.rearrange(x, 'b c (p1 w) (p2 h) -> (b w h) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[1])
        return x

    def forward(self, x, uv_mask=None):
        shape = x.shape
        x = self.process_x(x)
        if uv_mask is None:
            if 1:
                x = self.preprocess_nouv(x).transpose(0, 1)
                x = self.pos_fix(x)
            else:
                x = torch.cat([x, self.pos_embedding.repeat(x.shape[0], 1, 1)], 2)
                x = self.preprocess(x).transpose(0,1)
            x = self.encoder(x).transpose(0, 1)
            x = x.sum(1)
        else:
            uv_mask = self.process_x(uv_mask)
            pos = torch.cat([
                torch.cos(uv_mask[..., :2]),
                torch.sin(uv_mask[..., :2]),
            ], 2)
            mask = ~uv_mask[..., -1].bool()
            x = torch.cat([x, pos], 2)
            x = self.preprocess(x).transpose(0,1)
            x = self.encoder(x, src_key_padding_mask=mask).transpose(0,1)
            x = (x * mask[..., None]).sum(1) / mask.sum(1, True)
        x = self.mlp_head(x)
        x = einops.rearrange(x, '(b w h) s -> b s w h', w=shape[2] // self.patch_size[0], h=shape[3] // self.patch_size[1])
        return x



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

