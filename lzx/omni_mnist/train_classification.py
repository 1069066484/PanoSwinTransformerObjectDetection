import torch

from lzx.omni_mnist.omni_mnist import *
from mmdet.models.backbones.panoswin_transformer import *
import sys
import torch.nn as nn
from lzx.utils import FakeFn


def make_arg():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', default='OmniMNIST', choices=['OmniMNIST', 'OmniFashionMNIST'], help='which dataset to use')
    parser.add_argument(
        '--save_dir', default='save_{}_classification', help='which dataset to use')
    parser.add_argument(
        '--lr', default=1e-4, help='which dataset to use')
    parser.add_argument(
        '--sz', default=64, help='which dataset to use')
    parser.add_argument(
        '--batch_size', default=32, help='batch size')
    # log_interval
    parser.add_argument(
        '--log_interval', default=1000, help='batch size')
    parser.add_argument(
        '--device', default='cuda:0')
    args = parser.parse_args()
    return args


def make_tiny_swin():
    # is_win32 = sys == 'win32'
    embed_dim = 96
    patch_size = 4
    MODEL = dict(
            in_chans=3,
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            patch_size=patch_size,
            emb_conv_type='cnn',
            basketball_trans=False,
            ape=True,
            drop_path_rate=0.1,
            patch_norm=True,
            use_checkpoint=False
        )
    # print(MODEL); exit()
    model = PanoSwinTransformer(**MODEL)
    return model


def _make_dataset(dataset, batch_size, sz):
    datasets = dict([(split, make_dataset(split, dataset, sz=sz)) for split in ['train', 'test']])
    loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
               'test': torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)}
    return datasets, loaders


def make_classifier(datasets, swin):
    item, label = datasets['train'][0]
    item2 = item[None,:,:,None].repeat(2,1,1,3).permute(0,3,1,2)
    res = swin(item2, [[0.0, 1.0, None]] * 2)
    classfier_input_dim = res[-1].shape[1]
    classfier = nn.Sequential(
        FakeFn(lambda l: l[-1]),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(classfier_input_dim, 10))
    return classfier


def _train_epoch(train_loader, model, optimizer, args, epoch=0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args._device), target.to(args._device)
        # print(data.shape); exit()
        # cv_show1(data[:,None], w=True)
        optimizer.zero_grad()
        if data.dim() == 3: data = data.unsqueeze(1).repeat(1,3,1,1)  # (B, H, W) -> (B, C, H, W)
        output = model[0](data, [[0.0, 1.0, data.shape[-1]] for _ in range(data.shape[0])])
        output = model[1](output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def _test_epoch(test_loader, model, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args._device), target.to(args._device)
            if data.dim() == 3: data = data.unsqueeze(1).repeat(1,3,1,1)  # (B, H, W) -> (B, C, H, W)
            output = model[0](data, [[0.0, 1.0, data.shape[-1]] for _ in range(data.shape[0])])
            output = model[1](output)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    args = make_arg()
    args.batch_size = 2
    args._device = torch.device(args.device)
    datasets, loaders = _make_dataset(args.dataset, args.batch_size, args.sz)
    swin = make_tiny_swin()
    classfier = make_classifier(datasets, swin)
    model = nn.Sequential(swin, classfier).to(args._device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    _test_epoch(loaders['test'], model, args)
    for epoch in range(10):
        _train_epoch(loaders['train'], model, optimizer, args, epoch)
        _test_epoch(loaders['test'], model, args)


if __name__=='__main__':
    main()


