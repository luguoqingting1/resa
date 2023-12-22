import torch.nn as nn
import torch
import torch.nn.functional as F

from models.registry import NET
from models.resnet import ResNetWrapper
from models.decoder import BUSD, PlainDecoder


# 定义了RESA这个模块的卷积块和索引，和这个模块怎么执行
class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.iter = cfg.resa.iter
        chan = cfg.resa.input_channel  # 128
        fea_stride = cfg.backbone.fea_stride  # 8
        self.height = cfg.img_height // fea_stride  # 288/8  高为36
        self.width = cfg.img_width // fea_stride  # 800/8 宽为100
        self.alpha = cfg.resa.alpha  # 2
        conv_stride = cfg.resa.conv_stride  # 9
        # 初始化两个垂直卷积和两个水平卷积，四个方向索引
        for i in range(self.iter):  # iter=4
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),  # conv_stride为9
                padding=(0, conv_stride // 2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)
            # 使用 setattr 方法将创建的卷积层和索引设置为模型的属性，以便在模型的前向传播中使用。属性的名称根据当前循环的索引进行命名
            setattr(self, 'conv_d' + str(i), conv_vert1)  # 把 conv_vert1赋给conv_di,conv_d0
            setattr(self, 'conv_u' + str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)

            setattr(self, 'conv_r' + str(i), conv_hori1)
            setattr(self, 'conv_l' + str(i), conv_hori2)
            # 建立四个索引
            idx_d = (torch.arange(2 * self.height // 3) + (self.height // 2 ** (self.iter - i))) % (
                    2 * self.height // 3)
            setattr(self, 'idx_d' + str(i), idx_d)  # idx_d0  idx_d1  idx_d2..

            idx_u = (torch.arange(2 * self.height // 3) - (self.height // 2 ** (self.iter - i))) % (
                    2 * self.height // 3)
            setattr(self, 'idx_u' + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_r' + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, x):
        a = x[:, :, 12:, :]
        a = a.clone()
        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                a.add_(self.alpha * F.relu(conv(a[..., idx, :])))

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                a.add_(self.alpha * F.relu(conv(a[..., idx])))
        a = torch.cat((x[:, :, :12, :], a), dim=2)
        return a


class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)  # ???
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)  # 输入128 输出5

        stride = cfg.backbone.fea_stride * 2  # 16
        self.fc9 = nn.Linear(  # 输入维度：类别数×有效宽高，输出维度
            int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128)
        self.fc10 = nn.Linear(128, cfg.num_classes - 1)  # 输入维度，输出维度4

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, dim=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.view(-1, x.numel() // x.shape[0])  # 第一维行，列为计算每个样本（每个批次）的元素数量。每一行对应一个样本，每一列对应一个样本中的元素。
        print("x.view", x.shape)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = torch.sigmoid(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(  # 使用1x1卷积代替线性层，可以不用调整tensor的形状
            nn.Conv2d(self.channels, self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x)  # (b, c, 1, 1)
        avgout = self.avgpool(x)  # (b, c, 1, 1)

        maxout = self.mlp(maxout)  # (b, c, 1, 1)
        avgout = self.mlp(avgout)  # (b, c, 1, 1)

        attention = self.sigmoid(maxout + avgout)  # (b, c, 1, 1)
        return attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1,
                              kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxpool = x.argmax(dim=1, keepdim=True)  # (b, 1, h, w)
        avgpool = x.mean(dim=1, keepdim=True)  # (b, 1, h, w)

        out = torch.cat([maxpool, avgpool], dim=1)  # (b, 2, h, w)
        out = self.conv(out)  # (b, 1, h, w)

        attention = self.sigmoid(out)  # (b, 1, h, w)
        return attention


# 整个网络结构，先是resnet，再是resa 再是decoder和exist
@NET.register_module
class RESANet(nn.Module):
    def __init__(self, cfg):
        super(RESANet, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)

        self.resa = RESA(cfg)
        self.decoder = eval(cfg.decoder)(cfg)  # 通过eval将字符串转换为实际的类对象，从而创建了一个 decoder 对象。
        self.heads = ExistHead(cfg)

        self.channel_attention = ChannelAttention(channels=128)
        self.spatial_attention = SpatialAttention()

    def forward(self, batch):
        fea = self.backbone(batch)

        channel_attention = self.channel_attention(fea)
        spatial_attention = self.spatial_attention(fea)

        fea_with_attention = fea * channel_attention + fea * spatial_attention

        fea = self.resa(fea_with_attention)  # RESA处理
        seg = self.decoder(fea)  # decoder里的plain decoder处理
        exist = self.heads(fea)  # ExistHead处理

        output = {'seg': seg, 'exist': exist}

        return output
