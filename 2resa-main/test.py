import torch
from torch import nn
from torch.utils import model_zoo


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


class BasicBlock(nn.Module):
    expansion = 1  # 通道升降维倍数

    def __init__(self, in_channels, channels, stride=1, downsample=None, attention=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels,
                               kernel_size=3, stride=stride, padding=1)  # 第一个卷积层，通过stride进行下采样
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)  # 第二个卷积层，不进行下采样
        self.bn2 = nn.BatchNorm2d(channels)

        self.downsample = downsample
        self.attention = attention  # CBAM模块
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))

        if self.attention is not None:
            out = self.attention[0](out) * out  # 先进行通道注意力
            self.attention_weights = self.attention[1](out)  # CBAM的注意力图
            out = self.attention_weights * out  # 然后进行空间注意力
        else:
            self.attention_weights = None

        if self.downsample is not None:
            residual = self.downsample(x)  # 通道数不变，1x1卷积层仅用于降采样

        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        self.layers = layers
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一个残差层不进行下样  3层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3层

        self.attention_layer = [self.layer3, self.layer4]  # 仅在最后两个layer上添加注意力

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):  # block：basicblock or bottleneck
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:  # 需要下采样or要融合通道
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))  # 第一个残差块

        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            attention = None
            if i > 1:  # 在第3层、第4层才添加cbam
                attention = nn.Sequential(
                    ChannelAttention(self.in_channels),
                    SpatialAttention())

            layers.append(block(self.in_channels, channels, attention=attention))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self._attention_weights = [None] * len(self.attention_layer)  # 将带有注意力层的注意力权重拿出来

        for i, layer in enumerate(self.attention_layer):
            for j, (name, blk) in enumerate(layer.named_children()):
                self._attention_weights[i] = blk.attention_weights  # 覆盖，仅获取最后一个block的注意力

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @property
    def attention_weights(self):
        return self._attention_weights


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet34_cbam(pretrained=False, num_class=1000):

    model = ResNet(BasicBlock, [3, 4, 6, 3], num_class)  # 每个阶段中的基本块数量分别是 3、4、6、3
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])  # 预训练resnet的权重字典
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items()  # 除去全连接层的预训练权重
                                 if (k in pretrained_state_dict and 'fc' not in k)}

        new_state_dict = model.state_dict()
        new_state_dict.update(pretrained_state_dict)  # 将预训练权重通过dict的update方式更新

        model.load_state_dict(new_state_dict)         # 将更新的网络权重载入到注意力resnet中

    return model
