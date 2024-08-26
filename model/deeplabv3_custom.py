import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

__all__ = ['ResNetV1b', 'resnet18_v1b', 'resnet50_v1b',
           'resnet101_v1b', 'resnet18_v1s', 'resnet50_v1s', 'resnet101_v1s']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1b(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        self.deep_stem = deep_stem
        if deep_stem:
            '''
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
            '''
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = norm_layer(64)
            self.relu1 = nn.ReLU(True)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = norm_layer(64)
            self.relu2 = nn.ReLU(True)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = norm_layer(128)
            self.relu3 = nn.ReLU(True)
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = norm_layer(64)
            self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.deep_stem:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x



def resnet18_v1b(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    return model


def resnet50_v1b(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)
    return model



def resnet101_v1b(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], **kwargs)
    return model


def resnet18_v1s(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], deep_stem=True, **kwargs)
    return model

def resnet50_v1s(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    return model


def resnet101_v1s(pretrained=False, local_rank=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, **kwargs)
    return model

"""Base Model for Semantic Segmentation"""
import torch.nn as nn

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', local_rank=None, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone
        if backbone == 'resnet18':
            self.pretrained = resnet18_v1s(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)

        elif backbone == 'resnet18_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101_original':
            self.pretrained = resnet101_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""

        if self.backbone.split('_')[-1] == 'original':
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu1(x)

            x = self.pretrained.conv2(x)
            x = self.pretrained.bn2(x)
            x = self.pretrained.relu2(x)

            x = self.pretrained.conv3(x)
            x = self.pretrained.bn3(x)
            x = self.pretrained.relu3(x)
            x = self.pretrained.maxpool(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get_deeplabv3']


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        if backbone == 'resnet18':
            in_channels = 512
        else:
            in_channels = 2048

        self.head = _DeepLabHead(in_channels, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
    def forward(self, x):
        auxout = None
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        x, x_feat_after_aspp = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)

        return OrderedDict([
            ('out', x),
            ('aux', auxout)
        ])


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(in_channels, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, nclass, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.block[0:4](x)
        x_feat_after_aspp = x
        x = self.block[4](x)
        return x, x_feat_after_aspp


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        if in_channels == 512:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 256
        else:
            raise

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_deeplabv3(backbone, num_classes, args, local_rank=None, pretrained=False,
                  pretrained_base=False):

    model = DeepLabV3(nclass=(num_classes-1), backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base).to(args.device)
    if pretrained:
        print(f"Loading deeplabv3_resnet101_cirkd.pth.....")
        model.load_state_dict(torch.load('deeplabv3_resnet101_cirkd.pth', map_location=torch.device(args.device)))
        model.head.block[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.auxlayer.block[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        for name, param in model.named_parameters():
            if "head.block.4" not in name and "auxlayer.block.4" not in name:
                param.requires_grad = False
    else:
        model.head.block[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.auxlayer.block[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model