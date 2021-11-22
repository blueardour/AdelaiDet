
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY as BACKBONES
from detectron2.modeling.backbone import Backbone as BaseModule

from adet.modeling.layers import qconv, norm, actv, shuffle, concat, split, add

class _ShuffleBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, mid_stride=1, mid_groups=6, out_groups=2, args=None):
        super(_ShuffleBottleneck, self).__init__()
        QConv2d = partial(qconv, args=args)
        BatchNorm2d = partial(norm, args=args)
        QClippedReLU = partial(actv, args=args)
        QShuffleOp = partial(shuffle, args=args)

        mid_channels = mid_channels or in_channels
        self.conv1 = QConv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=2, bias=False)
        self.bn1 = BatchNorm2d(mid_channels)
        self.relu1 = QClippedReLU()

        self.shfl_op = QShuffleOp(2)

        self.conv2 = QConv2d(mid_channels, mid_channels, kernel_size=3, stride=mid_stride, padding=1, groups=mid_channels//4, bias=False)
        self.bn2 = BatchNorm2d(mid_channels)
        self.relu2 = QClippedReLU()

        self.conv3 = QConv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=out_groups, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        self.relu3 = QClippedReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.shfl_op(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class _ShuffleResUnitC(nn.Module):
    """C is short for Concat"""
    def __init__(self, in_channels, out_channels, mid_channels=None, mid_groups=6, args=None):
        super(_ShuffleResUnitC, self).__init__()
        QConcat = partial(concat, args=args)

        mid_channels = mid_channels or in_channels
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ShuffleBottleneck(in_channels, out_channels - in_channels, mid_channels, mid_stride=2, args=args)
        self.concat = QConcat()

    def forward(self, x):
        return self.concat(self.pooling(x), self.bottleneck(x))



class _ShuffleResUnitE_branch(nn.Module):
    def __init__(self, in_channels, mid_stride, args=None):
        super(_ShuffleResUnitE_branch, self).__init__()
        QConv2d = partial(qconv, args=args)
        BatchNorm2d = partial(norm, args=args)
        QClippedReLU = partial(actv, args=args)

        self.conv1 = QConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(in_channels)
        self.relu1 = QClippedReLU()
        
        self.conv2 = QConv2d(in_channels, in_channels, kernel_size=3, stride=mid_stride, padding=1, groups=in_channels//4, bias=False)
        self.bn2 = BatchNorm2d(in_channels)
        self.relu2 = QClippedReLU()

        self.conv3 = QConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(in_channels)
        self.relu3 = QClippedReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class _ShuffleResUnitE(nn.Module):
    """E is short for Eltwise-Add"""
    def __init__(self, in_channels, out_channels, stride, mid_channels=None, mid_groups=6, args=None):
        super(_ShuffleResUnitE, self).__init__()
        QConcat = partial(concat, args=args)
        QShuffleOp = partial(shuffle, args=args)
        QHalfSplit = partial(split, args=args)

        self.stride = stride
        assert stride in [1, 2]
        self.out_channels = out_channels
    
        self.concat_res = QConcat()
        self.shfl_op = QShuffleOp(2)
        self.banch2 = _ShuffleResUnitE_branch(out_channels//2, self.stride, args=args)
        self.first_half = partial(QHalfSplit, dim=1, first_half=True)()
        self.second_split = partial(QHalfSplit, dim=1, first_half=False)()

    def forward(self, x):
        x1 = self.first_half(x)
        x2 = self.second_split(x)
        out = self.concat_res(x1, self.banch2(x2))
        return self.shfl_op(out)

@BACKBONES.register_module()
class ShuffleNet(BaseModule):
    """ShuffleNet implementation.
    """
    def __init__(self, args=None, is_large=None, out_features=['res4'], **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        self.args = args
        self.out_features = out_features
        if not isinstance(is_large, bool):
            if args is not None and hasattr(args, 'keyword'):
                self.large = False if 'ShufflenetS' in args.keyword else True
            else:
                raise RuntimeError("either args or is_large should be provided")
        else:
            self.large = is_large

        QConv2d = partial(qconv, args=args)
        BatchNorm2d = partial(norm, args=args)
        QClippedReLU = partial(actv, args=args)

        self.conv1 = QConv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_bn = BatchNorm2d(24)
        self.conv1_ReLU = QClippedReLU()

        self.res1a = _ShuffleResUnitC(24, 48, args=args)
        self.res1b = _ShuffleResUnitE(48, 48, stride=1, args=args)
        self.res1c = _ShuffleResUnitE(48, 48, stride=1, args=args)
        self.res1d = _ShuffleResUnitE(48, 48, stride=1, args=args)

        self.res2a = _ShuffleResUnitC(48, 72, args=args)
        self.res2b = _ShuffleResUnitE(72, 72, stride=1, args=args)
        self.res2c = _ShuffleResUnitE(72, 72, stride=1, args=args)
        self.res2d = _ShuffleResUnitE(72, 72, stride=1, args=args)
        if not self.large:
            self.res2e = _ShuffleResUnitE(72, 72, stride=1, args=args)
            self.res2f = _ShuffleResUnitE(72, 72, stride=1, args=args)

        self.res3a = _ShuffleResUnitC(72, 144, args=args)
        self.res3b = _ShuffleResUnitE(144, 144, stride=1, args=args)
        self.res3c = _ShuffleResUnitE(144, 144, stride=1, args=args)
        self.res3d = _ShuffleResUnitE(144, 144, stride=1, args=args)
        if self.large:
            self.res3e = _ShuffleResUnitE(144, 144, stride=1, args=args)
            self.res3f = _ShuffleResUnitE(144, 144, stride=1, args=args)

        self.res4a = _ShuffleResUnitC(144, 288, args=args)
        self.res4b = _ShuffleResUnitE(288, 288, stride=1, args=args)
        self.res4c = _ShuffleResUnitE(288, 288, stride=1, args=args)
        self.res4d = _ShuffleResUnitE(288, 288, stride=1, args=args)
        if self.large:
            self.res4e = _ShuffleResUnitE(288, 288, stride=1, args=args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []

        x = self.conv1_ReLU(self.conv1_bn(self.conv1(x)))
        if 'stem' in self.out_features:
            outputs.append(x)

        # res1
        x = self.res1a(x)
        x = self.res1b(x)
        x = self.res1c(x)
        x = self.res1d(x)
        if 'res1' in self.out_features:
            outputs.append(x)

        # res2
        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)
        x = self.res2d(x)
        if not self.large:
            x = self.res2e(x)
            x = self.res2f(x)
        if 'res2' in self.out_features:
            outputs.append(x)

        # res3
        x = self.res3a(x)
        x = self.res3b(x)
        x = self.res3c(x)
        x = self.res3d(x)
        if self.large:
            x = self.res3e(x)
            x = self.res3f(x)
        if 'res3' in self.out_features:
            outputs.append(x)

        # res4
        x = self.res4a(x)
        x = self.res4b(x)
        x = self.res4c(x)
        x = self.res4d(x)
        if self.large:
            x = self.res4e(x)
        if 'res4' in self.out_features:
            outputs.append(x)

        return outputs

class ShuffleNet_(nn.Module):
    def __init__(self, args=None, is_large=None):
        super(ShuffleNet_, self).__init__()
        self.args = args

        self.backbone = ShuffleNet(args=args, is_large=is_large)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        outplanes = self.backbone.res4d.out_channels
        fc_function = nn.Linear
        self.fc = fc_function(outplanes, args.num_classes)

    def forward(self, x):
        outputs = self.backbone(x)
        assert len(outputs) == 1
        x = outputs[0]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
def shufflenet_large(args=None):
    return ShuffleNet_(args=args, is_large=True)

def shufflenet_small(args=None):
    return ShuffleNet_(args=args, is_large=False)


