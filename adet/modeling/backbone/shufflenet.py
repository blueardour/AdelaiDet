
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

from adet.modeling.layers import qconv, norm, actv, shuffle, concat, split, add

from functools import partial

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY as BACKBONES
from detectron2.modeling.backbone import Backbone as BaseModule

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

@BACKBONES.register()
class ShuffleNet(BaseModule):
    """ShuffleNet implementation.
    """
    def __init__(self, args=None, is_large=None, cfg="large_9cls", out_features=['res4'], init_cfg=None):
        if hasattr(super(ShuffleNet, self), 'size_divisibility'):
            super(ShuffleNet, self).__init__()
        else:
            super(ShuffleNet, self).__init__(init_cfg)

        self.args = args
        self.out_features = out_features

        if is_large is not None:
            print("is_large option is deprecated, use cfg instead")

        _cfg = {
            'large_7cls': [(24, None), (72, 4), (120, 5), (240, 8), (480, 5)],
            'large_9cls': [(24, None), (48, 4), (72, 4), (144, 6), (288, 5)],
            'small_9cls': [(24, None), (48, 4), (72, 6), (144, 4), (288, 4)],
        }
        assert cfg in _cfg, "cfg not found in the model definition"
        self.cfg = cfg
        self._cfg = _cfg

        QConv2d = partial(qconv, args=args)
        BatchNorm2d = partial(norm, args=args)
        QClippedReLU = partial(actv, args=args)

        in_channel, _ = _cfg[cfg][0]
        self.conv1 = QConv2d(3, in_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_bn = BatchNorm2d(in_channel)
        self.conv1_ReLU = QClippedReLU()

        self.name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        for i, (out_channel, items) in enumerate(_cfg[cfg]):
            if items is None or i == 0:
                continue

            for j in range(items):
                if j == 0:
                    self.add_module("res{}{}".format(i, self.name[j]), _ShuffleResUnitC(in_channel, out_channel, args=args))
                else:
                    self.add_module("res{}{}".format(i, self.name[j]), _ShuffleResUnitE(out_channel, out_channel, stride=1, args=args))
            in_channel = out_channel

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

        for i, (out_channel, items) in enumerate(self._cfg[self.cfg]):
            if i == 0 or items is None:
                continue

            for j in range(items):
                x = getattr(self, "res{}{}".format(i, self.name[j]))(x)

            if 'res{}'.format(i) in self.out_features:
                outputs.append(x)

        return outputs

class ShuffleNet_(nn.Module):
    def __init__(self, args=None, cfg=None):
        super(ShuffleNet_, self).__init__()
        self.args = args

        self.backbone = ShuffleNet(args=args, cfg=cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        outplanes = self.backbone.res4d.out_channels
        fc_function = nn.Linear
        self.fc = fc_function(outplanes, getattr(args, 'num_classes', 1000))

    def forward(self, x):
        outputs = self.backbone(x)
        assert len(outputs) == 1
        x = outputs[0]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
def shufflenet_large(args=None):
    return ShuffleNet_(args=args, cfg='large_9cls')

def shufflenet_small(args=None):
    return ShuffleNet_(args=args, cfg='small_9cls')

def main():
    model = shufflenet_large()

if __name__ == "__main__":
    main()

