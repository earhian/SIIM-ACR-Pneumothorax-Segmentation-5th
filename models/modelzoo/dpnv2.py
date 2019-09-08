from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from models.utils import *

from models.convert_from_mxnet import *


class CenterBlock(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels, rates=[1,2,4,6,8,12,16], feature_size=16):
        super(CenterBlock, self).__init__()
        self.feature_size = feature_size
        # assert in_channels == mid_channels
        self.in_channels = in_channels
        self.DC_layers = nn.ModuleList()
        self.gobal_info_layer = nn.Sequential(
            nn.Linear(self.feature_size * self.feature_size, self.feature_size * self.feature_size),
            nn.BatchNorm1d(self.feature_size*self.feature_size),
            nn.ReLU()
        )
        self.gobal_same_channel = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.DC_layers.append(
            nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        )
        for rate in rates:
            self.DC_layers.append(
                nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(mid_channels),
                     nn.ReLU()
            )
            )
        self.final = nn.Conv2d(mid_channels * (len(rates) + 2) , out_channels, kernel_size=1)
        self.se = Selayer(out_channels)
    def forward(self, x):

        b = x.size(0)
        results = []
        for index in range(len(self.DC_layers)):
            results.append(self.DC_layers[index](x))
        x = x.view(-1, self.feature_size * self.feature_size)
        x = self.gobal_info_layer(x)
        x = x.view(b, -1, self.feature_size, self.feature_size)
        x = self.gobal_same_channel(x)
        results.append(x)
        results = torch.cat(results, 1)
        self.final(results)
        results = self.final(results)
        # 1 kernel_size high_rate_DC and gobal_info 有比较高的权重 gobal_info有最高的分数
        return self.se(results)



__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']


model_urls = {
    'dpn68':
        'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
    'dpn68b-extra':
        'http://data.lip6.fr/cadene/pretrainedmodels/'
        'dpn68b_extra-84854c156.pth',
    'dpn92': '',
    'dpn92-extra':
        'http://data.lip6.fr/cadene/pretrainedmodels/'
        'dpn92_extra-b040e4a9b.pth',
    'dpn98':
        'http://data.lip6.fr/cadene/pretrainedmodels/dpn98-5b90dec4d.pth',
    'dpn131':
        'http://data.lip6.fr/cadene/pretrainedmodels/dpn131-71dfe43e0.pth',
    'dpn107-extra':
        'http://data.lip6.fr/cadene/pretrainedmodels/'
        'dpn107_extra-1ac7121e2.pth'
}


def dpn68(num_classes=1000, pretrained=False, inchannels=1,feature_size=16,test_time_pool=True):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes,inchannels=inchannels,feature_size=feature_size, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn68']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn68']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn68b(num_classes=1000, pretrained=False, test_time_pool=True, inchannels=1):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn68b-extra']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn68b-extra']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn92(num_classes=1000, pretrained=False, test_time_pool=True, extra=True):
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        # there are both imagenet 5k trained, 1k finetuned 'extra' weights
        # and normal imagenet 1k trained weights for dpn92
        key = 'dpn92'
        if extra:
            key += '-extra'
        if model_urls[key]:
            model.load_state_dict(model_zoo.load_url(model_urls[key]))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/' + key)
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn98(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn98']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn98']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn98')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn131(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn131']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn131']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn131')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn107(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn107-extra']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn107-extra']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn107-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,inchannels=1,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            inchannels, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        self.in_chs = in_chs
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
        # self.in1 = nn.InstanceNorm2d(num_1x1_c, affine=True)
        # if not self.has_proj:
        #     self.in2 = nn.InstanceNorm2d(in_chs - num_1x1_c + inc, affine=True)
        # else:
        #     self.in2 = nn.InstanceNorm2d(inc * 3, affine=True)
        # self.relu = nn.ReLU()
    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        # resid = self.relu(self.in1(resid))
        dense = torch.cat([x_s2, out2], dim=1)
        # dense = self.relu(self.in2(dense))

        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False, feature_size=16,inchannels=3):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()
        incsincs = []
        self.inchannels = inchannels
        print('inchannels:', self.inchannels)
        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=5, padding=2, inchannels=inchannels)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3, inchannels=inchannels)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        incs.append(in_chs)
        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        incs.append(in_chs)

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        incs.append(in_chs)

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        incs.append(in_chs)
        # blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.incs = incs
        print(incs)

        self.features = nn.Sequential(blocks)



        self.k_sec = k_sec
    def forward(self, x):
        k_sec = self.k_sec
        x1 = self.features[:1](x)
        start = 1

        x2 = self.features[start:start + k_sec[0]](x1)
        start += k_sec[0]

        x3 = self.features[start:start + k_sec[1]](x2)
        start += k_sec[1]

        x4 = self.features[start:start + k_sec[2]](x3)
        start += k_sec[2]

        x5 = self.features[start:start + k_sec[3]](x4)

        return x2, x3, x4, x5

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    #
    import torch
    model = dpn68(num_classes=2, pretrained=True, inchannels=2, feature_size=10).cuda()
    input = torch.rand((8, 2, 160, 160)).cuda()
    outs = model(input)
    print(outs.shape)
