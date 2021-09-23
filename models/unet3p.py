# -*- coding: utf-8 -*-
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore.common.initializer import HeNormal, Constant

from .layers import GlobalAveragePool, unetConv2, UpSample, TruncatedNormal

'''
    UNet 3+
'''

trunc_normal_ = TruncatedNormal(sigma=0.02)
he_normal_ = HeNormal(mode='fan_in')
constant0_ = Constant(value=0)
constant1_ = Constant(value=1)


class UNet_3Plus(nn.Cell):

    def __init__(self, in_channels=3, num_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.layer11 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.layer2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.layer3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.layer4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.layer5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(kernel_size=8, stride=8, pad_mode='valid')
        self.h1_PT_hd4_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h2_PT_hd4_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h3_PT_hd4_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->20*20, hd4->40*40, UpSample 2 times
        self.hd5_UT_hd4 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn4d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h1_PT_hd3_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h2_PT_hd3_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, UpSample 2 times
        self.hd4_UT_hd3 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, UpSample 4 times
        self.hd5_UT_hd3 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn3d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h1_PT_hd2_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, UpSample 2 times
        self.hd3_UT_hd2 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, UpSample 4 times
        self.hd4_UT_hd2 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, UpSample 8 times
        self.hd5_UT_hd2 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn2d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, UpSample 2 times
        self.hd2_UT_hd1 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, UpSample 4 times
        self.hd3_UT_hd1 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True, weight_init='HeNormal')
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, UpSample 8 times
        self.hd4_UT_hd1 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True, weight_init='HeNormal')
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, UpSample 16 times
        self.hd5_UT_hd1 = UpSample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True, weight_init='HeNormal')
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True, weight_init='HeNormal')  # 16
        self.bn1d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu1d_1 = nn.ReLU()

        # output
        self.outconv1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True, weight_init='HeNormal')

        # initialise weights
        # self._init_weights()

    # def _init_weights(self):
    #     for _, m in self.cells_and_names():
    #         if isinstance(m, nn.Dense):
    #             he_normal_._initialize(m.weight.data)
    #             if isinstance(m, nn.Dense) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
    #             he_normal_._initialize(m.weight.data)
    #             if isinstance(m, nn.Conv2d) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             trunc_normal_._initialize(m.gamma)
    #             constant0_._initialize(m.beta)

    def construct(self, inputs):
        ## -------------Encoder-------------
        h1 = self.layer11(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.layer2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.layer3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.layer4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.layer5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            P.Concat(1)((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            P.Concat(1)((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            P.Concat(1)((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            P.Concat(1)((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*num_classes
        return P.Sigmoid()(d1)


'''
    UNet 3+ with deep supervision
'''


class UNet_3Plus_DeepSup(nn.Cell):
    def __init__(self, in_channels=3, num_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(kernel_size=8, stride=8, pad_mode='valid')
        self.h1_PT_hd4_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h2_PT_hd4_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h3_PT_hd4_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->20*20, hd4->40*40, UpSample 2 times
        self.hd5_UT_hd4 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn4d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h1_PT_hd3_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h2_PT_hd3_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, UpSample 2 times
        self.hd4_UT_hd3 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, UpSample 4 times
        self.hd5_UT_hd3 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn3d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h1_PT_hd2_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, UpSample 2 times
        self.hd3_UT_hd2 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, UpSample 4 times
        self.hd4_UT_hd2 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, UpSample 8 times
        self.hd5_UT_hd2 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn2d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, UpSample 2 times
        self.hd2_UT_hd1 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, UpSample 4 times
        self.hd3_UT_hd1 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, UpSample 8 times
        self.hd4_UT_hd1 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, UpSample 16 times
        self.hd5_UT_hd1 = UpSample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn1d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu1d_1 = nn.ReLU()

        # -------------Bilinear Upsampling--------------
        self.upscore6 = UpSample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = UpSample(scale_factor=16, mode='bilinear')
        self.upscore4 = UpSample(scale_factor=8, mode='bilinear')
        self.upscore3 = UpSample(scale_factor=4, mode='bilinear')
        self.upscore2 = UpSample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv2 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv3 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv4 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv5 = nn.Conv2d(in_channels=filters[4], out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)

        # self._init_weights()

    # def _init_weights(self):
    #     for _, m in self.cells_and_names():
    #         if isinstance(m, nn.Dense):
    #             he_normal_._initialize(m.weight)
    #             if isinstance(m, nn.Dense) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
    #             he_normal_._initialize(m.weight)
    #             if isinstance(m, nn.Conv2d) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             trunc_normal_._initialize(m.gamma)
    #             constant0_._initialize(m.beta)

    def construct(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            P.Concat(1)((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            P.Concat(1)((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            P.Concat(1)((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            P.Concat(1)((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256
        return P.Sigmoid()(d1), P.Sigmoid()(d2), P.Sigmoid()(d3), P.Sigmoid()(d4), P.Sigmoid()(d5)


'''
    UNet 3+ with deep supervision and class-guided module
'''


class UNet_3Plus_DeepSup_CGM(nn.Cell):

    def __init__(self, in_channels=3, num_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(kernel_size=8, stride=8, pad_mode='valid')
        self.h1_PT_hd4_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h2_PT_hd4_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h3_PT_hd4_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->20*20, hd4->40*40, UpSample 2 times
        self.hd5_UT_hd4 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn4d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(kernel_size=4, stride=4, pad_mode='valid')
        self.h1_PT_hd3_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h2_PT_hd3_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, UpSample 2 times
        self.hd4_UT_hd3 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, UpSample 4 times
        self.hd5_UT_hd3 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn3d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.h1_PT_hd2_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                        pad_mode='pad', padding=1, has_bias=True)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, UpSample 2 times
        self.hd3_UT_hd2 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, UpSample 4 times
        self.hd4_UT_hd2 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, UpSample 8 times
        self.hd5_UT_hd2 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn2d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, UpSample 2 times
        self.hd2_UT_hd1 = UpSample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, UpSample 4 times
        self.hd3_UT_hd1 = UpSample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, UpSample 8 times
        self.hd4_UT_hd1 = UpSample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, UpSample 16 times
        self.hd5_UT_hd1 = UpSample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=3,
                                         pad_mode='pad', padding=1, has_bias=True)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(num_features=self.CatChannels, momentum=0.9)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=self.UpChannels, kernel_size=3,
                                  pad_mode='pad', padding=1, has_bias=True)  # 16
        self.bn1d_1 = nn.BatchNorm2d(num_features=self.UpChannels, momentum=0.9)
        self.relu1d_1 = nn.ReLU()

        # -------------Bilinear Upsampling--------------
        self.upscore6 = UpSample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = UpSample(scale_factor=16, mode='bilinear')
        self.upscore4 = UpSample(scale_factor=8, mode='bilinear')
        self.upscore3 = UpSample(scale_factor=4, mode='bilinear')
        self.upscore2 = UpSample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv2 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv3 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv4 = nn.Conv2d(in_channels=self.UpChannels, out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.outconv5 = nn.Conv2d(in_channels=filters[4], out_channels=num_classes, kernel_size=3, pad_mode='pad',
                                  padding=1, has_bias=True)

        self.cls = nn.SequentialCell([
            nn.Dropout(keep_prob=0.5),
            nn.Conv2d(in_channels=filters[4], out_channels=2, kernel_size=1, pad_mode='pad', has_bias=True),
            GlobalAveragePool(),
            nn.Sigmoid()])

        # initialise weights
        # self._init_weights()

    # def _init_weights(self):
    #     for _, m in self.cells_and_names():
    #         if isinstance(m, nn.Dense):
    #             he_normal_._initialize(m.weight)
    #             if isinstance(m, nn.Dense) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
    #             he_normal_._initialize(m.weight)
    #             if isinstance(m, nn.Conv2d) and m.bias is not None:
    #                 constant0_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             trunc_normal_._initialize(m.gamma)
    #             constant0_._initialize(m.beta)

    def dotProduct(self, seg, cls):
        B, N, H, W = P.Shape()(seg)
        seg = P.Reshape()(seg, (B, N, H * W,))
        cls = P.ExpandDims()(cls, -1)
        final = seg * cls
        final = P.Reshape()(final, (B, N, H, W,))
        return final

    def construct(self, inputs):
        B = inputs.shape[0]
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Classification-------------
        cls_branch = self.cls(hd5)
        cls_branch_max = P.Argmax(axis=1, output_type=mstype.int32)(cls_branch)
        cls_branch_max = P.Reshape()(cls_branch_max, (B, 1))
        # cls_branch_max = P.ExpandDims(-1)(cls_branch_max)
        # cls_branch_max = cls_branch_max[:, np.newaxis].float()

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            P.Concat(1)((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            P.Concat(1)((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            P.Concat(1)((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            P.Concat(1)((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)

        return P.Sigmoid()(d1), P.Sigmoid()(d2), P.Sigmoid()(d3), P.Sigmoid()(d4), P.Sigmoid()(d5)


# from mindspore import context
# from mindspore import Tensor
# import numpy as np

# context.set_context(mode=context.PYNATIVE_MODE)
# model = UNet_3Plus_DeepSup_CGM()
# data = Tensor(np.random.randn(2, 3, 224, 224), mstype.float32)
# print(model(data)[0].shape, model(data)[1].shape)
