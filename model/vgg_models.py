import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from model.vgg import B2_VGG


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """
    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """
    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """
    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class Pred_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Pred_endecoder, self).__init__()
        self.vgg = B2_VGG()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 128)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            # Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.vgg.conv1(x)  ## 352*352*64
        x1 = self.vgg.conv2(x)  ## 176*176*128
        x2 = self.vgg.conv3(x1)  ## 88*88*256
        x3 = self.vgg.conv4(x2)  ## 44*44*512
        x4 = self.vgg.conv5(x3)  ## 22*22*512
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        conv4_feat = self.path4(conv4_feat)
        # print('path 4 size: {}'.format(conv4_feat.shape))
        conv43 = self.path3(conv4_feat, conv3_feat)
        # print('path 3 size: {}'.format(conv43.shape))
        conv432 = self.path2(conv43, conv2_feat)
        # print('path 2 size: {}'.format(conv432.shape))
        conv4321 = self.path1(conv432, conv1_feat)
        # print('path 1 size: {}'.format(conv4321.shape))

        pred = self.output_conv(conv4321)
        # print('pred shape: {}'.format(pred.shape))

        return pred

    # def initialize_weights(self):
    #     res50 = models.resnet50(pretrained=True)
    #     pretrained_dict = res50.state_dict()
    #     all_params = {}
    #     for k, v in self.resnet.state_dict().items():
    #         if k in pretrained_dict.keys():
    #             v = pretrained_dict[k]
    #             all_params[k] = v
    #         elif '_1' in k:
    #             name = k.split('_1')[0] + k.split('_1')[1]
    #             v = pretrained_dict[name]
    #             all_params[k] = v
    #         elif '_2' in k:
    #             name = k.split('_2')[0] + k.split('_2')[1]
    #             v = pretrained_dict[name]
    #             all_params[k] = v
    #     assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
    #     self.resnet.load_state_dict(all_params)



class Pred_endecoder_VGG16(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Pred_endecoder_VGG16, self).__init__()
        self.vgg = B2_VGG()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 128)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            # Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.vgg.conv1(x)  ## 352*352*64
        x1 = self.vgg.conv2(x)  ## 176*176*128
        x2 = self.vgg.conv3(x1)  ## 88*88*256
        x3 = self.vgg.conv4(x2)  ## 44*44*512
        x4 = self.vgg.conv5(x3)  ## 22*22*512
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        conv4_feat = self.path4(conv4_feat)
        # print('path 4 size: {}'.format(conv4_feat.shape))
        conv43 = self.path3(conv4_feat, conv3_feat)
        # print('path 3 size: {}'.format(conv43.shape))
        conv432 = self.path2(conv43, conv2_feat)
        # print('path 2 size: {}'.format(conv432.shape))
        conv4321 = self.path1(conv432, conv1_feat)
        # print('path 1 size: {}'.format(conv4321.shape))

        pred = self.output_conv(conv4321)
        # print('pred shape: {}'.format(pred.shape))

        return pred