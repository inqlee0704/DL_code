# model_util.py
# ##############################################################################
# 20210307, In Kyu Lee
# Desc: Deep Learning Models
# ##############################################################################
# Models:
#  - Classification:
#    - ResNext
#    - EfficientNet
#    - ViT
#  - Semantic Segmentation:
#    - UNet
#    - Recursive UNet
# ##############################################################################
# How to use:
# from model_util import get_model
# model = get_model(CFG)
# ##############################################################################
import torch
import torch.nn as nn
import torchvision
import timm

def get_model(CFG):
# Classificaiton models
    if CFG.model == 'ResNext':
        MODEL = torchvision.models.resnext50_32x4d(pretrained=True)
        model = ResNextModel(model=MODEL,num_classes=CFG.target_size)
    elif CFG.model == 'EfficientNet':
        MODEL = timm.create_model('tf_efficientnet_b4_ns',pretrained=True)
        model = EffNetModel(model=MODEL,num_classes=CFG.target_size)
    elif CFG.model == 'ViT':
        MODEL = timm.create_model('vit_base_patch16_384', pretrained=True)
        model = ViTBase(model=MODEL, num_classes=CFG.target_size)
# Segmentation models
    elif CFG.model == 'UNet':
        model = UNet()
    elif CFG.model == 'RecursiveUNet':
        model = RecursiveUNet()

    return model

# ##############################################################################
# Classification Models
# ##############################################################################

# Custom Resnext
class ResNextModel(nn.Module):
    def __init__(self,model,num_classes):
        super().__init__()
        self.convnet = model
        n_features = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(n_features, num_classes)    
    
    def forward(self, img):
        outputs = self.convnet(img)
        return outputs


# EfficientNet
class EffNetModel(nn.Module):
    def __init__(self,model,num_classes):
        super().__init__()
        self.convnet = model
        n_features = self.convnet.classifier.in_features
        self.convnet.classifier = nn.Linear(n_features, num_classes)    
    
    def forward(self, img):
        outputs = self.convnet(img)
        return outputs

# Vit
class ViTBase(nn.Module):
    def __init__(self, model, num_classes):
        super(ViTBase, self).__init__()
        self.model = model
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

# ##############################################################################
# Segmentation Models
# ##############################################################################

# UNet
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    # batch_size, channel, height, width
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(1024,512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_conv_2 = double_conv(512,256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_conv_3 = double_conv(256,128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1)

    def forward(self,image):
        # encoder
        x1 = self.down_conv_1(image) # ---->
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) # --->
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) # -->
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) # ->
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7,x)
        x = self.up_conv_1(torch.cat([x, y],1))
        x = self.up_trans_2(x)
        y = crop_img(x5,x)
        x = self.up_conv_2(torch.cat([x, y],1))
        x = self.up_trans_3(x)
        y = crop_img(x3,x)
        x = self.up_conv_3(torch.cat([x, y],1))
        x = self.up_trans_4(x)
        y = crop_img(x1,x)
        x = self.up_conv_4(torch.cat([x, y],1))
        x = self.out(x)
        return x

# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
class RecursiveUNet(nn.Module):
    def __init__(self,
                num_classes=2,
                in_channels=1,
                initial_filter_size=64,
                kernel_size=3,
                num_downs=4,
                norm_layer=nn.InstanceNorm2d,
                activation=nn.LeakyReLU(inplace=True)):
#       InstancNorm performs better than BatchNorm for airway segmentation
        super(RecursiveUNet, self).__init__()
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1),
                                             out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes,
                                             kernel_size=kernel_size,
                                             norm_layer=norm_layer,
                                             innermost=True,
                                             activation=activation)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes,
                                                 kernel_size=kernel_size,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 activation=activation)

        unet_block = UnetSkipConnectionBlock(in_channels=in_channels,
                                             out_channels=initial_filter_size,
                                             num_classes=num_classes,
                                             kernel_size=kernel_size,
                                             submodule=unet_block,
                                             norm_layer=norm_layer,
                                             outermost=True,
                                             activation=activation)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 num_classes=1,
                 kernel_size=3,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 activation=nn.LeakyReLU(inplace=True)):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              norm_layer=norm_layer,
                              activation=activation)
        conv2 = self.contract(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              norm_layer=norm_layer,
                              activation=activation)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            activation=activation)
        conv4 = self.expand(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            activation=activation)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d,activation=nn.LeakyReLU(inplace=True)):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            activation)
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3, activation=nn.LeakyReLU(inplace=True)):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            activation,
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)

