import logging
import math
import os
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from bis3d_v2.networks.nets.transformer.transformer_model import TransConfig, TransModel2d
import math


class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.trans_model = TransModel2d(config)
        self.final_dense = nn.Linear(config.hidden_size,
                                     config.patch_size[0] * config.patch_size[1] * config.out_channels)
        self.patch_size = config.patch_size

    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1
        ww = w // p2

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)

        encode_x = self.trans_model(x)[-1]  # 取出来最后一层

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=hh, w=ww, c=self.config.out_channels)
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.config = config
        self.trans_model = TransModel2d(config)
        self.conv_embed = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(config.hidden_size, config.in_channels, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(config.in_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.conv_out2 = nn.Conv2d(config.in_channels, config.in_channels, kernel_size=3, padding=1)
        self.bn_out2 = nn.BatchNorm2d(config.in_channels)
        self.relu_out2 = nn.ReLU(inplace=True)
    def forward(self, x):
        ## x:(b, c, w, h)

        x = self.conv_embed(x)
        b, c, h, w = x.shape

        x = x.view(b, -1, c)

        encoder_x = self.trans_model(x)[-1]
        encoder_x = encoder_x.view(b, h, w, c)
        encoder_x = encoder_x.permute(0, 3, 1, 2)
        encoder_x = self.conv_out(encoder_x)
        encoder_x = self.bn_out(encoder_x)
        encoder_x = self.relu_out(encoder_x)
        encoder_x = self.conv_out2(encoder_x)
        encoder_x = self.bn_out2(encoder_x)
        encoder_x = self.relu_out2(encoder_x)
        return encoder_x


