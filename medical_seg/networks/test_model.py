
from os import path
from typing import Sequence, Union

import torch
import torch.nn as nn

from medical_seg.networks.blocks import Convolution, UpSample
from medical_seg.networks.layers.factories import Conv, Pool
from medical_seg.utils import ensure_tuple_rep
from medical_seg.networks.layers.transformer import TransformerLayerMulti
from medical_seg.networks.layers.spatial_image_transformer import SpatialTransformerLayer

class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, mlp_size, num_layer):
        super().__init__()
        self.z_layer = SpatialTransformerLayer(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                patch_size=patch_size,
                                                img_size=img_size, mlp_size=mlp_size,
                                                num_layers=num_layer,
                                                types=1)
        self.xy_layer = SpatialTransformerLayer(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                patch_size=patch_size,
                                                img_size=img_size, mlp_size=mlp_size,
                                                num_layers=num_layer,
                                                types=2)
    def forward(self, x):
        z_out = self.z_layer(x)
        xy_out = self.xy_layer(z_out)
        return xy_out


class TransNet(nn.Module):
    def __init__(self):
        super().__init__()


    
class TwoConv(nn.Sequential):
    """two convolutions."""
    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
        pool_size = (2, 2, 2)
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, pool_size, mode=upsample)
        self.convs = TwoConv(dim, up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x = self.upsample(x)

        x = self.convs(x)  # input channels: (cat_chns + up_chns)
        return x

class Test(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, patch_size, img_size, mlp_size):
        super().__init__()
        fea = [16, 16, 32, 64, 128, 16]
        self.two_conv_feature_layer = TwoConv(dim=3, in_chns=in_channels, out_chns=hidden_size)
        self.two_conv_feature_layer_2 = TwoConv(dim=3, in_chns=hidden_size, out_chns=fea[5])

        self.trans_layer = TransformerLayerMulti(in_channels=in_channels, 
                                                out_channels=hidden_size, 
                                                patch_size=patch_size, 
                                                img_size=img_size, 
                                                mlp_size=mlp_size)
        

        self.upcat_4 = UpCat(3, fea[4], fea[3])
        self.upcat_3 = UpCat(3, fea[3], fea[2])
        self.upcat_2 = UpCat(3, fea[2], fea[1])
        self.upcat_1 = UpCat(3, fea[1], fea[5])
        self.out_conv = nn.Conv3d(in_channels=fea[5], out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feaures = self.two_conv_feature_layer(x)
        trans_features = self.trans_layer(x)
        # print(trans_features.shape)
        trans_features = self.upcat_4(trans_features)
        trans_features = self.upcat_3(trans_features)
        trans_features = self.upcat_2(trans_features)
        trans_features = self.upcat_1(trans_features)
        # print(feaures.shape)
        # print(trans_features.shape)
        feaures = feaures + trans_features
        out = self.out_conv(feaures)
        return out 
