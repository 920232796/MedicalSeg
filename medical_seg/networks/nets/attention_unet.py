# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import torch
import torch.nn as nn
import os
from medical_seg.networks.blocks import Convolution, UpSample
from medical_seg.networks.layers.factories import Conv, Pool
from medical_seg.utils import ensure_tuple_rep

class TwoConv(nn.Sequential):
    """two convolutions."""
    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
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

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
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

        max_pooling = Pool["MAX", dim](kernel_size=2)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, dim, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        if dim == 2:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        elif dim == 3:
            self.W_g = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )
        else:
            os._exit(0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
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
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.attention_block = Attention_block(dim, F_g=up_chns, F_l=up_chns, F_int=in_chns)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
        x_a = self.attention_block(g=x_0, x=x_e)
        x = self.convs(torch.cat([x_a, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class AttentionUNet(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = AttentionUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"UNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)

        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return logits

class AttentionUNetUncert(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = AttentionUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"UNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)

        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)
        self.uncer_drop = nn.Dropout(0.5)
        self.uncer_sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        uncer = self.uncer_drop(logits).mean(dim=1)
        uncer_out = self.uncer_sig(uncer)
        return logits, uncer_out.detach()


class AttentionUNetUncertV2(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = AttentionUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = AttentionUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"UNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)

        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)
        self.uncer_drop = nn.Dropout(0.5)
        self.uncer_sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return logits

    def uncer(self, x):
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        uncer = self.uncer_drop(logits).mean(dim=1)
        uncer_out = self.uncer_sig(uncer)
        return uncer_out

if __name__ == '__main__':
   t1 = torch.rand(1, 3, 50, 50)
   net1 = AttentionUNet(dimensions=2, in_channels=3, out_channels=1, features=[32, 32, 64, 128, 256, 32], dropout=0.1)

   print(net1(t1).shape)

   t2 = torch.rand(2, 3, 50, 50, 50)
   net2 = AttentionUNet(dimensions=3, in_channels=3, out_channels=1, features=[32, 32, 64, 128, 256, 32], dropout=0.1)
   print(net2(t2).shape)

   t3 = torch.rand(1, 3, 50, 50, 50)
   net3 = AttentionUNetUncert(dimensions=3, in_channels=3, out_channels=2, features=[32, 32, 64, 128, 256, 32], dropout=0.1)
   # print(net3(t3)[0].shape)
   out, uncer = net3(t3)
   print(out.shape)
   print(uncer)