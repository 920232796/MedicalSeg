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
        pool_size=(2, 2, 2),
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

        max_pooling = Pool["MAX", dim](kernel_size=pool_size)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


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
        up_size = (2, 2, 2)
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
        self.upsample = UpSample(dim, in_chns, up_chns, up_size, mode=upsample)
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

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class BasicUNetTwoStream(nn.Module):
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
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, 32, act, norm, dropout)
        self.down_1 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_2 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_3 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_4 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(2, 2, 2))

        self.conv_0_2 = TwoConv(dimensions, in_channels, 32, act, norm, dropout)
        self.down_1_2 = Down(dimensions,32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_2_2 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_3_2 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_4_2 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(2, 2, 2))

        self.conv_0_1_2 = nn.Conv3d(2 * 32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_1_1_2 = nn.Conv3d(2 * 32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_2_1_2 = nn.Conv3d(2 * 32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_3_1_2 = nn.Conv3d(2 * 32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_4_1_2 = nn.Conv3d(2 * 32, 32, kernel_size=1, stride=1, padding=0)

        self.final_conv = Conv["conv", dimensions](32*5, out_channels, kernel_size=1)

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
        x_t1 = x[:, 0].unsqueeze(dim=1)
        x_t2 = x[:, 1].unsqueeze(dim=1)
        x0 = self.conv_0(x_t1)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        x0_2 = self.conv_0_2(x_t2)
        x1_2 = self.down_1_2(x0_2)
        x2_2 = self.down_2_2(x1_2)
        x3_2 = self.down_3_2(x2_2)
        x4_2 = self.down_4_2(x3_2)

        x0_1_2 = torch.cat([x0, x0_2], dim=1)
        x0_1_2 = self.conv_0_1_2(x0_1_2)
        x1_1_2 = torch.cat([x1, x1_2], dim=1)
        x1_1_2 = self.conv_1_1_2(x1_1_2)
        x2_1_2 = torch.cat([x2, x2_2], dim=1)
        x2_1_2 = self.conv_2_1_2(x2_1_2)
        x3_1_2 = torch.cat([x3, x3_2], dim=1)
        x3_1_2 = self.conv_3_1_2(x3_1_2)
        x4_1_2 = torch.cat([x4, x4_2], dim=1)
        x4_1_2 = self.conv_4_1_2(x4_1_2)

        x1 = torch.nn.functional.interpolate(x1_1_2, scale_factor=(1, 2, 2))
        x2 = torch.nn.functional.interpolate(x2_1_2, scale_factor=(1, 4, 4))
        x3 = torch.nn.functional.interpolate(x3_1_2, scale_factor=(1, 8, 8))
        x4 = torch.nn.functional.interpolate(x4_1_2, scale_factor=(2, 16, 16))

        x4 = torch.cat([x0, x1, x2, x3, x4], dim=1)

        logits = self.final_conv(x4)
        return logits

class BasicUNetOneStream(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
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
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, 32, act, norm, dropout)
        self.down_1 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_2 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_3 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(1, 2, 2))
        self.down_4 = Down(dimensions, 32, 32, act, norm, dropout, pool_size=(2, 2, 2))


        self.final_conv = Conv["conv", dimensions](32*5, out_channels, kernel_size=1)

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
        x1 = torch.nn.functional.interpolate(x1, scale_factor=(1, 2, 2))
        x2 = torch.nn.functional.interpolate(x2, scale_factor=(1, 4, 4))
        x3 = torch.nn.functional.interpolate(x3, scale_factor=(1, 8, 8))
        x4 = torch.nn.functional.interpolate(x4, scale_factor=(2, 16, 16))

        x4 = torch.cat([x0, x1, x2, x3, x4], dim=1)

        logits = self.final_conv(x4)
        return logits

class BasicUNetEncoder(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            in_channels: int = 1,
            features: Sequence[int] = (16, 16, 32, 64, 128, 16),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    ):
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, fea[0], act, norm, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout, pool_size=pool_size[0])
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout, pool_size=pool_size[1])
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout, pool_size=pool_size[2])
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout, pool_size=pool_size[3])

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

        return x4, x3, x2, x1, x0
