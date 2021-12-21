# # Copyright 2020 MONAI Consortium
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from typing import Optional, Sequence, Union
# from warnings import filters

# import numpy as np
# import torch
# import torch.nn.functional as F
# from einops import rearrange

# from medical_seg.networks.blocks.segresnet_block import *
# from medical_seg.networks.layers.factories import Act, Dropout
# from medical_seg.utils import UpsampleMode
# from medical_seg.networks.layers.multi_attention import SpatialTransformerLayer

# class ModalitySelfAttention(nn.Module):
#     def __init__(self, model_num, hidden_size):
#         super().__init__()
#         self.q = nn.Linear(in_features=hidden_size, out_features=hidden_size)
#         self.k = nn.Linear(in_features=hidden_size, out_features=hidden_size)
#         self.v = nn.Linear(in_features=hidden_size, out_features=hidden_size)
#         self.out_conv = nn.Conv3d(model_num*hidden_size, hidden_size, 1, stride=1, padding=0)
    
#     def forward(self, x, model_num):
#         ## x: (batch, modal, feature, d, w, h)
#         x = rearrange(x, "b (m f) d w h -> b d w h m f", m=model_num)
#         q_out = self.q(x)
#         k_out = self.k(x)
#         v_out = self.v(x)
        
#         attention_score = torch.einsum("b d w h m f, b d w h f n -> b d w h m n", q_out, k_out.transpose(-1, -2))
#         modality_att_out = torch.einsum("b d w h m n, b d w h n f -> b d w h m f", attention_score, v_out)
#         modality_att_out = rearrange(modality_att_out, "b d w h m f -> b (m f) d w h")
#         modality_att_out = self.out_conv(modality_att_out)
#         return modality_att_out

# class MultiAttentionNet(nn.Module):
#     """
#     SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
#     <https://arxiv.org/pdf/1810.11654.pdf>`_.
#     The module does not include the variational autoencoder (VAE).
#     The model supports 2D or 3D inputs.

#     Args:
#         spatial_dims: spatial dimension of the input data. Defaults to 3.
#         init_filters: number of output channels for initial convolution layer. Defaults to 8.
#         in_channels: number of input channels for the network. Defaults to 1.
#         out_channels: number of output channels for the network. Defaults to 2.
#         dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
#         norm_name: feature normalization type, this module only supports group norm,
#             batch norm and instance norm. Defaults to ``group``.
#         num_groups: number of groups to separate the channels into. Defaults to 8.
#         use_conv_final: if add a final convolution block to output. Defaults to ``True``.
#         blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
#         blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
#         upsample_mode: [``"transpose"``, ``"nontrainable"``, ``"pixelshuffle"``]
#             The mode of upsampling manipulations.
#             Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

#             - ``transpose``, uses transposed convolution layers.
#             - ``nontrainable``, uses non-trainable `linear` interpolation.
#             - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

#     """

#     def __init__(
#         self,
#         spatial_dims: int = 3,
#         image_size = (1, 1, 1),
#         init_filters: int = 8,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         window_size = (2, 4, 4),
#         dropout_prob: Optional[float] = None,
#         norm_name: str = "group",
#         num_groups: int = 8,
#         use_conv_final: bool = True,
#         blocks_down: tuple = (1, 2, 2, 4),
#         blocks_up: tuple = (1, 1, 1),
#         upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
#     ):
#         super().__init__()

#         assert spatial_dims == 2 or spatial_dims == 3, "spatial_dims can only be 2 or 3."

#         self.spatial_dims = spatial_dims
#         self.init_filters = init_filters
#         self.blocks_down = blocks_down
#         self.blocks_up = blocks_up
#         self.dropout_prob = dropout_prob
#         self.norm_name = norm_name
#         self.num_groups = num_groups
#         self.upsample_mode = UpsampleMode(upsample_mode)
#         self.use_conv_final = use_conv_final
#         self.convInit = get_conv_layer(spatial_dims, 1, init_filters)
#         self.down_layers = nn.ModuleList()
#         self.up_modality_connection = nn.ModuleList()
#         for i in range(in_channels):
#             self.down_layers.append(self._make_down_layers())

#         for i in range(len(blocks_up)):
#             self.up_modality_connection.append(ModalitySelfAttention(model_num=in_channels, 
#                                                                     hidden_size=init_filters*2**i))
#         self.up_layers, self.up_samples = self._make_up_layers()
#         self.relu = Act[Act.RELU](inplace=True)
#         self.conv_final = self._make_final_conv(out_channels)
#         self.modality_layer = ModalitySelfAttention(model_num=in_channels, hidden_size=init_filters*2**(len(blocks_down)-1))
        

#         if dropout_prob:
#             self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

#         self.multi_att = SpatialTransformerLayer(in_channels=init_filters*2**(len(blocks_down)-1), out_channels=init_filters*2**(len(blocks_down)-1), 
#                                                     patch_size=(1, 1, 1), 
#                                                     img_size=image_size, 
#                                                     num_heads=8, 
#                                                     mlp_size=128,
#                                                     num_layers=2,
#                                                     window_size=window_size)
#     def _make_down_layers(self):
#         down_layers = nn.ModuleList()
#         blocks_down, spatial_dims, filters, norm_name, num_groups = (
#             self.blocks_down,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm_name,
#             self.num_groups,
#         )
#         for i in range(len(blocks_down)):
#             layer_in_channels = filters * 2 ** i ## (8, 16, 32, 64)
#             # print(layer_in_channels)
#             pre_conv = (
#                 get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#             down_layer = nn.Sequential(
#                 pre_conv,
#                 *[
#                     ResBlock(spatial_dims, layer_in_channels, norm_name=norm_name, num_groups=num_groups)
#                     for _ in range(blocks_down[i])
#                 ],
#             )
#             down_layers.append(down_layer)
#         return down_layers

#     def _make_up_layers(self):
#         up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
#         upsample_mode, blocks_up, spatial_dims, filters, norm_name, num_groups = (
#             self.upsample_mode,
#             self.blocks_up,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm_name,
#             self.num_groups,
#         )
#         # print(upsample_mode)
#         n_up = len(blocks_up)
#         for i in range(n_up):
#             sample_in_channels = filters * 2 ** (n_up - i)
#             up_layers.append(
#                 nn.Sequential(
#                     *[
#                         ResBlock(spatial_dims, sample_in_channels // 2, norm_name=norm_name, num_groups=num_groups)
#                         for _ in range(blocks_up[i])
#                     ]
#                 )
#             )
#             up_samples.append(
#                 nn.Sequential(
#                     *[
#                         get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
#                         get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
#                     ]
#                 )
#             )
#         return up_layers, up_samples

#     def _make_final_conv(self, out_channels: int):
#         return nn.Sequential(
#             get_norm_layer(self.spatial_dims, self.init_filters, norm_name=self.norm_name, num_groups=self.num_groups),
#             self.relu,
#             get_conv_layer(self.spatial_dims, self.init_filters, out_channels=out_channels, kernel_size=1, bias=True),
#         )

#     def forward(self, x):

#         ## x: (batch, channel, d, w, h)
#         # print(x.shape)
#         model_num = x.shape[1]
#         x = x.unsqueeze(dim=2)
#         init_x = []
#         for i in range(model_num):
#             out = self.convInit(x[:, i])
#             if self.dropout_prob:
#                 out = self.dropout(out)
#             init_x.append(out)
        
#         # print(x.shape)

#         down = [None] * len(self.blocks_down)
#         for i in range(model_num):
#             xx = init_x[i]
#             for j in range(len(self.blocks_down)):
#                 xx = self.down_layers[i][j](xx)
#                 if down[j] is None :
#                     down[j] = xx
#                 else :
#                     down[j] = torch.cat([down[j], xx], dim=1)

#         down.reverse()

#         # for j in range(len(self.blocks_down)):
#             # print(f"down{j} is {down[j].shape}")

#         modality_out = self.modality_layer(down[0], model_num=model_num)

#         multi_att_out = self.multi_att(modality_out)
#         # print(f"multi_att_out is {multi_att_out.shape}")

#          # 拿到modality attention out 在上采样的过程中对cat feature进行 乘法， 做模态特征的筛选。
#         # modality_up_out = self.modality_attention_out()

#         # modality_up_out = torch.softmax(modality_up_out/2.0, dim=1)

#         for i in range(len(self.blocks_up)):
#             if i == 0:
#                 skip_x = self.up_modality_connection[len(self.blocks_up) -1 - i](down[i+1], model_num=model_num)
#                 x = self.up_samples[i](multi_att_out) + skip_x
#                 x = self.up_layers[i](x)
#                 # print(x.shape)
#             else :
#                 skip_x = self.up_modality_connection[len(self.blocks_up) -1 - i](down[i+1], model_num=model_num)
#                 x = self.up_samples[i](x) + skip_x
#                 x = self.up_layers[i](x)
#                 # print(x.shape)

#         if self.use_conv_final:
#             x = self.conv_final(x)
#         return x




# # if __name__ == '__main__':
# #     net = SegResNet(out_channels=1)
# #     t1 = torch.rand((1, 1, 32, 64, 64))
# #     print(net(t1)[0].shape)
#     # print(net(t1)[1])

pass