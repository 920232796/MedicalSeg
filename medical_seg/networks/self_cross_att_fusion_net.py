
from operator import mod
from medical_seg.networks.layers.image_transformer import PoolFormer
from medical_seg.networks.layers.multi_attention import MultiAttentionTransformer
from medical_seg.networks.nets.co_unet import BasicUNet
from medical_seg.networks.nets.basic_unet_encoder import BasicUNetEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
# from medical_seg.networks.layers.spatial_image_transformer import SpatialTransformerLayer
from medical_seg.networks.layers.fusion_transformer import FusionSelfCrossTrans
from typing import Sequence, Union
from medical_seg.networks.nets.co_unet import UpCat
from medical_seg.networks.layers.cpc import ImageCPC
from einops import rearrange
from medical_seg.networks.nets.basic_pool_former import BasicPoolFormerEncoder

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            
        )
    def forward(self, x):
        return self.net(x)


class ModalitySelfAttention(nn.Module):
    def __init__(self, model_num, hidden_size):
        super().__init__()
        self.q = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.k = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.v = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.out_conv = CNN(in_channels=model_num*hidden_size, out_channels=hidden_size)
    
    def forward(self, x, model_num):
        ## x: (batch, modal, feature, d, w, h)
        x = rearrange(x, "b m f d w h -> b d w h m f", m=model_num)
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        
        attention_score = torch.einsum("b d w h m f, b d w h f n -> b d w h m n", q_out, k_out.transpose(-1, -2))
        modality_att_out = torch.einsum("b d w h m n, b d w h n f -> b d w h m f", attention_score, v_out)
        modality_att_out = rearrange(modality_att_out, "b d w h m f -> b (m f) d w h")
        modality_att_out = self.out_conv(modality_att_out)
        return modality_att_out

class MCNNEncoder(nn.Module):
    def __init__(self, model_num, fea=[16, 16, 32, 64, 128, 16], 
                    pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]):

        super().__init__()
        self.model_num = model_num
        self.unets = nn.ModuleList([])
        for i in range(model_num):
            unet = BasicUNetEncoder(dimensions=3, in_channels=1,
                                         features=fea, pool_size=pool_size)
            self.unets.append(unet)

    def forward(self, x):
        encoder_out = []
        x = x.unsqueeze(dim=2)
        for i in range(self.model_num):
            encoder_out.append(self.unets[i](x[:, i]))

        return encoder_out
    
class PoolFormerEncoders(nn.Module):

    def __init__(self, model_num, fea=[16, 16, 32, 64, 128, 16], 
                    pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]):

        super().__init__()
        self.model_num = model_num
        self.encoders = nn.ModuleList([])
        for i in range(model_num):
            encoder = BasicPoolFormerEncoder(dimensions=3,
                                in_channels=1,
                                pool_size=pool_size,
                                features=fea)

            self.encoders.append(encoder)

    def forward(self, x):
        encoder_out = []
        x = x.unsqueeze(dim=2)
        for i in range(self.model_num):
            encoder_out.append(self.encoders[i](x[:, i]))

        return encoder_out

class PoolFormerUpcat(nn.Module):
    def __init__(self, in_channels, out_channels, up_size=(2, 2, 2), num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_size = up_size
        up_ = up_size[0] * up_size[1] * up_size[2]
        self.patch_expand = nn.Conv3d(in_channels, in_channels*up_, 1, 1, 0)
        self.pool_former = PoolFormer(in_channels+out_channels, out_channels, patch_size=(1, 1, 1), mlp_size=out_channels, num_layers=num_layers)

    def forward(self, x, x_cat):
        pass
        b, c, d, w, h = x.shape
        
        x = self.patch_expand(x)
        x = rearrange(x, "b (p1 p2 p3 c) d w h -> b c (d p1) (w p2) (h p3)", c = c, p1=self.up_size[0], p2=self.up_size[1], p3=self.up_size[2])
        x = torch.cat([x, x_cat], dim=1)
        x = self.pool_former(x)
        return x 


class SCAFNet(nn.Module):
    def __init__(self, model_num, out_channels, image_size, 
                act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm = ("GROUP", {"num_groups": 8, "affine": False}),
                dropout: Union[float, tuple] = 0.0,
                upsample: str = "deconv",
                fea = [16, 16, 32, 64, 128, 16],
                window_size=(2, 4, 4),
                pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
                patch_size=(1, 1, 1), 
                self_num_layer=2, 
                token_mixer_size=32,
                encoder="cnn",
                corss_attention=True,
                modality_gate=True):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.pool_size = pool_size
        self.cross_attention = corss_attention
        self.modality_gate = modality_gate

        pool_size_all = [1, 1, 1]
        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]

        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]
        if encoder == "cnn":
            print("use cnn encoder")
            self.multicnn_encoder = MCNNEncoder(model_num=model_num, fea=fea, pool_size=pool_size)
        elif encoder == "poolformer":
            print("use poolformer encoder")
            self.multicnn_encoder = PoolFormerEncoders(model_num=model_num, fea=fea, pool_size=pool_size)
        else :
            import os 
            print("model is error")
            os._exit(0)
        
        if self.cross_attention:

            print("use cross attention fusion module")
            self.cross_fusion_trans = FusionSelfCrossTrans(model_num=model_num, 
                                                        in_channels=fea[4], 
                                                        hidden_size=fea[4],
                                                        patch_size=patch_size, 
                                                        img_size=new_image_size,
                                                        mlp_size=2*fea[4], self_num_layer=self_num_layer,
                                                        window_size=window_size, token_mixer_size=token_mixer_size)

        else :
            # 不用crss attention
            print("no cross attention fusion module")
        self.fusion_conv_5 = CNN(model_num*fea[4], fea[4], 3, 1)

        if self.modality_gate:
            print("use modality gate module")
            self.gate_layer = nn.Conv3d(fea[4], 2, 1, 1, 0)
            
        else :
            print("no modality gate module")
        self.fusion_conv_1 = CNN(model_num*fea[0], fea[0], 3, 1)
        self.fusion_conv_2 = CNN(model_num*fea[1], fea[1], 3, 1)
        self.fusion_conv_3 = CNN(model_num*fea[2], fea[2], 3, 1)
        self.fusion_conv_4 = CNN(model_num*fea[3], fea[3], 3, 1)
        
        
        # self.upcat_4 = PoolFormerUpcat(fea[4], fea[3], up_size=pool_size[3], num_layers=2)
        # self.upcat_3 = PoolFormerUpcat(fea[3], fea[2], up_size=pool_size[2], num_layers=2)
        # self.upcat_2 = PoolFormerUpcat(fea[2], fea[1], up_size=pool_size[1], num_layers=2)
        # self.upcat_1 = PoolFormerUpcat(fea[1], fea[5], up_size=pool_size[0], num_layers=2)
        
        self.upcat_4 = UpCat(3, fea[4], fea[3], fea[3], act, norm, dropout, upsample, pool_size=pool_size[3])
        self.upcat_3 = UpCat(3, fea[3], fea[2], fea[2], act, norm, dropout, upsample, pool_size=pool_size[2])
        self.upcat_2 = UpCat(3, fea[2], fea[1], fea[1], act, norm, dropout, upsample, pool_size=pool_size[1])
        self.upcat_1 = UpCat(3, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False, pool_size=pool_size[0])
        
        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        encoder_x = self.multicnn_encoder(x)

        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)
        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)
        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)
        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)            
        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)            

        if self.cross_attention:
            fusion_out = self.cross_fusion_trans(encoder_5)
            encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")
            fusion_out_cnn = self.fusion_conv_5(encoder_5)
            fusion_out = fusion_out + fusion_out_cnn
        else :
            # 不用cross attention
            encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")
            fusion_out = self.fusion_conv_5(encoder_5)

        if self.modality_gate:
            # 使用modality gate
            fusion_out_tmp = self.gate_layer(fusion_out)
            fusion_out_2 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_tmp, scale_factor=self.pool_size[3], mode="trilinear"))
            fusion_out_4 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_2, scale_factor=self.pool_size[2], mode="trilinear"))
            fusion_out_8 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_4, scale_factor=self.pool_size[1], mode="trilinear"))
            fusion_out_16 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_8, scale_factor=self.pool_size[0], mode="trilinear"))
            # 筛选
            encoder_1 = rearrange(encoder_1 * fusion_out_16.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_2 = rearrange(encoder_2 * fusion_out_8.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_3 = rearrange(encoder_3 * fusion_out_4.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_4 = rearrange(encoder_4 * fusion_out_2.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")

        else :
            # 不筛选
            encoder_1 = rearrange(encoder_1 , "b n c d w h -> b (n c) d w h")
            encoder_2 = rearrange(encoder_2 , "b n c d w h -> b (n c) d w h")
            encoder_3 = rearrange(encoder_3 , "b n c d w h -> b (n c) d w h")
            encoder_4 = rearrange(encoder_4 , "b n c d w h -> b (n c) d w h")

        encoder_1_cnn = self.fusion_conv_1(encoder_1)
        encoder_2_cnn = self.fusion_conv_2(encoder_2)
        encoder_3_cnn = self.fusion_conv_3(encoder_3)
        encoder_4_cnn = self.fusion_conv_4(encoder_4)

        u4 = self.upcat_4(fusion_out, encoder_4_cnn)
        u3 = self.upcat_3(u4, encoder_3_cnn)
        u2 = self.upcat_2(u3, encoder_2_cnn)
        u1 = self.upcat_1(u2, encoder_1_cnn)

        logits = self.final_conv(u1)
        
        return logits


class SCAFNetNoCross(nn.Module):
    def __init__(self, model_num, out_channels, image_size, 
                act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm = ("GROUP", {"num_groups": 8, "affine": False}),
                dropout: Union[float, tuple] = 0.0,
                upsample: str = "deconv",
                fea = [16, 16, 32, 64, 128, 16],
                window_size=(2, 4, 4),
                pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
                patch_size=(1, 1, 1), 
                self_num_layer=2, 
                token_mixer_size=32,
                encoder="cnn",
                corss_attention=True,
                modality_gate=True):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.pool_size = pool_size
        self.cross_attention = corss_attention
        self.modality_gate = modality_gate

        pool_size_all = [1, 1, 1]
        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]

        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]
        if encoder == "cnn":
            print("use cnn encoder")
            self.multicnn_encoder = MCNNEncoder(model_num=model_num, fea=fea, pool_size=pool_size)
        elif encoder == "poolformer":
            print("use poolformer encoder")
            self.multicnn_encoder = PoolFormerEncoders(model_num=model_num, fea=fea, pool_size=pool_size)
        else :
            import os 
            print("model is error")
            os._exit(0)
        
        if self.cross_attention:

            print("use cross attention fusion module")
            self.cross_fusion_trans = MultiAttentionTransformer(
                                                            in_channels=model_num*fea[4],
                                                            out_channels=fea[4],
                                                            patch_size=(1,1,1),
                                                            img_size=new_image_size,
                                                            mlp_size=2*fea[4],
                                                            window_size=window_size,
            )
            # self.cross_fusion_trans = FusionSelfCrossTrans(model_num=model_num, 
            #                                             in_channels=fea[4], 
            #                                             hidden_size=fea[4],
            #                                             patch_size=patch_size, 
            #                                             img_size=new_image_size,
            #                                             mlp_size=2*fea[4], self_num_layer=self_num_layer,
            #                                             window_size=window_size, token_mixer_size=token_mixer_size)

        else :
            # 不用crss attention
            print("no cross attention fusion module")
        self.fusion_conv_5 = CNN(model_num*fea[4], fea[4], 3, 1)

        if self.modality_gate:
            print("use modality gate module")
            self.gate_layer = nn.Conv3d(fea[4], 2, 1, 1, 0)
            
        else :
            print("no modality gate module")
        self.fusion_conv_1 = CNN(model_num*fea[0], fea[0], 3, 1)
        self.fusion_conv_2 = CNN(model_num*fea[1], fea[1], 3, 1)
        self.fusion_conv_3 = CNN(model_num*fea[2], fea[2], 3, 1)
        self.fusion_conv_4 = CNN(model_num*fea[3], fea[3], 3, 1)
        
        
        # self.upcat_4 = PoolFormerUpcat(fea[4], fea[3], up_size=pool_size[3], num_layers=2)
        # self.upcat_3 = PoolFormerUpcat(fea[3], fea[2], up_size=pool_size[2], num_layers=2)
        # self.upcat_2 = PoolFormerUpcat(fea[2], fea[1], up_size=pool_size[1], num_layers=2)
        # self.upcat_1 = PoolFormerUpcat(fea[1], fea[5], up_size=pool_size[0], num_layers=2)
        
        self.upcat_4 = UpCat(3, fea[4], fea[3], fea[3], act, norm, dropout, upsample, pool_size=pool_size[3])
        self.upcat_3 = UpCat(3, fea[3], fea[2], fea[2], act, norm, dropout, upsample, pool_size=pool_size[2])
        self.upcat_2 = UpCat(3, fea[2], fea[1], fea[1], act, norm, dropout, upsample, pool_size=pool_size[1])
        self.upcat_1 = UpCat(3, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False, pool_size=pool_size[0])
        
        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        encoder_x = self.multicnn_encoder(x)

        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)
        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)
        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)
        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)            
        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)            

        if self.cross_attention:
            encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")
            fusion_out = self.cross_fusion_trans(encoder_5)
            fusion_out_cnn = self.fusion_conv_5(encoder_5)
            fusion_out = fusion_out + fusion_out_cnn
        else :
            # 不用cross attention
            encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")
            fusion_out = self.fusion_conv_5(encoder_5)

        if self.modality_gate:
            # 使用modality gate
            fusion_out_tmp = self.gate_layer(fusion_out)
            fusion_out_2 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_tmp, scale_factor=self.pool_size[3], mode="trilinear"))
            fusion_out_4 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_2, scale_factor=self.pool_size[2], mode="trilinear"))
            fusion_out_8 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_4, scale_factor=self.pool_size[1], mode="trilinear"))
            fusion_out_16 = torch.sigmoid(torch.nn.functional.interpolate(fusion_out_8, scale_factor=self.pool_size[0], mode="trilinear"))
            # 筛选
            encoder_1 = rearrange(encoder_1 * fusion_out_16.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_2 = rearrange(encoder_2 * fusion_out_8.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_3 = rearrange(encoder_3 * fusion_out_4.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")
            encoder_4 = rearrange(encoder_4 * fusion_out_2.unsqueeze(dim=2), "b n c d w h -> b (n c) d w h")

        else :
            # 不筛选
            encoder_1 = rearrange(encoder_1 , "b n c d w h -> b (n c) d w h")
            encoder_2 = rearrange(encoder_2 , "b n c d w h -> b (n c) d w h")
            encoder_3 = rearrange(encoder_3 , "b n c d w h -> b (n c) d w h")
            encoder_4 = rearrange(encoder_4 , "b n c d w h -> b (n c) d w h")

        encoder_1_cnn = self.fusion_conv_1(encoder_1)
        encoder_2_cnn = self.fusion_conv_2(encoder_2)
        encoder_3_cnn = self.fusion_conv_3(encoder_3)
        encoder_4_cnn = self.fusion_conv_4(encoder_4)

        u4 = self.upcat_4(fusion_out, encoder_4_cnn)
        u3 = self.upcat_3(u4, encoder_3_cnn)
        u2 = self.upcat_2(u3, encoder_2_cnn)
        u1 = self.upcat_1(u2, encoder_1_cnn)

        logits = self.final_conv(u1)
        
        return logits


        
        




            
