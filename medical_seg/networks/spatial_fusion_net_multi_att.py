
from operator import mod
from medical_seg.networks.nets.co_unet import BasicUNet
from medical_seg.networks.nets.basic_unet_encoder import BasicUNetEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
# from medical_seg.networks.layers.spatial_image_transformer import SpatialTransformerLayer
from medical_seg.networks.layers.multi_attention import SpatialTransformerLayer
from typing import Sequence, Union
from medical_seg.networks.nets.co_unet import UpCat
from medical_seg.networks.layers.cpc import ImageCPC
from einops import rearrange

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
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
        self.out_conv = nn.Conv3d(model_num*hidden_size, hidden_size, 1, stride=1, padding=0)
    
    def forward(self, x, model_num):
        ## x: (batch, modal, feature, d, w, h)
        x = rearrange(x, "b (m f) d w h -> b d w h m f", m=model_num)
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        
        attention_score = torch.einsum("b d w h m f, b d w h f n -> b d w h m n", q_out, k_out.transpose(-1, -2))
        modality_att_out = torch.einsum("b d w h m n, b d w h n f -> b d w h m f", attention_score, v_out)
        modality_att_out = rearrange(modality_att_out, "b d w h m f -> b (m f) d w h")
        modality_att_out = self.out_conv(modality_att_out)
        return modality_att_out

class SpatialTransNetV3(nn.Module):
    def __init__(self, model_num, out_channels=1, image_size=(8, 8, 8), 
                act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm: Union[str, tuple] = ("instance", {"affine": True}),
                dropout: Union[float, tuple] = 0.0,
                upsample: str = "deconv",
                fea = [16, 16, 32, 64, 128, 16],
                cpc_layer_num=1,
                window_size=(4, 4, 4),
                pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],**kwargs):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.unets = nn.ModuleList([])
        self.loss = nn.CrossEntropyLoss()
        self.cpc_layers = nn.ModuleList([])
        self.pool_size = pool_size

        self.modality_attention = ModalitySelfAttention(model_num=model_num, hidden_size=128)
        self.modality_attention_out = nn.Conv3d(128, model_num, 1, 1, 0)# 模态特征筛选。
        self.multi_att = SpatialTransformerLayer(in_channels=128, out_channels=128, 
                                                    patch_size=(1, 1, 1), 
                                                    img_size=image_size, 
                                                    num_heads=8, 
                                                    mlp_size=128,
                                                    num_layers=2,
                                                    window_size=window_size)

        self.cnn_fusion_layer = CNN(128*model_num, 128)

        for i in range(model_num):
            unet = BasicUNetEncoder(dimensions=3, in_channels=1,
                                         features=fea, pool_size=pool_size)
            self.unets.append(unet)

            cpc_layer = ImageCPC(x_size=128, y_size=128, n_layers=cpc_layer_num)
            self.cpc_layers.append(cpc_layer)

        self.modality_fusion_layer = nn.Conv3d(fea[4]*3, fea[4], kernel_size=1, stride=1)
        self.modality_fusion_ins = nn.BatchNorm3d(128)

        fusion_conv_1 = nn.Conv3d(model_num*fea[0], fea[0], 1, 1)
        fusion_conv_2 = nn.Conv3d(model_num*fea[1], fea[1], 1, 1)
        fusion_conv_3 = nn.Conv3d(model_num*fea[2], fea[2], 1, 1)
        fusion_conv_4 = nn.Conv3d(model_num*fea[3], fea[3], 1, 1)
        self.fusion_list = nn.ModuleList([fusion_conv_1, fusion_conv_2, fusion_conv_3, fusion_conv_4])

        self.upcat_4 = UpCat(3, fea[4], fea[3], fea[3], act, norm, dropout, upsample, pool_size=pool_size[3])
        self.upcat_3 = UpCat(3, fea[3], fea[2], fea[2], act, norm, dropout, upsample, pool_size=pool_size[2])
        self.upcat_2 = UpCat(3, fea[2], fea[1], fea[1], act, norm, dropout, upsample, pool_size=pool_size[1])
        self.upcat_1 = UpCat(3, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False, pool_size=pool_size[0])
        
        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

    def image_flatten(self, x:torch.Tensor):
        x = x.flatten(dim=2)
        x = x.transpose(-1, -2)
        return x 
    
    def recons_image(self, x, size):
        batch_size = x.shape[0]
        ## x: (1, seq_len, features)
        ## size: (1, features, d, w, h)
        x = x.transpose(-1, -2)
        out_size = (size[0],
                    size[1],
                    size[2])

        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))

    def forward(self, x, is_cpc=False):
        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        x_down = []
        x_fusion = []
        for i in range(self.model_num):
            x_i = x[:, i].unsqueeze(dim=1)
            x4, x3, x2, x1, x0 = self.unets[i](x_i)
            x_fusion.append(x4)
            x_down.append([x3, x2, x1, x0])
        ## fusion
        cat_feature = torch.cat(x_fusion, dim=1)

        ### cnn的输出。
        cnn_out = self.cnn_fusion_layer(cat_feature)

        modality_out = self.modality_attention(cat_feature, model_num=self.model_num)
        multi_att_out = self.multi_att(modality_out)

        fusion_out = multi_att_out + cnn_out

        ## 融合特征对原特征做cpc score
        cpc_loss = 0.0
        for i in range(self.model_num):
            cpc_loss += self.cpc_layers[i](x_fusion[i], fusion_out)
        

        # 拿到modality attention out 在上采样的过程中对cat feature进行 乘法， 做模态特征的筛选。
        modality_up_out = self.modality_attention_out(fusion_out)
        modality_up_out = torch.softmax(modality_up_out/2.0, dim=1)
        logits = []
        x_up = []   
        for i in range(4):
            down_feaure = [x_down[j][i] for j in range(self.model_num)]
            down_feaure = torch.stack(down_feaure, dim=1)
            modality_up_out = torch.nn.functional.interpolate(modality_up_out, scale_factor=self.pool_size[3 - i], mode="trilinear")
            tmp = modality_up_out.unsqueeze(dim=2)

            down_feaure = rearrange(down_feaure + (down_feaure * tmp), "b m f d w h -> b (m f) d w h")
            down_feaure = self.fusion_list[3 - i](down_feaure)

            x_up.append(down_feaure)
        
        u4 = self.upcat_4(fusion_out, x_up[0])
        u3 = self.upcat_3(u4, x_up[1])
        u2 = self.upcat_2(u3, x_up[2])
        u1 = self.upcat_1(u2, x_up[3])

        logits = self.final_conv(u1)
        if is_cpc is False:
            return logits
        return logits, cpc_loss


        
        




            
