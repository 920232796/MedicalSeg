
from operator import mod
from matplotlib.pyplot import sca

from numpy.lib.npyio import savez_compressed
from torch.utils.data.dataset import T
from medical_seg.networks.nets.co_unet import BasicUNet
from medical_seg.networks.nets.basic_unet_encoder import BasicUNetEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
# from medical_seg.networks.layers.spatial_image_transformer import SpatialTransformerLayer
from medical_seg.networks.layers.multi_attention import MultiAttentionTransformer
from typing import Sequence, Union
from medical_seg.networks.nets.co_unet import UpCat
from medical_seg.networks.layers.cpc import ImageCPC
from einops import rearrange
from medical_seg.networks.nets.vit import ViT


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm3d(num_features=out_channels), 
        )
    def forward(self, x):
        return self.net(x)


class ModalitySelfAttention(nn.Module):
    def __init__(self, model_num, hidden_size, out_size):
        super().__init__()
        self.model_num = model_num
        # 降维
        self.q = nn.Linear(in_features=hidden_size, out_features=1)
        self.k = nn.Linear(in_features=hidden_size, out_features=1)
        self.v = nn.Linear(in_features=hidden_size, out_features=1)
        # self.out_conv = nn.Conv3d(model_num*hidden_size, out_size, 1, stride=1, padding=0)
        self.out_conv = CNN(in_channels=model_num * hidden_size, out_channels=out_size)

    def forward(self, x):
        ## x: (batch, modal, feature, d, w, h)
        h = rearrange(x, "b (m f) d w h -> b m f d w h", m=self.model_num)
        x = rearrange(x, "b (m f) d w h -> b d w h m f", m=self.model_num)
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        
        attention_score = torch.einsum("b d w h m f, b d w h f n -> b d w h m n", q_out, k_out.transpose(-1, -2))
        modality_att_out = torch.einsum("b d w h m n, b d w h n f -> b d w h m f", attention_score, v_out)
        modality_att_out = rearrange(modality_att_out, "b d w h m f -> b m f d w h")
        # modality_att_out = self.out_conv(modality_att_out)
        # modality_att_out = torch.squeeze(dim=2)
        modality_att_out = torch.sigmoid(modality_att_out)
        h = h * modality_att_out
        h = rearrange(h, "b m f d w h -> b (m f) d w h")
        cnn_out = self.out_conv(h)

        return cnn_out

class MultiAttentionEncoder(nn.Module):
    def __init__(self, model_num, img_size, patch_size, 
                    hidden_size=128, mlp_size=256, num_layers=9, window_size=(8, 8, 8)):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.model_num = model_num

        # self.vit = ViT(
        #     in_channels=1,
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=3072,
        #     num_layers=self.num_layers,
        #     num_heads=12,
        #     pos_embed="conv",
        #     classification=self.classification,
        #     dropout_rate=0.0,
        #     spatial_dims=3,
        #     num_layers=9
        # )

        for i in range(model_num):
            self.encoder.append(MultiAttentionTransformer(in_channels=1, out_channels=hidden_size,
                                                            patch_size=patch_size, img_size=img_size, 
                                                            num_layers=num_layers, window_size=window_size, out_hidden=True,
                                                            mlp_size=mlp_size))
            self.encoder.append(ViT(
                                in_channels=1,
                                img_size=img_size,
                                patch_size=patch_size,
                                hidden_size=hidden_size,
                                mlp_dim=3072,
                                num_heads=12,
                                pos_embed="conv",
                                classification=False,
                                dropout_rate=0.0,
                                spatial_dims=3,
                                num_layers=9
                            ))

   


    def forward(self, x):
        pass
        out = [None] * self.model_num
        x = x.unsqueeze(dim=2)
        for i in range(self.model_num):
            _, out[i] = self.encoder[i](x[:, i])

        return out

class FusionUpsample(nn.Module):
    def __init__(self, model_num, hidden_size, out_size, scale_factor=(2, 2, 2)):
        super().__init__()

        self.fusion_layer = CNN(model_num*hidden_size, out_channels=out_size, kernel_size=1, stride=1, padding=0)
        # self.fusion_layer = ModalitySelfAttention(model_num, hidden_size, out_size)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    
    def forward(self, x):
        
        x = self.fusion_layer(x)
        x = self.upsample(x)
        return x 

# class UpsampleCNN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")
#         self.cnn_layer = CNN(in_channels=in_channels, out_channels=out_channels)

#     def forward(self, x):
#         pass

class MultiAttentionNet(nn.Module):
    def __init__(self, model_num, out_channels, image_size, patch_size,
                act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm: Union[str, tuple] = ("instance", {"affine": True}),
                dropout: Union[float, tuple] = 0.0,
                upsample: str = "nontrainable",
                fea = [32, 64, 128, 768, 16],
                window_size=(4, 4, 4),
                mlp_head=256,
               ):

        super().__init__()

        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(image_size, patch_size))
        self.out_channels = out_channels
        self.model_num = model_num
        ## 对原始图像进行conv提取特征。
        self.conv_init = CNN(in_channels=model_num, out_channels=fea[4])

        self.multi_att_encoder = MultiAttentionEncoder(model_num=model_num, hidden_size=fea[3], 
                                                    patch_size=patch_size, 
                                                    img_size=image_size, 
                                                    mlp_size=mlp_head,
                                                    window_size=window_size)

        
        # fusion_modality_1 = ModalitySelfAttention(model_num=model_num, hidden_size=fea[3], out_size=fea[1])
        # fusion_modality_2 = ModalitySelfAttention(model_num=model_num, hidden_size=fea[3], out_size=fea[2])
        # self.fusion_modality_3 = ModalitySelfAttention(model_num=model_num, hidden_size=fea[3], out_size=fea[3])
        
        # fusion_modality_1 = CNN(in_channels=model_num*fea[3], out_channels=fea[1])
        # fusion_modality_2 = CNN(in_channels=model_num*fea[3], out_channels=fea[2])
        self.fusion_modality_3 = CNN(in_channels=model_num*fea[3], out_channels=fea[3])

        # self.fusion_list = nn.ModuleList([fusion_modality_1, fusion_modality_2, fusion_modality_3])

        self.upcat_4 = UpCat(3, fea[3], fea[2], fea[2], act, norm, dropout, upsample, pool_size=(2, 2, 2))
        self.upcat_3 = UpCat(3, fea[2], fea[1], fea[1], act, norm, dropout, upsample, pool_size=(2, 2, 2))
        self.upcat_2 = UpCat(3, fea[1], fea[4], fea[4], act, norm, dropout, upsample, pool_size=(2, 2, 2))
        
        # self.upsample_4 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")
        # self.upsample_8 = nn.Upsample(scale_factor=(4, 4, 4), mode="trilinear")
        self.upsample_4 = FusionUpsample(model_num=model_num, hidden_size=fea[3], out_size=fea[2], scale_factor=(2, 2, 2))
        self.upsample_8 = FusionUpsample(model_num=model_num, hidden_size=fea[3], out_size=fea[1], scale_factor=(4, 4, 4))
        
        self.final_conv = nn.Conv3d(fea[4], out_channels, 1, 1)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x):
        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        x_in = self.conv_init(x)
        multi_hidden_states = self.multi_att_encoder(x) ## [hidden1, hidden2]
        ## fusion
        down_features_3 = []
        down_features_6 = []
        down_features_9 = []
        for i in range(self.model_num):
            down_features_3.append(self.proj_feat(multi_hidden_states[i][2], hidden_size=768, feat_size=self.feat_size))
            down_features_6.append(self.proj_feat(multi_hidden_states[i][5], hidden_size=768, feat_size=self.feat_size))
            down_features_9.append(self.proj_feat(multi_hidden_states[i][8], hidden_size=768, feat_size=self.feat_size))

        down_features_3 = torch.cat(down_features_3, dim=1)
        down_features_6 = torch.cat(down_features_6, dim=1)
        down_features_9 = torch.cat(down_features_9, dim=1)

        x = self.fusion_modality_3(down_features_9)
        # down_features_6 = self.fusion_list[1](down_features_6)
        # down_features_3 = self.fusion_list[0](down_features_3)

        # down_features_6 = self.upsample_4(down_features_6)
        # down_features_3 = self.upsample_8(down_features_3)

        down_features_6 = self.upsample_4(down_features_6)
        down_features_3 = self.upsample_8(down_features_3)

        # print(f"x shape is {x.shape}")
        # print(f"6 shape is {down_features_6.shape}")
        # print(f"3 shape is {down_features_3.shape}")
        
        u4 = self.upcat_4(x, down_features_6)
        u3 = self.upcat_3(u4, down_features_3)
        u2 = self.upcat_2(u3, x_in)

        logits = self.final_conv(u2)

        # print(logits)
        return logits


        
        




            
