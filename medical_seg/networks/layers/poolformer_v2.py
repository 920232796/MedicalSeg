
import numpy as np
import math
import torch.nn as nn
import torch
import ml_collections
from einops import rearrange

def get_config(in_channels=1, hidden_size=128, img_size=(1, 1, 1), patch_size=(1, 1, 1),  mlp_dim=256, num_heads=8):
    config = ml_collections.ConfigDict()
    config.hidden_size = hidden_size
    config.in_channels = in_channels
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = mlp_dim
    config.transformer.num_heads = num_heads
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = patch_size
    config.img_size = img_size

    return config

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Conv3d(config.hidden_size, config.transformer.mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(config.transformer.mlp_dim, config.hidden_size, 1)
        self.drop = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        in_channels = config.in_channels
        patch_size = config["patch_size"]

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        # self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = self.norm(x)

        return x


class GlobalPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        all_size = self.img_size[0] * self.img_size[1] * self.img_size[2]
        self.global_layer = nn.Linear(1, all_size)

    def forward(self, x):
        # x: (batch, c, d, w, h)
        x = rearrange(x, "b c d w h -> b c (d w h)")

        x = x.mean(dim=-1, keepdims=True)

        x = self.global_layer(x)

        x = rearrange(x, "b c (d w h) -> b c d w h", d=self.img_size[0], w=self.img_size[1], h=self.img_size[2])
        return x 

        

class BlockPool(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.mixer_token = nn.AvgPool3d(3, 1, padding=1)
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        # self.attn = nn.AvgPool3d(3, 1, padding=1)
        self.attn = GlobalPool(config)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x) + x
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class PoolFormerV2(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, mlp_size=256, num_layers=1):
        super().__init__()
        # global_pool_size = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
        # for index, (i, p) in enumerate(zip(img_size, patch_size)):
        #         global_pool_size[index][0] = i[0] // p[0]
        #         global_pool_size[index][1] = i[1] // p[1]
        #         global_pool_size[index][2] = i[2] // p[2]

        self.config = get_config(in_channels=in_channels, hidden_size=out_channels,
                                 patch_size=patch_size, mlp_dim=mlp_size, img_size=img_size)
        
        
        self.block_list = nn.ModuleList([BlockPool(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config)

    def forward(self, x, out_hidden=False):
        pass
        
        x = self.embeddings(x)
        hidden_state = []
        for l in self.block_list:
            x = l(x)
            hidden_state.append(x)
        if out_hidden:

            return x, hidden_state
        return x 




if __name__ == '__main__':
   pass 

