
import torch
import torch.nn as nn
import einops


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)
        y = y.permute(0, 2, 1)
        y = self.mlp_token(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.layer_norm_2(x)
        return x + self.mlp_channel(y)

class MlpMixerBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_token_dim, mlp_channel_dim, patch_size, img_size, num_block):
        super().__init__()
        self.token_dim = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv_embedding = nn.Conv2d(in_dim, hidden_dim, stride=patch_size, kernel_size=patch_size, padding=0)
        self.num_block = num_block
        self.blocks = nn.ModuleList([MixerBlock(hidden_dim, self.token_dim, mlp_token_dim, mlp_channel_dim) for _ in range(num_block)])
        self.head_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.conv_embedding(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        for l in self.blocks:
            x = l(x)

        x = self.head_layer_norm(x)

        return x

class MlpMixerBlock3d(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_token_dim, mlp_channel_dim, patch_size, img_size, num_block):
        super().__init__()
        self.token_dim = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.conv_embedding = nn.Conv3d(in_dim, hidden_dim, stride=patch_size, kernel_size=patch_size, padding=0)
        self.num_block = num_block
        self.blocks = nn.ModuleList([MixerBlock(hidden_dim, self.token_dim, mlp_token_dim, mlp_channel_dim) for _ in range(num_block)])
        self.head_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch, c, h, w, d = x.shape
        x = self.conv_embedding(x)
        x = einops.rearrange(x, 'n c h w d -> n (h w d) c')
        for l in self.blocks:
            x = l(x)

        x = self.head_layer_norm(x)
        x = x.view(1, -1, h, w, d)
        return x

class MlpMixer(nn.Module):

    def __init__(self, in_dim, hidden_dim, mlp_token_dim, mlp_channel_dim, patch_size, img_size, num_block, num_class):
        super().__init__()
        self.token_dim = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv_embedding = nn.Conv2d(in_dim, hidden_dim, stride=patch_size, kernel_size=patch_size, padding=0)
        self.num_block = num_block
        self.blocks = nn.ModuleList([MixerBlock(hidden_dim, self.token_dim, mlp_token_dim, mlp_channel_dim) for _ in range(num_block)])
        self.head_layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        x = self.conv_embedding(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        for l in self.blocks:
            x = l(x)

        x = self.head_layer_norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x


if __name__ == '__main__':
    t1 = torch.rand(1, 3, 128, 128)

    net = MlpMixer(3, 32, 32, 32, (16, 16), (128, 128), 2, 6)

    out = net(t1)


    print(out.shape)