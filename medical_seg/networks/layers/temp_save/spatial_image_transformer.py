
import numpy as np
import math
from torch import nn, einsum
from einops import rearrange, repeat
import torch
import ml_collections

def get_config(in_channels=1, hidden_size=128, img_size=(1, 1, 1), patch_size=(1, 1, 1),  mlp_dim=256, num_heads=8, window_size=(8, 8, 8)):
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
    config.window_size = window_size

    return config

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if attention:
            return attention_output, attention_probs
        return attention_output

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            min_indice = self.relative_indices.min()
            self.relative_indices += (-min_indice)
            max_indice = self.relative_indices.max().item()
            self.pos_embedding = nn.Parameter(torch.randn(max_indice + 1, max_indice + 1, max_indice + 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
      
        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]
        nw_d = n_d // self.window_size[2]

        ## h 为注意力头的个数 nw_h 为h（长）的维度上窗口个数 wh为窗口的长  nw_w同理
        ## 如何去进行窗口内部的attention计算呢，其实就是设置成这个shape (b, 注意力头个数，窗口个数，窗口面积，hidden size)
        ## 这样就做到了在窗口面积内进行attention计算。
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        # 注意力结果为 （b，注意力头个数， 窗口个数， 窗口长度，窗口宽度） 所以注意力表示的意思呢 就是每个窗口内互相的注意力大小
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1], self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embedding

        # if self.shifted:
        #     dots[:, :, -nw_w:] += self.upper_lower_mask
        #     dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d = self.window_size[2], nw_h=nw_h, nw_w=nw_w, nw_d=nw_d)
        out = self.to_out(out)

        return out

class MultiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.v_attention = Attention(config)
        self.h_attention = Attention(config)
        self.window_attention = WindowAttention(config.hidden_size, 
                                                config.num_heads, config.hidden_size // config.num_heads, 
                                                config.window_size, relative_pos_embedding=True)
    
    def forward(self, x):
        batch_size, hidden_size, D, W, H = x.shape
        # z direction 
        x_1 = x.view((-1, hidden_size, W, H)).contiguous()
        # x_1 = x_1.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        # elif self.types == 2:
        #     x = x.view((-1, self.config.hidden_size, out_size[0])).contiguous()
        #     x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, types=0):
        super(Embeddings, self).__init__()
        self.types = types
        self.config = config
        img_size = config.img_size
        in_channels = config.in_channels
        patch_size = config["patch_size"]
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        if types == 0:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        elif types == 1:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), config.hidden_size))
        elif types == 2:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]), config.hidden_size))

            
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        b, hidden, zz, xx, yy = x.shape
        if self.types == 0:
           pass 
        elif self.types == 1:
            ## z directions
            x = x.view(b*zz, hidden, xx, yy)
        elif self.types == 2:
            x = x.view(xx*yy*b, hidden, zz)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)


        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
    
        return embeddings

class BlockMulti(nn.Module):
    def __init__(self, config):
        super(BlockMulti, self).__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, attention=False):

        h = x
        x = self.attention_norm(x)
        if attention:
            x, att = self.attn(x, attention)
        else :
            x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        if attention:
            return x, att
        return x

# class GlobalAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads=1, mlp_size=256, num_layers=1, types=0):
#         super().__init__()
#         self.config = get_config(in_channels=in_channels, hidden_size=out_channels,
#                                  patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size, num_heads=num_heads)
#         self.attention = Attention(self.config)

#         self.embeddings = Embeddings(self.config, types=types)
#         self.types = types

#     def forward(self, x):
#         input_shape = self.config.img_size
#         batch_size = x.shape[0]
#         x = self.embeddings(x)
#         return x 

class SpatialTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads=8, mlp_size=256, num_layers=1, types=0):
        super().__init__()
        self.config = get_config(in_channels=in_channels, hidden_size=out_channels,
                                 patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size, num_heads=num_heads)
        self.block_list = nn.ModuleList([BlockMulti(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config, types=types)
        self.types = types

    def forward(self, x, attention=False):
        input_shape = self.config.img_size
        batch_size = x.shape[0]
        x = self.embeddings(x)

        for l in self.block_list:
            if attention:
                x, att = l(x, attention=attention)
            else :
                x = l(x)
        
        x = x.transpose(-1, -2)
        out_size = (input_shape[0] // self.config.patch_size[0],
                    input_shape[1] // self.config.patch_size[1],
                    input_shape[2] // self.config.patch_size[2],)
        if self.types == 0:
            x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        elif self.types == 1:
            # z direction 
            x = x.view((-1, self.config.hidden_size, out_size[1], out_size[2])).contiguous()
            x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        elif self.types == 2:
            x = x.view((-1, self.config.hidden_size, out_size[0])).contiguous()
            x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        
        if attention:
            return x, att 
        return x


if __name__ == '__main__':

    # t1 = torch.rand(1, 2, 32, 128, 128)
    # in_channels = 2
    # out_channels = 3
    # img_size = (32, 128, 128)
    # patch_size = (16, 16, 16)
    # num_layer = 1
    # mlp_size = 32
    # hidden_size = 128
    # conv3d = nn.Conv3d(3, 6, kernel_size=(8, 32, 32), stride=(8, 32, 32))
    #
    # out = conv3d(t1)
    #
    # print(out.shape)
    #
    # out = out.flatten(2)
    # print(out.shape)
    # config = get_config()
    # b = TransformerLayer(in_channels=3, out_channels=64, patch_size=(16, 16, 16), img_size=(32, 128, 128))

    # out = b(t1)
    # print(out.shape)

    # b = TransformerLayerMulti(in_channels=2, out_channels=2, patch_size=(16, 16, 16), img_size=(32, 128, 128), num_layers=2)
    #
    # print(b(t1).shape)

    # config = get_config(in_channels=in_channels, out_channels=hidden_size,
    #                              patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
    # cross_att = BlockMultiCrossModal(config)
    #
    # t1 = torch.rand(1, 10, 128)
    # t2 = torch.rand(1, 10, 128)
    #
    # out = cross_att(t1, t2)
    #
    # print(out.shape)

    # b = TransformerLayerMultiCrossModal(in_channels=2, out_channels=2, patch_size=(4, 16, 16), img_size=(32, 128, 128),
    #                           num_layers=2)
    # t2 = torch.rand(1, 2,  32, 128, 128)
    #
    # out = b(t1, t2)
    #
    # print(out.shape)

    # net = DoubleRouteTransformer(in_channels=2, out_channels=2, patch_size=(4, 16, 16), img_size=(32, 128, 128), num_layer=2)
    # out = net(t1, t2)

    # print(out.shape)

    t1 = torch.rand(1, 16, 16, 16, 128)

    # net = SpatialTransformerLayer(2, 64, (1, 1, 1), (16, 16, 16), mlp_size=128, num_layers=2, types=2)


    # out = net(t1)
    # print(out.shape)
    net = WindowAttention(dim=128, heads=16, head_dim=8, window_size=(4, 4, 4), relative_pos_embedding=True)
    out = net(t1)

    print(out.shape)