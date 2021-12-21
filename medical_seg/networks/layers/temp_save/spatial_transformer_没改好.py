
# ## 空间transformer 改进attention 模块 减少运算量
# import numpy as np
# import math
# from torch import nn, einsum
# from einops import rearrange, repeat
# import torch
# import ml_collections
# from typing import Union
# from einops import rearrange
# from medical_seg.networks.nets.co_unet import UpCat


# def get_config(in_channels=1, hidden_size=128, img_size=(1, 1, 1), patch_size=(1, 1, 1),  mlp_dim=256, num_heads=8, window_size=(8, 8, 8)):
#     config = ml_collections.ConfigDict()
#     config.hidden_size = hidden_size
#     config.in_channels = in_channels
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = mlp_dim
#     config.transformer.num_heads = num_heads
#     config.transformer.num_layers = 1
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.patch_size = patch_size
#     config.img_size = img_size
#     config.window_size = window_size
#     config.num_heads = num_heads
#     config.window_size = window_size

#     return config

# def swish(x):
#     return x * torch.sigmoid(x)

# ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# class Attention(nn.Module):
#     def __init__(self, config):
#         super(Attention, self).__init__()
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.out = nn.Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = nn.Softmax(dim=-1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states, attention=False):
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         if attention:
#             return attention_output, attention_probs
#         return attention_output


# class MultiAttention(nn.Module):
#     def __init__(self, config, is_position=False):
#         super().__init__()
#         self.config = config
#         self.is_position = is_position

#         self.v_attention = Attention(config)
#         self.h_attention = Attention(config)
    
#         if is_position:
#             self.pos_embedding_1 = Postion_embedding(config, types=1)
#             self.pos_embedding_2 = Postion_embedding(config, types=2)

#     def forward(self, x):
#         batch_size, hidden_size, D, W, H = x.shape

#         x_1 = rearrange(x, "b c d w h -> (b d) (w h) c")
#         x_2 = rearrange(x, "b c d w h -> (b w h) d c")

#         if self.is_position: 
#             x_1 = self.pos_embedding_1(x_1)
#             x_2 = self.pos_embedding_2(x_2)

#         x_1 = self.v_attention(x_1)
#         x_2 = self.h_attention(x_2)

#         x_1 = rearrange(x_1, "(b d) (w h) c -> b (d w h) c", d=D, w=W, h=H)

#         x_2 = rearrange(x_2, "(b w h) d c -> b (d w h) c", d=D, w=W, h=H)

#         return x_1 + x_2

# class Mlp(nn.Module):
#     def __init__(self, config):
#         super(Mlp, self).__init__()
#         self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
#         self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = nn.Dropout(config.transformer["dropout_rate"])

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self, config, types=0, is_window=False):
#         super(Embeddings, self).__init__()
#         self.is_window = is_window
#         self.types = types
#         self.config = config
#         img_size = config.img_size
#         in_channels = config.in_channels
#         patch_size = config["patch_size"]

#         self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
#                                        out_channels=config.hidden_size,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
       
#         self.dropout = nn.Dropout(config.transformer["dropout_rate"])

#     def forward(self, x):
        
#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
    
#         return x

# class Postion_embedding(nn.Module):
#     def __init__(self, config, types=0):
#         super().__init__()
#         img_size = config.img_size
#         patch_size = config.patch_size
        
#         if types == 0:
#             self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), config.hidden_size))
#         elif types == 1:
#             self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), config.hidden_size))
#         elif types == 2:
#             self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]), config.hidden_size))
    
#     def forward(self, x):
        
#         return x + self.position_embeddings


# class BlockMulti(nn.Module):
#     def __init__(self, config, is_position=False):
#         super(BlockMulti, self).__init__()
#         self.config = config
#         self.input_shape = config.img_size
#         self.hidden_size = config.hidden_size
#         self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = MultiAttention(config, is_position=is_position)

#     def forward(self, x):
#         # print(x.shape)
#         batch_size, hidden_size, D, W, H = x.shape
#         x = rearrange(x, "b c d w h -> b (d w h) c")
#         # x = x.view(batch_size, D*W*H, hidden_size)
#         h = x
#         x = self.attention_norm(x)

#         x = rearrange(x, "b (d w h) c -> b c d w h", d=D, w=W, h=H)

#         x = self.attn(x)

#         x = x + h

#         h = x

#         x = self.ffn_norm(x)
        
#         x = self.ffn(x)
        
#         x = x + h

#         x = x.transpose(-1, -2)
#         out_size = (self.input_shape[0] // self.config.patch_size[0],
#                     self.input_shape[1] // self.config.patch_size[1],
#                     self.input_shape[2] // self.config.patch_size[2],)
#         x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2])).contiguous()

#         return x

# class VHTransformer(nn.Module):
#     def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads=8, mlp_size=256, num_layers=1):
#         super().__init__()
#         self.config = get_config(in_channels=in_channels, hidden_size=out_channels,
#                                  patch_size=patch_size, img_size=img_size, 
#                                  mlp_dim=mlp_size, num_heads=num_heads)
#         self.block_list = nn.ModuleList([BlockMulti(self.config, is_position=True) if i == 0 
#                                         else BlockMulti(self.config) for i in range(num_layers)])

#         self.embeddings = Embeddings(self.config)

#     def forward(self, x):
#         x = self.embeddings(x)

#         for l in self.block_list:
           
#             x = l(x)
        
#         return x

# class MultiAttentionCross(nn.Module):
#     ## VH attention cross modality
#     def __init__(self, config, is_position=False):
#         super().__init__()
#         self.config = config
#         self.is_position = is_position

#         self.v_attention = AttentionCrossModal(config)
#         self.h_attention = AttentionCrossModal(config)
    
#         if is_position:
#             self.pos_embedding_1 = Postion_embedding(config, types=1)
#             self.pos_embedding_2 = Postion_embedding(config, types=2)

#     def forward(self, q, kv):
#         ## q (batch, c, d, w, h) # 做完了self attention
#         ## kv: 
#         batch_size, hidden_size, D, W, H = q.shape

        
#         x_1 = rearrange(x, "b c d w h -> (b d) (w h) c")
#         x_2 = rearrange(x, "b c d w h -> (b w h) d c")

#         if self.is_position: 
#             x_1 = self.pos_embedding_1(x_1)
#             x_2 = self.pos_embedding_2(x_2)

#         x_1 = self.v_attention(x_1)
#         x_2 = self.h_attention(x_2)

#         x_1 = rearrange(x_1, "(b d) (w h) c -> b (d w h) c", d=D, w=W, h=H)

#         x_2 = rearrange(x_2, "(b w h) d c -> b (d w h) c", d=D, w=W, h=H)

#         return x_1 + x_2

# class EmbeddingsCross(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self, config):
#         super(EmbeddingsCross, self).__init__()
#         self.config = config
#         img_size = config.img_size
#         in_channels = config.in_channels
#         patch_size = config["patch_size"]
#         n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

#         self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
#                                        out_channels=config.hidden_size,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

#         self.dropout = nn.Dropout(config.transformer["dropout_rate"])

#     def forward(self, x):

#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

#         x = x.flatten(2)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#         return embeddings

# class AttentionCrossModal(nn.Module):
#     def __init__(self, config):
#         super(AttentionCrossModal, self).__init__()
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.out = nn.Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = nn.Softmax(dim=-1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states, kv):
#         ## kv 是别的模态的特征。
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(kv)
#         mixed_value_layer = self.value(kv)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         return attention_output


# class BlockCrossAtt(nn.Module):
#     ## 跨模态的attention
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
#         self.attention_norm_cross = nn.LayerNorm(self.hidden_size, eps=1e-6)
#         self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn_cross = MultiAttentionCross(config)
       

#     def forward(self, q, kv):

#         ##q (batch, c, d, w, h) # 做完了self attention
#         ## kv (batch, n_patch*n, hidden)
#         # q是其他模态特征。
#         h = q
     
#         x = self.attn_cross(q, kv)
#         x = x + h
#         x = self.attention_norm_cross(x)

#         h = x
#         x = self.ffn(x)
#         x = x + h
#         x = self.ffn_norm(x)

#         return x


# class CrossVHTransLayer(nn.Module):
#     def __init__(self, model_num, in_channels, hidden_size, patch_size, img_size, mlp_size=256):
#         super().__init__()
#         self.embeddings = nn.ModuleList([])
#         self.config = get_config(in_channels=in_channels, hidden_size=hidden_size, patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
#         self.model_num = model_num

#         for i in range(model_num):
#             self.embeddings.append(EmbeddingsCross(self.config))
        
#         self.cross_attention = BlockCrossAtt(config=self.config)
        
#     def forward(self, q, kv):
#         pass
#         embed_x = []
#         for i in range(self.model_num):
#             embed_x.append(self.embeddings[i](kv[:, i]))
        
#         embed_x = torch.cat(embed_x, dim=1) ## embed_x : (batch, n_patch * n, hidden_size)

#         corss_out = self.cross_attention(q, embed_x)

#         return corss_out

# if __name__ == '__main__':

#     # t1 = torch.rand(1, 2, 32, 128, 128)
#     # in_channels = 2
#     # out_channels = 3
#     # img_size = (32, 128, 128)
#     # patch_size = (16, 16, 16)
#     # num_layer = 1
#     # mlp_size = 32
#     # hidden_size = 128
#     # conv3d = nn.Conv3d(3, 6, kernel_size=(8, 32, 32), stride=(8, 32, 32))
#     #
#     # out = conv3d(t1)
#     #
#     # print(out.shape)
#     #
#     # out = out.flatten(2)
#     # print(out.shape)
#     # config = get_config()
#     # b = TransformerLayer(in_channels=3, out_channels=64, patch_size=(16, 16, 16), img_size=(32, 128, 128))

#     # out = b(t1)
#     # print(out.shape)

#     # b = TransformerLayerMulti(in_channels=2, out_channels=2, patch_size=(16, 16, 16), img_size=(32, 128, 128), num_layers=2)
#     #
#     # print(b(t1).shape)

#     # config = get_config(in_channels=in_channels, out_channels=hidden_size,
#     #                              patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
#     # cross_att = BlockMultiCrossModal(config)
#     #
#     # t1 = torch.rand(1, 10, 128)
#     # t2 = torch.rand(1, 10, 128)
#     #
#     # out = cross_att(t1, t2)
#     #
#     # print(out.shape)

#     # b = TransformerLayerMultiCrossModal(in_channels=2, out_channels=2, patch_size=(4, 16, 16), img_size=(32, 128, 128),
#     #                           num_layers=2)
#     # t2 = torch.rand(1, 2,  32, 128, 128)
#     #
#     # out = b(t1, t2)
#     #
#     # print(out.shape)

#     # net = DoubleRouteTransformer(in_channels=2, out_channels=2, patch_size=(4, 16, 16), img_size=(32, 128, 128), num_layer=2)
#     # out = net(t1, t2)

#     # print(out.shape)

#     # t1 = torch.rand(1, 16, 16, 16, 128)

#     # net = SpatialTransformerLayer(2, 64, (1, 1, 1), (16, 16, 16), mlp_size=128, num_layers=2, types=2)


#     # out = net(t1)
#     # print(out.shape)
#     # net = WindowAttention(dim=128, heads=16, head_dim=8, window_size=(4, 4, 4), relative_pos_embedding=True)
#     # out = net(t1)

#     # print(out.shape)  
#     # config = get_config(in_channels=128,
#     #                              patch_size=(1, 1, 1), img_size=(16, 16, 16), mlp_dim=64)

#     # net = MultiAttention(config)
#     # t1 = torch.rand(1, 128, 16, 16, 16)

#     # out = net(t1)

#     # print(out.shape)

#     #########
#     # t1 = torch.rand(1, 3, 128, 128, 128)

#     # # net = SpatialTransformerLayer(in_channels=64, out_channels=128, patch_size=(2, 2, 2), img_size=(16, 16, 16), num_layers=1, window_size=(8, 8, 8))
#     # net = MyTransformer(in_channels=3, out_channels=2, img_size=(128, 128, 128))
#     # out = net(t1)

#     # print(out.shape)
#     pass 