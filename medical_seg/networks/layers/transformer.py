
import numpy as np
import math
import torch.nn as nn
import torch
import ml_collections


def get_config(in_channels, out_channels, patch_size, img_size, mlp_dim=256):
    config = ml_collections.ConfigDict()
    config.in_channels = in_channels
    config.hidden_size = out_channels
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = mlp_dim
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = patch_size
    config.img_size = img_size

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

    def forward(self, hidden_states):
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
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

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
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = config.img_size
        in_channels = config.in_channels
        patch_size = config["patch_size"]
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        input_shape = x.shape
        x = self.embeddings(x)
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.transpose(-1, -2)
        out_size = (input_shape[2] // self.config.patch_size[0],
                    input_shape[3] // self.config.patch_size[1],
                    input_shape[4] // self.config.patch_size[2],)
        x = x.view((input_shape[0], self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        return x


class TransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, mlp_size=256):
        super().__init__()
        self.config = get_config(in_channels=in_channels, out_channels=out_channels,
                                 patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
        self.block = Block(self.config)

    def forward(self, x):
        x = self.block(x)
        return x

class BlockMulti(nn.Module):
    def __init__(self, config):
        super(BlockMulti, self).__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):

        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class TransformerLayerMulti(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, mlp_size=256, num_layers=1):
        super().__init__()
        self.config = get_config(in_channels=in_channels, out_channels=out_channels,
                                 patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
        self.block_list = nn.ModuleList([BlockMulti(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config)

    def forward(self, x, encoder=False):
        input_shape = self.config.img_size
        batch_size = x.shape[0]
        x = self.embeddings(x)

        for l in self.block_list:
            x = l(x)
        if encoder:
            return x
        x = x.transpose(-1, -2)
        out_size = (input_shape[0] // self.config.patch_size[0],
                    input_shape[1] // self.config.patch_size[1],
                    input_shape[2] // self.config.patch_size[2],)
        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        return x

class AttentionCrossModal(nn.Module):
    def __init__(self, config):
        super(AttentionCrossModal, self).__init__()
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

    def forward(self, hidden_states, kv):
        ## kv 是别的模态的特征。
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(kv)
        mixed_value_layer = self.value(kv)

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
        return attention_output

class BlockMultiCrossModal(nn.Module):
    ## 跨模态的attention
    def __init__(self, config):
        super(BlockMultiCrossModal, self).__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_cross = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.attn_cross = AttentionCrossModal(config)

    def forward(self, x, kv):

        ## self attention
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        # cross attention
        h = x
        x = self.attention_norm_cross(x)
        x = self.attn_cross(x, kv)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class TransformerLayerMultiCrossModal(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, mlp_size=256, num_layers=1):
        super().__init__()
        self.config = get_config(in_channels=in_channels, out_channels=out_channels,
                                 patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
        self.block_list = nn.ModuleList([BlockMultiCrossModal(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config)

    def forward(self, x, kv):
        input_shape = self.config.img_size
        batch_size = x.shape[0]
        x = self.embeddings(x)

        for l in self.block_list:
            x = l(x, kv)

        x = x.transpose(-1, -2)
        out_size = (input_shape[0] // self.config.patch_size[0],
                    input_shape[1] // self.config.patch_size[1],
                    input_shape[2] // self.config.patch_size[2],)
        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2]))
        return x

class DoubleRouteTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_layer=3, mlp_size=128):
        super().__init__()
        if type(num_layer) is int :
            num_layer = (num_layer, num_layer)
        self.patch_size = patch_size
        # self.config = get_config(in_channels=in_channels, out_channels=out_channels, patch_size=p)
        self.self_transformer = TransformerLayerMulti(in_channels=in_channels, out_channels=out_channels,
                                                      patch_size=patch_size, img_size=img_size, num_layers=num_layer[0], mlp_size=mlp_size)
        self.cross_transformer = TransformerLayerMultiCrossModal(in_channels=in_channels, out_channels=out_channels,
                                                                 patch_size=patch_size,img_size=img_size,num_layers=num_layer[1], mlp_size=mlp_size)


    def forward(self, main_x, second_x):
        #传入降采样的特征map即可
        batch_size, hidden_size, x, y, z = main_x.shape
        second_x = self.self_transformer(second_x, encoder=True)

        main_x = self.cross_transformer(main_x, second_x)

        main_x = main_x.transpose(-1, -2)
        out_size = (x // self.patch_size[0],
                    y // self.patch_size[1],
                    z // self.patch_size[2],)
        main_x = main_x.view((batch_size, hidden_size, out_size[0], out_size[1], out_size[2]))

        return main_x

class DoubleRouteTransformerSelf(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_layer=3, mlp_size=128):
        super().__init__()
        if type(num_layer) is int :
            num_layer = (num_layer, num_layer)
        self.patch_size = patch_size
        self.self_transformer = TransformerLayerMulti(in_channels=in_channels, out_channels=out_channels,
                                                      patch_size=patch_size, img_size=img_size, num_layers=num_layer[0], mlp_size=mlp_size)
        self.img_size = img_size
        self.out_chennels = out_channels
    def forward(self, main_x):
        #传入降采样的特征map即可
        batch_size, hidden_size, x, y, z = main_x.shape
        main_x = self.self_transformer(main_x, encoder=True)

        main_x = main_x.transpose(-1, -2)
        out_size = (self.img_size[0] // self.patch_size[0],
                    self.img_size[1] // self.patch_size[1],
                    self.img_size[2] // self.patch_size[2],)

        main_x = main_x.view((batch_size, self.out_chennels, out_size[0], out_size[1], out_size[2]))

        return main_x

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

    t1 = torch.rand(1, 2, 32, 32, 32)

    net = TransformerLayer(2, 5, (1, 1, 1), (32, 32, 32), mlp_size=128)


    out = net(t1)
    print(out.shape)