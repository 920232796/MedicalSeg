import torch
from medical_seg.networks.layers.transformer import get_config, Embeddings, BlockMulti
import torch.nn as nn 

class TransformerLayers(nn.Module):
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

    
class TransformerEncoder(nn.Module):
    def __init__(self,  in_channels, out_channels, mlp_size=256, num_layers=1):
        super().__init__()
        self.config = get_config(in_channels=in_channels, out_channels=out_channels,
                                 mlp_dim=mlp_size)
        self.block_list = nn.ModuleList([BlockMulti(self.config) for i in range(num_layers)])

    
    def forward(self, x):
        for l in self.block_list:
            x = l(x)
      
        return x
    

