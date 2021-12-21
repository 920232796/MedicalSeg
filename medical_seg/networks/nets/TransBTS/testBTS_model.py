from matplotlib import cm

import matplotlib.pyplot as plt 
import glob 
import numpy as np 
import torch 
import numpy 
from medical_seg.networks.nets.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

if __name__ == '__main__':
    with torch.no_grad():
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        ## 降采样三次！！！
        cuda0 = torch.device('cpu')
        # x = torch.rand((1, 2, 32, 256, 256), device=cuda0)
        # _, model = TransBTS(image_size=(4, 32, 32), patch_dim = (1, 1, 1), in_channels=2, out_channels=3, _conv_repr=True, _pe_type="learned")
        # y = model(x)
        # print(y.shape)


        x = torch.rand((1, 4, 64, 64, 64), device=cuda0)
        _, model = TransBTS(image_size=(64, 64, 64), patch_dim = (8, 8, 8), in_channels=4, out_channels=2, _conv_repr=True, _pe_type="learned")
        y = model(x)
        print(y.shape)