
from medical_seg.networks.nets.basic_unet import BasicUNet
import torch.nn as nn 
import torch 

import torch.nn.functional as F 

class CoNet(nn.Module):
    def __init__(self, mode_num, in_channels=1, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode_num = mode_num
        self.unets = []
        for i in range(mode_num):
            unet = BasicUNet(dimensions=3, in_channels=self.in_channels, 
                            out_channels=self.out_channels, features=[16, 16, 32, 64, 128, 16])
            self.unets.append(unet)

        self.fusion_layer = nn.Conv3d(self.mode_num * 128, 128, 1)

    def forward(self, x):

        assert x.shape[1] == self.mode_num, "输入模态不一致，请检查"
        x_down = []
        for i in range(self.mode_num):
            x_down.append(self.unets[i].down_pass(x[:, i]))
        
        ## fusion
        fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))

        logits = []
        uncers = []
        for i in range(self.mode_num):
            logit = self.unets[i].up_pass(fusion_down)
            logits.append(logit)
            uncer = self.unets[i].uncer_pass(logit)
            uncers.append(uncer)
        
        logits = torch.cat(logits, dim=1)
        uncers = torch.cat(uncers, dim=1)
        uncers = uncers.argmax(dim=1)

        uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, 2)
        uncers = uncers.permute(0, 4, 1, 2, 3)
        out_teacher = (logits * uncers).sum(dim=1)

        return out_teacher


        
        




            
