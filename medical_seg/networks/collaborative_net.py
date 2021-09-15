
from medical_seg.networks.nets.basic_unet import BasicUNet
import torch.nn as nn 
import torch 
from medical_seg.networks.nets.transformer import TransformerLayers
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



class CoTNet(nn.Module):
    def __init__(self, model_num, image_size, num_layers=2, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.unets = []
        for i in range(model_num):
            unet = BasicUNet(dimensions=3, in_channels=1, 
                            out_channels=self.out_channels, features=[16, 16, 32, 64, 128, 16])
            self.unets.append(unet)

        self.fusion_layer = TransformerLayers(in_channels=self.model_num * 128, out_channels=128, 
                                                patch_size=(1, 1, 1), img_size=image_size, mlp_size=128, 
                                                num_layers=num_layers)
        # self.fusion_layer = nn.Conv3d(self.mode_num * 128, 128, 1)

    def forward(self, x):

        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        x_down = []
        for i in range(self.model_num):
            x_i = x[:, i].unsqueeze(dim=1)
            # print(f"x_i shape is {x_i.shape}")
            self.unets[i].down_pass(x_i)
            x_down.append(self.unets[i].x4)
        
        ## fusion
        # print(torch.cat(x_down, dim=1).shape)
        fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))
        # print(fusion_down.shape)

        logits = []
        uncers = []
        for i in range(self.model_num):
            logit = self.unets[i].up_pass(fusion_down)
            logits.append(logit)
            uncer = self.unets[i].uncer_pass(logit)
            uncers.append(uncer)
        
        logits = torch.stack(logits, dim=1) ## (1, model_num, out_channels, 32, 256, 256)
        # print(logits.shape)
        uncers = torch.cat(uncers, dim=1) ## (1, model_num, 32, 256, 256) 
        # print(uncers.shape)
        uncers = uncers.argmax(dim=1)

        uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, model_num)
        uncers = uncers.permute(0, 4, 1, 2, 3)
        uncers = uncers.unsqueeze(dim=2)
        
        out_teacher = (logits * uncers).sum(dim=1)

        return out_teacher, logits


        
        




            
