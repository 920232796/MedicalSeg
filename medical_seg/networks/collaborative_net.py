
from medical_seg.networks.nets.co_unet import BasicUNet
import torch.nn as nn 
import torch 
from medical_seg.networks.nets.transformer import TransformerLayers, TransformerEncoder
import torch.nn.functional as F 

class CoNet(nn.Module):
    def __init__(self, model_num, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.unets = []
        for i in range(model_num):
            unet = BasicUNet(dimensions=3, in_channels=1, 
                            out_channels=self.out_channels, features=[16, 16, 32, 64, 128, 16])
            self.unets.append(unet)

        self.fusion_layer = nn.Conv3d(self.model_num * 128, 128, 1)

    def forward(self, x):

        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        x_down = []
        for i in range(self.model_num):
            x_i = x[:, i].unsqueeze(dim=1)
            self.unets[i].down_pass(x_i)
            x_down.append(self.unets[i].x4)
        
        ## fusion
        fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))

        logits = []
        uncers = []
        for i in range(self.model_num):
            logit = self.unets[i].up_pass(fusion_down)
            logits.append(logit)
            uncer = self.unets[i].uncer_pass(logit)
            uncers.append(uncer)
        
        logits = torch.stack(logits, dim=1) ## (1, model_num, out_channels, 32, 256, 256)
        uncers = torch.cat(uncers, dim=1) ## (1, model_num, 32, 256, 256) 
        uncers = uncers.argmax(dim=1)

        uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, model_num)
        uncers = uncers.permute(0, 4, 1, 2, 3)
        uncers = uncers.unsqueeze(dim=2)
        
        out_teacher = (logits * uncers).sum(dim=1)

        return out_teacher, logits




# class MMTNet(nn.Module):
#     def __init__(self, model_num, image_size, num_layers=2, out_channels=1):
#         super().__init__()
#         self.out_channels = out_channels
#         self.model_num = model_num
#         self.unets = nn.ModuleList([])
#         self.conv1_1 = nn.ModuleList([])
#         self.gates = nn.ModuleList([])
#         for i in range(model_num):
#             unet = BasicUNet(dimensions=3, in_channels=1,
#                             out_channels=self.out_channels, features=[16, 16, 32, 64, 128, 16])
#             # conv = nn.Conv3d(128, 1, 1, 1, 0)## 将特征降为1
#             # self.conv1_1.append(conv)
#             self.gates.append(nn.Sigmoid()) ## gate
#             self.unets.append(unet)
#
#         self.fusion_layer = TransformerLayers(in_channels=self.model_num * 128, out_channels=128,
#                                                 mlp_size=128, num_layers=num_layers)
#         # self.fusion_layer = nn.Conv3d(self.mode_num * 128, 128, 1)
#
#     def forward(self, x):
#
#         assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
#         x_down = []
#         for i in range(self.model_num):
#             x_i = x[:, i].unsqueeze(dim=1)
#             self.unets[i].down_pass(x_i)
#             x_down.append(self.unets[i].x4)
#
#
#         ## fusion
#         fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))
#
#         logits = []
#         uncers = []
#         for i in range(self.model_num):
#             logit = self.unets[i].up_pass(fusion_down)
#             logits.append(logit)
#             uncer = self.unets[i].uncer_pass(logit)
#             uncers.append(uncer)
#
#         logits = torch.stack(logits, dim=1) ## (1, model_num, out_channels, 32, 256, 256)
#         uncers = torch.cat(uncers, dim=1) ## (1, model_num, 32, 256, 256)
#         uncers = uncers.argmax(dim=1)
#
#         uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, model_num)
#         uncers = uncers.permute(0, 4, 1, 2, 3)
#         uncers = uncers.unsqueeze(dim=2)
#
#         out_teacher = (logits * uncers).sum(dim=1)
#
#         return out_teacher, logits


class MMTNet_2(nn.Module):
    def __init__(self, model_num, image_size, num_layers=2, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.unets = nn.ModuleList([])
        self.gates = nn.ModuleList([])
        for i in range(model_num):
            unet = BasicUNet(dimensions=3, in_channels=1, 
                            out_channels=self.out_channels, features=[16, 16, 32, 32, 64, 16])
            gate = nn.Sequential(nn.Conv3d(64, 1, 1, 1, 0),
                                 nn.Sigmoid())

            self.unets.append(unet)
            self.gates.append(gate)

        self.fusion_layer = TransformerLayers(in_channels=self.model_num * 64, out_channels=64,
                                                patch_size=(1, 1, 1), img_size=image_size, mlp_size=64,
                                                num_layers=num_layers)
        # self.fusion_layer = nn.Conv3d(self.mode_num * 128, 128, 1)

    def forward(self, x):

        assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
        x_down = []
        for i in range(self.model_num):
            x_i = x[:, i].unsqueeze(dim=1)
            self.unets[i].down_pass(x_i)
            x_down.append(self.unets[i].x4)
        
        ## fusion
        fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))

        ## gate out
        fusion_out = 0
        for i in range(self.model_num):
            gate_i = self.gates[i](x_down[i])
            fusion_out = fusion_out + gate_i * fusion_down

        logits = []
        uncers = []
        for i in range(self.model_num):
            logit = self.unets[i].up_pass(fusion_out)
            logits.append(logit)
            uncer = self.unets[i].uncer_pass(logit)
            uncers.append(uncer)
        
        logits = torch.stack(logits, dim=1) ## (1, model_num, out_channels, 32, 256, 256)
        uncers = torch.cat(uncers, dim=1) ## (1, model_num, 32, 256, 256) 
        uncers = uncers.argmax(dim=1)

        uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, model_num)
        uncers = uncers.permute(0, 4, 1, 2, 3)
        uncers = uncers.unsqueeze(dim=2)
        
        out_teacher = (logits * uncers).sum(dim=1)

        return out_teacher, logits


# class MMTNet(nn.Module):
#     def __init__(self, model_num, image_size, num_layers=2, out_channels=1):
#         super().__init__()
#         self.out_channels = out_channels
#         self.model_num = model_num
#         self.unets = nn.ModuleList([])
#         self.gates = nn.ModuleList([])
#         for i in range(model_num):
#             unet = BasicUNet(dimensions=3, in_channels=1,
#                              out_channels=self.out_channels, features=[16, 16, 32, 64, 128, 16])
#             gate = nn.Sequential(nn.Conv3d(128, 1, 1, 1, 0),
#                                  nn.Sigmoid())
#
#             self.unets.append(unet)
#             self.gates.append(gate)
#
#         self.fusion_layer = TransformerLayers(in_channels=self.model_num * 128, out_channels=128,
#                                               patch_size=(1, 1, 1), img_size=image_size, mlp_size=128,
#                                               num_layers=num_layers)
#         # self.fusion_layer = nn.Conv3d(self.mode_num * 128, 128, 1)
#
#     def forward(self, x):
#
#         assert x.shape[1] == self.model_num, "输入模态不一致，请检查"
#         x_down = []
#         for i in range(self.model_num):
#             x_i = x[:, i].unsqueeze(dim=1)
#             self.unets[i].down_pass(x_i)
#             x_down.append(self.unets[i].x4)
#
#         ## fusion
#         fusion_down = self.fusion_layer(torch.cat(x_down, dim=1))
#
#         ## gate out
#         fusion_out = 0
#         for i in range(self.model_num):
#             gate_i = self.gates[i](x_down[i])
#             fusion_out = fusion_out + gate_i * fusion_down
#
#         logits = []
#         uncers = []
#         for i in range(self.model_num):
#             logit = self.unets[i].up_pass(fusion_out)
#             logits.append(logit)
#             uncer = self.unets[i].uncer_pass(logit)
#             uncers.append(uncer)
#
#         logits = torch.stack(logits, dim=1) ## (1, model_num, out_channels, 32, 256, 256)
#         uncers = torch.cat(uncers, dim=1) ## (1, model_num, 32, 256, 256)
#         uncers = uncers.argmax(dim=1)
#
#         uncers = F.one_hot(uncers, num_classes=logits.shape[1])##(1, 32, 256, 256, model_num)
#         uncers = uncers.permute(0, 4, 1, 2, 3)
#         uncers = uncers.unsqueeze(dim=2)
#
#         out_teacher = (logits * uncers).sum(dim=1)
#
#         return out_teacher, logits



        
        




            
