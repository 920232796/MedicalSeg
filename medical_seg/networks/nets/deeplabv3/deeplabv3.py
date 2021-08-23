# camera-ready
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from bis3d_v2.networks.nets.deeplabv3.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8, ResNet50_OS8
from bis3d_v2.networks.nets.deeplabv3.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, out_channels):
        super(DeepLabV3, self).__init__()

        self.num_classes = out_channels

        # self.create_model_dirs()

        self.resnet = ResNet34_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        # print(feature_map.shape)
        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

    # def create_model_dirs(self):
    #     self.logs_dir = self.project_dir + "/training_logs"
    #     self.model_dir = self.logs_dir + "/model_%s" % self.model_id
    #     self.checkpoints_dir = self.model_dir + "/checkpoints"
    #     if not os.path.exists(self.logs_dir):
    #         os.makedirs(self.logs_dir)
    #     if not os.path.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #         os.makedirs(self.checkpoints_dir)

if __name__ == '__main__':
    net = DeepLabV3(out_channels=1)
    t1 = torch.rand(1, 1, 256, 256)
    net.eval()
    out = net(t1)
    print(out.shape)