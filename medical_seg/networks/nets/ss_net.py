import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

import math


class GCN(nn.Module):
    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_t = nn.Conv2d(inplanes, planes, kernel_size=ks, padding=int(ks/2))

    def forward(self, x):
        x = self.conv_t(x)
        return x


class Refine(nn.Module):
    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out


class SSNet(nn.Module):
    def __init__(self, num_classes):
        super(SSNet, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.num_classes)
        self.gcn2 = GCN(1024, self.num_classes)
        self.gcn3 = GCN(512, self.num_classes)
        self.gcn4 = GCN(256, self.num_classes)
        self.gcn5 = GCN(64, self.num_classes)

        self.refine1 = Refine(self.num_classes)
        self.refine2 = Refine(self.num_classes)
        self.refine3 = Refine(self.num_classes)
        self.refine4 = Refine(self.num_classes)
        self.refine5 = Refine(self.num_classes)
        self.refine6 = Refine(self.num_classes)
        self.refine7 = Refine(self.num_classes)
        self.refine8 = Refine(self.num_classes)
        self.refine9 = Refine(self.num_classes)
        self.refine10 = Refine(self.num_classes)


    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(fm1))
        gcfm5 = self.refine5(self.gcn5(pool_x))

        fs1 = self.refine6(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)
        fs2 = self.refine7(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)
        fs3 = self.refine8(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)
        fs4 = self.refine9(F.upsample_bilinear(fs3, pool_x.size()[2:]) + gcfm5)
        out = self.refine10(F.upsample_bilinear(fs4, input.size()[2:]))

        return out

if __name__ == '__main__':
    net = SSNet(1)
    t1 = torch.rand(1, 3, 128, 128)
    out = net(t1)

    print(out.shape)