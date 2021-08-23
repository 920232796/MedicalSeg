import torch
import torch.nn as nn
import torch.nn.functional as F

class kiunet3d(nn.Module):  #

    def __init__(self, c=4, n=16, num_classes=5):
        super(kiunet3d, self).__init__()

        # Entry flow
        self.encoder1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=1, bias=False)  # H//2
        self.encoder2 = nn.Conv3d(n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.encoder3 = nn.Conv3d(2 * n, 4 * n, kernel_size=3, padding=1, stride=1, bias=False)

        self.kencoder1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder2 = nn.Conv3d(n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder3 = nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)

        self.downsample1 = nn.MaxPool3d(2, stride=2)
        self.downsample2 = nn.MaxPool3d(2, stride=2)
        self.downsample3 = nn.MaxPool3d(2, stride=2)
        self.kdownsample1 = nn.MaxPool3d(2, stride=2)
        self.kdownsample2 = nn.MaxPool3d(2, stride=2)
        self.kdownsample3 = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.kupsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.kupsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.kupsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2

        self.decoder1 = nn.Conv3d(4 * n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.decoder2 = nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.decoder3 = nn.Conv3d(2 * n, c, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder1 = nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder2 = nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder3 = nn.Conv3d(2 * n, c, kernel_size=3, padding=1, stride=1, bias=False)

        self.intere1_1 = nn.Conv3d(n, n, 3, stride=1, padding=1)
        # self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv3d(2 * n, 4 * n, 3, stride=1, padding=1)
        # self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv3d(n, n, 3, stride=1, padding=1)
        # self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv3d(4 * n, 2 * n, 3, stride=1, padding=1)
        # self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv3d(n, n, 3, stride=1, padding=1)
        # self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv3d(2 * n, 2 * n, 3, stride=1, padding=1)
        # self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv3d(n, n, 3, stride=1, padding=1)
        # self.intd3_2bn = nn.BatchNorm2d(64)

        self.seg = nn.Conv3d(c, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # U-Net branch
        out1 = F.relu(F.interpolate(self.kencoder1(x), scale_factor=2, mode='trilinear'))  # Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intere1_1(out1)), scale_factor=0.25, mode='trilinear'))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.relu(self.intere1_2(tmp)), scale_factor=4, mode='trilinear'))  # CRFB

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        out1 = F.relu(F.interpolate(self.kencoder2(out1), scale_factor=2, mode='trilinear'))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intere2_1(out1)), scale_factor=0.0625, mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intere2_2(tmp)), scale_factor=16, mode='trilinear'))

        u2 = out
        o2 = out1

        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        out1 = F.relu(F.interpolate(self.kencoder3(out1), scale_factor=2, mode='trilinear'))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intere3_1(out1)), scale_factor=0.015625, mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intere3_2(tmp)), scale_factor=64, mode='trilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='trilinear'))  # U-NET
        out1 = F.relu(F.max_pool3d(self.kdecoder1(out1), 2, 2))  # Ki-NET
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.interd1_1(out1)), scale_factor=0.0625, mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.interd1_2(tmp)), scale_factor=16, mode='trilinear'))

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder2(out1), 2, 2))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.interd2_1(out1)), scale_factor=0.25, mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.interd2_2(tmp)), scale_factor=4, mode='trilinear'))

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder3(out1), 2, 2))

        out = torch.add(out, out1)  # fusion of both branches

        out = F.relu(self.seg(out))  # 1*1 conv

        # out = self.soft(out)
        return out

if __name__ == '__main__':
    t1 = torch.rand(1, 1, 32, 32, 32)
    net = kiunet3d(c=1, n=1, num_classes=2)
    out = net(t1)
    print(out.shape)