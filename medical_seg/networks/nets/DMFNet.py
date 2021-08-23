import torch.nn as nn
import torch.nn.functional as F
import torch

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        # print("~~~")
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print("~~~")

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        b, c, d, h, w = shortcut.shape
        x4 = nn.functional.interpolate(x4, size=(d, h, w), mode="trilinear")
        return x4 + shortcut

class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        #
        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):

        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        x4 = self.conv3x3x3_m2(x3)
        shortcut = x
        # print("~~~")
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print("~~~")
        # dimensions = len(x.shape) - 2
        # sp = [0] * (dimensions * 2)
        # for i in range(dimensions):
        #     if x.shape[-i - 1] != x4.shape[-i - 1]:
        #         sp[i * 2 + 1] = 1
        # x4 = torch.nn.functional.pad(y1, sp, "replicate")
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        # dimensions = len(x.shape) - 2
        # sp = [0] * (dimensions * 2)
        # for i in range(dimensions):
        #     if shortcut.shape[-i - 1] != x4.shape[-i - 1]:
        #         sp[i * 2 + 1] = 1
        # shortcut = torch.nn.functional.pad(shortcut, sp, "replicate")
        return x4 + shortcut


class MFNet(nn.Module): #
    # [96]   Flops:  13.361G  &  Params: 1.81M
    # [112]  Flops:  16.759G  &  Params: 2.46M
    # [128]  Flops:  20.611G  &  Params: 3.19M
    def __init__(self, c=4,n=32,channels=128,groups = 16,norm='bn', num_classes=4):
        super(MFNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d( c, n, kernel_size=2, stride=2, bias=False, padding=0)# H//2
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels*2, g=groups, stride=2, norm=norm), # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            MFunit(channels*2, channels*3, g=groups, stride=2, norm=norm), # H//16
            MFunit(channels*3, channels*3, g=groups, stride=1, norm=norm),
            MFunit(channels*3, channels*2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.decoder_block1 = MFunit(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.decoder_block2 = MFunit(channels*2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)# H//2 down

        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # x4 = self.encoder_block4(x3) # H//16
        # Decoder
        # y1 = self.upsample1(x4)# H//8
        dimensions = len(x.shape) - 2
        # sp = [0] * (dimensions * 2)
        # for i in range(dimensions):
        #     if y1.shape[-i - 1] != x3.shape[-i - 1]:
        #         sp[i * 2 + 1] = 1
        # y1 = torch.nn.functional.pad(y1, sp, "replicate")
        # y1 = torch.cat([x3,y1],dim=1)
        # y1 = self.decoder_block1(y1)

        # y2 = self.upsample2(y1)# H//4
        y2 = self.upsample2(x3)  # H//4
        # print(y2.shape)
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if y2.shape[-i - 1] != x2.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        y2 = torch.nn.functional.pad(y2, sp, "replicate")
        y2 = torch.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)
        # print(y2.shape)

        y3 = self.upsample3(y2)# H//2
        # print(y3.shape)
        # print(x1.shape)
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if y3.shape[-i - 1] != x1.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        y3 = torch.nn.functional.pad(y3, sp, "replicate")
        y3 = torch.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        for i in range(dimensions):
            if y4.shape[-i - 1] != x.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        y4 = torch.nn.functional.pad(y4, sp, "replicate")
        y4 = self.seg(y4)
        if hasattr(self,'softmax'):
            y4 = self.softmax(y4)
        return y4


class DMFNet(MFNet): # softmax
    # [128]  Flops:  27.045G  &  Params: 3.88M
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='bn', num_classes=4):
        super(DMFNet, self).__init__(c,n,channels,groups, norm, num_classes)

        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2,3]),# H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3]), # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2,3]), # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3]),# Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )


if __name__ == '__main__':

    x = torch.rand((1,1,24,128,128)) # [bsize,channels,Height,Width,Depth]
    model = MFNet(c=1, groups=16, norm='bn', num_classes=1)
    y = model(x)
    print(y.shape)