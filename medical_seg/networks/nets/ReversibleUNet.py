# A Partially Reversible U-Net for Memory-Efficient Volumetric Image Segmentation (MICCAI 2019)
# https://github.com/RobinBruegger/PartiallyReversibleUnet

import torch.nn as nn
import torch
import torch.nn.functional as F
import revtorch.revtorch as rv

CHANNELS = [60, 120, 240, 360, 480]

class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(F.leaky_relu(self.gn(x), inplace=True))
        return x


def makeReversibleSequence(channels):
    innerChannels = channels // 2
    groups = CHANNELS[0] // 2
    fBlock = ResidualInner(innerChannels, groups)
    gBlock = ResidualInner(innerChannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

def getChannelsAtIndex(index):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]

class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outChannels, depth)

    def forward(self, x):
        if self.downsample:
            x = F.max_pool3d(x, 2)
            x = self.conv(x) #increase number of channels
        x = self.reversibleBlocks(x)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.reversibleBlocks = makeReversibleComponent(inChannels, depth)
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(inChannels, outChannels, 1)

    def forward(self, x):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class NoNewReversible(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(NoNewReversible, self).__init__()
        depth = 1
        self.levels = 5

        self.firstConv = nn.Conv3d(in_channels, CHANNELS[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        self.lastConv = nn.Conv3d(CHANNELS[0], out_channels, 1, bias=True)

        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getChannelsAtIndex(i - 1), getChannelsAtIndex(i), depth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels - i - 1), getChannelsAtIndex(self.levels - i - 2), depth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        x = self.firstConv(x)
        #x = self.dropout(x)

        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            # print(x.shape)
            if i < self.levels - 1:
                inputStack.append(x)
        # print("~~~")
        for i in range(self.levels):

            x = self.decoders[i](x)
            # print(x.shape)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        return x

if __name__ == "__main__":
    net = NoNewReversible(in_channels=1, out_channels=1)
    t1 = torch.rand(1, 1, 32, 128, 128)
    out = net(t1)
    print(out.shape)
    # print(net)