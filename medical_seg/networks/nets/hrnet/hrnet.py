from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

import os, sys
import torch
import torch.nn.functional as functional
from queue import Queue


from os import path

import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

# _src_path = path.join(path.dirname(path.abspath(__file__)), "src")
# _backend = load(name="inplace_abn",
#                 extra_cflags=["-O3"],
#                 sources=[path.join(_src_path, f) for f in [
#                     "inplace_abn.cpp",
#                     "inplace_abn_cpu.cpp",
#                     "inplace_abn_cuda.cu"
#                 ]],
#                 extra_cuda_cflags=["--expt-extended-lambda"])

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


# def _act_forward(ctx, x):
#     if ctx.activation == ACT_LEAKY_RELU:
#         _backend.leaky_relu_forward(x, ctx.slope)
#     elif ctx.activation == ACT_ELU:
#         _backend.elu_forward(x)
#     elif ctx.activation == ACT_NONE:
#         pass


# def _act_backward(ctx, x, dx):
#     if ctx.activation == ACT_LEAKY_RELU:
#         _backend.leaky_relu_backward(x, dx, ctx.slope)
#     elif ctx.activation == ACT_ELU:
#         _backend.elu_backward(x, dx)
#     elif ctx.activation == ACT_NONE:
#         pass


# class InPlaceABN(autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, bias, running_mean, running_var,
#                 training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
#         # Save context
#         ctx.training = training
#         ctx.momentum = momentum
#         ctx.eps = eps
#         ctx.activation = activation
#         ctx.slope = slope
#         ctx.affine = weight is not None and bias is not None
#
#         # Prepare inputs
#         count = _count_samples(x)
#         x = x.contiguous()
#         weight = weight.contiguous() if ctx.affine else x.new_empty(0)
#         bias = bias.contiguous() if ctx.affine else x.new_empty(0)
#
#         if ctx.training:
#             mean, var = _backend.mean_var(x)
#
#             # Update running stats
#             running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
#             running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))
#
#             # Mark in-place modified tensors
#             ctx.mark_dirty(x, running_mean, running_var)
#         else:
#             mean, var = running_mean.contiguous(), running_var.contiguous()
#             ctx.mark_dirty(x)
#
#         # BN forward + activation
#         _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
#         _act_forward(ctx, x)
#
#         # Output
#         ctx.var = var
#         ctx.save_for_backward(x, var, weight, bias)
#         return x
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dz):
#         z, var, weight, bias = ctx.saved_tensors
#         dz = dz.contiguous()
#
#         # Undo activation
#         _act_backward(ctx, z, dz)
#
#         if ctx.training:
#             edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
#         else:
#             # TODO: implement simplified CUDA backward for inference mode
#             edz = dz.new_zeros(dz.size(1))
#             eydz = dz.new_zeros(dz.size(1))
#
#         dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
#         dweight = dweight if ctx.affine else None
#         dbias = dbias if ctx.affine else None
#
#         return dx, dweight, dbias, None, None, None, None, None, None, None
#
#
# class InPlaceABNSync(autograd.Function):
#     @classmethod
#     def forward(cls, ctx, x, weight, bias, running_mean, running_var,
#                 extra, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
#         # Save context
#         cls._parse_extra(ctx, extra)
#         ctx.training = training
#         ctx.momentum = momentum
#         ctx.eps = eps
#         ctx.activation = activation
#         ctx.slope = slope
#         ctx.affine = weight is not None and bias is not None
#
#         # Prepare inputs
#         count = _count_samples(x) * (ctx.master_queue.maxsize + 1)
#         x = x.contiguous()
#         weight = weight.contiguous() if ctx.affine else x.new_empty(0)
#         bias = bias.contiguous() if ctx.affine else x.new_empty(0)
#
#         if ctx.training:
#             mean, var = _backend.mean_var(x)
#
#             if ctx.is_master:
#                 means, vars = [mean.unsqueeze(0)], [var.unsqueeze(0)]
#                 for _ in range(ctx.master_queue.maxsize):
#                     mean_w, var_w = ctx.master_queue.get()
#                     ctx.master_queue.task_done()
#                     means.append(mean_w.unsqueeze(0))
#                     vars.append(var_w.unsqueeze(0))
#
#                 means = comm.gather(means)
#                 vars = comm.gather(vars)
#
#                 mean = means.mean(0)
#                 var = (vars + (mean - means) ** 2).mean(0)
#
#                 tensors = comm.broadcast_coalesced((mean, var), [mean.get_device()] + ctx.worker_ids)
#                 for ts, queue in zip(tensors[1:], ctx.worker_queues):
#                     queue.put(ts)
#             else:
#                 ctx.master_queue.put((mean, var))
#                 mean, var = ctx.worker_queue.get()
#                 ctx.worker_queue.task_done()
#
#             # Update running stats
#             running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
#             running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))
#
#             # Mark in-place modified tensors
#             ctx.mark_dirty(x, running_mean, running_var)
#         else:
#             mean, var = running_mean.contiguous(), running_var.contiguous()
#             ctx.mark_dirty(x)
#
#         # BN forward + activation
#         _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
#         _act_forward(ctx, x)
#
#         # Output
#         ctx.var = var
#         ctx.save_for_backward(x, var, weight, bias)
#         return x
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dz):
#         z, var, weight, bias = ctx.saved_tensors
#         dz = dz.contiguous()
#
#         # Undo activation
#         _act_backward(ctx, z, dz)
#
#         if ctx.training:
#             edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
#
#             if ctx.is_master:
#                 edzs, eydzs = [edz], [eydz]
#                 for _ in range(len(ctx.worker_queues)):
#                     edz_w, eydz_w = ctx.master_queue.get()
#                     ctx.master_queue.task_done()
#                     edzs.append(edz_w)
#                     eydzs.append(eydz_w)
#
#                 edz = comm.reduce_add(edzs) / (ctx.master_queue.maxsize + 1)
#                 eydz = comm.reduce_add(eydzs) / (ctx.master_queue.maxsize + 1)
#
#                 tensors = comm.broadcast_coalesced((edz, eydz), [edz.get_device()] + ctx.worker_ids)
#                 for ts, queue in zip(tensors[1:], ctx.worker_queues):
#                     queue.put(ts)
#             else:
#                 ctx.master_queue.put((edz, eydz))
#                 edz, eydz = ctx.worker_queue.get()
#                 ctx.worker_queue.task_done()
#         else:
#             edz = dz.new_zeros(dz.size(1))
#             eydz = dz.new_zeros(dz.size(1))
#
#         dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
#         dweight = dweight if ctx.affine else None
#         dbias = dbias if ctx.affine else None
#
#         return dx, dweight, dbias, None, None, None, None, None, None, None, None
#
#     @staticmethod
#     def _parse_extra(ctx, extra):
#         ctx.is_master = extra["is_master"]
#         if ctx.is_master:
#             ctx.master_queue = extra["master_queue"]
#             ctx.worker_queues = extra["worker_queues"]
#             ctx.worker_ids = extra["worker_ids"]
#         else:
#             ctx.master_queue = extra["master_queue"]
#             ctx.worker_queue = extra["worker_queue"]
#
#
# inplace_abn = InPlaceABN.apply
# inplace_abn_sync = InPlaceABNSync.apply
#
# __all__ = ["inplace_abn", "inplace_abn_sync", "ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU", "ACT_NONE"]
#
#
# class ABN(nn.Module):
#     """Activated Batch Normalization
#     This gathers a `BatchNorm2d` and an activation function in a single module
#     """
#
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
#         """Creates an Activated Batch Normalization module
#         Parameters
#         ----------
#         num_features : int
#             Number of feature channels in the input and output.
#         eps : float
#             Small constant to prevent numerical issues.
#         momentum : float
#             Momentum factor applied to compute running statistics as.
#         affine : bool
#             If `True` apply learned scale and shift transformation after normalization.
#         activation : str
#             Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
#         slope : float
#             Negative slope for the `leaky_relu` activation.
#         """
#         super(ABN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
#         self.momentum = momentum
#         self.activation = activation
#         self.slope = slope
#         if self.affine:
#             self.weight = nn.Parameter(torch.ones(num_features))
#             self.bias = nn.Parameter(torch.zeros(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.constant_(self.running_mean, 0)
#         nn.init.constant_(self.running_var, 1)
#         if self.affine:
#             nn.init.constant_(self.weight, 1)
#             nn.init.constant_(self.bias, 0)
#
#     def forward(self, x):
#         x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
#                                   self.training, self.momentum, self.eps)
#
#         if self.activation == ACT_RELU:
#             return functional.relu(x, inplace=True)
#         elif self.activation == ACT_LEAKY_RELU:
#             return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
#         elif self.activation == ACT_ELU:
#             return functional.elu(x, inplace=True)
#         else:
#             return x
#
#     def __repr__(self):
#         rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
#               ' affine={affine}, activation={activation}'
#         if self.activation == "leaky_relu":
#             rep += ', slope={slope})'
#         else:
#             rep += ')'
#         return rep.format(name=self.__class__.__name__, **self.__dict__)
#
#
# class InPlaceABN(ABN):
#     """InPlace Activated Batch Normalization"""
#
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
#         """Creates an InPlace Activated Batch Normalization module
#         Parameters
#         ----------
#         num_features : int
#             Number of feature channels in the input and output.
#         eps : float
#             Small constant to prevent numerical issues.
#         momentum : float
#             Momentum factor applied to compute running statistics as.
#         affine : bool
#             If `True` apply learned scale and shift transformation after normalization.
#         activation : str
#             Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
#         slope : float
#             Negative slope for the `leaky_relu` activation.
#         """
#         super(InPlaceABN, self).__init__(num_features, eps, momentum, affine, activation, slope)
#
#     def forward(self, x):
#         return inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var,
#                            self.training, self.momentum, self.eps, self.activation, self.slope)
#
#
# class InPlaceABNSync(ABN):
#     """InPlace Activated Batch Normalization with cross-GPU synchronization
#     This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
#     """
#
#     def __init__(self, num_features, devices=None, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
#                  slope=0.01):
#         """Creates a synchronized, InPlace Activated Batch Normalization module
#         Parameters
#         ----------
#         num_features : int
#             Number of feature channels in the input and output.
#         devices : list of int or None
#             IDs of the GPUs that will run the replicas of this module.
#         eps : float
#             Small constant to prevent numerical issues.
#         momentum : float
#             Momentum factor applied to compute running statistics as.
#         affine : bool
#             If `True` apply learned scale and shift transformation after normalization.
#         activation : str
#             Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
#         slope : float
#             Negative slope for the `leaky_relu` activation.
#         """
#         super(InPlaceABNSync, self).__init__(num_features, eps, momentum, affine, activation, slope)
#         self.devices = devices if devices else list(range(torch.cuda.device_count()))
#
#         # Initialize queues
#         self.worker_ids = self.devices[1:]
#         self.master_queue = Queue(len(self.worker_ids))
#         self.worker_queues = [Queue(1) for _ in self.worker_ids]
#
#     def forward(self, x):
#         if x.get_device() == self.devices[0]:
#             # Master mode
#             extra = {
#                 "is_master": True,
#                 "master_queue": self.master_queue,
#                 "worker_queues": self.worker_queues,
#                 "worker_ids": self.worker_ids
#             }
#         else:
#             # Worker mode
#             extra = {
#                 "is_master": False,
#                 "master_queue": self.master_queue,
#                 "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
#             }
#
#         return inplace_abn_sync(x, self.weight, self.bias, self.running_mean, self.running_var,
#                                 extra, self.training, self.momentum, self.eps, self.activation, self.slope)
#
#     def __repr__(self):
#         rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
#               ' affine={affine}, devices={devices}, activation={activation}'
#         if self.activation == "leaky_relu":
#             rep += ', slope={slope})'
#         else:
#             rep += ')'
#         return rep.format(name=self.__class__.__name__, **self.__dict__)



# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
# BatchNorm2d = nn.BatchNorm2d()
from torch.nn import BatchNorm2d
BN_MOMENTUM = 0.01
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../src'))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, in_channels=1, out_channels=1):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=out_channels,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(w, h), mode="bilinear")
        return x

    def init_weights(self, pretrained='', ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


from yacs.config import CfgNode as CN
import os

def get_seg_model(in_channels=3, out_channels=1):
    _C = CN()
    config_path = "./bis3d_v2/networks/nets/hrnet/seg_hrnet_w48_train.yaml"
    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.EXTRA = CN(new_allowed=True)

    cfg = _C

    cfg.defrost()
    cfg.merge_from_file(config_path)

    model = HighResolutionNet(cfg, in_channels=in_channels, out_channels=out_channels)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model

if __name__ == '__main__':

    hr_model = get_seg_model(in_channels=1, out_channels=1)

    t1 = torch.rand(1, 1, 256, 256)
    out = hr_model(t1)
    print(out.shape)



