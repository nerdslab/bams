# Code adapted from https://github.com/locuslab/TCN/blob/2f8c2b817050206397458dfd1f5a25ce8a32fe65/TCN/tcn.py#L48


import copy

import torch.nn as nn
from torch.nn.utils import weight_norm
from einops import rearrange


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, shift=0):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.shift = shift

    def forward(self, x):
        if self.chomp_size + self.shift > 0:
            return x[:, :, : -self.chomp_size - 2 * self.shift].contiguous()
        else:
            return x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        n_layers=2,
        dropout=0.2,
        shift=0,
    ):
        super(TemporalBlock, self).__init__()
        assert n_layers in [1, 2]

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU(init=0.1)
        self.dropout1 = nn.Dropout(dropout)

        if n_layers == 1:
            assert shift == 0
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        else:
            self.conv2 = weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding + shift,
                    dilation=dilation,
                )
            )
            self.chomp2 = Chomp1d(padding, shift)
            self.relu2 = nn.PReLU(init=0.1)
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.chomp2,
                self.relu2,
                self.dropout2,
            )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1, padding=shift // dilation)
            if n_inputs != n_outputs
            else nn.ConstantPad1d(shift // dilation, 0)
        )
        self.chompd = Chomp1d(0, shift // dilation)
        self.relu = nn.PReLU(init=0.1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if hasattr(self, "conv2"):
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None and hasattr(self.downsample, "weight"):
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        res = self.chompd(res)
        return self.relu(out + res)


def prepare_argument(arg, num_levels):
    if isinstance(arg, int):
        arg = [arg] * num_levels
    else:
        assert len(arg) == num_levels
    return arg


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        num_layers_per_block=2,
        dropout=0.2,
        shift=0,
        dilation=2,
    ):
        super(TemporalConvNet, self).__init__()

        self.num_levels = len(num_channels)
        self.kernel_size = prepare_argument(kernel_size, self.num_levels)
        self.num_layers_per_block = prepare_argument(
            num_layers_per_block, self.num_levels
        )
        self.dilation = dilation
        self.feat_dim = num_channels[-1]
        layers = []
        for i in range(self.num_levels):
            dilation_size = dilation**i
            shift_ = shift if i == (self.num_levels - 1) else 0
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            kernel_size = self.kernel_size[i]
            num_layers_per_block = self.num_layers_per_block[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    n_layers=num_layers_per_block,
                    dropout=dropout,
                    shift=shift_ * dilation_size,
                )
            ]

        self.network = nn.Sequential(*layers)

    @property
    def receptive_field(self):
        return compute_receiptive_field(
            kernel_size=self.kernel_size,
            num_blocks=self.num_levels,
            num_layers_per_block=self.num_layers_per_block,
            dilation=self.dilation,
        )

    def forward(self, x):
        x = rearrange(x, "b l k -> b k l")
        ret = self.network(x)
        ret = rearrange(ret, "b k l -> b l k")
        return ret


def compute_receiptive_field(
    kernel_size, num_blocks, num_layers_per_block, dilation, sample_size=100
):
    a = [
        {
            i,
        }
        for i in range(sample_size)
    ]
    for i in range(num_blocks):
        dilation_size = dilation**i
        for j in range(num_layers_per_block[i]):  # 2 conv layers
            a_copy = copy.deepcopy(a)
            for k in range(len(a)):
                for l in range(kernel_size[i]):
                    p = dilation_size * l
                    if 0 <= k - p:
                        a[k] = a[k].union(a_copy[k - p])
    receiptive_field = len(a[-1])
    if receiptive_field >= sample_size:
        return compute_receiptive_field(
            kernel_size, num_blocks, num_layers_per_block, dilation, 10 * sample_size
        )
    return receiptive_field
