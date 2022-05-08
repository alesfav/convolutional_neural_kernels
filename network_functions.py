import torch
import torch.nn as nn
import torch.nn.functional as F


class LocBlock1dNT(nn.Module):
    """
    1d non-overlapping locally-connected block with NTK initialization.
    Input should be of shape (num_points, in_channels, dimension).

    Args:
        out_channels: number of output channels (neurons)
        in_channels: number of channels of the input
        num_patches: number of input patches (dimension / filter_size)
        filter_size: size of filter and stride.
    """

    def __init__(self, out_channels, in_channels, num_patches, filter_size):
        super().__init__()
        self.w = nn.Parameter(
            torch.randn(1, out_channels, in_channels, num_patches, filter_size)
        )

    def forward(self, x):

        filter_size = self.w.size(4)
        in_channels = self.w.size(2)
        x = x.unfold(2, filter_size, filter_size)
        y = (x.unsqueeze(1) * self.w).sum([2, 4])
        y = F.relu(y / (filter_size * in_channels) ** 0.5)

        return y


class LocBlock2dNT(nn.Module):
    """
    2d non-overlapping locally-connected block with NTK initialization.
    Input should be of shape (num_points, in_channels, dimension, dimension).

    Args:
        out_channels: number of output channels (neurons)
        in_channels: number of channels of the input
        num_patches: number of input patches (dimension / filter_size)
        filter_size: size of filter and stride.
    """

    def __init__(self, out_channels, in_channels, num_patches, filter_size):
        super().__init__()
        self.w = nn.Parameter(
            torch.randn(
                1, out_channels, in_channels, num_patches, num_patches, filter_size ** 2
            )
        )
        self.in_channels = in_channels
        self.filter_size = filter_size

    def forward(self, x):

        filter_size = self.filter_size
        in_channels = self.in_channels
        x = x.unfold(2, filter_size, filter_size).unfold(3, filter_size, filter_size)
        x = x.reshape(*x.size()[:-2], -1)
        y = (x.unsqueeze(1) * self.w).sum([2, 5])
        y = F.relu(y / (filter_size ** 2 * in_channels) ** 0.5)

        return y


class LocNet1dNT(nn.Module):
    """
    1d non-overlapping locally-connected network with NTK initialization.
    Input should be of shape (num_points, in_channels, dimension).

    Args:
        h: number channels (width)
        in_channels: number of channels of the input
        dimension: input dimension
        filter_sizes: sizes of filters and strides.
    """

    def __init__(self, h, in_channels, dimension, filter_sizes, out_dim=1):

        super().__init__()
        self.conv = nn.Sequential(
            LocBlock1dNT(h, in_channels, dimension // filter_sizes[0], filter_sizes[0]),
            *[
                LocBlock1dNT(
                    h,
                    h,
                    dimension // torch.tensor(filter_sizes[: block + 1]).prod().item(),
                    filter_sizes[block],
                )
                for block in range(1, len(filter_sizes))
            ]
        )
        self.beta = nn.Parameter(
            torch.randn(
                h * dimension // torch.tensor(filter_sizes).prod().item(), out_dim
            )
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        y = y @ self.beta / self.beta.size(0) ** 0.5

        return y


class LocNet2dNT(nn.Module):
    """
    2d non-overlapping locally-connected network with NTK initialization.
    Input should be of shape (num_points, in_channels, dimension).

    Args:
        h: number channels (width)
        in_channels: number of channels of the input
        dimension: input dimension
        filter_sizes: sizes of filters and strides.
    """

    def __init__(self, h, in_channels, dimension, filter_sizes, out_dim=1):

        super().__init__()
        self.conv = nn.Sequential(
            LocBlock2dNT(h, in_channels, dimension // filter_sizes[0], filter_sizes[0]),
            *[
                LocBlock2dNT(
                    h,
                    h,
                    dimension // torch.tensor(filter_sizes[: block + 1]).prod().item(),
                    filter_sizes[block],
                )
                for block in range(1, len(filter_sizes))
            ]
        )
        self.beta = nn.Parameter(
            torch.randn(
                h * (dimension // torch.tensor(filter_sizes).prod().item()) ** 2,
                out_dim,
            )
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        y = y @ self.beta / self.beta.size(0) ** 0.5

        return y
