import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """Encapsulates the Conv2D > BatchNorm > ReLU pipeline.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Filter size.
    padding : int
        Number of pixels added around the border.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()

        # Components needed in a convolution layer
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode="circular",
        )
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))
    

class InceptionModule(nn.Module):
    """A single inception module as described for the v1 architecture.
    It performs 4 different convolutions in parallel en concatenates the outputs
        a) 1x1 convolution\\
        b) 1x1 convolution, followed by 3x3 convolution\\
        c) 1x1 convolution, followed by 5x5 convolution\\
        d) MaxPool, followed by 1x1 convolution

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_1x1 : int
        Number of output channels of branch a).
    red_3x3 : int
        Reduced 3x3, output of 1x1 in branch b).
    out_3x3 : int
        Number of output channels of branch b).
    red_5x5 : int
        Reduced 5x5, output of 1x1 in branch c).
    out 5x5 : int
        Number of output channels of branch d).
    out_1x1_pool : int
        Number of output channels of branch d).
    """

    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool
    ):
        super(InceptionModule, self).__init__()

        # Branch A - conv(kernel 1, padding 0)
        self.branch_a = ConvolutionBlock(in_channels, out_1x1, 1, 1, 0)

        # Branch B - conv(kernel 1, padding 0) -> conv(kernel 3, padding 1)
        self.branch_b = nn.Sequential(
            ConvolutionBlock(in_channels, red_3x3, 1, 1, 0),
            ConvolutionBlock(red_3x3, out_3x3, 3, 1, 1),
        )

        # Branch C - conv(kernel 1, padding 0) -> conv(kernel 5, padding 2)
        self.branch_c = nn.Sequential(
            ConvolutionBlock(in_channels, red_5x5, 1, 1, 0),
            ConvolutionBlock(red_5x5, out_5x5, 5, 1, 2),
        )

        # Branch D - pool(kernel 3, padding 1) -> conv(kernel 1, padding 0)
        self.branch_d = nn.Sequential(
            nn.MaxPool2d(3, 1, 1), ConvolutionBlock(in_channels, out_1x1_pool, 1, 1, 0)
        )

    def forward(self, x):
        """Calculate a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
        """
        out = torch.cat(
            [self.branch_a(x), self.branch_b(x), self.branch_c(x), self.branch_d(x)],
            dim=1,
        )
        return out
