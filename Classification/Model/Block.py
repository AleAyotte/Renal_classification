"""
    @file:              Block.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 03/2021

    @Description:       This file contain some generic module used to create several model like the ResNet,
                        MultiLevelResNet and CapsNet. The module are DynamicHighCapsule, PrimaryCapsule, Resblock and
                        ResBottleneck.
"""
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers import split_args
from monai.networks.layers.factories import Act, Norm
import numpy as np
import torch
from torch import nn
from typing import Sequence, Union


class PreResBlock(nn.Module):
    """
    A 3D version of the preactivation residual bottleneck block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    first_activation: nn.module
        The non linear activation function that is applied before the forward pass in the residual mapping.
    first_normalization: nn.module
        The normalization layer that is applied before the forward pass in the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = True,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out, kernel_size=1, stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation](**args)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=kernel, stride=strides,
                      padding=padding, bias=bias,
                      groups=groups if split_layer is False else 1),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                      padding=padding, bias=bias,
                      groups=groups)
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.first_normalization(x)
        out = self.first_activation(out)

        if self.__subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class PreResBottleneck(nn.Module):
    """
    A 3D version of the preactivation residual block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    first_activation: nn.module
        The non linear activation function that is applied before the forward pass in the residual mapping.
    first_normalization: nn.module
        The normalization layer that is applied before the forward pass in the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 4

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = False,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation](**args)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups),
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.first_normalization(x)
        out = self.first_activation(out)

        if self.__subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class ResBlock(nn.Module):
    """
    A 3D version of the residual block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    res: ResidualUnit
        A MONAI implementation of the Residual block. Act like an nn.Sequential.
    """
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "ReLU",
                 bias: bool = True,
                 drop_rate: float = 0,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a Residual Block using MONAI

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param strides: Convolution strides.
        """
        super().__init__()

        self.res = ResidualUnit(dimensions=3, in_channels=fmap_in, out_channels=fmap_out,
                                kernel_size=kernel, strides=strides, dropout=drop_rate,
                                dropout_dim=3, norm=norm, last_conv_only=False, bias=bias,
                                act=activation if activation != "PReLU" else ("prelu", {"init": 0.01}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res(x)

        return out


class ResBottleneck(nn.Module):
    """
    A 3D version of the residual bottleneck block as described in Ref 1).
    (Conv(1x1x1), Norm, Act, Conv(kernel), Norm, Act, Conv(1x1x1), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    last_activation: nn.module
        The non linear activation function that is applied after adding the shorcut to the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 4

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = True,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out*self.expansion)
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)
        self.last_activation = Act[activation](**args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        if self.__subsample:
            shortcut = self.sub_conv(x)
        else:
            shortcut = x

        out = self.residual_layer(x) + shortcut

        return self.last_activation(out)
