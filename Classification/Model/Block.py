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
from torch.nn import functional as F
from typing import Sequence, Union


class DynamicHighCapsule(nn.Module):
    """
    An implementation of an high level capsule layer that use the dynamic routing-by-agreements.

    ...
    Attributes
    ----------
    __max_min_norm : bool
        If true, the routing-by-agreements has describe in ref 2) will be used. Else, the standard dynamical
        routing-by-agreements define by sabour et al. (2017) will be used.
    __num_iter : int
        The number of iteration in the dynamical routing-by-agreements
    priors_weights : torch.nn.Parameter
        The weight that will be used to compute the priors probability u_{j|i} with the inputs capsules.
    """
    def __init__(self,
                 in_caps: int,
                 in_caps_dim: int,
                 out_caps: int,
                 out_caps_dim: int,
                 max_min_norm: bool = False,
                 num_routing_iter: int = 3):
        """
        The constructor of the DynamicHighCapsule class

        :param in_caps: Number of input capsules.
        :param in_caps_dim: Dimension of the input capsules.
        :param out_caps: The number of outputs capsules.
        :param out_caps_dim: The dimension of the outputs capsules.
        :param max_min_norm: If True, the routing-by-agreements describe in ref 2) will be used
                             the routing-by-agreements
        :param num_routing_iter: The number of iteration in the dynamical routing-by-agreements.
        """

        super().__init__()
        assert num_routing_iter > 0, "The number of routing iter should be greater than 0."

        self.__max_min_norm = max_min_norm
        self.__num_iter = num_routing_iter
        self.priors_weights = nn.Parameter(torch.randn(out_caps, in_caps, in_caps_dim, out_caps_dim),
                                           requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        priors = torch.matmul(
            x[:, None, :, None, :],
            self.priors_weights[None, :, :, :, :]
        ).transpose(2, 4)
        coupling_coef = torch.zeros(*priors.size()).cuda()

        # Min max normalization
        if self.__max_min_norm:
            probs = torch.ones(*priors.size()).cuda()

            # The routing by agreements
            with torch.no_grad():
                for it in range(self.__num_iter - 1):
                    out = squash((probs * priors).sum(dim=-1, keepdims=True), dim=2)

                    delta_logits = (priors * out).sum(dim=2, keepdims=True)
                    coupling_coef = coupling_coef + delta_logits

                    # Max-Min Normalization
                    min_coeff = coupling_coef.min(dim=-1, keepdim=True).values
                    max_coeff = coupling_coef.max(dim=-1, keepdim=True).values
                    probs = (coupling_coef - min_coeff) / (max_coeff - min_coeff)

        # Softmax normalization
        else:
            # The routing by agreements
            with torch.no_grad():
                for it in range(self.__num_iter - 1):
                    probs = F.softmax(coupling_coef, dim=-1)
                    out = squash((probs * priors).sum(dim=-1, keepdims=True), dim=2)

                    delta_logits = (priors * out).sum(dim=2, keepdims=True)
                    coupling_coef = coupling_coef + delta_logits

            probs = F.softmax(coupling_coef, dim=-1)
        return squash((probs * priors).sum(dim=-1).squeeze())


def squash(capsules: torch.Tensor,
           dim: int = -1) -> torch.Tensor:
    """
    Reduce the norm of a vector between 0 and 1 without changing the orientation.

    :param capsules: A tensor that represent the capsules to normalize.
    :param dim: Along which dimension should we normalize the capsules.
    :return: A torch.Tensor that represent the normalized capsules.
    """
    norm = torch.linalg.norm(capsules, dim=dim, keepdim=True)
    return norm / (1 + norm ** 2) * capsules


def tilt(capsules: torch.Tensor,
         dim: int = -1) -> torch.Tensor:
    """
    Based on @"Wasserstein Routed Capsule Networks" that proposed the tilt non-linearity to replace the squash function.
    :param capsules:
    :param dim:
    :return:
    """
    rotated_caps = (1+F.softmax(capsules, dim=dim))/2
    return torch.mul(capsules, rotated_caps)


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
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 groups: int = 1,
                 split_layer: bool = False,
                 drop_rate: float = 0,
                 activation: str = "relu",
                 norm: str = "batch"):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param groups: Number of group in the convolutions.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used in the model.
        :param norm: The normalization layer name that will be used in the model.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out, kernel_size=1, stride=strides, bias=False,
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
                      padding=padding, bias=False,
                      groups=groups if split_layer is False else 1),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                      padding=padding, bias=False,
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
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 groups: int = 1,
                 split_layer: bool = False,
                 drop_rate: float = 0,
                 activation: str = "relu",
                 norm: str = "batch"):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param groups: Number of group in the convolutions.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used in the model.
        :param norm: The normalization layer name that will be used in the model.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=False,
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
                      stride=1, bias=False,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=False,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=False,
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


class PrimaryCapsule(nn.Module):
    """
    An implementation of a primary capsule layer.

    ...
    Attributes
    ----------
    __caps_dim : int
        The lenght of each capsule.
    conv : Union[torch.nn.Conv2d, torch.nn.Conv3d]
        The convolution layer that will be used to transform an input images into capsules.
    """
    def __init__(self,
                 dimension: int,
                 caps_dim: int,
                 in_channels: int,
                 kernel: Union[int, Sequence[int]],
                 num_primary_caps: int,
                 padding: Union[Sequence[int], int] = 0,
                 stride: int = 1):
        """
        The construtor of the PrimaryCapsule class

        :param dimension: The number of dimension of the convolution.
        :param caps_dim: The output dimension of the capsules.
        :param in_channels: The number of input channels for the convolution.
        :param kernel: The convolution's kernel size.
        :param num_primary_caps: The number of primary capsule block. The number of outputs channels will be
                                 equal to num_primary_caps * caps_dim.
        :param padding: The convolution's padding size.
        :param stride: The convolution stride parameter.
        """

        super().__init__()
        assert dimension in [2, 3], "The PrimaryCapule layer can only be of dimension 2 or 3."

        self.__caps_dim = caps_dim

        if dimension == 2:
            self.conv = nn.Conv2d(kernel_size=kernel, in_channels=in_channels,
                                  out_channels=num_primary_caps * caps_dim,
                                  stride=stride, padding=padding)
        else:
            self.conv = nn.Conv3d(kernel_size=kernel, in_channels=in_channels,
                                  out_channels=num_primary_caps * caps_dim,
                                  stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).view(x.size()[0], self.__caps_dim, -1).transpose(1, 2)
        return squash(out)


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
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 drop_rate: float = 0,
                 activation: str = "ReLU",
                 norm: str = "batch"):
        """
        Create a PreActivation Residual Block using MONAI

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used
        :param norm: The normalization layer name that will be used in the model.
        """
        super().__init__()

        self.res = ResidualUnit(dimensions=3, in_channels=fmap_in, out_channels=fmap_out,
                                kernel_size=kernel, strides=strides, dropout=drop_rate,
                                dropout_dim=3, norm=norm, last_conv_only=False,
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
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 groups: int = 1,
                 split_layer: bool = False,
                 drop_rate: float = 0,
                 activation: str = "relu",
                 norm: str = "batch"):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param groups: Number of group in the convolutions.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used in the model.
        :param norm: The normalization layer name that will be used in the model.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=False,
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
                      stride=1, bias=False,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=False,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=False,
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
