"""
    @file:              Module.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       This file contain some generic module used to create several model like the ResNet,
                        MultiLevelResNet and SharedNet. The module are SluiceUnit, CrossStitchUnit, Mixup and
                        UncertaintyLoss.

    @Reference:         1) I. Misra, et al. Cross-stitch networks formulti-task learning. IEEE Conference on Computer
                           Vision and PatternRecognition (CVPR), 2016
                        2) S. Ruder et al. Latent Multi-Task Architecture Learning. Proceedings of the AAAI
                           Conferenceon Artificial Intelligence, 2019
                        3) R. Cipolla et al. Multi-task Learning UsingUncertainty to Weigh Losses for Scene Geometry
                           and Semantics. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018
"""

import numpy as np
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Act, Norm
import torch
from torch import nn
from typing import Tuple, Sequence, Union


class CrossStitchUnit(torch.nn.Module):
    """
    A cross stitch unit implementation as described in Ref 1).

    ...
    Attributes
    ----------
    alpha: torch.nn.Parameter
        A torch Tensor that represent parameters of the Cross-Stitch Unit. Those parameters are
        a matrix NxN that learned how the information between N subspace should be shared.
    """
    def __init__(self,
                 nb_channels: int,
                 nb_task: int,
                 c=0.9,
                 spread=0.1):
        """

        :param nb_task: The number of network to combine.
        :param nb_channels: Number of channels in the latent space PER NETWORK.
        :param c: A float that represent the conservation parameter.
        :param spread: A float that represent the spread parameters.
        """
        super().__init__()
        mean = (1 - c) / (nb_task - 1)
        std = spread / (nb_task - 1)

        alpha = []
        for t in range(nb_task):
            temp = np.random.normal(mean, std, (nb_channels, nb_task))
            temp[:, t] = c
            alpha.append(temp)

        alpha = torch.from_numpy(np.array(alpha)).float()
        alpha = torch.transpose(alpha, 1, 2)  # Output, Task, Channels

        self.alpha = torch.nn.Parameter(data=alpha, requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        # Output, Task, Batch, Channels, Depth, Width, Height
        out = torch.mul(self.alpha[:, :, None, :, None, None, None], x[None, :, :, :, :, :, :])
        return out.sum(1)


class Mixup(torch.nn.Module):
    """
    An implementation of a Mixup module that can be used to create both input mixup and manifold mixup.

    ...
    Attributes
    ----------
    __beta: float
        The parameter of beta distribution. The lamb parameter will be sample in a distribution Beta(__beta, __beta)
    lamb: float
        The parameter that will be used to compute the simplex between two images.
    __batch_size: int
        The length of the image batch that will mix.
    permut: Sequence[int]
        The permutation indices that will be used to shuffle the features before the next foward pass.
    enable: bool
        If true, the current module will mix the next batch of images.
    Methods
    -------
    get_mix_params() -> Tuple[float, Sequence[int]]:
        Return the last sampled lambda parameter that will used in the forward pass to perform a linear combination
        on the data of same batch and the permutation index list that indicate how the data will be combine.
    sample() -> Tuple[float, Sequence[int]]:
        Sample a point in a beta distribution and define the permutation index list to prepare the mixup module.
    set_batch_size(b_size: int) -> None:
        Change the value of the __batch_size attribut.
    """
    def __init__(self, beta_params):
        """
        The constructor of a mixup module.
        :param beta_params: One single value for the two parameters of a beta distribution
        """

        super().__init__()
        self.__beta = beta_params
        self.lamb = 1
        self.__batch_size = 0
        self.permut = None
        self.enable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mixup module. This is here were we mix the data.

        :return: Return mixed data if the network is in train mode and the module is enable.
        """

        if self.training and self.enable:
            device = x.get_device()
            lamb = torch.from_numpy(np.array([self.lamb]).astype('float32')).to(device)
            lamb = torch.autograd.Variable(lamb)
            return lamb*x + (1 - lamb)*x[self.permut.to(device)]
        else:
            return x

    def get_mix_params(self) -> Tuple[float, Sequence[int]]:
        """
        Return the last sampled lambda parameter that will used in the forward pass to perform a linear combination
        on the data of same batch and the permutation index list that indicate how the data will be combine.

        :return: The constant that will be use to mixup the data for the next iteration and a list of index that
                 represents the permutation used for the mixing process.
        """

        return self.lamb, self.permut

    def sample(self) -> Tuple[float, Sequence[int]]:
        """
        Sample a point in a beta distribution and define the permutation index list to prepare the mixup module.

        :return: The coefficient used to mixup the training features during the next foward pass.
                 The permutation indices that will be used to shuffle the features before the next foward pass.
        """

        if self.__beta > 0:
            # We activate the module and we sample a value in his beta distribution
            self.enable = True
            self.lamb = np.random.beta(self.__beta, self.__beta)
        else:
            self.lamb = 1

        self.permut = torch.randperm(self.__batch_size)

        return self.lamb, self.permut

    def set_batch_size(self, b_size: int) -> None:
        """
        Change the value of the __batch_size attribut.

        :param b_size: A integer that indicate the size of the next mini_batch of data.
        """
        self.__batch_size = b_size


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

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation]()

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=kernel, stride=strides,
                      padding=padding, bias=False,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](),
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

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation]()

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=False,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=False,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](),
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
                                dropout_dim=3, act=activation, norm=norm,
                                last_conv_only=False)

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

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=False,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=False,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=False,
                      groups=groups),
            Norm[norm, 3](fmap_out*self.expansion)
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)
        self.last_activation = Act[activation]()

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


class SluiceUnit(torch.nn.Module):
    """
    An implementation of the sluice unit as described in Ref 2)

    ...
    Attributes
    ----------
    alpha: torch.nn.Parameter
        A torch Tensor that represent parameters of the Sluice Unit. Those parameters are
        a matrix NxN that learned how the information between N subspace should be shared.
    """

    def __init__(self,
                 nb_subspace: int,
                 c: float = 0.9,
                 spread: float = 0.1):
        """
        Initialize a sluice units.

        :param nb_subspace: The number of subspace that will be shared.
        :param c: A float that represent the conservation parameter.
        :param spread: A float that represent the spread parameters.
        """
        super().__init__()
        mean = (1 - c) / (nb_subspace - 1)
        std = spread / (nb_subspace - 1)

        alpha = np.random.normal(mean, std, (nb_subspace, nb_subspace))
        alpha[np.diag_indices_from(alpha)] = c

        alpha = torch.from_numpy(alpha).float()
        self.alpha = torch.nn.Parameter(data=alpha, requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        # Batch, Subspace, Channels, Depth, Height, Width -> Batch, Width, Channels, Depth, Height, Subspace
        x = x.transpose(1, x.ndim - 1)  # We transpose the subspace dimension with the last dimension
        out = torch.matmul(x[:, :, :, :, :, None, :], self.alpha).squeeze()

        # Batch, Width, Channels, Depth, Height, Subspace -> Batch, Subspace, Channels, Depth, Height, Width
        return out.transpose(1, x.ndim - 1)


class UncertaintyLoss(torch.nn.Module):
    """
    An implementation of the Uncertainty loss as described in Ref 3). This loss is used to scale and combine the loss
    of different task in a multi-task learning.

    ...
    Attributes
    ----------
    phi: torch.nn.Parameter

    """
    def __init__(self, num_task):
        """
        Initialize a Uncertainty loss module.

        :param num_task: the number of task that will be combined with the uncertainty loss.
        """
        super().__init__()
        self.phi = torch.nn.Parameter(data=torch.zeros(num_task), requires_grad=True)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute the uncertainty loss

        :param losses: A torch.Tensor that represent the vector of lenght 3 that contain the losses.
        :return: A torch.Tensor that represent the uncertainty loss (multi-task loss).
        """
        return torch.dot(torch.exp(-self.phi), losses) + torch.sum(self.phi / 2)
