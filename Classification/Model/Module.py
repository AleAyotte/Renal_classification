"""
    @file:              Module.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       This file contain some generic module used to create several model like
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
import torch
from torch import nn
from torch.nn import functional as F
from Trainer.Utils import to_one_hot
from typing import Sequence, Tuple


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


class MarginLoss(nn.Module):
    """
    An implementation of the margin loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        labels = to_one_hot(labels, num_classes=2)
        left = F.relu(0.9 - pred, inplace=True) ** 2
        right = F.relu(pred - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        return margin_loss.mean()


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
            lamb = torch.Tensor([self.lamb]).float().to(device)
            lamb.requires_grad = True
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
