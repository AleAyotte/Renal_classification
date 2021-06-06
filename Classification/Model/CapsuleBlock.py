"""
    @file:              CapsuleBlock.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 06/2021

    @Description:       This file contain blocks and function that will be used to build a CapsNet.

    @Reference:         1) Wasserstein Routed Capsule Networks, Fuchs, A. and Pernkopf, .F, Arxiv 2020
"""

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
    Based on ref 1) that proposed the tilt non-linearity to replace the squash function.

    :param capsules: A tensor that represent the capsules to normalize.
    :param dim: Along which dimension should we normalize the capsules.
    :return: A torch.Tensor that represent the normalized capsules.
    """
    rotated_caps = (1+F.softmax(capsules, dim=dim))/2
    return torch.mul(capsules, rotated_caps)
