"""
    @file:              CapsNet2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 03/2021

    @Description:       This file contain the CapsNet2D class that inherit from the NeuralNet class.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
                        2) Capsule Networks with Max-Min Normalization, Zhao et al. arxiv 2019
"""
from Model.CapsuleBlock import PrimaryCapsule, DynamicHighCapsule
from Model.NeuralNet import NeuralNet
from monai.networks.layers.factories import Act
import numpy as np
from Trainer.Utils import init_weights, to_one_hot
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Union


class CapsNet2D(NeuralNet):
    """
    An implementation of the CapsNet2D that use the Dynamical routing-by-agreements.

    ...
    Attributes
    ----------
    conv : torch.nn.Sequential
        The first convolution layer in the network followed by an activation function.
    prim : PrimaryCapsule
        The primary capsule layer
    high_layer : torch.nn.Sequential
        All the high level capsule layers in the network.
    """
    def __init__(self,
                 in_shape: Sequence[int],
                 activation: str = "ReLU",
                 first_kernel: Union[Sequence[int], int] = 9,
                 high_caps_dim: Union[Sequence[int], int] = 16,
                 num_high_caps: Union[Sequence[int], int] = 2,
                 num_prim_caps: int = 32,
                 num_routing_iter: int = 3,
                 prim_caps_kernel: Union[Sequence[int], int] = 9,
                 prim_caps_dim: int = 8,
                 out_channels: int = 256):
        """
        The CapsNet 2d construtor

        :param in_shape: The data input shape.
        :param activation: A string that represent the activation function that will be used in the NeuralNet.
                           (Default=ReLU)
        :param first_kernel: The kernel shape of the first convolution layer. (Default=9)
        :param high_caps_dim: A list of int where each element represent the dimension of the output capsule of a
                              high level capsule layer. (Default=16)
        :param num_high_caps: A list of int where each elemet represent the number of output capsule of a high level
                              capsule layer. Should be the same lenght as high_caps_dim. (Default=16)
        :param num_prim_caps: The number of capsules at the output of the primary capsule layer. (Default=32)
        :param num_routing_iter: The number of iteration in the routing-by-agreements algorithm. (Default=3)
        :param prim_caps_kernel: The kernel size of the convolution that will be used in the primary capsule layer.
                                 (Default=9)
        :param prim_caps_dim: The number of dimension for the output capsules of the primary capsule layer. (Default=8)
        :param out_channels: The number of output channels of the first convolution.
        """
        super().__init__()

        h_caps_dim = high_caps_dim if type(high_caps_dim) == list else [high_caps_dim]
        num_h_caps = num_high_caps if type(num_high_caps) == list else [num_high_caps]

        assert len(h_caps_dim) == len(num_h_caps), "You need to specify the number of capsule and the capsule " \
                                                   "dimension for each high level capsule layer."

        image_shape = np.array(in_shape[1:])

        self.conv = nn.Sequential(nn.Conv2d(kernel_size=first_kernel,
                                            in_channels=in_shape[0],
                                            out_channels=out_channels),
                                  Act[activation.lower()]())
                                  
        image_shape -= (first_kernel - 1)

        self.prim = PrimaryCapsule(2, caps_dim=prim_caps_dim,
                                   in_channels=out_channels, kernel=prim_caps_kernel,
                                   num_primary_caps=num_prim_caps, stride=2)

        image_shape -= (prim_caps_kernel - 1)
        image_shape = np.ceil(image_shape / 2).astype(int)
        num_capsule = np.prod(image_shape) * num_prim_caps
        caps_dim = prim_caps_dim

        high_layer = []
        for i in range(len(h_caps_dim)):
            high_layer.append(DynamicHighCapsule(num_capsule,
                                                 in_caps_dim=caps_dim,
                                                 out_caps=num_h_caps[i],
                                                 out_caps_dim=h_caps_dim[i],
                                                 num_routing_iter=num_routing_iter))
            num_capsule = num_h_caps[i]
            caps_dim = h_caps_dim[i]

        self.high_layer = nn.Sequential(*high_layer)

        self.apply(init_weights)

    def forward(self, images):
        if images.dim() == 5:
            images = images[:, 0, :, :, :]
        else:
            images = images[0, :, :, :].unsqueeze(dim=0)

        out = self.conv(images)
        out = self.prim(out)
        out = self.high_layer(out)

        # print(out)
        pred = torch.linalg.norm(out, dim=-1)
        return F.softmax(pred, dim=-1)
