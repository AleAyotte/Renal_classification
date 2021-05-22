"""
    @file:              ResNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       This file contain the classe ResNet that inherit from the NeuralNet class. This ResNet version
                        can only be use for single task trainer on 3D images.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
"""

from Model.Block import PreResBlock, PreResBottleneck, ResBlock, ResBottleneck
from Model.Module import Mixup, UncertaintyLoss
from monai.networks.blocks.convolutions import Convolution
from Model.NeuralNet import NeuralNet, init_weights
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union


class ResNet(NeuralNet):
    """
    A ResNet 3D implementation for single task training.

    ...
    Attributes
    ----------
    conv: Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layer: nn.Linear
        The last fully connected layer.
    __in_channels: int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
    layers1: nn.Sequential
        The first series of residual block.
    layers2: nn.Sequential
        The second series of residual block.
    layers3: nn.Sequential
        The third series of residual block.
    layers4: nn.Sequential
        The last series of residual block.
    mixup: nn.ModuleDict
        A dictionnary that contain all the mixup module.
    __num_flat_features: int
        Number of features at the output of the last convolution.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    set_mixup(b_size : int)
        Set the b_size parameter of each mixup module.
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    """
    def __init__(self,
                 depth: int = 18,
                 first_channels: int = 16,
                 groups: int = 1,
                 num_classes: int = 2,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 first_kernel: Union[Sequence[int], int] = 3,
                 kernel: Union[Sequence[int], int] = 3,
                 mixup: Sequence[int] = None,
                 num_in_chan: int = 4,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 act: str = "ReLU",
                 norm: str = "batch",
                 pre_act: bool = True):
        """
        Create a pre activation or post activation 3D Residual Network.

        :param depth: The number of convolution and fully connected layer in the neural network. (Default=18)
        :param first_channels: The number of channels at the output of the first convolution layer. (Default=16)
        :param num_classes: The number of features at the output of the neural network. (Default=2)
        :param in_shape: The image shape at the input of the neural network. (Default=(64, 64, 16))
        :param first_kernel: The kernel shape of the first convolution layer. (Default=3)
        :param kernel: The kernel shape of all convolution layer except the first one. (Default=3)
        :param mixup: The The alpha parameter of each mixup module. Those alpha parameter are used to sample the
                      dristribution Beta(alpha, alpha).
        :param drop_rate: The maximal dropout rate used to configure the dropout layer. See drop_type (Default=0)
        :param drop_type: If drop_type == 'flat' every dropout layer will have the same drop rate.
                          Else if, drop_type == 'linear' the drop rate will grow linearly at each dropout layer
                          from 0 to 'drop_rate'. (Default='Flat')
        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param pre_act: If true, the PreResBlock or the PreResBottleneck will be used instead of ResBlock or
                        ResBottleneck. (Defaut=True)
        """
        super().__init__()

        # --------------------------------------------
        #                   MIXUP
        # --------------------------------------------
        assert mixup is None or len(mixup) == 4, "You should specify the 4 mixup parameters."
        mixup = [0, 0, 0, 0] if mixup is None else mixup

        for i in range(len(mixup)):
            if mixup[i] > 0:
                self.mixup[str(i)] = Mixup(mixup[i])

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if pre_act:
            block = {18: PreResBlock, 34: PreResBlock, 50: PreResBottleneck, 101: PreResBottleneck}
        else:
            block = {18: ResBlock, 34: ResBlock, 50: ResBottleneck, 101: ResBottleneck}

        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

        # --------------------------------------------
        #                   DROPOUT
        # --------------------------------------------
        num_block = int(np.sum(layers[depth]))

        if drop_type.lower() == "flat":
            temp = [drop_rate for _ in range(num_block)]
        elif drop_type.lower() == "linear":
            temp = [1 - (1 - (drop_rate * i / (num_block - 1))) for i in range(num_block)]
        else:
            raise NotImplementedError

        dropout = []
        for i in range(4):
            first = int(np.sum(layers[depth][0:i]))
            last = int(np.sum(layers[depth][0:i+1]))
            dropout.append(temp[first:last])

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        self.__in_channels = first_channels
        self.conv = Convolution(dimensions=3,
                                in_channels=num_in_chan,
                                out_channels=self.__in_channels,
                                kernel_size=first_kernel,
                                act=act,
                                groups=groups,
                                conv_only=pre_act)

        self.layers1 = self.__make_layer(block[depth], layers[depth][0],
                                         first_channels, kernel=kernel,
                                         strides=[2, 2, 1], norm=norm,
                                         drop_rate=dropout[0], act=act)

        self.layers2 = self.__make_layer(block[depth], layers[depth][1],
                                         first_channels * 2, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[1], act=act)

        self.layers3 = self.__make_layer(block[depth], layers[depth][2],
                                         first_channels * 4, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[2], act=act)

        self.layers4 = self.__make_layer(block[depth], layers[depth][3],
                                         first_channels * 8, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[3], act=act)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 16), int(in_shape[1] / 16),  int(in_shape[2] / 8)]

        self.avg_pool = nn.AvgPool3d(kernel_size=out_shape)

        self.__num_flat_features = self.__in_channels

        self.fc_layer = nn.Linear(self.__num_flat_features, num_classes)

        torch.manual_seed(66)
        self.apply(init_weights)

    def __make_layer(self,
                     block: Union[PreResBlock, PreResBottleneck, ResBlock, ResBottleneck],
                     num_block: int,
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     strides: Union[Sequence[int], int] = 1,
                     drop_rate: Sequence[float] = None,
                     act: str = "ReLU",
                     norm: str = "batch") -> nn.Sequential:
        layers = []

        for i in range(num_block):
            layers.append(block(fmap_in=self.__in_channels, fmap_out=fmap_out,
                                kernel=kernel,
                                strides=strides if i == 0 else 1,
                                drop_rate=drop_rate[i],
                                activation=act,
                                norm=norm))
            self.__in_channels = fmap_out * block.expansion if i == 0 else self.__in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixup_key_list = list(self.mixup.keys())

        out = self.mixup["0"](x) if "0" in mixup_key_list else x
        out = self.conv(out)
        out = self.layers1(out)

        out = self.mixup["1"](out) if "1" in mixup_key_list else out
        out = self.layers2(out)

        out = self.mixup["2"](out) if "2" in mixup_key_list else out
        out = self.layers3(out)

        out = self.mixup["3"](out) if "3" in mixup_key_list else out
        out = self.layers4(out)
        out = self.avg_pool(out)

        features = out.view(-1, self.__num_flat_features)
        out = self.fc_layer(features)

        return out
