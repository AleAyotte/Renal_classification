"""
    @file:              ResNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 06/2021

    @Description:       This file contain the classe ResNet that inherit from the NeuralNet class. This ResNet version
                        can only be use for single task trainer on 3D images.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
"""

from Constant import BlockType, DropType
from Model.Block import PreResBlock, PreResBottleneck, ResBlock, ResBottleneck
from Model.Module import Mixup, UncertaintyLoss
from monai.networks.blocks.convolutions import Convolution
from Model.NeuralNet import NeuralNet, init_weights
import numpy as np
import torch
import torch.nn as nn
from typing import Final, List, Sequence, Tuple, Type, Union


NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4


class ResNet(NeuralNet):
    """
    A ResNet 3D implementation for single task training.

    ...
    Attributes
    ----------
    conv : Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    __in_channels : int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
    last_layers : nn.Sequential
        A sequential that contain the average pooling and the fully connected layer.
    layers1 : nn.Sequential
        The first series of residual block.
    layers2 : nn.Sequential
        The second series of residual block.
    layers3 : nn.Sequential
        The third series of residual block.
    layers4 : nn.Sequential
        The last series of residual block.
    mixup : nn.ModuleDict
        A dictionnary that contain all the mixup module.
    __num_flat_features : int
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
                 act: str = "ReLU",
                 blocks_type: Union[BlockType, List[BlockType]] = BlockType.PREACT,
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: DropType = DropType.FLAT,
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 groups: int = 1,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 kernel: Union[Sequence[int], int] = 3,
                 mixup: Sequence[int] = None,
                 norm: str = "batch",
                 num_classes: int = 2,
                 num_in_chan: int = 4):
        """
        Create a pre activation or post activation 3D Residual Network.

        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param blocks_type: A string or a list of string that indicate the type of block that will be used at each
                            level. If only a string is gived, all blocks in the model will be of the same type.
                            (Options: see BlockType in Constant.py) (Defaut=BlockType.PREACT).
                            The 2 following example give the same result.
                            Example 1)
                                blocks_type = BlockType.PREACT
                            Example 2)
                                blocks_type = [BlockType.PREACT for _ in range(4)]
        :param depth: The number of convolution and fully connected layer in the neural network. (Default=18)
        :param drop_rate: The maximal dropout rate used to configure the dropout layer. See drop_type (Default=0)
        :param drop_type: If drop_type == 'flat' every dropout layer will have the same drop rate.
                          Else if, drop_type == 'linear' the drop rate will grow linearly at each dropout layer
                          from 0 to 'drop_rate'. (Default='Flat')
        :param first_channels: The number of channels at the output of the first convolution layer. (Default=16)
        :param first_kernel: The kernel shape of the first convolution layer. (Default=3)
        :param groups: An integer that indicate in how many groups the first convolution will be separated.
                       (Options=[1, 2]) (Default=1)
        :param in_shape: The image shape at the input of the neural network. (Default=(64, 64, 16))
        :param kernel: The kernel shape of all convolution layer except the first one. (Default=3)
        :param mixup: The The alpha parameter of each mixup module. Those alpha parameter are used to sample the
                      dristribution Beta(alpha, alpha).
        :param num_classes: The number of features at the output of the neural network. (Default=2)
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param num_classes: A positive integer that represent the number of classes on which the model will be train.
        :param num_in_chan: A positive integer that represent the number of channels of the input images.
        """
        super().__init__()

        # --------------------------------------------
        #                   MIXUP
        # --------------------------------------------
        assert mixup is None or len(mixup) == NB_LEVELS, "You should specify the 4 mixup parameters."
        mixup = [0, 0, 0, 0] if mixup is None else mixup

        for i in range(len(mixup)):
            if mixup[i] > 0:
                self.mixup[str(i)] = Mixup(mixup[i])

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if type(blocks_type) is DropType:
            block_type_list = [blocks_type for _ in range(4)]
        elif type(blocks_type) is list:
            assert len(blocks_type) == NB_LEVELS, "You should specify one or 4 pre_act parameters."
            block_type_list = blocks_type
        else:
            raise Exception(f"blocks_type should be a BlockType or a list of BlockType. get {blocks_type}")

        block_list = []
        for block_type in block_type_list:
            if block_type == BlockType.PREACT:
                block_list.append(PreResBlock if depth <= 34 else PreResBottleneck)
            elif block_type == BlockType.POSTACT:
                block_list.append(ResBlock if depth <= 34 else ResBottleneck)
            else:
                raise Exception(f"The block_type is not an option: {block_type}, see BlockType Enum in Constant.py.")

        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

        # --------------------------------------------
        #                   DROPOUT
        # --------------------------------------------
        num_block = int(np.sum(layers[depth]))

        assert type(drop_type) is DropType
        if drop_type is DropType.FLAT:
            temp = [drop_rate for _ in range(num_block)]
        elif drop_type is DropType.LINEAR:
            temp = [1 - (1 - (drop_rate * i / (num_block - 1))) for i in range(num_block)]
        else:
            raise NotImplementedError

        dropout = []
        for i in range(NB_LEVELS):
            first = int(np.sum(layers[depth][0:i]))
            last = int(np.sum(layers[depth][0:i+1]))
            dropout.append(temp[first:last])

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        self.__in_channels = first_channels
        self.conv = Convolution(dimensions=NB_DIMENSIONS,
                                in_channels=num_in_chan,
                                out_channels=self.__in_channels,
                                kernel_size=first_kernel,
                                act=act,
                                groups=groups,
                                conv_only=block_type_list[0] == BlockType.PREACT,
                                norm="batch")

        self.layers1 = self.__make_layer(block_list[0], dropout[0],
                                         first_channels, kernel,
                                         num_block=layers[depth][0],
                                         strides=[2, 2, 1], norm=norm,
                                         act=act)

        self.layers2 = self.__make_layer(block_list[1], dropout[1],
                                         first_channels * 2, kernel,
                                         num_block=layers[depth][1],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act)

        self.layers3 = self.__make_layer(block_list[2], dropout[2],
                                         first_channels * 4, kernel,
                                         num_block=layers[depth][2],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act)

        self.layers4 = self.__make_layer(block_list[3], dropout[3],
                                         first_channels * 8, kernel,
                                         num_block=layers[depth][3],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        self.__num_flat_features = self.__in_channels

        self.last_layers = nn.Sequential(nn.AvgPool3d(kernel_size=out_shape),
                                         nn.Flatten(start_dim=1),
                                         torch.nn.Linear(self.__num_flat_features, num_classes))

        self.apply(init_weights)

    def __make_layer(self,
                     block: Union[Type[PreResBlock], Type[PreResBottleneck], Type[ResBlock], Type[ResBottleneck]],
                     drop_rate: List[float],
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     num_block: int,
                     act: str = "ReLU",
                     norm: str = "batch",
                     strides: Union[Sequence[int], int] = 1) -> nn.Sequential:
        """
        Create a sequence of layer of a given class and of lenght num_block.

        :param block: A class type that indicate which block should be create in the sequence.
        :param drop_rate: A list of float that indicate the drop_rate for each block.
        :param fmap_out: fmap_out*block.expansion equal the number of output feature maps of each block.
        :param kernel: An integer or a list of integer that indicate the convolution kernel size.
        :param num_block: An integer that indicate how many block will contain the sequence.
        :param act: The activation function that will be used in each block.
        :param norm: The normalization layer that will be used in each block.
        :param strides: An integer or a list of integer that indicate the strides of the first convolution of the
                        first block.
        :return: A nn.Sequential that represent the sequence of layer.
        """
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

        out = self.last_layers(out)

        return out
