"""
    @file:              mtan.py
    @Author:            Alexandre Ayotte

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains the class STAN that inherit from the NeuralNet class. This Single task
                        Attention Network can only be use with single task trainer on 3D images.

    @Reference:         1) End-To-End Multi-Task Learning With Attention, Liu, S. et al, CVPR 2019
"""
import torch
import torch.nn as nn
from typing import Dict, Final, List, Sequence, Tuple, Union

from constant import AttentionBlock, BlockType, DropType
from model.block import CBAM, ChannelAttBlock, PreResBlock, PreResBottleneck, ResBlock, ResBottleneck, SpatialAttBlock
from model.neural_net import NeuralNet, init_weights
from model.resnet import ResNet

NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4
SQUEEZE_FACTOR_LIST: Final = [4, 8, 8, 8]
STRIDES: Final = [2, 2, 2]


class STAN(NeuralNet):
    """
    A 3D version of the SingleTask Attention Network adapted to classification and regression task and inspired from the
    MultiTask Attention Network describe in Ref) 1.

    ...
    Attributes
    ----------
    att_layers : nn.ModuleList
        A list of nn.ModuleDict that contain the attention module of each task for each level.
    conv : Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layers : nn.Sequential
        A dictionary of nn.Sequential that contain the pooling layer and the last fully connected of each task.
    shared_layers1_base : nn.Sequentiel
        The n-1 first block of the self.layers1 sequential in the ResNet. Its output will be use as input in the first
        attention module.
    shared_layers1_last : nn.Sequentiel
        The last block of the self.layers1 sequential in the ResNet. The mask produced by the first attention module
        will be applied on its output.
    shared_layers2_base : nn.Sequentiel
        The n-1 first block of the self.layers2 sequential in the ResNet. Its output will be use as input in the second
        attention module.
    shared_layers2_last : nn.Sequentiel
        The last block of the self.layers2 sequential in the ResNet. The mask produced by the second attention module
        will be applied on its output.
    shared_layers3_base : nn.Sequentiel
        The n-1 first block of the self.layers3 sequential in the ResNet. Its output will be use as input in the third
        attention module.
    shared_layers3_last : nn.Sequentiel
        The last block of the self.layers3 sequential in the ResNet. The mask produced by the third attention module
        will be applied on its output.
    shared_layers4_base : nn.Sequentiel
        The n-1 first block of the self.layers4 sequential in the ResNet. Its output will be use as input in the fourth
        attention module.
    shared_layers4_last : nn.Sequentiel
        The last block of the self.layers4 sequential in the ResNet. The mask produced by the fourth attention module
        will be applied on its output.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    """
    def __init__(self,
                 act: str = "ReLU",
                 att_type: AttentionBlock = SpatialAttBlock,
                 blocks_type: Union[BlockType, List[BlockType]] = BlockType.PREACT,
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: DropType = DropType.FLAT,
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 groups: int = 1,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 num_classes: int = 2,
                 num_in_chan: int = 4) -> None:
        """
        Create a pre activation or post activation 3D Residual Network.

        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param att_type: Indicate which type of attention module will be used. (Options: See AttentionBlock in
                          constant.py) (Default=AttentionBlock.SPATIAL)
        :param blocks_type: A string or a list of string that indicate the type of block that will be used at each
                            level. If only a string is gived, all blocks in the model will be of the same type.
                            (Options: see BlockType in constant.py) (Defaut=BlockType.PREACT).
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
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param num_classes: The number of features at the output of the neural network. (Default=2)
        :param num_in_chan: A positive integer that represent the number of channels of the input images.
        """
        super().__init__()

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if type(blocks_type) is BlockType:
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
                raise Exception(f"The block_type is not an option: {block_type}, see BlockType Enum in constant.py.")

        # We create the base of the network.
        basenet = ResNet(act=act, blocks_type=blocks_type, depth=depth,
                         drop_rate=drop_rate, drop_type=drop_type,
                         first_channels=first_channels, first_kernel=first_kernel,
                         groups=groups, in_shape=in_shape, kernel=kernel,
                         norm=norm, num_in_chan=num_in_chan)

        # --------------------------------------------
        #                SHARED LAYERS
        # --------------------------------------------
        self.conv = basenet.conv
        self.shared_layers1_base = basenet.layers1[:-1]
        self.shared_layers1_last = basenet.layers1[-1]

        self.shared_layers2_base = basenet.layers2[:-1]
        self.shared_layers2_last = basenet.layers2[-1]

        self.shared_layers3_base = basenet.layers3[:-1]
        self.shared_layers3_last = basenet.layers3[-1]

        self.shared_layers4_base = basenet.layers4[:-1]
        self.shared_layers4_last = basenet.layers4[-1]

        self.att_layers = nn.ModuleList()

        # --------------------------------------------
        #               ATTENTION LAYERS
        # --------------------------------------------
        num_out_chan = first_channels
        factor = 1  # factor = 1 if input is not concatenate with the last attention block output.
        for count, (block, sq_fact) in enumerate(zip(block_list, SQUEEZE_FACTOR_LIST)):

            # We create the subsample module
            num_in_chan = num_out_chan * block.expansion
            num_out_chan *= 2 if count < 3 else 1
            subsample = block(fmap_in=num_in_chan, fmap_out=num_out_chan,
                              kernel=kernel, strides=STRIDES,
                              activation=act, norm=norm)

            if att_type == AttentionBlock.CHANNEL:
                att_block = ChannelAttBlock
            elif att_type == AttentionBlock.SPATIAL:
                att_block = SpatialAttBlock
            else:
                att_block = CBAM

            self.att_layers.append(
                att_block(fmap_in=num_in_chan * block.expansion * factor,
                          fmap_out=num_in_chan*block.expansion,
                          squeeze_factor=sq_fact,
                          subsample=subsample if count < 3 else None)
            )

            factor = 2

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        num_flat_features = num_out_chan*block_list[0].expansion

        self.fc_layers = nn.Sequential(nn.AvgPool3d(kernel_size=out_shape),
                                       nn.Flatten(start_dim=1),
                                       torch.nn.Linear(num_flat_features, num_classes))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Single-Task Attention Network

        :param x: A batch of 3D images that we want to classify.
        :return: A torch.Tensor that represent the model output.
        """
        out = self.conv(x)

        out1_base = self.shared_layers1_base(out)
        out1_last = self.shared_layers1_last(out1_base)

        out2_base = self.shared_layers2_base(out1_last)
        out2_last = self.shared_layers2_last(out2_base)

        out3_base = self.shared_layers3_base(out2_last)
        out3_last = self.shared_layers3_last(out3_base)

        out4_base = self.shared_layers4_base(out3_last)
        out4_last = self.shared_layers4_last(out4_base)

        att_out = self.att_layers[0](out1_base, out1_last)
        att_out = self.att_layers[1](torch.cat((out2_base, att_out), dim=1), out2_last)
        att_out = self.att_layers[2](torch.cat((out3_base, att_out), dim=1), out3_last)
        att_out = self.att_layers[3](torch.cat((out4_base, att_out), dim=1), out4_last)

        pred = self.fc_layers(att_out)
        return pred
