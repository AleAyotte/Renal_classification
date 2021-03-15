import copy
from Model.Module import Mixup, PreResBlock, PreResBottleneck, ResBlock, ResBottleneck, UncertaintyLoss
from Model.NeuralNet import NeuralNet
from monai.networks.blocks.convolutions import Convolution
import numpy as np
from Trainer.Utils import init_weights
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union


NB_TASK = 2


class MultiLevelResNet(NeuralNet):
    """
    A  multiLevel 3D Residual Network implementation for multi task learning.

    ...
    Attributes
    ----------
    backend_task1 : bool
        If True the common layers will be optimizes for the task 1. Else, they will be optimizes for the task 2.
    common_layers : nn.Sequential
        The layers that are shared by both tasks.
    __in_channels : int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
    __num_flat_features : int
        Number of features at the output of the last convolution.
    __split : int
        Indicate where the network will be split to a multi-task network. Should be an integer between 1 and 5.
        1 indicate that the network will be split before the first series of residual block.
        5 indicate that the network will be split after the last series of residual block.
    task1_fc_layer : nn.Linear
        The last fully connected layer that will be used to classify the first task.
    task2_fc_layer : nn.Linear
        The first fully connected layer that will be used to classify the second task.
    task1_layers : nn.Sequential:
        The convolutionnal layers that will be used to extract features for the first task.
    task2_layers : nn.Sequential:
        The convolutionnal layers that will be used to extract features for the second task.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    """
    def __init__(self,
                 act: str = "ReLU",
                 backend_task1: bool = True,
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 pre_act: bool = True,
                 split_level: int = 4):
        """
        Create a pre activation or post activation 3D Residual Network for multi-task learning.

        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param backend_task1: If True the common layers will be optimizes for the task 1. Else, they will be
                              optimizes for the task 2.
        :param depth: The number of convolution and fully connected layer in the neural network. (Default=18)
        :param drop_rate: The maximal dropout rate used to configure the dropout layer. See drop_type (Default=0)
        :param drop_type: If drop_type == 'flat' every dropout layer will have the same drop rate.
                          Else if, drop_type == 'linear' the drop rate will grow linearly at each dropout layer
                          from 0 to 'drop_rate'. (Default='Flat')
        :param first_channels: The number of channels at the output of the first convolution layer. (Default=16)
        :param first_kernel: The kernel shape of the first convolution layer. (Default=3)
        :param in_shape: The image shape at the input of the neural network. (Default=(64, 64, 16))
        :param kernel: The kernel shape of all convolution layer except the first one. (Default=3)
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param pre_act: If true, the PreResBlock or the PreResBottleneck will be used instead of ResBlock or
                        ResBottleneck. (Defaut=True)
        :param split_level: At which level the multi level resnet should split into sub net. (Default=4)
                                1: After the first convolution,
                                2: After the first residual level,
                                3: After the second residual level,
                                4: After the third residual level,
                                5: After the last residual level so just before the fully connected layers.
        """
        super().__init__()
        self.__split = split_level
        self.__in_channels = first_channels
        self.__backend_task1 = backend_task1
        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        self.uncertainty_loss = UncertaintyLoss(num_task=NB_TASK)

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if pre_act:
            block = {18: PreResBlock, 34: PreResBlock, 50: PreResBottleneck, 101: PreResBottleneck}
        else:
            block = {18: ResBlock, 34: ResBlock}

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
        common_layers = []
        task1_layers = []
        task2_layers = []

        assert 1 <= split_level <= 5, "The split level should be an integer between 1 and 5."
        common_layers.append(Convolution(dimensions=3,
                                         in_channels=3,
                                         out_channels=self.__in_channels,
                                         kernel_size=first_kernel,
                                         act=act,
                                         conv_only=pre_act))

        for i in range(4):
            strides = [2, 2, 1] if i == 0 else [2, 2, 2]
            layers = self.__make_layer(block[depth], layers[depth][i],
                                       first_channels * (2**i), kernel=kernel,
                                       strides=strides, norm=norm,
                                       drop_rate=dropout[i], act=act)
            if split_level < i+1:
                common_layers.append(layers)
            else:
                task1_layers.append(layers)
                task2_layers.append(copy.deepcopy(layers))

        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 16), int(in_shape[1] / 16), int(in_shape[2] / 8)]

        task1_layers.append(nn.AvgPool3d(kernel_size=out_shape))
        task2_layers.append(nn.AvgPool3d(kernel_size=out_shape))

        self.common_layers = nn.Sequential(*common_layers)
        self.task1_layers = nn.Sequential(*task1_layers)
        self.task2_layers = nn.Sequential(*task2_layers)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        self.__num_flat_features = self.__in_channels
        self.task1_fc_layer = torch.nn.Sequential(torch.nn.Linear(self.__num_flat_features, 2))
        self.task2_fc_layer = torch.nn.Sequential(torch.nn.Linear(self.__num_flat_features, 2))

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.common_layers(x)

        # We split the neural network in two.
        out_task1 = self.task1_layers(out if self.__backend_task1 else out.detach())
        out_task2 = self.task2_layers(out.detach() if self.__backend_task1 else out)

        task1_features = out_task1.view(-1, self.__num_flat_features)
        task2_features = out_task2.view(-1, self.__num_flat_features)

        task1_pred = self.task1_fc_layer(task1_features)
        task2_pred = self.task2_fc_layer(task2_features)

        return task1_pred, task2_pred
