"""
    @file:              HardSharedResNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 03/2021

    @Description:       This file contain the classe HardSharedResNet that inherit from the NeuralNet class.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
"""

from Model.Block import PreResBlock, PreResBottleneck, ResBlock, ResBottleneck
from Model.Module import Mixup, UncertaintyLoss
from monai.networks.blocks.convolutions import Convolution
from Model.NeuralNet import NeuralNet
import numpy as np
from Trainer.Utils import init_weights
import torch
import torch.nn as nn
from typing import Dict, Sequence, Tuple, Union


class HardSharedResNet(NeuralNet):
    """
    A hard shared 3D Residual Network implementation for multi task learning.

    ...
    Attributes
    ----------
    conv: Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layer : torch.nn.ModuleDict
        A dictionnary that contain the fully connected of each task.
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
    __split: int
        Indicate where the network will be split to a multi-task network. Should be an integer between 1 and 5.
        1 indicate that the network will be split before the first series of residual block.
        5 indicate that the network will be split after the last series of residual block.
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
                 tasks: Sequence[str],
                 num_classes: Dict[str, int],
                 depth: int = 18,
                 first_channels: int = 16,
                 split_level: int = 4,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 first_kernel: Union[Sequence[int], int] = 3,
                 kernel: Union[Sequence[int], int] = 3,
                 mixup: Sequence[float] = None,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 act: str = "ReLU",
                 norm: str = "batch",
                 pre_act: bool = True):
        """
        Create a pre activation or post activation 3D Residual Network for multi-task learning.

        :param tasks: A list of tasks on which the model will be train.
        :param depth: The number of convolution and fully connected layer in the neural network. (Default=18)
        :param first_channels: The number of channels at the output of the first convolution layer. (Default=16)
        :param split_level: At which level the multi level resnet should split into sub net. (Default=4)
                                1: After the first convolution,
                                2: After the first residual level,
                                3: After the second residual level,
                                4: After the third residual level,
                                5: After the last residual level so just before the fully connected layers.
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
        assert len(tasks) > 0, "You should specify the name of each task"
        super().__init__()
        self.__split = split_level
        self.__in_channels = first_channels
        self.__tasks = tasks
        nb_task = len(tasks)

        # --------------------------------------------
        #                NUM_CLASSES
        # --------------------------------------------
        # If num_classes has not been defined, then we assume that every task are binary classification.
        if num_classes is None:
            num_classes = {}
            for task in self.__tasks:
                num_classes[task] = 2

        # If num_classes has been defined for some tasks but not all, we assume that the remaining are regression task
        else:
            key_set = set(num_classes.keys())
            tasks_set = set(self.__tasks)
            missing_tasks = tasks_set - key_set
            assert missing_tasks == (tasks_set ^ key_set), f"The following tasks are present in num_classes " \
                                                           "but not in tasks {}".format(key_set - tasks_set)
            for task in list(missing_tasks):
                num_classes[task] = 1

        # --------------------------------------------
        #                   MIXUP
        # --------------------------------------------
        assert mixup is None or len(mixup) == 4, "You should specify the 4 mixup parameters."
        mixup = [0, 0, 0, 0] if mixup is None else mixup

        for i in range(len(mixup)):
            if mixup[i] > 0:
                self.mixup[str(i)] = Mixup(mixup[i])

        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        self.uncertainty_loss = UncertaintyLoss(num_task=nb_task)

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
        assert 1 <= split_level <= 5, "The split level should be an integer between 1 and 5."
        self.conv = Convolution(dimensions=3,
                                in_channels=3,
                                out_channels=self.__in_channels,
                                kernel_size=first_kernel,
                                act=act,
                                conv_only=pre_act)

        self.layers1 = self.__make_layer(block[depth], layers[depth][0],
                                         first_channels, kernel=kernel,
                                         strides=[2, 2, 1], norm=norm,
                                         drop_rate=dropout[0], act=act,
                                         split_layer=(1 == split_level),
                                         groups=nb_task if 1 >= split_level else 1)

        self.layers2 = self.__make_layer(block[depth], layers[depth][1],
                                         first_channels * 2, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[1], act=act,
                                         split_layer=(2 == split_level),
                                         groups=nb_task if 2 >= split_level else 1)

        self.layers3 = self.__make_layer(block[depth], layers[depth][2],
                                         first_channels * 4, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[2], act=act,
                                         split_layer=(3 == split_level),
                                         groups=nb_task if 3 >= split_level else 1)

        self.layers4 = self.__make_layer(block[depth], layers[depth][3],
                                         first_channels * 8, kernel=kernel,
                                         strides=[2, 2, 2], norm=norm,
                                         drop_rate=dropout[3], act=act,
                                         split_layer=(4 == split_level),
                                         groups=nb_task if 4 >= split_level else 1)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 16), int(in_shape[1] / 16),  int(in_shape[2] / 8)]

        self.avg_pool = nn.AvgPool3d(kernel_size=out_shape)

        if self.__split == 5:
            self.__num_flat_features = self.__in_channels
        else:
            self.__num_flat_features = int(self.__in_channels / nb_task)

        self.fc_layer = nn.ModuleDict()
        for task in tasks:
            self.fc_layer[task] = torch.nn.Linear(self.__num_flat_features,
                                                  num_classes[task])

        self.apply(init_weights)

    def __make_layer(self,
                     block: Union[PreResBlock, PreResBottleneck, ResBlock, ResBottleneck],
                     num_block: int,
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     strides: Union[Sequence[int], int] = 1,
                     drop_rate: Sequence[float] = None,
                     act: str = "ReLU",
                     norm: str = "batch",
                     groups: int = 1,
                     split_layer: bool = False) -> nn.Sequential:

        fmap_out = fmap_out * groups
        layers = []
        for i in range(num_block):
            layers.append(block(fmap_in=self.__in_channels, fmap_out=fmap_out,
                                kernel=kernel,
                                strides=strides if i == 0 else 1,
                                drop_rate=drop_rate[i],
                                activation=act,
                                norm=norm,
                                groups=groups,
                                split_layer=split_layer))
            split_layer = False
            self.__in_channels = fmap_out * block.expansion if i == 0 else self.__in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        preds = {}
        if self.__split == 5:
            features = out.view(-1, self.__num_flat_features)

            for task in self.__tasks:
                preds[task] = self.fc_layer[task](features)

        else:
            features = out.view(-1, len(self.__tasks), self.__num_flat_features)

            for count, task in enumerate(self.__tasks):
                preds[task] = self.fc_layer[task](features[:, count, :])

        return preds