"""
    @file:              HardSharedResNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 06/2021

    @Description:       This file contain the classe HardSharedResNet that inherit from the NeuralNet class.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
"""

from Constant import BlockType, Tasks
from Model.Block import PreResBlock, PreResBottleneck, ResBlock, ResBottleneck
from Model.Module import Mixup, UncertaintyLoss
from monai.networks.blocks.convolutions import Convolution
from Model.NeuralNet import NeuralNet, init_weights
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Final, List, Optional, Sequence, Tuple, Type, Union


NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4


class HardSharedResNet(NeuralNet):
    """
    A hard shared 3D Residual Network implementation for multi task learning.

    ...
    Attributes
    ----------
    __tasks : List[str]
        The list of tasks on which the model will be train.
    __backend_tasks : List[str]
        The list of tasks on which the shared layers of the model will be train.
    __in_channels : int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
        The last series of residual block.
    __num_flat_features : int
        Number of features at the output of the last convolution.
    shared_layers : nn.Sequentiel
        A sequence of neurat network layer that will used for every task.
    __split : int
        Indicate where the network will be split to a multi-task network. Should be an integer between 1 and 5.
        1 indicate that the network will be split before the first series of residual block.
        5 indicate that the network will be split after the last series of residual block.
    tasks_layers : nn.ModuleDict[str, nn.Sequential]
        A dictionnary of Sequential module where the key is a task name and the value is a sequence of layer that will
        be used only for this task.
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
                 num_classes: Dict[str, int],
                 tasks: Sequence[str],
                 act: str = "ReLU",
                 backend_tasks: Optional[str] = None,
                 commun_block: Union[Sequence[str], str] = "PreAct",
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 num_in_chan: int = 4,
                 task_block: Union[Dict[str, Union[str, Sequence[str]]], str] = "PreAct",
                 split_level: int = 3):
        """
        Create a pre activation or post activation 3D Residual Network for multi-task learning.

        :param num_classes: A dictionnary that indicate the number of class for each task. For regression tasks,
                            the num_class shoule be equal to one.
        :param tasks: A list of tasks on which the model will be train.
        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param backend_tasks: The list of tasks on which the shared layers of the model will be train.
        :param commun_block: A string or a sequence of string that indicate which type of block will be use to
                             create the shared layers of the network. If it is a list, it should be equal in length to
                             split_level - 1.
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
        :param num_in_chan: An integer that indicate the number of channel of the input images.
        :param task_block: A dictionary of string, a dictionary of sequence of string or a string that indication which
                           block type will be used to create the task specifics layers. If a dictionary is used, then
                           they keys should be the task name. The 3 following example give the same result.
                           Example 1)
                            task_block = "preact"
                           Example 2)
                            task_block = {"malignancy": "preact", "subtype": "preact"}
                           Example 3)
                            task_block = {"malignancy": ["preact", "preact"], "subtype": ["preact", "preact"]}
        :param split_level: At which level the multi level resnet should split into sub net. (Default=4)
                                1: After the first convolution,
                                2: After the first residual level,
                                3: After the second residual level,
                                4: After the third residual level,
                                5: After the last residual level so just before the fully connected layers.
        """
        assert len(tasks) > 0, "You should specify the name of each task"
        super().__init__()
        self.__split = split_level
        self.__in_channels = first_channels
        self.__tasks = tasks
        self.__backend_tasks = self.__tasks if backend_tasks is None else backend_tasks

        assert set(self.__backend_tasks) <= set(self.__tasks), "Every backend tasks should be in tasks."
        nb_task = len(tasks)

        # --------------------------------------------
        #                NUM_CLASSES
        # --------------------------------------------
        # If num_classes has not been defined, then we assume that every task are binary classification.
        if num_classes is None:
            num_classes = {}
            for task in self.__tasks:
                num_classes[task] = Tasks.CLASSIFICATION

        # If num_classes has been defined for some tasks but not all, we assume that the remaining are regression task
        else:
            key_set = set(num_classes.keys())
            tasks_set = set(self.__tasks)
            missing_tasks = tasks_set - key_set
            assert missing_tasks == (tasks_set ^ key_set), f"The following tasks are present in num_classes " \
                                                           "but not in tasks {}".format(key_set - tasks_set)
            for task in list(missing_tasks):
                num_classes[task] = Tasks.REGRESSION

        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        self.uncertainty_loss = UncertaintyLoss(num_task=nb_task)

        # --------------------------------------------
        #                   DROPOUT
        # --------------------------------------------
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}
        num_block = int(np.sum(layers[depth]))

        if drop_type.lower() == "flat":
            temp = [drop_rate for _ in range(num_block)]
        elif drop_type.lower() == "linear":
            temp = [1 - (1 - (drop_rate * i / (num_block - 1))) for i in range(num_block)]
        else:
            raise NotImplementedError

        dropout = []
        for i in range(NB_LEVELS):
            first = int(np.sum(layers[depth][0:i]))
            last = int(np.sum(layers[depth][0:i+1]))
            dropout.append(temp[first:last])

        # --------------------------------------------
        #                SHARED BLOCKS
        # --------------------------------------------
        if type(commun_block) is not list:
            shared_blocks = [self.__get_block(commun_block, depth) for _ in range(split_level - 1)]
        else:
            assert len(commun_block) == split_level - 1, \
                "The lenght of commun_block do not match with the split_level."
            shared_blocks = [self.__get_block(block, depth) for block in commun_block]

        # --------------------------------------------
        #                TASKS BLOCKS
        # --------------------------------------------
        if type(task_block) is not dict:
            task_block = {task: task_block for task in tasks}

        task_block_list = {}
        for task, blocks in list(task_block.items()):
            if type(blocks) is not list:
                task_block_list[task] = [self.__get_block(blocks, depth) for _ in range(5 - split_level)]
            else:
                assert len(task_block) == 5 - split_level, \
                    "The lenght of shared_block do not match with the split_level."
                task_block_list[task] = [self.__get_block(block, depth) for block in blocks]

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        shared_layers = []
        task_specific_layer = {}
        for task in tasks:
            task_specific_layer[task] = []

        assert 1 <= split_level <= 5, "The split level should be an integer between 1 and 5."
        shared_layers.append(Convolution(dimensions=NB_DIMENSIONS,
                                         in_channels=num_in_chan,
                                         out_channels=self.__in_channels,
                                         kernel_size=first_kernel,
                                         act=act,
                                         norm=norm,
                                         conv_only=shared_blocks[0] in [PreResBlock, PreResBottleneck]))

        for i in range(NB_LEVELS):
            strides = [2, 2, 1] if i == 0 else [2, 2, 2]

            if split_level > i + 1:
                temp_layers = self.__make_layer(shared_blocks[i],
                                                act=act,
                                                drop_rate=dropout[i],
                                                fmap_out=first_channels * (2**i),
                                                kernel=kernel,
                                                norm=norm,
                                                num_block=layers[depth][i],
                                                strides=strides)
                shared_layers.append(temp_layers)

            else:
                in_channels = self.__in_channels
                for task in tasks:
                    idx = i - (split_level - 1)
                    self.__in_channels = in_channels
                    temp_layers = self.__make_layer(task_block_list[task][idx],
                                                    act=act,
                                                    drop_rate=dropout[i],
                                                    fmap_out=first_channels * (2**i),
                                                    kernel=kernel,
                                                    norm=norm,
                                                    num_block=layers[depth][i],
                                                    strides=strides)
                    task_specific_layer[task].append(temp_layers)
        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        self.__num_flat_features = self.__in_channels

        for task in tasks:
            task_specific_layer[task].extend([nn.AvgPool3d(kernel_size=out_shape),
                                              nn.Flatten(start_dim=1),
                                              torch.nn.Linear(self.__num_flat_features, num_classes[task])])

        self.shared_layers = nn.Sequential(*shared_layers)
        self.tasks_layers = nn.ModuleDict({task: nn.Sequential(*task_specific_layer[task]) for task in tasks})

        self.apply(init_weights)

    @staticmethod
    def __get_block(block_type: str,
                    depth: int) -> Union[Type[ResBlock], Type[ResBottleneck],
                                         Type[PreResBlock], Type[PreResBottleneck]]:
        """
        Return the correct block class according to the depth of the network and the block_type.

        :param block_type: The block type that would be used. (Options: ["preact", "postact"])
        :param depth: The depth of the network.
        :return: A type class that represent the corresponding block to use.
        """
        assert block_type.lower() in [BlockType.PREACT, BlockType.POSTACT], "The block type option " \
                                                                            "are 'preact' and 'postact'."
        if depth <= 34:
            return PreResBlock if block_type.lower() == BlockType.PREACT else ResBlock
        else:
            return PreResBottleneck if block_type.lower() == BlockType.PREACT else ResBottleneck

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
        :param drop_rate: A sequence of float that indicate the drop_rate for each block.
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.shared_layers(x)

        preds = {}
        for task in self.__tasks:
            preds[task] = self.tasks_layers[task](out if task in self.__backend_tasks else out.detach())

        return preds
