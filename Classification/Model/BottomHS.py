"""
    @file:              BottomHS.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 09/2021

    @Description:       This file contain the class of the BottomHS that inherit from the NeuralNet class.
                        The BottomHS is a form of Hard Sharing ResNet where the first layers and the normalization
                        layers are task specific and the last convolution layers are shared.

    @Reference:         1) Rethinking Hard-Parameter Sharing in Multi-Task Learning, Zhang, L. et al., arXiv 2021
"""

from Constant import BlockType, DropType, Loss, Tasks
from Model.Block import IndPreResBlock, IndResBlock
from Model.Module import Mixup, UncertaintyLoss, UniformLoss
from monai.networks.blocks.convolutions import Convolution
from Model.NeuralNet import NeuralNet, init_weights
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Final, List, Optional, Sequence, Tuple, Type, Union


NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4


class IndNormResNet(NeuralNet):
    """
    A hard shared 3D Residual Network implementation for multi task learning where bottom layers and the normalization
    layers are task specific. Inspired by ref 1).

    ...
    Attributes
    ----------

    fc_layers : nn.ModuleDict[str, Module]
        A dictionary that contain the classifier of each task.
    first_conv : nn.ModuleDict[str, Module]
        A dictionary that contain the first convolution of each task.
    __in_channels : int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
        The last series of residual block.
    loss : Union[UncertaintyLoss, UniformLoss]
        A torch.module that will be used to compute the multi-task loss during the training.
    __num_flat_features : int
        Number of features at the output of the last convolution.
    shared_layers : nn.ModuleList
        A sequence of neural network layer that will used for every task.
    __tasks : List[str]
        The list of tasks on which the model will be train.
    tasks_layers : nn.ModuleDict[str, nn.ModuleList]
        A dictionary of ModuleList module where the key is a task name and the value is a sequence of layer that will
        be used only for this task.
    Methods
    -------
    forward(x: torch.Tensor) -> Dict[str, torch.Tensor]
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
                 tasks: List[str],
                 act: str = "ReLU",
                 commun_block: Union[BlockType, List[BlockType]] = BlockType.PREACT,
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: DropType = DropType.FLAT,
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 kernel: Union[Sequence[int], int] = 3,
                 loss: Loss = Loss.UNCERTAINTY,
                 norm: str = "batch",
                 num_in_chan: int = 4,
                 task_block: Union[
                     Dict[str, Union[BlockType, List[BlockType]]],
                     BlockType
                 ] = BlockType.PREACT,
                 merge_level: int = 3):
        """
        Create a pre activation or post activation 3D Residual Network for multi-task learning.

        :param num_classes: A dictionnary that indicate the number of class for each task. For regression tasks,
                            the num_class shoule be equal to one.
        :param tasks: A list of tasks on which the model will be train.
        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param commun_block: A string or a sequence of string that indicate which type of block will be use to
                             create the shared layers of the network. If it is a list, it should be equal in length to
                             split_level - 1.
        :param depth: The ResNet configuration that will be used to build the network. (Default=18)
        :param drop_rate: The maximal dropout rate used to configure the dropout layer. See drop_type (Default=0)
        :param drop_type: If drop_type == 'flat' every dropout layer will have the same drop rate.
                          Else if, drop_type == 'linear' the drop rate will grow linearly at each dropout layer
                          from 0 to 'drop_rate'. (Default='Flat')
        :param first_channels: The number of channels at the output of the first convolution layer. (Default=16)
        :param first_kernel: The kernel shape of the first convolution layer. (Default=3)
        :param in_shape: The image shape at the input of the neural network. (Default=(64, 64, 16))
        :param kernel: The kernel shape of all convolution layer except the first one. (Default=3)
        :param loss: Indicate the MTL loss that will be used during the training. (Default=Loss.Uncertainty)
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param num_in_chan: An integer that indicate the number of channel of the input images.
        :param task_block: A dictionary of string, a dictionary of sequence of string or a string that indication which
                           block type will be used to create the task specifics layers. If a dictionary is used, then
                           they keys should be the task name. The 3 following example give the same result.
                           Example 1)
                            task_block = BlockType.PREACT
                           Example 2)
                            task_block = {"malignancy": BlockType.PREACT, "subtype": BlockType.PREACT}
                           Example 3)
                            task_block = {"malignancy": [BlockType.PREACT, BlockType.PREACT],
                                          "subtype": [BlockType.PREACT, BlockType.PREACT]}
        :param merge_level: At which level the subnet should shared the same layers. (Default=4)
                                1: After the first convolution,
                                2: After the first residual level,
                                3: After the second residual level,
                                4: After the third residual level,
        """
        assert len(tasks) > 0, "You should specify the name of each task"
        super().__init__()
        self.__split = merge_level
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
        if loss == Loss.UNCERTAINTY:
            self.loss = UncertaintyLoss(num_task=nb_task)
        else:
            self.loss = UniformLoss()

        # --------------------------------------------
        #                   DEPTH
        # --------------------------------------------
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
        #                SHARED BLOCKS
        # --------------------------------------------
        if type(commun_block) is not list:
            shared_blocks = [self.__get_block(commun_block) for _ in range(5 - merge_level)]
        else:
            assert len(commun_block) == 5 - merge_level, \
                "The lenght of commun_block do not match with the split_level."
            shared_blocks = [self.__get_block(block) for block in commun_block]

        # --------------------------------------------
        #                TASKS BLOCKS
        # --------------------------------------------
        if type(task_block) is not dict:
            task_block = {task: task_block for task in tasks}

        task_block_list = {}
        for task, blocks in list(task_block.items()):
            if type(blocks) is not list:
                task_block_list[task] = [self.__get_block(blocks) for _ in range(merge_level - 1)]
            else:
                assert len(task_block) == merge_level - 1, \
                    "The lenght of shared_block do not match with the split_level."
                task_block_list[task] = [self.__get_block(block) for block in blocks]

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        shared_layers = []
        task_specific_layer = {}
        for task in tasks:
            task_specific_layer[task] = []

        assert 1 <= merge_level <= 4, "The split level should be an integer between 1 and 4."
        self.first_conv = nn.ModuleDict({
            task: Convolution(dimensions=NB_DIMENSIONS,
                              in_channels=num_in_chan,
                              out_channels=self.__in_channels,
                              kernel_size=first_kernel,
                              act=act,
                              norm=norm,
                              conv_only=task_block_list[task][0] in [IndPreResBlock])
            for task in tasks
        })

        for i in range(NB_LEVELS):
            strides = [2, 2, 1] if i == 0 else [2, 2, 2]

            if merge_level < i + 1:
                temp_layers = self.__make_layer(shared_blocks[i],
                                                act=act,
                                                drop_rate=dropout[i],
                                                fmap_out=first_channels * (2**i),
                                                kernel=kernel,
                                                norm=norm,
                                                num_block=layers[depth][i],
                                                strides=strides,
                                                task_list=self.__tasks)
                shared_layers.extend(temp_layers)

            else:
                in_channels = self.__in_channels
                for task in tasks:
                    idx = i
                    self.__in_channels = in_channels
                    temp_layers = self.__make_layer(task_block_list[task][idx],
                                                    act=act,
                                                    drop_rate=dropout[i],
                                                    fmap_out=first_channels * (2**i),
                                                    kernel=kernel,
                                                    norm=norm,
                                                    num_block=layers[depth][i],
                                                    strides=strides,
                                                    task_list=[task])
                    task_specific_layer[task].extend(temp_layers)

        self.shared_layers = nn.ModuleList(shared_layers)
        self.tasks_layers = nn.ModuleDict({task: nn.ModuleList(task_specific_layer[task]) for task in tasks})

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        self.__num_flat_features = self.__in_channels
        self.flat_layers = nn.Sequential(nn.AvgPool3d(kernel_size=out_shape), nn.Flatten(start_dim=1))
        self.fc_layers = nn.ModuleDict({task: torch.nn.Linear(self.__num_flat_features, num_classes[task])
                                        for task in self.__tasks})

        self.apply(init_weights)

    @staticmethod
    def __get_block(block_type: BlockType) -> Union[Type[IndResBlock], Type[IndPreResBlock]]:
        """
        Return the correct block class according to the depth of the network and the block_type.

        :param block_type: The block type that would be used. (Options: ["preact", "postact"])
        :return: A type class that represent the corresponding block to use.
        """
        assert type(block_type) is BlockType, f"The block_type should be a BlockType, but get {block_type}."
        return IndPreResBlock if block_type is BlockType.PREACT else IndResBlock

    def __make_layer(self,
                     block: Union[Type[IndPreResBlock], Type[IndResBlock]],
                     drop_rate: List[float],
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     num_block: int,
                     task_list: List[str],
                     act: str = "ReLU",
                     norm: str = "batch",
                     strides: Union[Sequence[int], int] = 1) -> List:
        """
        Create a sequence of layer of a given class and of length num_block.

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
                                activation=act,
                                drop_rate=drop_rate[i],
                                kernel=kernel,
                                norm=norm,
                                strides=strides if i == 0 else 1,
                                task_list=task_list))
            self.__in_channels = fmap_out * block.expansion if i == 0 else self.__in_channels

        return layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform the forward pass on a given batch of 3D images

        :param x: A torch.Tensor that represent a batch of 3D images.
        :return: A dictionnary of torch.tensor that reprensent the output per task.
                 The keys correspond to the tasks name.
        """
        preds = {}

        for task in self.__tasks:
            temp = self.first_conv[task](x)
            for layer in self.tasks_layers[task]:
                temp = layer(task=task, x=temp)

            for layer in self.shared_layers:
                temp = layer(task=task, x=temp)

            temp = self.flat_layers(temp)
            preds[task] = self.fc_layers[task](temp)

        return preds
