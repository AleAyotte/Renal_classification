"""
    @file:              mtan.py
    @Author:            Alexandre Ayotte

    @Creation Date:     07/2021
    @Last modification: 07/2021

    @Description:       This file contain the classe MTAN that inherit from the NeuralNet class. This Multi-task
                        Attention Network can only be use for multitask trainer on 3D images.

    @Reference:         1) End-To-End Multi-Task Learning With Attention, Liu, S. et al, CVPR 2019
"""
import torch
import torch.nn as nn
from typing import Dict, Final, Iterator, List, Optional, Sequence, Tuple, Union

from constant import AttentionBlock, BlockType, DropType, Loss, Tasks
from model.block import CBAM, ChannelAttBlock, PreResBlock, PreResBottleneck, ResBlock, ResBottleneck, SpatialAttBlock
from model.module import UncertaintyLoss, UniformLoss
from model.neural_net import NeuralNet, init_weights
from model.resnet import ResNet

NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4
SQUEEZE_FACTOR_LIST: Final = [4, 8, 8, 8]
STRIDES: Final = [2, 2, 2]


class MTAN(NeuralNet):
    """
    A 3D version of the Multi-Task Attention Network adapted to classification and regression task from the
    version describe in Ref) 1.

    ...
    Attributes
    ----------
    att_layers: nn.ModuleList[nn.ModuleDict]
        A list of nn.ModuleDict that contain the attention module of each task for each level.
    conv : Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layers: nn.ModuleDict
        A dictionary of nn.Sequential that contain the pooling layer and the last fully connected of each task.
    loss: Union[UncertaintyLoss, UniformLoss]
        A torch.module that will be used to compute the multi-task loss during the training.
    shared_layers1_base : nn.Sequentiel
        The n-1 first block of the self.layers1 sequential in the ResNet. Its output will be use as input in the first
        attention module.
    shared_layers1_base : nn.Sequentiel
        The last block of the self.layers1 sequential in the ResNet. The mask produced by the first attention module
        will be applied on its output.
    shared_layers2_base : nn.Sequentiel
        The n-1 first block of the self.layers2 sequential in the ResNet. Its output will be use as input in the second
        attention module.
    shared_layers2_base : nn.Sequentiel
        The last block of the self.layers2 sequential in the ResNet. The mask produced by the second attention module
        will be applied on its output.
    shared_layers3_base : nn.Sequentiel
        The n-1 first block of the self.layers3 sequential in the ResNet. Its output will be use as input in the third
        attention module.
    shared_layers3_base : nn.Sequentiel
        The last block of the self.layers3 sequential in the ResNet. The mask produced by the third attention module
        will be applied on its output.
    shared_layers4_base : nn.Sequentiel
        The n-1 first block of the self.layers4 sequential in the ResNet. Its output will be use as input in the fourth
        attention module.
    shared_layers4_base : nn.Sequentiel
        The last block of the self.layers4 sequential in the ResNet. The mask produced by the fourth attention module
        will be applied on its output.
    __tasks : List[str]
        The list of tasks on which the model will be train.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    get_weights()
        Get the model parameters and the loss parameters.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 tasks: Sequence[str],
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
                 loss: Loss = Loss.UNCERTAINTY,
                 norm: str = "batch",
                 num_in_chan: int = 4):
        """
        Create a pre activation or post activation 3D Residual Network.

        :param num_classes: A dictionary that indicate the number of class for each task. For regression tasks,
                            the num_class should be equal to one.
        :param tasks: A list of tasks on which the model will be train.
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
        :param loss: Indicate the MTL loss that will be used during the training. (Default=Loss.Uncertainty)
        :param norm: A string that represent the normalization layers that will be used in the NeuralNet.
                     (Default=batch)
        :param num_in_chan: A positive integer that represent the number of channels of the input images.
        """
        super().__init__()

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

            self.att_layers.append(nn.ModuleDict())

            for task in self.__tasks:
                if att_type == AttentionBlock.CHANNEL:
                    att_block = ChannelAttBlock
                elif att_type == AttentionBlock.SPATIAL:
                    att_block = SpatialAttBlock
                else:
                    att_block = CBAM
                self.att_layers[-1][task] = att_block(fmap_in=num_in_chan * block.expansion * factor,
                                                      fmap_out=num_in_chan*block.expansion,
                                                      squeeze_factor=sq_fact,
                                                      subsample=subsample if count < 3 else None)
            factor = 2

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        num_flat_features = num_out_chan*block_list[0].expansion

        self.fc_layers = nn.ModuleDict()
        for task in tasks:
            self.fc_layers[task] = nn.Sequential(nn.AvgPool3d(kernel_size=out_shape),
                                                 nn.Flatten(start_dim=1),
                                                 torch.nn.Linear(num_flat_features, num_classes[task]))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Define the forward pass of the Multi-Task Attention Network

        :param x: A batch of 3D images that we want to classify.
        :return: A dictionary of torch.Tensor that represent the prediction per task for every element in the batch.
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

        preds = {}
        for task in self.__tasks:
            att_out = self.att_layers[0][task](out1_base, out1_last)
            att_out = self.att_layers[1][task](torch.cat((out2_base, att_out), dim=1), out2_last)
            att_out = self.att_layers[2][task](torch.cat((out3_base, att_out), dim=1), out3_last)
            att_out = self.att_layers[3][task](torch.cat((out4_base, att_out), dim=1), out4_last)

            preds[task] = self.fc_layers[task](att_out)

        return preds

    def get_weights(self) -> Tuple[List[Iterator[torch.nn.Parameter]],
                                   Optional[List[torch.nn.Parameter]]]:
        """
        Get the model parameters and the loss parameters.

        :return: A list of parameters that represent the weights of the network and another list of parameters
                 that represent the weights of the loss.
        """
        parameters = list(self.conv.parameters)
        parameters += list(self.shared_layers1_base) + list(self.shared_layers1_last)
        parameters += list(self.shared_layers2_base) + list(self.shared_layers2_last)
        parameters += list(self.shared_layers3_base) + list(self.shared_layers3_last)
        parameters += list(self.shared_layers4_base) + list(self.shared_layers4_last)
        parameters += list(self.att_layers.parameters()) + list(self.fc_layers.parameters())

        if isinstance(self.main_tasks_loss, UncertaintyLoss):
            loss_parameters = self.loss.parameters()
        else:
            loss_parameters = None
        return [parameters], loss_parameters
