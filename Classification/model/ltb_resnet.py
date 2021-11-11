"""
    @file:              ltb_resnet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     08/2021
    @Last modification: 10/2021

    @Description:       This file contain the class LTBResNet (Learn-To-Branch ResNet) that inherit from the NeuralNet
                        class.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
                        2) Learning to Branch for Multi-Task Learning, Guo, P. et al., CoRR 2020
"""
from monai.networks.blocks.convolutions import Convolution
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple, Type, Union

from constant import BlockType, DropType, Loss, Tasks
from model.block import BranchingBlock, PreResBlock, PreResBottleneck, ResBlock, ResBottleneck
from model.module import Mixup, UncertaintyLoss, UniformLoss
from model.neural_net import NeuralNet, init_weights

NB_DIMENSIONS: Final = 3
NB_LEVELS: Final = 4


class LTBResNet(NeuralNet):
    """
    An implementation of the Learn-To-Branch version of the ResNet3D inspired by Ref) 2

    ...
    Attributes
    ----------
    aux_tasks_loss : Union[UncertaintyLoss, UniformLoss]
        A torch.module that will be used to compute the multi-task loss on the auxiliary tasks.
    avg_pool :  nn.AvgPool3d
        The AvgPool3D module that is used after the last series of residual block.
    conv : nn.ModuleList
        First blocks of the network.
    __in_channels : int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
        The last series of residual block.
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
    loss : Callable
        A method that will combine the main task loss and the auxiliary task loss.
    main_tasks_loss : Union[UncertaintyLoss, UniformLoss]
        A torch.module that will be used to compute the multi-task loss on the main tasks.
    __num_flat_features : int
        Number of features at the output of the last convolution.
    __tasks : List[str]
        The list of tasks on which the model will be train.
    Methods
    -------
    forward(x: torch.Tensor) -> Dict[str, torch.Tensor]
        Execute the forward on a given torch.Tensor.
    update_epoch(num_epoch: Optional[int] = None) -> None
        Update the attribute named num_epoch of all gumbel_softmax layer in the LTBResNet.
    """
    def __init__(self,
                 block_type_list: Union[List[BlockType], List[List[BlockType]], BlockType],
                 main_tasks: List[str],
                 num_classes: Dict[str, int],
                 act: str = "ReLU",
                 aux_tasks: List[str] = None,
                 aux_tasks_coeff: float = 0.05,
                 block_width: Union[int, List[int]] = 2,
                 depth: int = 18,
                 drop_rate: float = 0,
                 drop_type: DropType = DropType.LINEAR,
                 first_channels: int = 16,
                 first_kernel: Union[Sequence[int], int] = 3,
                 in_shape: Union[Sequence[int], Tuple] = (96, 96, 32),
                 kernel: Union[Sequence[int], int] = 3,
                 loss: Loss = Loss.UNCERTAINTY,
                 norm: str = "batch",
                 num_in_chan: int = 4,
                 tau: float = 1) -> None:
        """

        :param block_type_list: A list of block that will be instanciate in the network.
        :param main_tasks: A list of tasks on which the model will be train.
        :param num_classes: A dictionnary that indicate the number of class for each task. For regression tasks,
                            the num_class shoule be equal to one. Regression task will be consider has auxiliary tasks.
        :param act: A string that represent the activation function that will be used in the NeuralNet. (Default=ReLU)
        :param aux_tasks_coeff: The coefficient that will multiply the loss of the auxiliary tasks.
        :param block_width: A integer or a list of integer that indicate the number of possible path at each layer.
        :param depth: The number of convolution and fully connected layer in the neural network. (Default=18)
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
        :param tau: The non-negative scalar temperature parameter of the gumble softmax operation.
        """
        assert len(main_tasks) > 0, "You should specify the name of each task"
        super().__init__()
        aux_tasks = [] if aux_tasks is None else aux_tasks
        self.aux_tasks_coeff = aux_tasks_coeff
        self.__tasks = main_tasks + aux_tasks
        # --------------------------------------------
        #                NUM_CLASSES
        # --------------------------------------------
        # If num_classes has not been defined, then we assume that every main task are binary classification and
        # every auxiliary task are regression.
        if num_classes is None:
            num_classes = {}
            for task in main_tasks:
                num_classes[task] = Tasks.CLASSIFICATION
            for task in aux_tasks:
                num_classes[task] = Tasks.REGRESSION
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
            self.main_tasks_loss = UncertaintyLoss(num_task=len(main_tasks))
            self.aux_tasks_loss = UncertaintyLoss(num_task=len(aux_tasks))
        else:
            self.main_tasks_loss = UniformLoss()
            self.aux_tasks_loss = UniformLoss()
        self.loss = self.__define_loss(aux_tasks_coeff=aux_tasks_coeff if len(aux_tasks) > 0 else 0.0)

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}
        net_width = [block_width for _ in range(NB_LEVELS)] if isinstance(block_width, int) else block_width

        if type(block_type_list) is BlockType:
            block_type_list = [[block_type_list] for _ in range(NB_LEVELS)]
        elif type(block_type_list) is list:
            if len(block_type_list) == 4:
                block_type_list = [[block_type] for block_type in block_type_list]
            else:
                block_type_list = [block_type_list for _ in range(NB_LEVELS)]

        block_list = []
        for count, block_level in enumerate(block_type_list):
            block_list.append([])
            for block_type in block_level:
                for _ in range(net_width[count]):
                    if block_type is BlockType.PREACT:
                        block_list[count].append(PreResBlock if depth <= 34 else PreResBottleneck)
                    elif block_type is BlockType.POSTACT:
                        block_list[count].append(ResBlock if depth <= 34 else ResBottleneck)
                    else:
                        raise Exception(f"The block_type is not an option: {block_type}, "
                                        f"see BlockType Enum in Constant.py.")

        width_factor = [len(blocks_type) for blocks_type in block_type_list]
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
        self.conv = nn.ModuleList()
        for block_type in block_type_list[0]:
            self.conv.append(
                Convolution(dimensions=NB_DIMENSIONS,
                            in_channels=num_in_chan,
                            out_channels=self.__in_channels,
                            kernel_size=first_kernel,
                            act=act,
                            conv_only=block_type is BlockType.PREACT,
                            norm="batch")
            )

        self.layers1 = self.__make_layer(block_list[0], dropout[0],
                                         first_channels, kernel,
                                         num_block=layers[depth][0],
                                         num_input=len(block_type_list[0]),
                                         strides=[2, 2, 1], norm=norm,
                                         act=act, tau=tau)

        self.layers2 = self.__make_layer(block_list[1], dropout[1],
                                         first_channels * 2, kernel,
                                         num_block=layers[depth][1],
                                         num_input=net_width[0] * width_factor[0],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act, tau=tau)

        self.layers3 = self.__make_layer(block_list[2], dropout[2],
                                         first_channels * 4, kernel,
                                         num_block=layers[depth][2],
                                         num_input=net_width[1] * width_factor[1],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act, tau=tau)

        self.layers4 = self.__make_layer(block_list[3], dropout[3],
                                         first_channels * 8, kernel,
                                         num_block=layers[depth][3],
                                         num_input=net_width[2] * width_factor[2],
                                         strides=[2, 2, 2], norm=norm,
                                         act=act, tau=tau)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 2**NB_LEVELS),
                     int(in_shape[1] / 2**NB_LEVELS),
                     int(in_shape[2] / 2**(NB_LEVELS - 1))]

        self.avg_pool = nn.Sequential(nn.AvgPool3d(kernel_size=out_shape),
                                      nn.Flatten(start_dim=1))
        self.__num_flat_features = self.__in_channels

        self.last_layers = nn.ModuleDict()
        for task in self.__tasks:
            self.last_layers[task] = BranchingBlock([torch.nn.Linear],
                                                    num_input=net_width[3] * width_factor[3],
                                                    tau=tau,
                                                    in_features=self.__num_flat_features,
                                                    out_features=num_classes[task])
        self.apply(init_weights)

    def __define_loss(self, aux_tasks_coeff: float) -> Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                             Callable[[torch.Tensor], torch.Tensor]]:
        """
        Build the method that will be used to compute the loss.

        :param aux_tasks_coeff: The coefficient that will multiply the loss of the auxiliary tasks.
        :return: A function that compute the multi-task loss.
        """
        if aux_tasks_coeff:
            def multi_task_loss(losses: torch.Tensor,
                                aux_tasks_losses: torch.Tensor) -> torch.Tensor:
                aux_coeff = aux_tasks_coeff / aux_tasks_losses.size(dim=0)
                return self.main_tasks_loss(losses) + aux_coeff * self.aux_tasks_loss(aux_tasks_losses)
        else:
            def multi_task_loss(losses: torch.Tensor) -> torch.Tensor:
                return self.loss_module(losses)

        return multi_task_loss

    def __make_layer(self,
                     block_list: List[Union[Type[PreResBlock], Type[PreResBottleneck],
                                            Type[ResBlock], Type[ResBottleneck]]],
                     drop_rate: List[float],
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     num_block: int,
                     num_input: int,
                     tau: float,
                     act: str = "ReLU",
                     norm: str = "batch",
                     strides: Union[Sequence[int], int] = 1) -> nn.Sequential:
        """
        Create a sequence of layer of a given class and of lenght num_block.

        :param block_list: A list of class type that indicate which blocks should be create in the sequence.
        :param drop_rate: A list of float that indicate the drop_rate for each block.
        :param fmap_out: fmap_out*block.expansion equal the number of output feature maps of each block.
        :param kernel: An integer or a list of integer that indicate the convolution kernel size.
        :param num_block: An integer that indicate how many block will contain the sequence.
        :param num_input: An integer that indicate the number of parent node of the first block.
        :param tau: non-negative scalar temperature parameter of the gumble softmax operation.
        :param act: The activation function that will be used in each block.
        :param norm: The normalization layer that will be used in each block.
        :param strides: An integer or a list of integer that indicate the strides of the first convolution of the
                        first block.
        :return: A nn.Sequential that represent the sequence of layer.
        """
        layers = []

        for i in range(num_block):
            layers.append(BranchingBlock(block_list=block_list,
                                         num_input=num_input if i == 0 else len(block_list),
                                         tau=tau,
                                         fmap_in=self.__in_channels,
                                         fmap_out=fmap_out,
                                         kernel=kernel,
                                         strides=strides if i == 0 else 1,
                                         drop_rate=drop_rate[i],
                                         activation=act,
                                         norm=norm))

            self.__in_channels = fmap_out * block_list[0].expansion if i == 0 else self.__in_channels

        return nn.Sequential(*layers)

    def get_weights(self) -> Tuple[List[List[torch.nn.Parameter]],
                                   Optional[List[torch.nn.Parameter]]]:
        """
        get the branching and the nodes parameters.

        :return: A list of torch.Parameter that represent the weights of the nodes and the branching and another
                 list of torch.Parameter that represent the weight of the loss.
        """
        weights = list(self.conv.parameters())
        gumbel_softmax_weights = []

        for layers in [self.layers1, self.layers2, self.layers3, self.layers4]:
            for layer in layers:
                weights += layer.get_weights(gumbel_softmax_weights=False)
                gumbel_softmax_weights += layer.get_weights(gumbel_softmax_weights=True)

        for task in self.__tasks:
            weights += self.last_layers[task].get_weights(gumbel_softmax_weights=False)
            gumbel_softmax_weights += self.last_layers[task].get_weights(gumbel_softmax_weights=True)

        if isinstance(self.main_tasks_loss, UncertaintyLoss):
            loss_parameters = list(self.main_tasks_loss.parameters()) + list(self.aux_tasks_loss.parameters())
        else:
            loss_parameters = None
        return [weights, gumbel_softmax_weights], loss_parameters

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The forward pass of the LTBResNet

        :param x: A torch.Tensor that represent a batch of 3D images.
        :return: A dictionnary of torch.tensor that reprensent the output per task.
                 The keys correspond to the tasks name.
        """
        out = torch.stack([conv(x) for conv in self.conv])
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.layers4(out)

        out = torch.stack([self.avg_pool(out[i]) for i in range(out.size()[0])])

        pred = {task: self.last_layers[task](out).squeeze(dim=0) for task in self.__tasks}
        return pred

    def freeze_branching(self) -> Tuple[List[List[int]], List[str]]:
        """
        Freeze the gumbel softmax operation of all branching blocks in the model and return a list of list of int
        that represent the result of the architecture search. Finally the weights are re initialized.

        :return: A list that indicate which children are connected to which parents and the task list.
        """
        parents_list = [[]]
        unique_parents = []

        self.eval()

        for task in self.__tasks:
            parents, unique_parent = self.last_layers[task].freeze_branch()
            parents_list[0].extend(parents)
            unique_parents.extend(unique_parent)

        _, index = np.unique(np.array(unique_parents), return_index=True)
        unique_parents = list(np.array(unique_parents)[np.sort(index)])

        for layers in [self.layers4, self.layers3, self.layers2, self.layers1]:
            for layer in reversed(list(layers)):
                parents, unique_parents = layer.freeze_branch(unique_parents)
                parents_list.append(parents)

        parents_list.reverse()
        self.apply(init_weights)
        return parents_list, self.__tasks

    def update_epoch(self, num_epoch: Optional[int] = None) -> None:
        """
        Update the attribute named num_epoch of all gumbel_softmax layer in the LTBResNet.

        :param num_epoch: The current number of epoch as int. If None, self.__num_epoch will be incremented by 1.
        """
        for layers in [self.layers1, self.layers2, self.layers3, self.layers4]:
            for layer in layers:
                layer.gumbel_softmax.update_epoch(num_epoch)

        for task in self.__tasks:
            self.last_layers[task].gumbel_softmax.update_epoch(num_epoch)
