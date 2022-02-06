"""
    @file:              cross_stitch.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2022
    @Last modification: 02/2022

    @Description:       This file contain the CrossStitch class. This class used three different neural network and
                        connect them with CrossStitchUnit to create a unique multitask Network.

    @Reference:         1) I. Misra, et al. Cross-stitch networks for multi-task learning. IEEE Conference on Computer
                           Vision and PatternRecognition (CVPR), 2016
"""
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from typing import Dict, Final, Iterator, List, Optional, Sequence, Tuple, Union

from constant import Loss, SharingUnits
from model.module import CrossStitchUnitV2, SluiceUnit, UncertaintyLoss, UniformLoss
from model.neural_net import NeuralNet

NB_LEVELS: Final = 4


class CrossStitch(NeuralNet):
    """
    An implementation of a cross stitch neural network inspired by Ref 1) for multi-task learning problem.
    The cross stitch network is a alternative of the hard-sharing structure proposed by Caruana
    (Multi-task learning, 1997). It took several sub neural network that can be pretrained on their corresponding
    task and it connect them with 'cross stitch unit' to allow features sharing between the two subnetwork.

     ...
    Attributes
    ----------
    loss : Union[UncertaintyLoss, UniformLoss]
        A torch.module that will be used to compute the multi-task loss without penalty.
    nets : nn.ModuleDict
        A dictionary that contain several neural network. One for each task.
    __nb_task : int
        The number of tasks.
    sharing_units_dict : nn.ModuleDict
        A dictionary that contain all the sharing_unit module.
        These sharing unit modules are referenced by their level.
    __tasks : List[str]
        The list of all tasks name.
    Methods
    -------
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]:
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    forward(x: torch.Tensor) -> torch.Tensor:
        Execute the forward on a given torch.Tensor.
    set_mixup(b_size : int):
        Set the b_size parameter of each mixup module.
    shared_forward(sharing_level: int, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        Reshape the features and compute the forward pass of the shared unit at a given sharing level.
    save_histogram_sharing_unit(current_iter: int, writer: SummaryWriter, prefix: Optional[str] = "")
        Save an histogram of the weights of each sharing unit with a given tensorboard writer.

    """
    def __init__(self,
                 sub_nets: nn.ModuleDict,
                 c: float = 0.9,
                 loss: Loss = UncertaintyLoss,
                 num_shared_channels: Optional[Sequence[int]] = None,
                 spread: float = 0.1) -> None:
        """
        Create a Shared network with Shared Module like Sluice Unit or Cross-Stitch Unit.

        :param sub_nets: A dictionary of neural network where the key are task name associated to the network.
        :param c: A float that represent the conservation parameter of the sharing units.
        :param num_shared_channels: A list of int that indicate the number of channels per network before the
                                    cross-stitch unit. Only used if sharing_unit == "cross_stitch"
        :param spread: A float that represent the spread parameter of the sharing units.
        """
        super().__init__()

        # --------------------------------------------
        #                    NETS
        # --------------------------------------------
        self.nets = sub_nets
        self.__tasks = list(self.nets.keys())
        self.__nb_task = len(self.__tasks)

        # --------------------------------------------
        #               SHARING UNITS
        # --------------------------------------------
        self.sharing_units_dict = nn.ModuleDict()
        for i in range(1, NB_LEVELS+1):
            if num_shared_channels[i - 1] != 0:
                self.sharing_units_dict[str(i)] = CrossStitchUnitV2(nb_channels=num_shared_channels[i - 1],
                                                                    nb_task=self.__nb_task,
                                                                    c=c,
                                                                    spread=spread)

        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        if loss == Loss.UNCERTAINTY:
            self.loss = UncertaintyLoss(num_task=self.__nb_task)
        else:
            self.loss = UniformLoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The forward pass of the CrossStitch network

        :param x: A torch.Tensor that represent a batch of 3D images.
        :return: A dictionary of torch.tensor that represent the output per task.
                 The keys correspond to the tasks name.
        """
        mixup_key_list = list(self.mixup.keys())

        out = self.mixup["0"](x) if "0" in mixup_key_list else x

        # --------------------------------------------
        #                   CONV
        # --------------------------------------------
        outs = []
        for task in self.__tasks:
            outs.append(self.nets[task].conv(out))

        # --------------------------------------------
        #                   LEVEL 1
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers1(outs[count])
        outs = self.shared_forward(1, outs)

        # --------------------------------------------
        #                   LEVEL 2
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers2(outs[count])
        outs = self.shared_forward(2, outs)

        # --------------------------------------------
        #                   LEVEL 3
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers3(outs[count])
        outs = self.shared_forward(3, outs)

        # --------------------------------------------
        #                   LEVEL 4
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers4(outs[count])
        outs = self.shared_forward(4, outs)

        # --------------------------------------------
        #              POOLING + FC_LAYERS
        # --------------------------------------------
        preds = {}
        for count, task in enumerate(self.__tasks):
            preds[task] = self.nets[task].last_layers(outs[count])

        return preds

    def get_weights(self) -> Tuple[List[Iterator[torch.nn.Parameter]],
                                   Optional[List[torch.nn.Parameter]]]:
        """
        Get the model parameters and the loss parameters.

        :return: A list of parameters that represent the weights of the network and another list of parameters
                 that represent the weights of the loss.
        """
        parameters = [self.nets.parameters(),
                      self.sharing_units_dict.parameters()]

        loss_parameters = self.loss.parameters() if isinstance(self.loss, UncertaintyLoss) else None

        return parameters, loss_parameters

    def shared_forward(self,
                       sharing_level: int,
                       x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reshape the features and compute the forward pass of the shared unit at the sharing level i.

        :param sharing_level: An integer that indicate which sharing unit will be used.
        :param x: A list of torch.Tensor that represent the features extract by each sub neural net before the
                  given sharing level.
        :return: A list of torch.Tensor that represent the shared features representation for each sub neural net.
        """

        if str(sharing_level) in list(self.sharing_units_dict.keys()):
            return self.sharing_units_dict[str(sharing_level)](x)

        # Skip the sharing level
        else:
            return x

    def save_histogram_sharing_unit(self,
                                    current_iter: int,
                                    writer: SummaryWriter,
                                    prefix: Optional[str] = "") -> None:
        """
        Save an histogram of the weights of each sharing unit with a given tensorboard writer.

        :param current_iter: An integer that indicate the current iteration.
        :param writer: The tensorboard writer that will be used to save the histogram.
        :param prefix: A string that will be used as prefix of the histogram name.
        """
        for key, mod in self.sharing_units_dict.items():
            weights = mod.alpha.detach().cpu().numpy()
            writer.add_histogram(prefix + f"Sharing units {key}", weights.flatten(), current_iter)
