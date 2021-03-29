"""
    @file:              SharedNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       This file contain the SharedNet class. This class used three different neural network and
                        connect them with SluiceUnit or CrossStitchUnit to create a unique MultiTaskNetwork.

    @Reference:         1) I. Misra, et al. Cross-stitch networks formulti-task learning. IEEE Conference on Computer
                           Vision and PatternRecognition (CVPR), 2016
                        2) S. Ruder et al. Latent Multi-Task Architecture Learning. Proceedings of the AAAI
                           Conferenceon Artificial Intelligence, 2019
"""

from Model.Module import CrossStitchUnit, Mixup, SluiceUnit, UncertaintyLoss
from Model.NeuralNet import NeuralNet
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Sequence, Union


class SharedNet(NeuralNet):
    """
    An implemantation of a general shared neural network inpired by Ref 1) and 2) for multi-task learning problem.
    The SharedNet is a alternative of the hard-sharing structure proposed by Caruana (Multi-task learning, 1997).
    It took 2 sub neural network that can be pretrained on their corresponding task and it connect them with
    'sharing unit' to allow features sharing between the two network. These 'sharing utit' perform a linear
    interpolation between different subspace of the sub neural networks.

     ...
    Attributes
    ----------
    mixup: nn.ModuleDict
        A dictionnary that contain all the mixup modules. These mixup modules are referenced by their level.
    nets: nn.ModuleDict
        A dictionnary that contain two neural network. One for each task.
        They are referenced by the shortname of their corresponding task. (mal, sub)
    __sharing_unit: str
        A string that indicate which type of sharing unit are present in the neural network.
    sharing_units_dict: nn.ModuleDict
        A dictionnary that contain all the sharing_unit module.
        These sharing unit modules are referenced by their level.
    __subspace: Sequence[int]
        A list of int that indicate the number of subspace for each network (mal, sub).
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Execute the forward on a given torch.Tensor.
    set_mixup(b_size : int):
        Set the b_size parameter of each mixup module.
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]:
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    shared_forward(x: Sequence[torch.Tensor], sharing_level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Reshape the features and compute the forward pass of the shared unit at the sharing level i.
    """
    def __init__(self,
                 sub_nets: nn.ModuleDict,
                 sharing_unit: str = "sluice",
                 mixup: Sequence[float] = None,
                 subspace_1: Union[Sequence[int], None] = None,
                 subspace_2: Union[Sequence[int], None] = None,
                 subspace_3: Union[Sequence[int], None] = None,
                 subspace_4: Union[Sequence[int], None] = None,
                 num_shared_channels: [Sequence[int], None] = None,
                 c: float = 0.9,
                 spread: float = 0.1):
        """
        Create a Shared network with Shared Module like Sluice Unit or Cross-Stitch Unit.

        :param sub_nets:
        :param sharing_unit: The sharing unit that will be used to shared the information between the 3 network.
        :param mixup: A list of int that indicate the beta parameters of each mixup modules.
        :param subspace_1: A list of int that indicate the number of subspace in each network before the sluice unit 1.
        :param subspace_2: A list of int that indicate the number of subspace in each network before the sluice unit 2.
        :param subspace_3: A list of int that indicate the number of subspace in each network before the sluice unit 3.
        :param subspace_4: A list of int that indicate the number of subspace in each network before the sluice unit 4.
        :param num_shared_channels: A list of int that indicate the number of channels per network before the
                                    cross-stitch unit. Only used if sharing_unit == "cross_stitch"
        :param c: A float that represent the conservation parameter of the sharing units.
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
        self.uncertainty_loss = UncertaintyLoss(num_task=self.__nb_task)

        # --------------------------------------------
        #               SHARING UNITS
        # --------------------------------------------
        assert sharing_unit.lower() in ["sluice", "cross_stitch"], \
            "The sharing unit can only be a sluice or cross_stitch one."

        self.__sharing_unit = sharing_unit
        self.__subspace = np.array([subspace_1, subspace_2, subspace_3, subspace_4])

        if sharing_unit.lower() == "sluice":
            for it in range(4):
                assert len(self.__subspace[it]) == self.__nb_task, "You must give the number of subspace of each " \
                                                                   "network in this order: Malignant, Subtype, Grade."

        else:
            assert len(num_shared_channels) == 4, "You must give the number of shared channels PER NETWORK for each " \
                                                  "shared unit. Only {} were gived".format(len(num_shared_channels))

        self.sharing_units_dict = nn.ModuleDict()
        for i in range(1, 5):
            if sharing_unit.lower() == "sluice":
                self.sharing_units_dict[str(i)] = SluiceUnit(self.__subspace[i - 1].sum(),
                                                             c,
                                                             spread)
            else:
                self.sharing_units_dict[str(i)] = CrossStitchUnit(nb_channels=num_shared_channels[i - 1],
                                                                  nb_task=self.__nb_task,
                                                                  c=c,
                                                                  spread=spread)

    def shared_forward(self,
                       x: Sequence[torch.Tensor],
                       sharing_level: int) -> List[torch.Tensor]:
        """
        Reshape the features and compute the forward pass of the shared unit at the sharing level i.

        :param x: A list of torch.Tensor that represent the features extract by each sub neural net before the
                  given sharing level.
        :param sharing_level: An integer that indicate which sharing unit will be used.
        :return: A tuple of torch.Tensor that represent the shared features representation for each sub neural net.
        """

        num_chan = [tensor.size()[1] for tensor in x]
        b_size, _, depth, width, height = x[0].size()
        num_sub = self.__subspace[sharing_level - 1]

        if self.__sharing_unit == "sluice":
            out = torch.cat(
                [x[i].view(b_size, num_sub[i], int(num_chan[i] / num_sub[i]), depth, width, height)
                    for i in range(self.__nb_task)],
                dim=1
            )

            out = self.sharing_units_dict[str(sharing_level)](out)
            out = list(torch.split(out, tuple(num_sub), dim=1))

            for i in range(len(out)):
                out[i] = out[i].reshape(b_size, num_chan[i], depth, width, height)

        else:
            out = list(self.sharing_units_dict[str(sharing_level)](torch.stack(x, dim=0)))

        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mixup_key_list = list(self.mixup.keys())
        b_size = x.size()[0]

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
        outs = self.shared_forward(outs, 1)
        if "1" in mixup_key_list:
            for count, task in enumerate(self.__tasks):
                outs[count] = self.mixup["1"](outs[count])

        # --------------------------------------------
        #                   LEVEL 2
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers2(outs[count])

        outs = self.shared_forward(outs, 2)
        if "2" in mixup_key_list:
            for count, task in enumerate(self.__tasks):
                outs[count] = self.mixup["2"](outs[count])

        # --------------------------------------------
        #                   LEVEL 3
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers3(outs[count])

        outs = self.shared_forward(outs, 3)

        if "3" in mixup_key_list:
            for count, task in enumerate(self.__tasks):
                outs[count] = self.mixup["3"](outs[count])

        # --------------------------------------------
        #                   LEVEL 4
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].layers4(outs[count])

        outs = self.shared_forward(outs, 4)

        # --------------------------------------------
        #                   POOLING
        # --------------------------------------------
        for count, task in enumerate(self.__tasks):
            outs[count] = self.nets[task].avg_pool(outs[count])

        # --------------------------------------------
        #                   FC_LAYERS
        # --------------------------------------------
        preds = {}
        for count, task in enumerate(self.__tasks):
            preds[task] = self.nets[task].fc_layer(outs[count].view(b_size, -1))

        return preds
