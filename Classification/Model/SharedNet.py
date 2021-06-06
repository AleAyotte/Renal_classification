"""
    @file:              SharedNet.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 06/2021

    @Description:       This file contain the SharedNet class. This class used three different neural network and
                        connect them with SluiceUnit or CrossStitchUnit to create a unique MultiTaskNetwork.

    @Reference:         1) I. Misra, et al. Cross-stitch networks formulti-task learning. IEEE Conference on Computer
                           Vision and PatternRecognition (CVPR), 2016
                        2) S. Ruder et al. Latent Multi-Task Architecture Learning. Proceedings of the AAAI
                           Conferenceon Artificial Intelligence, 2019
"""

from Constant import SharingUnits
from Model.Module import CrossStitchUnit, SluiceUnit, UncertaintyLoss
from Model.NeuralNet import NeuralNet
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Final, List, Optional, Sequence, Union


NB_LEVELS: Final = 4


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
    nets : nn.ModuleDict
        A dictionnary that contain two neural network. One for each task.
        They are referenced by the shortname of their corresponding task. (mal, sub)
    __nb_task : int
        The number of tasks.
    __sharing_unit : SharingUnits
        Indicate which type of sharing unit are present in the neural network.
    sharing_units_dict : nn.ModuleDict
        A dictionnary that contain all the sharing_unit module.
        These sharing unit modules are referenced by their level.
    __subspace : np.array[int]
        A list of int that indicate the number of subspace for each network.
    __tasks : List[str]
        The list of all tasks name.
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
    shared_forward(sharing_level: int, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        Reshape the features and compute the forward pass of the shared unit at a gived sharing level.
    """
    def __init__(self,
                 sub_nets: nn.ModuleDict,
                 c: float = 0.9,
                 num_shared_channels: Optional[Sequence[int]] = None,
                 sharing_unit: SharingUnits = SharingUnits.SLUICE,
                 spread: float = 0.1,
                 subspace_1: Union[Dict[str, int], int] = 0,
                 subspace_2: Union[Dict[str, int], int] = 0,
                 subspace_3: Union[Dict[str, int], int] = 0,
                 subspace_4: Union[Dict[str, int], int] = 0):
        """
        Create a Shared network with Shared Module like Sluice Unit or Cross-Stitch Unit.

        :param sub_nets:
        :param c: A float that represent the conservation parameter of the sharing units.
        :param num_shared_channels: A list of int that indicate the number of channels per network before the
                                    cross-stitch unit. Only used if sharing_unit == "cross_stitch"
        :param sharing_unit: The sharing unit that will be used to shared the information between the 3 network.
        :param spread: A float that represent the spread parameter of the sharing units.
        :param subspace_1: A list of int that indicate the number of subspace in each network before the sluice unit 1.
        :param subspace_2: A list of int that indicate the number of subspace in each network before the sluice unit 2.
        :param subspace_3: A list of int that indicate the number of subspace in each network before the sluice unit 3.
        :param subspace_4: A list of int that indicate the number of subspace in each network before the sluice unit 4.
        """
        super().__init__()

        # --------------------------------------------
        #                    NETS
        # --------------------------------------------
        self.nets = sub_nets
        self.__tasks = list(self.nets.keys())
        self.__nb_task = len(self.__tasks)

        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        self.uncertainty_loss = UncertaintyLoss(num_task=self.__nb_task)

        # --------------------------------------------
        #               SHARING UNITS
        # --------------------------------------------
        assert type(sharing_unit) is SharingUnits, "The sharing_unit should of type SharingUnits. See Constant.py."
        self.__sharing_unit = sharing_unit

        self.__subspace = np.zeros((NB_LEVELS, self.__nb_tasks))
        for i in range(NB_LEVELS):
            subspace = [subspace_1, subspace_2, subspace_3, subspace_4][i]

            if type(subspace) is not dict:
                self.__subspace[i, :] = np.array([subspace for _ in range(self.__nb_tasks)])
            else:
                self.__subspace[i, :] = np.array([subspace[task] for task in self.__tasks])

        else:
            assert len(num_shared_channels) == NB_LEVELS, "You must give the number of shared channels PER NETWORK " \
                                                          f"for each shared unit. Only {len(num_shared_channels)} " \
                                                           "were gived"

        self.sharing_units_dict = nn.ModuleDict()
        for i in range(1, NB_LEVELS+1):
            if sharing_unit is SharingUnits.SLUICE:
                if self.__subspace[i - 1].sum() != 0:
                    self.sharing_units_dict[str(i)] = SluiceUnit(self.__subspace[i - 1].sum(),
                                                                 c,
                                                                 spread)
            else:
                if num_shared_channels != 0:
                    self.sharing_units_dict[str(i)] = CrossStitchUnit(nb_channels=num_shared_channels[i - 1],
                                                                      nb_task=self.__nb_task,
                                                                      c=c,
                                                                      spread=spread)

    def shared_forward(self,
                       sharing_level: int,
                       x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reshape the features and compute the forward pass of the shared unit at the sharing level i.

        :param sharing_level: An integer that indicate which sharing unit will be used.
        :param x: A list of torch.Tensor that represent the features extract by each sub neural net before the
                  given sharing level.
        :return: A tuple of torch.Tensor that represent the shared features representation for each sub neural net.
        """

        if str(sharing_level) in list(self.sharing_units_dict.keys()):
            num_chan = [tensor.size()[1] for tensor in x]
            b_size, _, depth, width, height = x[0].size()

            if self.__sharing_unit is SharingUnits.SLUICE:
                num_sub = self.__subspace[sharing_level - 1]
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
        # Skip the sharing level
        else:
            return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
