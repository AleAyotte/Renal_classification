import numpy as np
from random import randint
from torch import nn
from typing import Sequence, Tuple, Union


class NeuralNet(nn.Module):
    """
    Define the Neural Network abstract class
     ...
    Attributes
    ----------
    mixup: nn.ModuleDict
        A dictionnary that contain all the mixup module.
    Methods
    -------
    set_mixup(b_size : int)
        Set the b_size parameter of each mixup module.
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    """
    def __init__(self):
        super().__init__()
        self.mixup = nn.ModuleDict()

    def forward()
        pass

    def activate_mixup(self) -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]:
        """
        Choose randomly a mixup module and activate it.
        """
        key_list = list(self.mixup.keys())
        rand_key = key_list[randint(0, len(key_list) - 1)]

        lamb, permut = self.mixup[rand_key].sample()
        return rand_key, lamb, permut

    def disable_mixup(self, key: int = -1):
        """
        Disable one or all mixup module.

        :param key: The name of the mixup module in the dictionary self.mixup.
                    If key=-1 then all mixup module will be disable.
        """
        if key == -1:
            for module in self.mixup.values():
                module.enable = False
        else:
            self.mixup[key].enable = False
    
    def set_mixup(self, b_size: int):
        """
        Set the b_size parameter of each mixup module.

        :param b_size: An integer that represent the batch size parameter.
        """
        for module in self.mixup.values():
            module.set_bacth_size(b_size)