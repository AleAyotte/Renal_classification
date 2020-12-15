import numpy as np
import torch
from typing import Tuple, Sequence


class Mixup(torch.nn.Module):
    """
    Create a pre activation or post activation 3D Residual Network.
     ...
    Attributes
    ----------
    """
    def __init__(self, beta_params):
        """
        The constructor of a mixup module.
        :param beta_params: One single value for the two parameters of a beta distribution
        """

        torch.nn.Module.__init__(self)
        self.__beta = beta_params
        self.lamb = 1
        self.__batch_size = 0
        self.permut = None
        self.enable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mixup module. This is here were we mix the data.

        :return: Return mixed data if the network is in train mode and the module is enable.
        """

        if self.training and self.enable:
            device = x.get_device()
            lamb = torch.from_numpy(np.array([self.lamb]).astype('float32')).to(device)
            lamb = torch.autograd.Variable(lamb)
            return lamb*x + (1 - lamb)*x[self.permut.to(device)]
        else:
            return x

    def get_mix_params(self) -> Tuple[float, Sequence[int]]:
        """
        :return: The constant that will be use to mixup the data for the next iteration and a list of index that
                 represents the permutation used for the mixing process.
        """

        return self.lamb, self.permut

    def sample(self) -> Tuple[float, Sequence[int]]:
        """
        Sample a point in a beta distribution to prepare the mixup
        :return: The coefficient used to mixup the training features during the next foward pass.
                 The permutation indices that will be used to shuffle the features before the next foward pass.
        """

        if self.__beta > 0:
            # We activate the module and we sample a value in his beta distribution
            self.enable = True
            self.lamb = np.random.beta(self.__beta, self.__beta)
        else:
            self.lamb = 1

        self.permut = torch.randperm(self.__batch_size)

        return self.lamb, self.permut

    def set_batch_size(self, b_size: int):
        self.__batch_size = b_size
