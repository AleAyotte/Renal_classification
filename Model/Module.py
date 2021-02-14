import numpy as np
import torch
from typing import Tuple, Sequence


class SluiceUnit(torch.nn.Module):
    """
    Create a SluiceUnit
    ...
    Attributes
    ----------
    alpha: torch.nn.Parameter
        A torch Tensor that represent parameters of the Sluice Unit. Those parameters are
        a matrix NxN that learned how the information between N subspace should be shared.
    """
    def __init__(self,
                 nb_subspace: int,
                 c: float = 0.9,
                 spread: float = 0.1):
        """
        Initialize a sluice units.

        :param nb_subspace: The number of subspace that will be shared.
        :param c: A float that represent the conservation parameter.
        :param spread: A float that represent the spread parameters.
        """
        super().__init__()
        mean = (1-c)/(nb_subspace - 1)
        std = spread/(nb_subspace - 1)

        alpha = np.random.normal(mean, std, (nb_subspace, nb_subspace))
        alpha[np.diag_indices_from(alpha)] = c

        alpha = torch.from_numpy(alpha).float()
        self.alpha = torch.nn.Parameter(data=alpha, requires_grad=True)
        # Batch, Subspace, Channels, Depth, Height, Width

    def forward(self, x):
        x = x.transpose(1, x.ndim - 1)  # We transpose the subspace dimension with the last dimension
        out = torch.matmul(x[:, :, :, :, :, None, :], self.alpha).squeeze()
        return out.transpose(1, 5)


class CrossStitchUnit(torch.nn.Module):
    """
    Create a Cross-Stitch Unit
    ...
    Attributes
    ----------
    alpha: torch.nn.Parameter
        A torch Tensor that represent parameters of the Cross-Stitch Unit. Those parameters are
        a matrix NxN that learned how the information between N subspace should be shared.
    """
    def __init__(self,
                 nb_channels: int,
                 nb_task: int,
                 c=0.9,
                 spread=0.1):
        """

        :param nb_task: The number of network to combine.
        :param nb_channels: Number of channels in the latent space PER NETWORK.
        :param c: A float that represent the conservation parameter.
        :param spread: A float that represent the spread parameters.
        """
        super().__init__()
        mean = (1 - c) / (nb_task - 1)
        std = spread / (nb_task - 1)

        alpha = []
        for t in range(nb_task):
            temp = np.random.normal(mean, std, (nb_channels, nb_task))
            temp[:, t] = c
            alpha.append(temp)

        alpha = torch.from_numpy(np.array(alpha)).float().swapaxes(1, 2)
        alpha = alpha.swapaxes(1, 2)  # Output, Task, Channels

        self.alpha = torch.nn.Parameter(data=alpha, requires_grad=True)

    def forward(self, x):
        # Output, Task, Batch, Channels, Depth, Width, Height
        out = torch.mul(self.alpha[:, :, None, :, None, None, None], x[None, :, :, :, :, :, :])
        return out.sum(1)


class Mixup(torch.nn.Module):
    """
    Create a MixUp module.
    ...
    Attributes
    ----------
    __beta: float
        The parameter of beta distribution. The lamb parameter will be sample in a distribution Beta(__beta, __beta)
    lamb: float
        The parameter that will be used to compute the simplex between two images.
    __batch_size: int
        The length of the image batch that will mix.
    permut: Sequence[int]
        The permutation indices that will be used to shuffle the features before the next foward pass.
    enable: bool
        If true, the current module will mix the next batch of images.
    """
    def __init__(self, beta_params):
        """
        The constructor of a mixup module.
        :param beta_params: One single value for the two parameters of a beta distribution
        """

        super().__init__()
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
