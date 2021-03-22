from Model.NeuralNet import NeuralNet
from monai.networks.layers.factories import Act
import numpy as np
from Trainer.Utils import init_weights, to_one_hot
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from typing import Sequence, Tuple, Union


def squash(capsules: torch.Tensor,
           dim: int = -1) -> torch.Tensor:
    """
    Reduce the norm of a vector between 0 and 1 without changing the orientation.

    :param capsules: A tensor that represent the capsules to normalize.
    :param dim: Along which dimension should we normalize the capsules.
    :return: A torch.Tensor that represent the normalized capsules.
    """
    norm = torch.linalg.norm(capsules, dim=dim, keepdim=True)
    return norm / (1 + norm ** 2) * capsules


class PrimaryCapsule(nn.Module):
    def __init__(self,
                 dimension: int,
                 caps_dim: int,
                 in_channels: int,
                 kernel: Union[int, Sequence[int]],
                 num_primary_caps: int,
                 padding: Union[Sequence[int], int] = 0,
                 stride: int = 1
                 ):
        """
        The construtor of the PrimaryCapsule class

        :param dimension: The number of dimension of the convolution.
        :param caps_dim: The output dimension of the capsules.
        :param in_channels: The number of input channels for the convolution.
        :param kernel: The convolution's kernel size.
        :param num_primary_caps: The number of primary capsule block. The number of outputs channels will be
                                 equal to num_primary_caps * caps_dim.
        :param padding: The convolution's padding size.
        :param stride: The convolution stride parameter.
        """

        super().__init__()
        assert dimension in [2, 3], "The PrimaryCapule layer can only be of dimension 2 or 3."

        self.caps_dim = caps_dim

        if dimension == 2:
            self.conv = nn.Conv2d(kernel_size=kernel, in_channels=in_channels,
                                  out_channels=num_primary_caps * caps_dim,
                                  stride=stride, padding=padding)
        else:
            self.conv = nn.Conv3d(kernel_size=kernel, in_channels=in_channels,
                                  out_channels=num_primary_caps * caps_dim,
                                  stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).view(x.size()[0], self.caps_dim, -1).transpose(1, 2)
        return squash(out)


class DynamicHighCapsule(nn.Module):
    """
    An implementation of an high level capusle layer that use the dynamic routing-by-agreements
    """
    def __init__(self,
                 in_caps: int,
                 in_caps_dim: int,
                 out_caps: int,
                 out_caps_dim: int,
                 num_routing_iter: int = 3):
        """
        The constructor of the DynamicHighCapsule class

        :param in_caps:
        :param in_caps_dim:
        :param out_caps:
        :param out_caps_dim:
        :param num_routing_iter:
        """

        super().__init__()
        assert num_routing_iter > 0, "The number of routing iter should be greater than 0."

        self.in_caps = in_caps
        self.num_iter = num_routing_iter
        self.out_caps = out_caps

        self.priors_weights = nn.Parameter(torch.randn(out_caps, in_caps, in_caps_dim, out_caps_dim),
                                           requires_grad=True)
        print(np.prod(self.priors_weights.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        priors = torch.matmul(
            x[:, None, :, None, :],
            self.priors_weights[None, :, :, :, :]
        ).transpose(2, 4)

        coupling_coef = Variable(torch.zeros(*priors.size())).cuda()

        # The routing by agreements
        with torch.no_grad():
            for it in range(self.num_iter - 1):
                probs = F.softmax(coupling_coef, dim=-1)
                out = squash((probs * priors).sum(dim=-1, keepdims=True), dim=2)

                delta_logits = (priors * out).sum(dim=2, keepdims=True)
                coupling_coef = coupling_coef + delta_logits

        probs = F.softmax(coupling_coef, dim=-1)
        return squash((probs * priors).sum(dim=-1).squeeze())


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        labels = to_one_hot(labels, num_classes=2)

        left = F.relu(0.9 - pred, inplace=True) ** 2
        right = F.relu(pred - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        return margin_loss.mean()


class CapsNet2D(NeuralNet):
    def __init__(self,
                 in_shape: Sequence[int],
                 activation: str = "ReLU",
                 first_kernel: Union[Sequence[int], int] = 9,
                 high_caps_dim: Union[Sequence[int], int] = 16,
                 margin_loss: bool = False,
                 num_high_caps: Union[Sequence[int], int] = 2,
                 num_prim_caps: int = 32,
                 num_routing_iter: int = 3,
                 prim_caps_kernel: Union[Sequence[int], int] = 9,
                 prim_caps_dim: int = 8,
                 out_channels: int = 256
                 ):
        super().__init__()

        h_caps_dim = high_caps_dim if type(high_caps_dim) == list else [high_caps_dim]
        num_h_caps = num_high_caps if type(num_high_caps) == list else [num_high_caps]

        assert len(h_caps_dim) == len(num_h_caps), "You need to specify the number of capsule and the capsule " \
                                                   "dimension for each high level capsule layer."

        self.margin_loss = margin_loss
        self.in_shape = in_shape
        image_shape = np.array(in_shape[1:])

        self.conv = nn.Sequential(nn.Conv2d(kernel_size=first_kernel,
                                            in_channels=self.in_shape[0],
                                            out_channels=out_channels),
                                  Act[activation.lower()]())
                                  
        image_shape -= (first_kernel - 1)

        self.prim = PrimaryCapsule(2, caps_dim=prim_caps_dim,
                                   in_channels=out_channels, kernel=prim_caps_kernel,
                                   num_primary_caps=num_prim_caps, stride=2)

        image_shape -= (prim_caps_kernel - 1)
        image_shape = np.ceil(image_shape / 2).astype(int)
        num_capsule = np.prod(image_shape) * num_prim_caps
        caps_dim = prim_caps_dim

        high_layer = []
        for i in range(len(h_caps_dim)):
            high_layer.append(DynamicHighCapsule(num_capsule,
                                                 in_caps_dim=caps_dim,
                                                 out_caps=num_h_caps[i],
                                                 out_caps_dim=h_caps_dim[i],
                                                 num_routing_iter=num_routing_iter))
            num_capsule = num_h_caps[i]
            caps_dim = h_caps_dim[i]

        self.high_layer = nn.Sequential(*high_layer)

        self.apply(init_weights)

    def forward(self, images):
        if images.dim() == 5:
            images = images[:, 0, :, :, :]
        else:
            images = images[0, :, :, :].unsqueeze(dim=0)

        out = self.conv(images)
        out = self.prim(out)
        out = self.high_layer(out)

        pred = torch.linalg.norm(out, dim=-1)
        return F.softmax(pred, dim=-1)
