from Model.Block import PrimaryCapsule, DynamicHighCapsule
from Model.NeuralNet import NeuralNet
from monai.networks.layers.factories import Act
import numpy as np
from Trainer.Utils import init_weights, to_one_hot
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Union


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

        # print(out)
        pred = torch.linalg.norm(out, dim=-1)
        return F.softmax(pred, dim=-1)
