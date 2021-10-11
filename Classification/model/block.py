"""
    @file:              block.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 08/2021

    @Description:       This file contain some generic module used to create several model like the ResNet,
                        MultiLevelResNet and CapsNet. The module are DynamicHighCapsule, PrimaryCapsule, Resblock and
                        ResBottleneck.

    @Reference:         1) Identity Mappings in Deep Residual Networks, He, K. et al., ECCV 2016
                        2) CBAM: Convolutional Block Attention Module, Woo, S et al., ECCV 2018
                        3) End-To-End Multi-Task Learning With Attention, Liu, S. et al, CVPR 2019
                        4) Learning to Branch for Multi-Task Learning, Guo, P. et al., CoRR 2020
"""
from __future__ import annotations

from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers import split_args
from monai.networks.layers.factories import Act, Norm
import numpy as np
import torch
from torch import nn
from typing import Final, List, NewType, Optional, Sequence, Tuple, Type, Union

from model.module import GumbelSoftmax


class BranchingBlock(nn.Module):
    """
    The Branching block that will be used in the Learn-To-Branch model as described in ref 4)

    ...
    Attributes
    ----------
    __active_children : List[int]
        A list of integer that indicate which children is use as parent in the next branching block.
    __active_parents :  List[int]
        A list of integer that indicate the index of the input that will be used by every children in the forward pass.
    blocks : nn.ModuleList
        A list of block that correspond to the children nodes.
    __frozen : bool
        A boolean that indicate if the architecture search is finish.
    gumbel_softmax : GumbelSoftmax
        The gumbel softmax layer that learn which parents should be connected to the childrens.
    __layer_type : str
        Indicate the type of layer that is use in the current block. Can be 'conv' or 'linear'.
    __num_block : int
        The number of children nodes.
    Methods
    -------
    get_weights() -> Union[torch.nn.Parameter, Iterator[torch.nn.Parameter]]
        Get the blocks or gumbel softmax parameters.
    forward() -> torch.Tensor
        The forward pass of the gumbel softmax layer.
    """
    BLOCK_LIST_TYPE: Final = Union[NewType("PreResBlock", type),
                                   NewType("PreResBottleneck", type),
                                   NewType("ResBlock", type),
                                   NewType("ResBottleneck", type),
                                   Type[nn.Linear]]

    def __init__(self,
                 block_list: Sequence[BLOCK_LIST_TYPE],
                 num_input: int,
                 tau: float = 1,
                 **kwargs):
        """
        Create a gumbel softmax block

        :param block_list: A list of block class to instantiate.
        :param num_input: The number of parent nodes.
        :param tau: non-negative scalar temperature parameter of the gumble softmax operation.
        :param kwargs: A dictionary of parameters that will be used to instantiate the blocks.
        """
        super().__init__()
        self.__num_block = len(block_list)
        self.__active_children = range(self.__num_block)
        self.__active_parents = range(num_input)
        self.__frozen = False
        self.gumbel_softmax = GumbelSoftmax(num_input=num_input, num_output=self.__num_block, tau=tau)
        self.blocks = nn.ModuleList()

        for block in block_list:
            if block is nn.modules.linear.Linear:
                self.__layer_type = "linear"
                self.blocks.append(block(in_features=kwargs["in_features"],
                                         out_features=kwargs["out_features"]))
            else:
                self.__layer_type = "conv"
                self.blocks.append(block(fmap_in=kwargs["fmap_in"],
                                         fmap_out=kwargs["fmap_out"],
                                         kernel=kwargs["kernel"],
                                         strides=kwargs["strides"],
                                         drop_rate=kwargs["drop_rate"],
                                         activation=kwargs["activation"],
                                         norm=kwargs["norm"]))

    def get_weights(self, gumbel_softmax_weights: bool = False) -> Union[torch.nn.Parameter,
                                                                         List[torch.nn.Parameter]]:
        """
        Get the blocks or gumbel softmax parameters.

        :param gumbel_softmax_weights: If true, the gumbel softmax layer weights will be returned. Else, it will be
                                       the blocks weights that will be returned.
        :return: The parameters blocks parameters or the gumbel softmax parameters.
        """

        if gumbel_softmax_weights:
            return list(self.gumbel_softmax.parameters())
        else:
            return list(self.blocks.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the GumbelSoftmaxBlock

        :param x: A torch.Tensor that represent the stacked output of the parent nodes.
        :return: A torch.Tensor that represent the stacked output of the children nodes.
        """
        if not self.__frozen:
            probs = self.gumbel_softmax()

            if self.__layer_type == "linear":
                # Output, Input, Batch, features
                out = torch.mul(probs[:, :, None, None], x[None, :, :, :]).sum(1)
            else:
                # Output, Input, Batch, Channel, Depth, Width, Height
                out = torch.mul(probs[:, :, None, None, None, None, None], x[None, :, :, :, :, :, :]).sum(1)
            return torch.stack([self.blocks[i](out[i]) for i in range(self.__num_block)])
        else:
            out = []
            for child, parent in zip(self.__active_children, self.__active_parents):
                out.append(self.blocks[child](x[parent]))
            return torch.stack(out)

    def freeze_branch(self, children: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """
        Freeze the branching operation for an optimized forward pass and return a list that indicate the active parents.

        :param children: A list of int that indicate the children that are use as parents
                         by next block's active children.
        :return: Return a list of int that indicate the parents that are used by the active children and the same list
                 but without repetition of parents.
        """
        self.__active_children = children if children is not None else self.__active_children
        self.__frozen = True
        parents = torch.argmax(self.gumbel_softmax(), dim=-1).cpu().numpy()
        active_parents = [parents[child] for child in self.__active_children]

        # Return the list of used parent without repetition. Ex: [1, 4, 2, 4] -> [1, 4, 2]
        _, index = np.unique(np.array(active_parents), return_index=True)
        unique_parent = list(np.array(active_parents)[np.sort(index)])

        # The parent's index of each child. Since only the active parents will produce an output, the length
        # of the input vector will be equal to the length of unique_parent and not the total number of parents.
        # So we need to shift the indices and put them in the correct order.
        # Ex: [1, 4, 2, 4] -> [0, 1, 2, 1]
        self.__active_parents = [unique_parent.index(x) for x in active_parents]

        return active_parents, unique_parent


class CBAM(nn.Module):
    """
    A 3D version of the Convolutional Block Attention Module as described in Ref 2).

    ...
    Attributes
    ----------
    channel_att_block: ChannelAttBlock
        A 3D version of the Channel Attention Block that will be applied before the Spatial Attention Block
    spatial_att_block: SpatialAttBlock
        A 3D version of the Spatial Attention Block
    """

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 1,
                 norm: str = "batch",
                 squeeze_factor: int = 8,
                 subsample: Optional[nn.Module] = None):
        """
        Create a Convolutional Block Attention Module.

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps of the mask.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param squeeze_factor: A coefficient that will divide fmap_in to determine the number of intermediate feature
                               maps.
        :param subsample: A block or a torch.nn.module that will applied to the masked tensor to reduce the dimension
                          of the output.
        """
        super().__init__()
        self.channel_att_block = ChannelAttBlock(fmap_in=fmap_in, fmap_out=fmap_out,
                                                 squeeze_factor=squeeze_factor)
        self.spatial_att_block = SpatialAttBlock(fmap_in=fmap_out, fmap_out=fmap_out,
                                                 kernel=kernel, norm=norm,
                                                 squeeze_factor=squeeze_factor,
                                                 subsample=subsample)

    def forward(self,
                x: torch.Tensor,
                to_mask: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Channel Attention Block.

        :param x: Input tensor of the attention module.
        :param to_mask: Tensor that will be multiply with the mask produced by the attention module.
        :return: The downsampled masked input as torch.Tensor.
        """

        out = self.channel_att_block(x, to_mask)
        return self.spatial_att_block(out, out)


class ChannelAttBlock(nn.Module):
    """
    A 3D version of the Channel Attention Block as described in Ref 2).

    ...
    Attributes
    ----------
    att: nn.Sequential
        A Sequential module that contain all layer that form the spatial attention block.
    subsample: nn.Module
        A pytorch module or a block that reduce the dimension of the output tensor. If subsample block is given during
        __init__, then an identity layer will be used.
    """

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 squeeze_factor: int,
                 subsample: Optional[nn.Module] = None):
        """
        Create a Channel Attention Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps of the mask.
        :param squeeze_factor: A coefficient that will divide fmap_in to determine the number of intermediate feature
                               maps.
        :param subsample: A block or a torch.nn.module that will applied to the masked tensor to reduce the dimension
                          of the output.
        """
        super().__init__()

        num_int_node = fmap_in // squeeze_factor
        self.avg = nn.AdaptiveAvgPool3d(output_size=1)
        self.max = nn.AdaptiveMaxPool3d(output_size=1)
        self.att = nn.Sequential(nn.Flatten(start_dim=2),
                                 nn.Linear(fmap_in, num_int_node),
                                 nn.Linear(num_int_node, fmap_out))
        self.sigmoid = nn.Sigmoid()

        self.subsample = subsample if subsample is not None else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                to_mask: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Channel Attention Block.

        :param x: Input tensor of the attention module.
        :param to_mask: Tensor that will be multiply with the mask produced by the attention module.
        :return: The downsampled masked input as torch.Tensor.
        """
        b, c = to_mask.size()[0:2]
        out = torch.cat((self.avg(x).unsqueeze(1), self.max(x).unsqueeze(1)), dim=1)
        out = self.att(out)
        out = self.sigmoid(torch.sum(out, dim=1)).view(b, c, 1, 1, 1)
        out = torch.mul(to_mask, out.expand_as(to_mask))

        return self.subsample(out)


class PreResBlock(nn.Module):
    """
    A 3D version of the preactivation residual bottleneck block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    first_activation: nn.module
        The non linear activation function that is applied before the forward pass in the residual mapping.
    first_normalization: nn.module
        The normalization layer that is applied before the forward pass in the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = True,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out, kernel_size=1, stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation](**args)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=kernel, stride=strides,
                      padding=padding, bias=bias,
                      groups=groups if split_layer is False else 1),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                      padding=padding, bias=bias,
                      groups=groups)
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.first_normalization(x)
        out = self.first_activation(out)

        if self.__subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class PreResBottleneck(nn.Module):
    """
    A 3D version of the preactivation residual block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    first_activation: nn.module
        The non linear activation function that is applied before the forward pass in the residual mapping.
    first_normalization: nn.module
        The normalization layer that is applied before the forward pass in the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 4

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = False,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        self.first_normalization = Norm[norm, 3](fmap_in)
        self.first_activation = Act[activation](**args)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups),
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.first_normalization(x)
        out = self.first_activation(out)

        if self.__subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class ResBlock(nn.Module):
    """
    A 3D version of the residual block as described in Ref 1).
    (Conv(kernel), Norm, Act, Conv(kernel), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    res: ResidualUnit
        A MONAI implementation of the Residual block. Act like an nn.Sequential.
    """
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "ReLU",
                 bias: bool = True,
                 drop_rate: float = 0,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a Residual Block using MONAI

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param strides: Convolution strides.
        """
        super().__init__()

        self.res = ResidualUnit(dimensions=3, in_channels=fmap_in, out_channels=fmap_out,
                                kernel_size=kernel, strides=strides, dropout=drop_rate,
                                dropout_dim=3, norm=norm, last_conv_only=False, bias=bias,
                                act=activation if activation != "PReLU" else ("prelu", {"init": 0.01}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res(x)

        return out


class ResBottleneck(nn.Module):
    """
    A 3D version of the residual bottleneck block as described in Ref 1).
    (Conv(1x1x1), Norm, Act, Conv(kernel), Norm, Act, Conv(1x1x1), Norm, Add, Act, Dropout)

    ...
    Attributes
    ----------
    last_activation: nn.module
        The non linear activation function that is applied after adding the shorcut to the residual mapping.
    residual_layer: nn.Sequential
        A serie of convolution, normalization and activation layer to play the role of residual mapping function.
    sub_conv: nn.Sequential
        A 3D convolution layer used to subsample the input features and to match the dimension of the shorcut output
        with the dimension of the residual mapping.
    __subsample: boolean
        A boolean that indicate if the input features will be subsample with a convolution layer with a stride of 2.
    """
    expansion = 4

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 activation: str = "relu",
                 bias: bool = True,
                 drop_rate: float = 0,
                 groups: int = 1,
                 kernel: Union[Sequence[int], int] = 3,
                 norm: str = "batch",
                 split_layer: bool = False,
                 strides: Union[Sequence[int], int] = 1):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps.
        :param activation: The activation function that will be used in the model.
        :param bias: The bias parameter of the convolution.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param groups: Number of group in the convolutions.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param strides: Convolution strides.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.__subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.__subsample = False
        else:
            self.__subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=bias,
                                      groups=groups if split_layer is False else 1)

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        # We initialize the PReLU with the LeakyReLU default parameter.
        if activation == "PReLU":
            _, args = split_args((activation, {"init": 0.01}))
        else:
            _, args = split_args(activation)

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups if split_layer is False else 1,),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out),
            Act[activation](**args),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=bias,
                      groups=groups),
            Norm[norm, 3](fmap_out*self.expansion)
        ]

        if drop_rate > 0:
            res_layer.extend([nn.Dropout3d(drop_rate)])

        self.residual_layer = nn.Sequential(*res_layer)
        self.last_activation = Act[activation](**args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        if self.__subsample:
            shortcut = self.sub_conv(x)
        else:
            shortcut = x

        out = self.residual_layer(x) + shortcut

        return self.last_activation(out)


class SpatialAttBlock(nn.Module):
    """
    A 3D version of the Spatial Attention Block as described in Ref 3)

    ...
    Attributes
    ----------
    att: nn.Sequential
        A Sequential module that contain all layer that form the spatial attention block.
    subsample: nn.Module
        A pytorch module or a block that reduce the dimension of the output tensor. If subsample block is given during
        __init__, then an identity layer will be used.
    """
    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 1,
                 norm: str = "batch",
                 squeeze_factor: int = 8,
                 subsample: Optional[nn.Module] = None):
        """
        Create a Spatial Attention Block.

        :param fmap_in: Number of input feature maps.
        :param fmap_out: Number of output feature maps of the mask.
        :param kernel: Kernel size as integer. (Example: 3.  For a 3x3 kernel)
        :param norm: The normalization layer name that will be used in the model.
        :param squeeze_factor: A coefficient that will divide fmap_in to determine the number of intermediate feature
                               maps.
        :param subsample: A block or a torch.nn.module that will applied to the masked tensor to reduce the dimension
                          of the output.
        """
        super().__init__()

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        num_int_map = fmap_in // squeeze_factor

        self.att = nn.Sequential(
            nn.Conv3d(fmap_in, num_int_map, kernel_size=kernel, padding=padding),
            Norm[norm, 3](num_int_map),
            nn.ReLU(),
            nn.Conv3d(num_int_map, fmap_out, kernel_size=kernel, padding=padding),
            Norm[norm, 3](fmap_out),
            nn.Sigmoid()
        )

        self.subsample = subsample if subsample is not None else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                to_mask: torch.tensor) -> torch.Tensor:
        """
        Define the forward pass of the Spatial Attention Block.

        :param x: Input tensor of the attention module.
        :param to_mask: Tensor that will be multiply with the mask produced by the attention module.
        :return: The downsampled masked input as torch.Tensor.
        """
        out = to_mask * self.att(x)

        return self.subsample(out)
