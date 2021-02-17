from Model.Module import Mixup, UncertaintyLoss
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act
from Model.NeuralNet import NeuralNet
import numpy as np
from Trainer.Utils import init_weights
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union


class PreResBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 groups: int = 1,
                 split_layer: bool = False,
                 drop_rate: float = 0,
                 activation: str = "relu"):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param groups: Number of group in the convolutions.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used in the model.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.subsample = False
        else:
            self.subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out, kernel_size=1, stride=strides, bias=False,
                                      groups=groups if split_layer is False else 1)

        self.bn1 = nn.BatchNorm3d(fmap_in)
        self.activation1 = Act[activation]()

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=kernel, stride=strides,
                      padding=padding, bias=False,
                      groups=groups if split_layer is False else 1,),
            nn.BatchNorm3d(fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                      padding=padding, bias=False,
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

        out = self.bn1(x)
        out = self.activation1(out)

        if self.subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class PreResBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 groups: int = 1,
                 split_layer: bool = False,
                 drop_rate: float = 0,
                 activation: str = "relu"):
        """
        Create a PreActivation Residual Block

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param groups: Number of group in the convolutions.
        :param split_layer: If true, then the first convolution and the shortcut
                            will ignore the groups parameter.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used in the model.
        """
        super().__init__()

        if type(strides) == int and strides == 1:
            self.subsample = False
        elif type(strides) == Sequence and np.sum(strides) == 3:
            self.subsample = False
        else:
            self.subsample = True
            self.sub_conv = nn.Conv3d(fmap_in, fmap_out*self.expansion, kernel_size=1,
                                      stride=strides, bias=False,
                                      groups=groups if split_layer is False else 1)

        self.bn1 = nn.BatchNorm3d(fmap_in)
        self.activation1 = Act[activation]()

        if type(kernel) == int:
            padding = int((kernel - 1)/2)
        else:
            padding = [int((ker - 1)/2) for ker in kernel]

        res_layer = [
            nn.Conv3d(fmap_in, fmap_out, kernel_size=1,
                      stride=1, bias=False,
                      groups=groups if split_layer is False else 1,),
            nn.BatchNorm3d(fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out, kernel_size=kernel,
                      stride=strides, padding=padding, bias=False,
                      groups=groups),
            nn.BatchNorm3d(fmap_out),
            Act[activation](),
            nn.Conv3d(fmap_out, fmap_out*self.expansion, kernel_size=1,
                      stride=1, bias=False,
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

        out = self.bn1(x)
        out = self.activation1(out)

        if self.subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 drop_rate: float = 0,
                 activation: str = "ReLU"):
        """
        Create a PreActivation Residual Block using MONAI

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used
        """
        super().__init__()

        self.res = ResidualUnit(dimensions=3, in_channels=fmap_in, out_channels=fmap_out,
                                kernel_size=kernel, strides=strides, dropout=drop_rate,
                                dropout_dim=3, act=activation, norm="BATCH",
                                last_conv_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res(x)

        return out


class PreResBlock2(nn.Module):
    expansion = 1

    def __init__(self,
                 fmap_in: int,
                 fmap_out: int,
                 kernel: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 1,
                 drop_rate: float = 0,
                 activation: str = "ReLU"):
        """
        Create a PreActivation Residual Block using MONAI
        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param strides: Convolution strides.
        :param drop_rate: The hyperparameter of the Dropout3D module.
        :param activation: The activation function that will be used
        """
        super().__init__()

        self.bn = nn.BatchNorm3d(fmap_in)
        self.act = Act[activation]()
        self.res = ResidualUnit(dimensions=3, in_channels=fmap_in, out_channels=fmap_out,
                                kernel_size=kernel, strides=strides, dropout=drop_rate,
                                dropout_dim=3, act=activation, norm="BATCH",
                                last_conv_only=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.act(out)
        out = self.res(out)

        return out


class ResNet(NeuralNet):
    """
    Create a pre activation or post activation 3D Residual Network.
     ...
    Attributes
    ----------
    conv: Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layer: nn.Linear
        The last fully connected layer.
    __in_channels: int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
    layers1: nn.Sequential
        The first series of residual block.
    layers2: nn.Sequential
        The second series of residual block.
    layers3: nn.Sequential
        The third series of residual block.
    layers4: nn.Sequential
        The last series of residual block.
    mixup: nn.ModuleDict
        A dictionnary that contain all the mixup module.
    __num_flat_features: int
        Number of features at the output of the last convolution.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    set_mixup(b_size : int)
        Set the b_size parameter of each mixup module.
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    """
    def __init__(self,
                 depth: int = 18,
                 first_channels: int = 16,
                 num_classes: int = 2,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 first_kernel: Union[Sequence[int], int] = 3,
                 kernel: Union[Sequence[int], int] = 3,
                 mixup: Sequence[int] = None,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 act: str = "ReLU",
                 pre_act: bool = True):

        super().__init__()

        # --------------------------------------------
        #                   MIXUP
        # --------------------------------------------
        assert mixup is None or len(mixup) == 4, "You should specify the 4 mixup parameters."
        mixup = [0, 0, 0, 0] if mixup is None else mixup

        for i in range(len(mixup)):
            if mixup[i] > 0:
                self.mixup[str(i)] = Mixup(mixup[i])

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if pre_act:
            block = {18: PreResBlock, 34: PreResBlock, 50: PreResBottleneck, 101: PreResBottleneck}
        else:
            block = {18: ResBlock, 34: ResBlock}

        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

        # --------------------------------------------
        #                   DROPOUT
        # --------------------------------------------
        num_block = int(np.sum(layers[depth]))

        if drop_type.lower() == "flat":
            temp = [drop_rate for _ in range(num_block)]
        elif drop_type.lower() == "linear":
            temp = [1 - (1 - (drop_rate * i / (num_block - 1))) for i in range(num_block)]
        else:
            raise NotImplementedError

        dropout = []
        for i in range(4):
            first = int(np.sum(layers[depth][0:i]))
            last = int(np.sum(layers[depth][0:i+1]))
            dropout.append(temp[first:last])

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        self.__in_channels = first_channels
        self.conv = Convolution(dimensions=3,
                                in_channels=3,
                                out_channels=self.__in_channels,
                                kernel_size=first_kernel,
                                act=act,
                                conv_only=pre_act)

        self.layers1 = self.__make_layer(block[depth], layers[depth][0],
                                         first_channels,
                                         kernel=kernel, strides=[2, 2, 1],
                                         drop_rate=dropout[0], act=act)
        self.layers2 = self.__make_layer(block[depth], layers[depth][1],
                                         first_channels * 2,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[1], act=act)
        self.layers3 = self.__make_layer(block[depth], layers[depth][2],
                                         first_channels * 4,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[2], act=act)
        self.layers4 = self.__make_layer(block[depth], layers[depth][3],
                                         first_channels * 8,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[3], act=act)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 16), int(in_shape[1] / 16),  int(in_shape[2] / 8)]

        self.avg_pool = nn.AvgPool3d(kernel_size=out_shape)

        self.__num_flat_features = self.__in_channels

        self.fc_layer = nn.Linear(self.__num_flat_features, num_classes)

        self.apply(init_weights)

    def __make_layer(self,
                     block,
                     num_block: int,
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     strides: Union[Sequence[int], int] = 1,
                     drop_rate: Sequence[float] = None,
                     act: str = "ReLU") -> nn.Sequential:
        layers = []

        for i in range(num_block):
            layers.append(block(fmap_in=self.__in_channels, fmap_out=fmap_out,
                                kernel=kernel,
                                strides=strides if i == 0 else 1,
                                drop_rate=drop_rate[i],
                                activation=act))
            self.__in_channels = fmap_out * block.expansion if i == 0 else self.__in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixup_key_list = list(self.mixup.keys())

        out = self.mixup["0"](x) if "0" in mixup_key_list else x
        out = self.conv(out)
        out = self.layers1(out)

        out = self.mixup["1"](out) if "1" in mixup_key_list else out
        out = self.layers2(out)

        out = self.mixup["2"](out) if "2" in mixup_key_list else out
        out = self.layers3(out)

        out = self.mixup["3"](out) if "3" in mixup_key_list else out
        out = self.layers4(out)
        out = self.avg_pool(out)

        features = out.view(-1, self.__num_flat_features)
        out = self.fc_layer(features)

        return out


class MultiLevelResNet(NeuralNet):
    """
    Create a pre activation or post activation 3D Residual Network.

    ...
    Attributes
    ----------
    conv: Convolution
        First block of the network. If pre_act is True then, its only a convolution. Else, its combination of
        convolution, activation et normalisation.
    fc_layer_mal: nn.Linear
        The last fully connected layer that will be used to classify the malignity of the tumor.
    fc_layer_sub_1: nn.Linear
        The first fully connected layer that will be used to classify the subtype of the tumor.
    fc_layer_sub_2: nn.Linear
        The last fully connected layer that will be used to classify the subtype of the tumor.
    fc_layer_grade_1: nn.Linear
        The first fully connected layer that will be used to classify the grade of the tumor.
    fc_layer_grade_2: nn.Linear
        The last fully connected layer that will be used to classify the grade of the tumor.
    __in_channels: int
        Number of output channels of the last convolution created. Used to determine the number of input channels of
        the next convolution to create.
    layers1: nn.Sequential
        The first series of residual block.
    layers2: nn.Sequential
        The second series of residual block.
    layers3: nn.Sequential
        The third series of residual block.
    layers4: nn.Sequential
        The last series of residual block.
    mixup: nn.ModuleDict
        A dictionnary that contain all the mixup module.
    __num_flat_features: int
        Number of features at the output of the last convolution.
    __split: int
        Indicate where the network will be split to a multi-task network. Should be an integer between 1 and 5.
        1 indicate that the network will be split before the first series of residual block.
        5 indicate that the network will be split after the last series of residual block.
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Execute the forward on a given torch.Tensor.
    set_mixup(b_size : int)
        Set the b_size parameter of each mixup module.
    activate_mixup() -> Tuple[int, Union[float, Sequence[float]], Sequence[int]]
        Choose randomly a mixup module and activate it.
    disable_mixup(key: int = -1):
        Disable a mixup module according to is key index in self.Mixup. If none is specified (key= -1), all mixup
        modules will be disable.
    """
    def __init__(self,
                 depth: int = 18,
                 first_channels: int = 16,
                 split_level: int = 4,
                 in_shape: Union[Sequence[int], Tuple] = (64, 64, 16),
                 first_kernel: Union[Sequence[int], int] = 3,
                 kernel: Union[Sequence[int], int] = 3,
                 mixup: Sequence[float] = None,
                 drop_rate: float = 0,
                 drop_type: str = "flat",
                 act: str = "ReLU",
                 pre_act: bool = True):

        super().__init__()
        self.__split = split_level
        self.__in_channels = first_channels
        # --------------------------------------------
        #                   MIXUP
        # --------------------------------------------
        assert mixup is None or len(mixup) == 4, "You should specify the 4 mixup parameters."
        mixup = [0, 0, 0, 0] if mixup is None else mixup

        for i in range(len(mixup)):
            if mixup[i] > 0:
                self.mixup[str(i)] = Mixup(mixup[i])

        # if depth in [50, 101]:
        #     raise NotImplementedError

        # --------------------------------------------
        #              UNCERTAINTY LOSS
        # --------------------------------------------
        self.uncertainty_loss = UncertaintyLoss(num_classes=3)

        # --------------------------------------------
        #                    BLOCK
        # --------------------------------------------
        if pre_act:
            block = {18: PreResBlock, 34: PreResBlock, 50: PreResBottleneck, 101: PreResBottleneck}
        else:
            block = {18: ResBlock, 34: ResBlock}

        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

        # --------------------------------------------
        #                   DROPOUT
        # --------------------------------------------
        num_block = int(np.sum(layers[depth]))

        if drop_type.lower() == "flat":
            temp = [drop_rate for _ in range(num_block)]
        elif drop_type.lower() == "linear":
            temp = [1 - (1 - (drop_rate * i / (num_block - 1))) for i in range(num_block)]
        else:
            raise NotImplementedError

        dropout = []
        for i in range(4):
            first = int(np.sum(layers[depth][0:i]))
            last = int(np.sum(layers[depth][0:i+1]))
            dropout.append(temp[first:last])

        # --------------------------------------------
        #                  CONV LAYERS
        # --------------------------------------------
        assert 1 <= split_level <= 5, "The split level should be an integer between 1 and 5."
        self.conv = Convolution(dimensions=3,
                                in_channels=3,
                                out_channels=self.__in_channels,
                                kernel_size=first_kernel,
                                act=act,
                                conv_only=pre_act)

        self.layers1 = self.__make_layer(block[depth], layers[depth][0],
                                         first_channels,
                                         kernel=kernel, strides=[2, 2, 1],
                                         drop_rate=dropout[0], act=act,
                                         split_layer=(1 == split_level),
                                         groups=3 if 1 >= split_level else 1)
        self.layers2 = self.__make_layer(block[depth], layers[depth][1],
                                         first_channels * 2,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[1], act=act,
                                         split_layer=(2 == split_level),
                                         groups=3 if 2 >= split_level else 1)
        self.layers3 = self.__make_layer(block[depth], layers[depth][2],
                                         first_channels * 4,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[2], act=act,
                                         split_layer=(3 == split_level),
                                         groups=3 if 3 >= split_level else 1)
        self.layers4 = self.__make_layer(block[depth], layers[depth][3],
                                         first_channels * 8,
                                         kernel=kernel, strides=[2, 2, 2],
                                         drop_rate=dropout[3], act=act,
                                         split_layer=(4 == split_level),
                                         groups=3 if 4 >= split_level else 1)

        # --------------------------------------------
        #                   FC LAYERS
        # --------------------------------------------
        in_shape = list(in_shape)
        out_shape = [int(in_shape[0] / 16), int(in_shape[1] / 16),  int(in_shape[2] / 8)]

        self.avg_pool = nn.AvgPool3d(kernel_size=out_shape)

        if self.__split == 5:
            self.__num_flat_features = self.__in_channels
        else:
            self.__num_flat_features = int(self.__in_channels / 3)

        self.fc_layer_mal = torch.nn.Sequential(torch.nn.Linear(self.__num_flat_features, 2))
        self.fc_layer_sub_1 = torch.nn.Sequential(torch.nn.Linear(self.__num_flat_features, 2))
        self.fc_layer_grade_1 = torch.nn.Sequential(torch.nn.Linear(self.__num_flat_features, 2))

        self.apply(init_weights)

    def __make_layer(self,
                     block,
                     num_block: int,
                     fmap_out: int,
                     kernel: Union[Sequence[int], int],
                     strides: Union[Sequence[int], int] = 1,
                     drop_rate: Sequence[float] = None,
                     act: str = "ReLU",
                     groups: int = 1,
                     split_layer: bool = False) -> nn.Sequential:

        fmap_out = fmap_out * groups
        layers = []
        for i in range(num_block):
            layers.append(block(fmap_in=self.__in_channels, fmap_out=fmap_out,
                                kernel=kernel,
                                strides=strides if i == 0 else 1,
                                drop_rate=drop_rate[i],
                                activation=act,
                                groups=groups,
                                split_layer=split_layer))
            split_layer = False
            self.__in_channels = fmap_out * block.expansion if i == 0 else self.__in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixup_key_list = list(self.mixup.keys())

        out = self.mixup["0"](x) if "0" in mixup_key_list else x
        out = self.conv(out)
        out = self.layers1(out)

        out = self.mixup["1"](out) if "1" in mixup_key_list else out
        out = self.layers2(out)

        out = self.mixup["2"](out) if "2" in mixup_key_list else out
        out = self.layers3(out)

        out = self.mixup["3"](out) if "3" in mixup_key_list else out
        out = self.layers4(out)
        out = self.avg_pool(out)

        if self.__split == 5:
            features = out.view(-1, self.__num_flat_features)

            mal_pred = self.fc_layer_mal(features)
            sub_pred = self.fc_layer_sub_1(features)
            grade_pred = self.fc_layer_grade_1(features)

        else:
            features = out.view(-1, 3, self.__num_flat_features)

            mal_pred = self.fc_layer_mal(features[:, 0, :])
            sub_pred = self.fc_layer_sub_1(features[:, 1, :])
            grade_pred = self.fc_layer_grade_1(features[:, 2, :])

        return mal_pred, sub_pred, grade_pred
