from typing import Sequence
from monai.networks.nets import UNet


def load_unet(out_channels: int = 3, channels: Sequence[int] = None,
              strides: Sequence[int] = None, num_res_units: int = 2, dropout: float = 0):

    channels = [16, 32, 64, 128, 256] if channels is None else channels
    strides = [2, 2, 2, 2] if strides is None else strides

    return UNet(dimensions=3, in_channels=3,
                out_channels=out_channels, channels=channels,
                strides=strides, num_res_units=num_res_units,
                dropout=dropout,
                )
