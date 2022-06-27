"""
    @file:              constant.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 08/2021

    @Description:       Contain several class that group constant and enum.
"""

from enum import Enum, unique
from typing import Final


class AttentionBlock(Enum):
    CHANNEL: Final = 0
    SPATIAL: Final = 1
    CBAM: Final = 2


class AuxTaskSet:
    SET0: Final = "Data/r_task_all.txt"
    SET1: Final = "Data/r_task_g1.txt"
    SET2: Final = "Data/r_task_g2.txt"
    SET3: Final = "Data/r_task_g3.txt"
    SET4: Final = "Data/r_task_g4.txt"
    SET5: Final = "Data/r_task_g5.txt"
    SET6: Final = "Data/r_task_g6.txt"
    SET7: Final = "Data/r_task_g7.txt"


@unique
class BlockType(Enum):
    PREACT: Final = 1
    POSTACT: Final = 2


# Configuration of the cross-stitch module. Should be multiply by args.num_channel
CS_CONFIG: Final = [
    [1, 2, 4, 8],  # Config 0
    [1, 2, 4, 0],  # Config 1
    [1, 2, 0, 0],  # Config 2
    [0, 2, 4, 0]   # Config 3
]


@unique
class DatasetName(Enum):
    BMETS: Final = 1
    RCC: Final = 2


@unique
class DropType(Enum):
    FLAT: Final = 1
    LINEAR: Final = 2


class Experimentation(Enum):
    SINGLE_TASK_2D: Final = 1
    STL_2D: Final = 1
    SINGLE_TASK_3D: Final = 2
    STL_3D: Final = 2
    HARD_SHARING: Final = 3
    HS: Final = 3
    SOFT_SHARING: Final = 4
    SS: Final = 4
    MTAN: Final = 5
    LTB: Final = 6
    TAG: Final = 7


@unique
class Loss(Enum):
    UNCERTAINTY: Final = 1
    UNIFORM: Final = 2


@unique
class LTBConfig(Enum):
    CONFIG1: Final = [BlockType.PREACT, BlockType.PREACT, BlockType.PREACT, BlockType.PREACT]
    CONFIG2: Final = [BlockType.PREACT, BlockType.PREACT, BlockType.POSTACT, BlockType.POSTACT]
    CONFIG3: Final = [BlockType.PREACT, BlockType.POSTACT]
    CONFIG4: Final = [BlockType.POSTACT, BlockType.POSTACT, BlockType.PREACT, BlockType.PREACT]


@unique
class ModelType(Enum):
    STANDARD: Final = 1
    SHARED_NET: Final = 2
    LTB_NET: Final = 3


@unique
class NetConfig(Enum):
    CONFIG0: Final = [BlockType.PREACT, BlockType.PREACT, BlockType.PREACT, BlockType.PREACT]
    CONFIG1: Final = [BlockType.POSTACT, BlockType.POSTACT, BlockType.POSTACT, BlockType.POSTACT]
    CONFIG2: Final = [BlockType.POSTACT, BlockType.POSTACT, BlockType.PREACT, BlockType.PREACT]
    CONFIG3: Final = [BlockType.PREACT, BlockType.PREACT, BlockType.POSTACT, BlockType.POSTACT]


@unique
class SharingUnits(Enum):
    CROSS_STITCH: Final = 1
    SLUICE: Final = 2


class SplitName:
    TRAIN: Final = "TRAIN"
    VALIDATION: Final = "VALIDATION"
    TEST: Final = "TEST"
    HOLDOUT: Final = "HOLD_OUT"


@unique
class SubNetDepth(Enum):
    CONFIG1: Final = {"malignancy": 18, "subtype": 18, "grade": 18}
    CONFIG2: Final = {"malignancy": 34, "subtype": 34, "grade": 34}
    CONFIG3: Final = {"malignancy": 18, "subtype": 18, "grade": 34}


class Tasks:
    ARE: Final = "are"
    GRADE: Final = "grade"
    LRF: Final = "lrf"
    MALIGNANCY: Final = "malignancy"
    SUBTYPE: Final = "subtype"
    CLASSIFICATION: Final = 2
    REGRESSION: Final = 1
