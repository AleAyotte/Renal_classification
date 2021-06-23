"""
    @file:              Constant.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 06/2021

    @Description:       Contain several class that group constant and enum.
"""

from enum import Enum, unique
from typing import Final


@unique
class BlockType(Enum):
    PREACT: Final = 1
    POSTACT: Final = 2


class DatasetName:
    TRAIN: Final = "TRAIN"
    VALIDATION: Final = "VALIDATION"
    TEST: Final = "TEST"
    HOLDOUT: Final = "HOLD_OUT"


@unique
class DropType(Enum):
    FLAT: Final = 1
    LINEAR: Final = 2


@unique
class Experimentation(Enum):
    SINGLE_TASK_2D: Final = 1
    SINGLE_TASK_3D: Final = 2
    HARD_SHARING: Final = 3
    SOFT_SHARING: Final = 4


@unique
class SharingUnits(Enum):
    CROSS_STITCH: Final = 1
    SLUICE: Final = 2


class SubNetDepth:
    CONFIG1: Final = {"malignancy": 18, "subtype": 18, "grade": 18}
    CONFIG2: Final = {"malignancy": 34, "subtype": 34, "grade": 34}
    CONFIG3: Final = {"malignancy": 18, "subtype": 18, "grade": 34}


class Tasks:
    MALIGNANCY: Final = "malignancy"
    SUBTYPE: Final = "subtype"
    GRADE: Final = "grade"
    REGRESSION: Final = 1
    CLASSIFICATION: Final = 2
