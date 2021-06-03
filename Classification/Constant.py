"""
    @file:              Constant.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 06/2021

    @Description:       Contain several class that group constant.
"""

from enum import Enum, unique
from typing import Final


class BlockType:
    PREACT: Final = "preact"
    POSTACT: Final = "postact"


class DatasetName:
    TRAIN: Final = "TRAIN"
    VALIDATION: Final = "VALIDATION"
    TEST: Final = "TEST"
    HOLDOUT: Final = "HOLD_OUT"


@unique
class Experimentation(Enum):
    SINGLE_TASK_2D: Final = 1
    SINGLE_TASK_3D: Final = 2
    HARD_SHARING: Final = 3
    SOFT_SHARING: Final = 4


class Tasks:
    MALIGNANCY: Final = "malignancy"
    SUBTYPE: Final = "subtype"
    GRADE: Final = "grade"
    REGRESSION: Final = 1
    CLASSIFICATION: Final = 2
