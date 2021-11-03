from typing import Final

# Sampler
AIM_TARGET: Final = "aim_target"
AIM_RATE: Final = "aim_rate"
GREATER: Final = "greater_rate"
LOWER: Final = "lower_rate"
NUM_TARGET: Final = "num_target"
NUM_POS: Final = "num_pos"
POS_RATE: Final = "pos_rate"

# DatasetBuilder
BMETS_A: Final = "Data/BrainMetsA.hdf5"
BMETS_B: Final = "Data/BrainMetsB.hdf5"
BMETS_SLICE: Final = [32, 32, 16]
BMETS_TOL_DICT = {"are": 0.01, "lrf": 0.02}
REJECT_FILE: Final = "Data/reject_list.txt"
RCC_2D: Final = "Data/2D_with_N4"
RCC_4CHAN: Final = "Data/RCC_4chan.hdf5"
RCC_3CHAN: Final = "Data/RCC_3chan.hdf5"
RCC_SLICE: Final = [48, 48, 16]
RCC_STRATIFICATION_KEYS: Final = ["malignancy", "subtype", "grade"]

# Data Augmentation Constant
ALIGN_CORNERS: Final = True
CROP_SIZE_RCC: Final = [64, 64, 24]
CROP_SIZE_BMET: Final = [48, 48, 24]
KEEP_SIZE: Final = False
MAX_ZOOM_2D: Final = 1.05
MAX_ZOOM_3D: Final = 1.23
MIN_ZOOM_2D: Final = 0.95
MIN_ZOOM_3D: Final = 0.77
MODE_2D: Final = "bilinear"
MODE_3D: Final = "trilinear"
PAD_MODE_AFFINE: Final = "zeros"
PAD_MODE_RESIZE: Final = "constant"
PROB: Final = 0.5
RANDOM_CENTER: Final = False
ROTATE_RANGE_2D: Final = 6.28
ROTATE_RANGE_3D: Final = [0.0, 0.0, 6.28]
SHEAR_RANGE_2D: Final = 0.5
SHEAR_RANGE_3D: Final = [0.4, 0.4, 0.0]
SPATIAL_AXIS: Final = [0]
SPLIT_SIZE: Final = 0.2
TRANSLATE_RANGE_2D: Final = 0.1
TRANSLATE_RANGE_3D: Final = 0.66


"""
ALIGN_CORNERS: Final = True
KEEP_SIZE: Final = False
MAX_ZOOM_2D: Final = 1.05
MAX_ZOOM_3D: Final = 1.05
MIN_ZOOM_2D: Final = 0.95
MIN_ZOOM_3D: Final = 0.95
MODE_2D: Final = "bilinear"
MODE_3D: Final = "trilinear"
PAD_MODE_AFFINE: Final = "zeros"
PAD_MODE_RESIZE: Final = "constant"
PROB: Final = 0.5
RANDOM_CENTER: Final = False
ROTATE_RANGE_2D: Final = 6.28
ROTATE_RANGE_3D: Final = [0.0, 0.0, 0.2]
SHEAR_RANGE_2D: Final = 0.5
SHEAR_RANGE_3D: Final = [0.1, 0.1, 0.0]
SPATIAL_AXIS: Final = [0]
SPLIT_SIZE: Final = 0.2
TRANSLATE_RANGE_2D: Final = 0.1
TRANSLATE_RANGE_3D: Final = 0.1
"""