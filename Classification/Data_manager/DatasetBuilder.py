"""
    @file:              DatasetBuilder.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 03/2021

    @Description:       This file contain a function that build the training, validation and testing set for both 2D
                        and 3D datasets.
"""

from Data_manager.RenalDataset import RenalDataset
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
from random import randint
from typing import Sequence, Tuple, Union


DATA_PATH_2D = "dataset_2D/Data_with_N4"
DATA_PATH_3D = "final_dtset/dataset_3D.hdf5"
# DATA_PATH_3D = "final_dtset/dataset_3_channel.hdf5"
PAD_MODE = "constant"  # choices=["constant", "edge", "reflect", "symmetric"]
REJECT_FILE = "Data_manager/reject_list.txt"
STRATIFICATION_KEYS = ["malignancy", "subtype", "grade"]


def get_data_augmentation(num_dimension: int = 3) -> Tuple[Compose, Compose]:
    if num_dimension == 3:
        """
        transform = Compose([
            AddChanneld(keys=["t1", "t2", "roi"]),
            RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=["t1", "t2"], factors=0.1, prob=0.5),
            RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                        rotate_range=[0, 0, 6.28], translate_range=0.1, padding_mode="zeros"),
            RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
            RandZoomd(keys=["t1", "t2", "roi"], prob=0.5, min_zoom=1.00, max_zoom=1.05,
                      keep_size=False, mode="trilinear", align_corners=True),
            ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode=args.pad_mode),
            ToTensord(keys=["t1", "t2", "roi"])
        ])
        """
        """
        # 4 CHANNELS
        transform = Compose([
            AddChanneld(keys=["t1", "t2", "roi_t1", "roi_t2"]),
            RandFlipd(keys=["t1", "t2", "roi_t1", "roi_t2"], spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
            RandAffined(keys=["t1", "t2", "roi_t1", "roi_t2"], prob=0.5, shear_range=[0.4, 0.4, 0],
                        rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
            # RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
            RandSpatialCropd(keys=["t1", "t2", "roi_t1", "roi_t2"], roi_size=[64, 64, 24], random_center=False),
            RandZoomd(keys=["t1", "t2", "roi_t1", "roi_t2"], prob=0.5, min_zoom=0.77, max_zoom=1.23,
                      keep_size=False, mode="trilinear", align_corners=True),
            ResizeWithPadOrCropd(keys=["t1", "t2", "roi_t1", "roi_t2"], spatial_size=[96, 96, 32], mode=PAD_MODE),
            ToTensord(keys=["t1", "t2", "roi_t1", "roi_t2"])
        ])
        test_transform = Compose([
            AddChanneld(keys=["t1", "t2", "roi_t1", "roi_t2"]),
            ToTensord(keys=["t1", "t2", "roi_t1", "roi_t2"])
        ])
        """
        # 3 CHANNELS
        transform = Compose([
            AddChanneld(keys=["t1", "t2", "roi"]),
            RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
            RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                        rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
            # RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
            RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 24], random_center=False),
            RandZoomd(keys=["t1", "t2", "roi"], prob=0.5, min_zoom=0.77, max_zoom=1.23,
                      keep_size=False, mode="trilinear", align_corners=True),
            ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode=PAD_MODE),
            ToTensord(keys=["t1", "t2", "roi"])
        ])
        test_transform = Compose([
            AddChanneld(keys=["t1", "t2", "roi"]),
            ToTensord(keys=["t1", "t2", "roi"])
        ])
    else:
        transform = Compose([
            RandFlipd(keys=["t1", "t2"], spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
            RandAffined(keys=["t1", "t2"], prob=0.8, shear_range=0.5,
                        rotate_range=6.28, translate_range=0.1),
            RandZoomd(keys=["t1", "t2"], prob=0.5, min_zoom=0.95, max_zoom=1.05,
                      keep_size=False),
            ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=[224, 224], mode="constant"),
            ToTensord(keys=["t1", "t2"])
        ])

        test_transform = Compose([
            AddChanneld(keys=["t1", "t2"]),
            ToTensord(keys=["t1", "t2"])
        ])

    return transform, test_transform


def build_datasets(tasks: Sequence[str],
                   clin_features: Union[Sequence[str], None] = None,
                   num_dimension: int = 3,
                   testset_name: str = "test") -> Tuple[RenalDataset, RenalDataset, RenalDataset]:
    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform, test_transform = get_data_augmentation(num_dimension=num_dimension)
    # imgs_keys = ["t1", "roi_t1", "t2", "roi_t2"] if num_dimension == 3 else ["t1", "t2"]
    imgs_keys = ["t1", "t2", "roi"] if num_dimension == 3 else ["t1", "t2"]
    data_path = DATA_PATH_3D if num_dimension == 3 else DATA_PATH_2D

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    with open(REJECT_FILE, 'r') as f:
        exclude_list = [line.strip() for line in f]

    trainset = RenalDataset(data_path,
                            transform=transform,
                            imgs_keys=imgs_keys,
                            clinical_features=clin_features,
                            tasks=tasks,
                            exclude_list=exclude_list,
                            stratification_keys=STRATIFICATION_KEYS)
    validset = RenalDataset(data_path,
                            transform=test_transform,
                            imgs_keys=imgs_keys,
                            clinical_features=clin_features,
                            tasks=tasks,
                            split=None,
                            stratification_keys=STRATIFICATION_KEYS)
    testset = RenalDataset(data_path,
                           transform=test_transform,
                           imgs_keys=imgs_keys,
                           clinical_features=clin_features,
                           tasks=tasks,
                           split=None if testset_name == "test" else testset_name,
                           stratification_keys=STRATIFICATION_KEYS)

    seed = randint(0, 10000)
    if testset_name == "test":
        trainset, testset = split_trainset(trainset, testset, validation_split=0.2, random_seed=0)

    trainset, validset = split_trainset(trainset, validset, validation_split=0.2, random_seed=seed)

    # --------------------------------------------
    #             NORMALIZE THE DATA
    # --------------------------------------------
    if clin_features is not None:
        mean, std = trainset.normalize_clin_data(get_norm_param=True)
        validset.normalize_clin_data(mean=mean, std=std)
        testset.normalize_clin_data(mean=mean, std=std)

    # --------------------------------------------
    #            REMOVE UNLABELED DATA
    # --------------------------------------------
    trainset.remove_unlabeled_data()
    validset.remove_unlabeled_data()
    testset.remove_unlabeled_data()

    return trainset, validset, testset


def split_trainset(trainset: RenalDataset,
                   validset: RenalDataset,
                   validation_split: float = 0.2,
                   random_seed: int = 0) -> Tuple[RenalDataset, RenalDataset]:
    """
    Transfer a part of the trainset into the validation set.

    :param trainset: A RenalDataset that contain the training and the validation data.
    :param validset: A empty RenalDataset with split = None that will be used to stock the validation data.
    :param validation_split: Proportion of the training set that will be used to create the validation set.
    :param random_seed: The random seed that will be used shuffle and split the data.
    :return: Two RenalDataset that will represent the trainset and the validation set respectively
    """

    data, label, patient_id, stratum_keys, clin = trainset.stratified_split(pop=True,
                                                                            sample_size=validation_split,
                                                                            random_seed=random_seed)
    validset.add_data(data, label, patient_id, stratum_keys, clin)

    return trainset, validset
