"""
    @file:              DatasetBuilder.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 03/2021

    @Description:       This file contain a function that build the training, validation and testing set for both 2D
                        and 3D datasets.
"""

from Data_manager.DataManager import RenalDataset, split_trainset
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
from random import randint
from typing import Sequence, Tuple, Union


DATA_PATH_3D = "final_dtset/all.hdf5"
DATA_PATH_2D = "dataset_2D/Data_with_N4"


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
            ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode="constant"),
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
    imgs_keys = ["t1", "t2", "roi"] if num_dimension == 3 else ["t1", "t2"]
    data_path = DATA_PATH_3D if num_dimension == 3 else DATA_PATH_2D

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset = RenalDataset(data_path,
                            transform=transform,
                            imgs_keys=imgs_keys,
                            clinical_features=clin_features,
                            tasks=tasks)
    validset = RenalDataset(data_path,
                            transform=test_transform,
                            imgs_keys=imgs_keys,
                            clinical_features=clin_features,
                            tasks=tasks,
                            split=None)
    testset = RenalDataset(data_path,
                           transform=test_transform,
                           imgs_keys=imgs_keys,
                           clinical_features=clin_features,
                           tasks=tasks,
                           split=None if testset_name == "test" else testset_name)

    if testset_name == "test":
        trainset, testset = split_trainset(trainset, testset, validation_split=0.2)

    seed = randint(0, 10000)
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
