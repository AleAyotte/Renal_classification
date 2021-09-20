"""
    @file:              DatasetBuilder.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 07/2021

    @Description:       This file contain a function that build the training, validation and testing set for both 2D
                        and 3D datasets.
"""

from Data_manager.RenalDataset import RenalDataset
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
from random import randint
from typing import Optional, Sequence, Tuple, Union


DATA_PATH_2D = "dataset_2D/Data_with_N4"
DATA_PATH_3D_4CHAN = "final_dtset/dataset_3D.hdf5"
DATA_PATH_3D_3CHAN = "final_dtset/dataset_3_channel.hdf5"
PAD_MODE = "constant"  # choices=["constant", "edge", "reflect", "symmetric"]
REJECT_FILE = "Data_manager/reject_list.txt"
SPLIT_SIZE = 0.2
STRATIFICATION_KEYS = ["malignancy", "subtype", "grade"]


def get_data_augmentation(num_chan: int = 4,
                          num_dimension: int = 3) -> Tuple[Compose, Compose]:
    """
    Create the Compose objects that will be use by the training set, the validation set and the test set.
    The compose object is a list of transformation function that will be use on the data.

    :param num_chan: The number of channels of the output tensor.
    :param num_dimension: The number of dimension of the images in the dataset.
    :return: A tuple of Compose object that list the transformation that will be applied on the images.
    """
    if num_dimension == 3:
        # 4 CHANNELS
        if num_chan == 4:
            transform = Compose([
                AddChanneld(keys=["t1", "t2", "roi_t1", "roi_t2"]),
                RandFlipd(keys=["t1", "t2", "roi_t1", "roi_t2"], spatial_axis=[0], prob=0.5),
                RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
                RandAffined(keys=["t1", "t2", "roi_t1", "roi_t2"], prob=0.5, shear_range=[0.4, 0.4, 0],
                            rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
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

        # 3 CHANNELS
        else:
            transform = Compose([
                AddChanneld(keys=["t1", "t2", "roi"]),
                RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
                RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
                RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                            rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
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
    # 2 Dimensions
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
                   num_chan: int = 4,
                   num_dimension: int = 3,
                   split_seed: Optional[int] = None,
                   testset_name: str = "test") -> Tuple[RenalDataset, RenalDataset, RenalDataset]:
    """

    :param tasks: A list of attribute that will be use to labeled the data.
    :param clin_features: A list of clinical features that will be used as input in the model.
    :param num_chan: The number images channels. Only use if num_dimension == 3.
    :param num_dimension: The number of dimension of the images in the dataset.
    :param testset_name: Determine the data that will be used in the testset. If testset == 'test' then the testset
                         will be sampled in the train set. Else, it will load the corresponding testset.
    :param split_seed: The seed that should be use to sample the testset and the validset in the training set.
    :return: 3 RenalDataset: Training set, validation set and testset
    """
    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform, test_transform = get_data_augmentation(num_dimension=num_dimension, num_chan=num_chan)
    if num_dimension == 3:
        assert num_chan in [3, 4], "num_chan should be equal to 3 or 4 if num_dimension == 3."
        if num_chan == 4:
            data_path = DATA_PATH_3D_4CHAN
            imgs_keys = ["t1", "roi_t1", "t2", "roi_t2"]
        else:
            data_path = DATA_PATH_3D_3CHAN
            imgs_keys = ["t1", "t2", "roi"]
    else:
        data_path = DATA_PATH_2D
        imgs_keys = ["t1", "t2"]

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

    seed_valid = randint(0, 10000) if split_seed is None else split_seed
    seed_test = randint(0, 10000) if split_seed is None else split_seed

    if testset_name == "test":
        testset = trainset.split(pop=True,
                                 sample_size=SPLIT_SIZE,
                                 random_seed=seed_test,
                                 transform=test_transform)
    else:
        testset = RenalDataset(data_path,
                               transform=test_transform,
                               imgs_keys=imgs_keys,
                               clinical_features=clin_features,
                               tasks=tasks,
                               split=testset_name,
                               stratification_keys=STRATIFICATION_KEYS)

    validset = trainset.split(pop=True,
                              sample_size=SPLIT_SIZE,
                              random_seed=seed_valid,
                              transform=test_transform)

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
