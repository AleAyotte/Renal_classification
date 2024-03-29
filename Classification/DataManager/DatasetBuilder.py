"""
    @file:              DatasetBuilder.py
    @Author:            Alexandre Ayotte

    @Creation Date:     03/2021
    @Last modification: 09/2021

    @Description:       This file contain a function that build the training, validation and testing set for both 2D
                        and 3D datasets.
"""

from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
from random import randint
from typing import Final, List, Optional, Sequence, Tuple, Union

from Constant import DatasetName, SplitName
from DataManager.BrainDataset import BrainDataset
from DataManager.RenalDataset import RenalDataset


BMETS_A: Final = "Data/BrainMetsA.hdf5"
BMETS_B: Final = "Data/BrainMetsB.hdf5"
BMETS_TOL_DICT = {"are": 0.01, "lrf": 0.02}
RCC_2D: Final = "Data/2D_with_N4"
RCC_4CHAN: Final = "Data/RCC_4chan.hdf5"
RCC_3CHAN: Final = "Data/RCC_3chan.hdf5"
PAD_MODE: Final = "constant"  # choices=["constant", "edge", "reflect", "symmetric"]
REJECT_FILE: Final = "DataManager/reject_list.txt"
RCC_STRATIFICATION_KEYS: Final = ["malignancy", "subtype", "grade"]
SPLIT_SIZE: Final = 0.2


def build_datasets(dataset_name: DatasetName,
                   tasks: Sequence[str],
                   clin_features: Optional[Sequence[str]] = None,
                   num_chan: int = 4,
                   num_dimension: int = 3,
                   split_seed: Optional[int] = None,
                   testset_name: str = "test") -> Union[Tuple[BrainDataset, BrainDataset, BrainDataset],
                                                        Tuple[RenalDataset, RenalDataset, RenalDataset]]:
    """
    Load the data, split them into 3 dataset (training, validation, test) and prepare the dataset.

    :param dataset_name: Indicate which dataset will be used. (See DatasetName in Constant.py)
    :param tasks: A list of attribute that will be use to labeled the data.
    :param clin_features: A list of clinical features that will be used as input in the model.
    :param num_chan: The number images channels. Only use if num_dimension == 3.
    :param num_dimension: The number of dimension of the images in the dataset.
    :param testset_name: Determine the data that will be used in the testset. If testset == 'test' then the testset
                         will be sampled in the train set. Else, it will load the corresponding testset.
    :param split_seed: The seed that should be use to sample the testset and the validset in the training set.
    :return: 3 HDF5Dataset: Training set, validation set and testset.
    """
    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform, test_transform = get_data_augmentation(dataset_name=dataset_name, num_dimension=num_dimension, num_chan=num_chan)

    if dataset_name is DatasetName.RCC:
        if num_dimension == 3:
            assert num_chan in [3, 4], "num_chan should be equal to 3 or 4 if num_dimension == 3."
            if num_chan == 4:
                data_path = RCC_4CHAN
                imgs_keys = ["t1", "roi_t1", "t2", "roi_t2"]
            else:
                data_path = RCC_3CHAN
                imgs_keys = ["t1", "t2", "roi"]
        else:
            data_path = RCC_2D
            imgs_keys = ["t1", "t2"]
    else:
        data_path = BMETS_A
        imgs_keys = ["t1ce", "dose", "roi"]

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    with open(REJECT_FILE, 'r') as f:
        exclude_list = [line.strip() for line in f]

    trainset = get_dataset(clinical_features=clin_features,
                           data_path=data_path,
                           dataset_name=dataset_name,
                           exclude_list=exclude_list,
                           imgs_keys=imgs_keys,
                           split=SplitName.TRAIN,
                           tasks=tasks,
                           transform=transform)

    seed_valid = randint(0, 10000) if split_seed is None else split_seed
    seed_test = randint(0, 10000) if split_seed is None else split_seed

    if testset_name == SplitName.HOLDOUT.lower():
        testset = get_dataset(clinical_features=clin_features,
                              data_path=data_path,
                              dataset_name=dataset_name,
                              exclude_list=exclude_list,
                              imgs_keys=imgs_keys,
                              split=SplitName.HOLDOUT,
                              tasks=tasks,
                              transform=test_transform)
    else:
        testset = split_dataset(dataset=trainset,
                                dataset_name=dataset_name,
                                random_seed=seed_test,
                                tasks=tasks,
                                transform=test_transform)

    validset = split_dataset(dataset=trainset,
                             dataset_name=dataset_name,
                             random_seed=seed_valid,
                             tasks=tasks,
                             transform=test_transform)

    # --------------------------------------------
    #             NORMALIZE THE DATA
    # --------------------------------------------
    if clin_features is not None:
        mean, std = trainset.normalize_clin_data(get_norm_param=True)
        validset.normalize_clin_data(mean=mean, std=std)
        testset.normalize_clin_data(mean=mean, std=std)

    # --------------------------------------------
    #               PREPARE DATASET
    # --------------------------------------------
    for dataset in [trainset, validset, testset]:
        dataset.remove_unlabeled_data() if dataset_name is DatasetName.RCC else dataset.prepare_dataset()

    return trainset, validset, testset


def get_data_augmentation(dataset_name: DatasetName,
                          num_chan: int = 4,
                          num_dimension: int = 3) -> Tuple[Compose, Compose]:
    """
    Create the Compose objects that will be use by the training set, the validation set and the test set.
    The compose object is a list of transformation function that will be use on the data.

    :param dataset_name: Indicate for which dataset we want to get the data augmentation.
    :param num_chan: The number of channels of the output tensor.
    :param num_dimension: The number of dimension of the images in the dataset.
    :return: A tuple of Compose object that list the transformation that will be applied on the images.
    """
    if num_dimension == 3:
        if dataset_name is DatasetName.RCC:
            all_imgs_keys = ["t1", "t2", "roi_t1", "roi_t2"] if num_chan == 4 else ["t1", "t2", "roi"]
            partial_imgs_key = ["t1", "t2"]
            crop_size = [64, 64, 24]
            intensity_factor = 0.2
            pad_size = [96, 96, 32]
        else:
            all_imgs_keys = ["t1ce", "dose", "roi"]
            partial_imgs_key = ["t1ce"]
            crop_size = [48, 48, 24]
            intensity_factor = 0.1
            pad_size = [64, 64, 32]

        transform = Compose([
            AddChanneld(keys=all_imgs_keys),
            RandFlipd(keys=all_imgs_keys, spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=partial_imgs_key, factors=intensity_factor, prob=0.5),
            RandAffined(keys=all_imgs_keys, prob=0.5, shear_range=[0.4, 0.4, 0],
                        rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
            RandSpatialCropd(keys=all_imgs_keys, roi_size=crop_size, random_center=False),
            RandZoomd(keys=all_imgs_keys, prob=0.5, min_zoom=0.77, max_zoom=1.23,
                      keep_size=False, mode="trilinear", align_corners=True),
            ResizeWithPadOrCropd(keys=all_imgs_keys, spatial_size=pad_size, mode=PAD_MODE),
            ToTensord(keys=all_imgs_keys)
        ])

    # 2 Dimensions
    else:
        all_imgs_keys = ["t1", "t2"]
        transform = Compose([
            RandFlipd(keys=all_imgs_keys, spatial_axis=[0], prob=0.5),
            RandScaleIntensityd(keys=all_imgs_keys, factors=0.2, prob=0.5),
            RandAffined(keys=all_imgs_keys, prob=0.8, shear_range=0.5,
                        rotate_range=6.28, translate_range=0.1),
            RandZoomd(keys=all_imgs_keys, prob=0.5, min_zoom=0.95, max_zoom=1.05,
                      keep_size=False),
            ResizeWithPadOrCropd(keys=all_imgs_keys, spatial_size=[224, 224], mode="constant"),
            ToTensord(keys=all_imgs_keys)
        ])

    test_transform = Compose([
        AddChanneld(keys=all_imgs_keys),
        ToTensord(keys=all_imgs_keys)
    ])

    return transform, test_transform


def get_dataset(clinical_features: Sequence[str],
                dataset_name: DatasetName,
                data_path: str,
                exclude_list: List[str],
                imgs_keys: List[str],
                split: SplitName,
                tasks: Sequence[str],
                transform: Compose) -> Union[BrainDataset, RenalDataset]:
    """
    Load the correct Dataset object according to the dataset that we want to load

    :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
    :param dataset_name: Indicate which dataset will be used. (See DatasetName in Constant.py)
    :param data_path: The filepath of the hdf5 file where the data has been stored.
    :param exclude_list: A list of patient_id to exclude in this dataset.
    :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
    :param split: A string that indicate which subset will be load. (Default="train")
    :param tasks: A list of clinical_features that will be used has labels for tasks.
    :param transform: A function/transform that will be applied on the images and the ROI.
    :return: An HDF5Dataset that contain the loaded data.
    """
    if dataset_name is DatasetName.RCC:
        dataset = RenalDataset(clinical_features=clinical_features,
                               exclude_list=exclude_list,
                               hdf5_filepath=data_path,
                               imgs_keys=imgs_keys,
                               split=split,
                               stratification_keys=RCC_STRATIFICATION_KEYS,
                               tasks=tasks,
                               transform=transform)
    else:
        dataset = BrainDataset(clinical_features=clinical_features,
                               hdf5_filepath=data_path,
                               imgs_keys=imgs_keys,
                               tasks=tasks,
                               transform=transform)
    return dataset


def split_dataset(dataset: Union[BrainDataset, RenalDataset],
                  dataset_name: DatasetName,
                  random_seed: int,
                  tasks: Sequence[str],
                  transform: Compose) -> Union[BrainDataset, RenalDataset]:
    """
    Split the dataset using the correct parameter according to the dataset name.

    :param dataset: The HDF5Dataset that will be split.
    :param dataset_name: Indicate which dataset will be used. (See DatasetName in Constant.py)
    :param random_seed: The seed that will be used to split the data.
    :param tasks: A list of attribute that will be use to labeled the data.
    :param transform: A function/transform that will be applied on the images and the ROI.
    :return: An HDF5Dataset that represent the validation/test set
    """
    if dataset_name is DatasetName.RCC:
        testset = dataset.split(pop=True,
                                random_seed=random_seed,
                                sample_size=SPLIT_SIZE,
                                transform=transform)
    else:
        testset = dataset.split(pop=True,
                                random_seed=random_seed,
                                sample_size=SPLIT_SIZE,
                                tol_dict={task: BMETS_TOL_DICT[task] for task in tasks},
                                transform=transform)
    return testset
