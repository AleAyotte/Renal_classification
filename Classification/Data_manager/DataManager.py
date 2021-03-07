"""
    @file:              DataManager.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 02/2021

    @Description:       This file contain the RenalDataset class, which is used to load and preprocess both 2D and 3D
                        data to train a model. It also contain the split_trainset function which is used to create the
                        train/validation split.
"""
import h5py
from monai.transforms import Compose
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Sequence, Tuple, Union


class RenalDataset(Dataset):
    """
    Renal classification dataset.

        ...
    Attributes
    ----------
    transform: Union[compose, None]
    __with_clinical: bool
        Indicate if the dataset should also store the clinical data.
    __imgs_keys: Union[Sequence[string], string]
        A string or a list of string that indicate The images name in the hdf5 file that will be load in the dataset
        (Exemple: "t1").
    __data: np.array
        A numpy array that contain the dataset medical images.
    __clinical_data: Union[np.array, None]
        If __with_clinical is True, then it will be a numpy array that contain the clinical of each patient in the
        dataset.
    __labels : np.array
        A numpy array that contain the labels of each data for each task.
    Methods
    -------
    add_data(data, label, clinical_data):
        Add a subset of images, labels and clinical data to the current dataset.
    extract_data(idx, pop):
        Extract data without applying transformation on the images. If pop is true, then the data are removed from
        the current dataset.
    normalize_clinical_data(mean, std, get_norm_param):
        Normalize the clinical substracting them the given mean and divide them by the given std. If no mean or std
        is given, the they will defined with current dataset clinical data.
    """
    def __init__(self,
                 hdf5_filepath: str,
                 imgs_keys: Union[Sequence[str], str],
                 split: Union[str, None] = "train",
                 transform: Union[Compose, None] = None,
                 clinical_features: Union[Sequence[str], str] = None):
        """
        Create a dataset by loading the renal image at the given path.

        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
        :param split: A string that indicate which subset will be load. (Option: train, test, test2). (Default="train")
        :param transform: A function/transform that will be applied on the images and the ROI.
        :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
        """
        assert split in ['train', 'test', 'test2', None]
        self.transform = transform
        self.__with_clinical = clinical_features is not None
        self.__imgs_keys = imgs_keys
        self.__data = np.array([])
        self.__clinical_data = None
        self.__labels = np.array([])

        if clinical_features is not None:
            self.__clinical_data = np.empty(shape=(0, len(clinical_features)))

        if split is not None:
            if self.__with_clinical and type(clinical_features) is not list:
                clinical_features = [clinical_features]

            self.__read_hdf5(hdf5_filepath, split, clinical_features)

    def add_data(self,
                 data: Sequence[dict],
                 label: Union[Sequence[dict], Sequence[int]],
                 clinical_data: Sequence[Sequence[int]] = None) -> None:
        """
        Add data to the dataset.

        :param data: A sequence of dictionary that contain the images.
        :param label: A sequence of dictionary or a sequence of int that contain the labels.
        :param clinical_data: A sequence of sequence of int that contain the clinical data.
        """
        self.__data = np.append(self.__data, data, 0)
        self.__labels = np.append(self.__labels, label, 0)
        if clinical_data is not None:
            self.__clinical_data = np.append(self.__clinical_data, clinical_data, 0)

    def extract_data(self,
                     idx: Sequence[int],
                     pop: bool = True) -> Tuple[Sequence[dict],
                                                Union[Sequence[dict], Sequence[int]],
                                                Sequence[Sequence[int]]]:
        """
        Extract data without applying transformation on the images.

        :param idx: The index of the data to extract.
        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :return: A tuple that contain the data (images), the labels and the clinical data.
        """
        mask = np.ones(len(self.__data), dtype=bool)
        mask[idx] = False

        data = self.__data[~mask]
        labels = self.__labels[~mask]
        clin = self.__clinical_data[~mask] if self.__with_clinical else None

        if pop:
            self.__data = self.__data[mask]
            self.__labels = self.__labels[mask]
            self.__clinical_data = self.__clinical_data[mask] if self.__with_clinical else None

        return data, labels, clin

    def labels_bincount(self) -> Sequence[np.array]:
        """
        Count the number of data per class for each task

        :return: A list of np.array where each np.array represent the number of data per class.
                 The length of the list is equal to the number of task.
        """

        all_labels = []
        if type(self.__labels[0]) == dict:
            for key in list(self.__labels[0].keys()):
                all_labels.append([int(label[key]) for label in self.__labels])
        else:
            all_labels.append(self.__labels.astype(int))

        return [np.bincount(label_list) for label_list in all_labels]

    def normalize_clin_data(self,
                            mean: Union[Sequence[float], np.array, None] = None,
                            std: Union[Sequence[float], np.array, None] = None,
                            get_norm_param: bool = False) -> Union[Tuple[Union[Sequence[float], np.array],
                                                                         Union[Sequence[float], np.array]],
                                                                   None]:
        """
        Normalize the clinical substracting them the given mean and divide them by the given std. If no mean or std
        is given, the they will defined with current dataset clinical data.

        :param mean: An array of length equal to the number of clinical features (not the number of clinical data)
                     that will be substract to the clinical features.
        :param std: An array of length equal to the number of clinical features (not the number of clinical data)
                     that will divide the substracted clinical features.
        :param get_norm_param: If True, the mean and the std of the current dataset are return.
        :return: If get_norm_param is True then the mean and the std of the current dataset will be return. Otherwise,
                 nothing will be return.
        """

        assert type(mean) == type(std), "The mean and the std should has the same type."
        assert self.__with_clinical, "No clinical has been loaded."
        if mean is None and std is None:
            mean = np.mean(self.__clinical_data, axis=0)
            std = np.std(self.__clinical_data, axis=0)

        self.__clinical_data = (self.__clinical_data - mean) / std

        if get_norm_param:
            return mean, std

    def __read_hdf5(self,
                    filepath: str,
                    split: str,
                    features_names: Sequence[str]) -> None:
        """
        Read the images and ROI from a given hdf5 filepath and store it in memory

        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        :param features_names: A list of string that indicate which clinical features will be used
                               to train the model.
        """
        f = h5py.File(filepath, 'r')
        dtset = f[split]
        self.__data = []
        self.__labels = []
        self.__clinical_data = []

        for key in list(dtset.keys()):
            imgs = {}
            for img_key in list(self.__imgs_keys):
                imgs[img_key] = dtset[key][img_key][:]

            self.__data.append(imgs)

            if "outcome" in list(dtset[key].attrs.keys()):
                self.__labels.append(dtset[key].attrs["outcome"])
            else:
                self.__labels.append({"malignant": dtset[key].attrs["malignant"],
                                      "subtype": dtset[key].attrs["subtype"],
                                      "grade": dtset[key].attrs["grade"]})
            if self.__with_clinical:
                attrs = dtset[key].attrs
                self.__clinical_data.append([attrs[feat_name] for feat_name in features_names])

        self.__data = np.array(self.__data)
        self.__labels = np.array(self.__labels)
        self.__clinical_data = np.array(self.__clinical_data)
        f.close()

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.transform(self.__data[idx])

        if type(idx) == int:
            idx = [idx]
            sample = [sample]

        # Stack or cat the images into one tensor.
        # If one dimension has been added to the images, we need to cat them.
        to_cat = next(iter(sample[0].values())).dim() > next(iter(self.__data[0].values())).ndim
        temp = []

        for samp in sample:
            if to_cat:
                temp.append(torch.cat(tuple(samp[img_key] for img_key in self.__imgs_keys), 0))
            else:
                temp.append(torch.stack([samp[img_key] for img_key in self.__imgs_keys]))
        sample = torch.stack(temp) if len(temp) > 1 else temp[0]

        # Extract and stack the labels in a dictionnary of torch tensor
        if type(self.__labels[0]) == dict:
            labels = {}
            for key in list(self.__labels[idx][0].keys()):
                labels[key] = torch.tensor([
                    _dict[key] for _dict in self.__labels[idx]
                ]).long().squeeze()
        # Extract and stack the labels in a torch tensor
        else:
            labels = torch.tensor(self.__labels[idx]).long().squeeze()

        if self.__with_clinical:
            features = torch.tensor(self.__clinical_data[idx]).long().squeeze()
            return {"sample": sample, "labels": labels, "features": features}

        else:
            return {"sample": sample, "labels": labels}


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
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    val_indices = indices[:split]
    data, label, clin = trainset.extract_data(idx=val_indices)
    validset.add_data(data, label, clin)

    return trainset, validset
