"""
    @file:              hdf5_dataset.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 10/2021

    @Description:       This file contain the HDF5Dataset class, which is used to load an HDF5 dataset and to
                        preprocess 2D and 3D images.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from monai.transforms import Compose
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from constant import SplitName


class HDF5Dataset(ABC, Dataset):
    """
    The HDF5Dataset can load data from an HDF5 file and the prepare the data to train a model.

    ...
    Attributes
    ----------
    _clinical_data : Union[np.array, None]
        If __with_clinical is True, then it will be a numpy array that contain the clinical of each patient in the
        dataset.
    _c_tasks : List[str]
        A list of features name that will be used has classification tasks.
    _data : np.array
        A numpy array that contain the dataset medical images.
    _features_name : Union[List[str], str]
        A list of string that indicate which clinical features will be used to train the model.
    _imgs_keys : Union[Sequence[string], string]
        A string or a list of string that indicate The images name in the hdf5 file that will be load in the dataset
        (Exemple: "t1").
    _labels : np.array
        A numpy array that contain the labels of each data for each task.
    _patient_id : np.array
        A list of string that indicate the patient id of each data in the dataset.
    _r_tasks : List[str]
        A list of features name that will be used has regression tasks.
    _tasks : List[str]
        A list of clinical_features that will be used has labels for tasks. (Default=['outcome'])
    transform : Union[compose, None]
        A function/transform that will be applied on the images and the ROI.
    _with_clinical : bool
        Indicate if the dataset should also store the clinical data.
    Methods
    -------
    add_data(data, label, clinical_data):
        Add a subset of images, labels and clinical data to the current dataset.
    extract_data(idx, pop):
        Extract data without applying transformation on the images. If pop is true, then the data are removed from
        the current dataset.
    labels_bincount():
        Count the number of data per class for each task.
    normalize_clinical_data(mean, std, get_norm_param):
        Normalize the clinical substracting them the given mean and divide them by the given std. If no mean or std
        is given, the they will defined with current dataset clinical data.
    remove_unlabeled_data():
        Remove the data that has no label for every task.
    split(pop, random_seed, sample_size, transform):
        Split the current dataset and return a new dataset with the data.
    """
    def __init__(self,
                 classification_tasks: List[str],
                 imgs_keys: Union[Sequence[str], str],
                 clinical_features: Optional[Union[List[str], str]] = None,
                 regression_tasks: Optional[List[str]] = None,
                 split: Optional[str] = SplitName.TRAIN,
                 transform: Optional[Compose] = None) -> None:
        """
        Create a dataset by loading the renal image at the given path.

        :param classification_tasks: A list of clinical_features that will be used has labels for classification tasks.
        :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
        :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
        :param regression_tasks: A list of clinical_features that will be used has labels for regression tasks.
        :param split: A string that indicate which subset will be load. (Default=DatasetName.TRAIN)
        :param transform: A function/transform that will be applied on the images and the ROI.
        """
        if split is not None:
            assert split.upper() in [SplitName.TRAIN, SplitName.HOLDOUT]
        super(ABC).__init__()

        self.transform = transform
        self._imgs_keys = imgs_keys
        self._c_tasks = classification_tasks
        self._r_tasks = regression_tasks if regression_tasks is not None else []
        self._tasks = self._c_tasks + self._r_tasks
        self._with_clinical = clinical_features is not None
        if self._with_clinical and type(clinical_features) is not list:
            self._features_name = [clinical_features]
        else:
            self._features_name = clinical_features

        self._data = np.array([])
        self._labels = np.array([])
        self._patient_id = np.array([])
        self._clinical_data = None

        if clinical_features is not None:
            self._clinical_data = np.empty(shape=(0, len(clinical_features)))

    @abstractmethod
    def _extract_data(self,
                      idx: Sequence[int],
                      pop: bool = True):
        """
        Extract data without applying transformation on the images.

        :param idx: The index of the data to extract.
        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :return: See the overrided method.
        """
        raise NotImplementedError("Must override _extract_data.")

    def get_patient_id(self) -> Sequence[str]:
        """
        Return a list of patient id.

        :return: A numpy array of string that correspond to the list of patient id.
        """
        return self._patient_id

    def labels_bincount(self) -> dict:
        """
        Count the number of data per class for each task

        :return: A list of np.array where each np.array represent the number of data per class.
                 The length of the list is equal to the number of task.
        """

        all_labels = {}
        for task in self._c_tasks:
            label_list = [int(label[task]) for label in self._labels if label[task] >= 0]
            all_labels[task] = np.bincount(label_list)

            # If there is more than 2 classes, we only take the two last.
            if len(all_labels[task]) > 2:
                all_labels[task] = all_labels[task][1:]

        return all_labels

    def normalize_clin_data(self,
                            mean: Union[Sequence[float], np.array, None] = None,
                            std: Union[Sequence[float], np.array, None] = None,
                            get_norm_param: bool = False) -> Union[Tuple[Union[Sequence[float], np.array],
                                                                         Union[Sequence[float], np.array]],
                                                                   None]:
        """
        Normalizes the clinical features by substracting them the given mean and divide them by the given std. If no
        mean or std is given, the they will defined with current dataset clinical data.

        :param mean: An array of length equal to the number of clinical features (not the number of clinical data)
                     that will be substract to the clinical features.
        :param std: An array of length equal to the number of clinical features (not the number of clinical data)
                     that will divide the substracted clinical features.
        :param get_norm_param: If True, the mean and the std of the current dataset are return.
        :return: If get_norm_param is True then the mean and the std of the current dataset will be return. Otherwise,
                 nothing will be return.
        """
        assert type(mean) == type(std), "The mean and the std should has the same type."
        assert self._with_clinical, "No clinical has been loaded."
        if mean is None and std is None:
            mean = np.mean(self._clinical_data, axis=0)
            std = np.std(self._clinical_data, axis=0)

        self._clinical_data = (self._clinical_data - mean) / std

        if get_norm_param:
            return mean, std

    def normalize_regression_labels(self,
                                    mean: Optional[Dict[str, float]] = None,
                                    std: Optional[Dict[str, float]] = None,
                                    get_norm_param: bool = False) -> Union[Tuple[Dict[str, float],
                                                                                 Dict[str, float]],
                                                                           None]:
        """
        Normalizes the labels of the regression tasks by substracting them the given mean and divide them by the given
        std. If no mean or std is given, the they will defined with current dataset clinical data.

        :param mean: A dictionary that indicate the mean value has float for each task.
        :param std: A dictionary that indicate the standard deviation has float for each task.
        :param get_norm_param: If True, the mean and the std of the current dataset are return.
        :return: If get_norm_param is True then the mean and the std of the current dataset will be return. Otherwise,
                 nothing will be return.
        """
        if mean is None and std is None:
            mean = {}
            std = {}
            for task in self._r_tasks:
                task_label = [pat[task] for pat in self._labels]
                mean[task] = np.mean(task_label)
                std[task] = np.std(task_label)

        for pat in self._labels:
            for task in self._r_tasks:
                pat[task] = (pat[task] - mean[task]) / std[task]

        if get_norm_param:
            return mean, std

    @abstractmethod
    def _read_hdf5(self,
                   to_exclude: Set[str],
                   filepath: str,
                   split: str) -> None:
        """
        Read the images and ROI from a given hdf5 filepath and store it in memory

        :param to_exclude: A list of patient_id that will not be load in the dataset.
        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        """
        raise NotImplementedError("Must override __read_hdf5.")

    def remove_unlabeled_data(self,
                              tasks_list: Optional[List[str]] = None) -> None:
        """
        Removes the data that has no label for every given task.

        :param tasks_list: A list of tasks for which the data should have a label for at least one of them.
                           If none, the list of tasks that has been gived at the instanciation of the dataset is used.
        """
        tasks_list = self._c_tasks if tasks_list is None else tasks_list

        unlabeled_idx = []
        for i in range(len(self._labels)):
            for task in tasks_list:
                if self._labels[i][task] != -1:
                    break
            else:
                unlabeled_idx.append(i)

        self._extract_data(idx=unlabeled_idx, pop=True)

    @abstractmethod
    def split(self,
              pop: bool = True,
              random_seed: int = 0,
              sample_size: float = 0.1,
              transform: Optional[Compose] = None,
              *args) -> HDF5Dataset:
        """
        Split the current dataset and return a new dataset with the data.

        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :param random_seed: The seed that will be used to split the data.
        :param sample_size: Proportion of the current set that will be used to create the new stratified set.
        :param transform: A function/transform that will be applied on the images and the ROI.
        :return: A Dataset that contain the subset of data
        """
        raise NotImplementedError("Must override split.")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.transform(self._data[idx])

        if type(idx) == int:
            idx = [idx]
            sample = [sample]

        # Stack or cat the images into one tensor.
        # If one dimension has been added to the images, we need to cat them.
        to_cat = next(iter(sample[0].values())).dim() > next(iter(self._data[0].values())).ndim
        temp = []

        for samp in sample:
            if to_cat:
                temp.append(torch.cat(tuple(samp[img_key] for img_key in self._imgs_keys), 0))
            else:
                temp.append(torch.stack([samp[img_key] for img_key in self._imgs_keys]))
        sample = torch.stack(temp) if len(temp) > 1 else temp[0]

        # Extract and stack the labels in a dictionnary of torch tensor
        labels = {}

        # Classification tasks
        for task in list(self._c_tasks):
            labels[task] = torch.tensor([
                _dict[task] for _dict in self._labels[idx]
            ]).long().squeeze()

        # Regression tasks
        for task in list(self._r_tasks):
            labels[task] = torch.tensor([
                _dict[task] for _dict in self._labels[idx]
            ]).float().squeeze()

        if self._with_clinical:
            features = torch.tensor(self._clinical_data[idx]).float().squeeze()
            return {"sample": sample, "labels": labels, "features": features}

        else:
            return {"sample": sample, "labels": labels}
