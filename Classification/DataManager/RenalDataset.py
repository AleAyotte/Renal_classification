"""
    @file:              RenalDataset.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 09/2021

    @Description:       This file contain the RenalDataset class, which is used to load and preprocess both 2D and 3D
                        data to train a model. It also contain the split_trainset function which is used to create the
                        train/validation split.
"""
from __future__ import annotations

import h5py
from monai.transforms import Compose
import numpy as np
import random
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from Constant import SplitName
from DataManager.HDF5Dataset import HDF5Dataset


class RenalDataset(HDF5Dataset):
    """
    Renal classification dataset.

    ...
    Attributes
    ----------
    transform: Union[compose, None]
    _clinical_data: Union[np.array, None]
        If __with_clinical is True, then it will be a numpy array that contain the clinical of each patient in the
        dataset.
    _data: np.array
        A numpy array that contain the dataset medical images.
    _imgs_keys: Union[Sequence[string], string]
        A string or a list of string that indicate The images name in the hdf5 file that will be load in the dataset
        (Exemple: "t1").
    _labels : np.array
        A numpy array that contain the labels of each data for each task.
    _patient_id : np.array
        A list of string that indicate the patient id of each data in the dataset.
    __strat_keys_name : List[str]
        A list of string that contain the name of the keys that will be used to create the stratum_keys.
    __stratum_keys : np.array
        A list of string that indicate to which stratum the data is associated.
    _tasks : Sequence[string]
        A list of clinical_features that will be used has labels for tasks. (Default=['outcome'])
    _with_clinical: bool
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
    stratified_stats():
        Calculate the number of data per encoding key.
    """
    def __init__(self,
                 imgs_keys: Union[Sequence[str], str],
                 tasks: Sequence[str],
                 clinical_features: Optional[Union[List[str], str]] = None,
                 exclude_list: Optional[List[str]] = None,
                 hdf5_filepath: Optional[str] = None,
                 split: Optional[str] = SplitName.TRAIN,
                 stratification_keys: Optional[List[str]] = None,
                 transform: Optional[Compose] = None) -> None:
        """
        Create a dataset by loading the renal image at the given path.

        :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
        :param tasks: A list of clinical_features that will be used has labels for tasks.
        :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
        :param exclude_list: A list of patient_id to exclude in this dataset.
        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2). (Default="train")
        :param stratification_keys: The names of the attributes that will be used to execute stratification sampling.
        :param transform: A function/transform that will be applied on the images and the ROI.
        """

        super().__init__(tasks=tasks,
                         imgs_keys=imgs_keys,
                         clinical_features=clinical_features,
                         split=split,
                         transform=transform)
        self.__stratum_keys = np.array([])

        if clinical_features is not None:
            self._clinical_data = np.empty(shape=(0, len(clinical_features)))

        to_exclude = set() if exclude_list is None else set(exclude_list)  # set because we do not care about order
        if split is not None:
            self.__strat_keys_name = [] if stratification_keys is None else stratification_keys
            self._read_hdf5(to_exclude=to_exclude,
                            filepath=hdf5_filepath,
                            split=split.lower() if split is not None else split)

    def add_data(self,
                 data: Sequence[dict],
                 label: Union[Sequence[dict], Sequence[int]],
                 patient_id: Sequence[str],
                 stratum_keys: Sequence[str],
                 clinical_data: Sequence[Sequence[int]] = None) -> None:
        """
        Add data to the dataset.

        :param data: A sequence of dictionary that contain the images.
        :param label: A sequence of dictionary or a sequence of int that contain the labels.
        :param patient_id: The patient id of each data that we want to add in the dataset.
        :param stratum_keys: A list of string that indicate to which stratum the data is associated.
        :param clinical_data: A sequence of sequence of int that contain the clinical data.
        """
        self._data = np.append(self._data, data, 0)
        self._labels = np.append(self._labels, label, 0)
        self._patient_id = np.append(self._patient_id, patient_id, 0)
        self.__stratum_keys = np.append(self.__stratum_keys, stratum_keys, 0)
        if clinical_data is not None:
            self._clinical_data = np.append(self._clinical_data, clinical_data, 0)

    def _extract_data(self,
                      idx: Sequence[int],
                      pop: bool = True) -> Tuple[Sequence[dict],
                                                 Union[Sequence[dict], Sequence[int]],
                                                 Sequence[str],
                                                 Sequence[str],
                                                 Sequence[Sequence[int]]]:
        """
        Extract data without applying transformation on the images.

        :param idx: The index of the data to extract.
        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :return: A tuple that contain the data (images), the labels, the patients id, the stratum_keys and
                 the clinical data.
        """
        mask = np.ones(len(self._data), dtype=bool)
        mask[idx] = False

        data = self._data[~mask]
        labels = self._labels[~mask]
        patient_id = self._patient_id[~mask]
        stratum_keys = self.__stratum_keys[~mask]
        clin = self._clinical_data[~mask] if self._with_clinical else None

        if pop:
            self._data = self._data[mask]
            self._labels = self._labels[mask]
            self._patient_id = self._patient_id[mask]
            self.__stratum_keys = self.__stratum_keys[mask]
            self._clinical_data = self._clinical_data[mask] if self._with_clinical else None

        return data, labels, patient_id, stratum_keys, clin

    def extract_data(self,
                     idx: Sequence[int],
                     pop: bool = True) -> Tuple[Sequence[dict],
                                                Union[Sequence[dict], Sequence[int]],
                                                Sequence[str],
                                                Sequence[str],
                                                Sequence[Sequence[int]]]:
        """
        Extract data without applying transformation on the images.

        :param idx: The index of the data to extract.
        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :return: A tuple that contain the data (images), the labels, the stratum_keys and the clinical data.
        """
        return self._extract_data(idx, pop)

    def _read_hdf5(self,
                   to_exclude: Set[str],
                   filepath: str,
                   split: str) -> None:
        """
        Read the images and ROI from a given hdf5 filepath and store it in memory

        :param to_exclude: A list of patient_id that will not be load in the dataset.
        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load.
        """
        f = h5py.File(filepath, 'r')
        dtset = f[split]
        self._data = []
        self._labels = []
        self._clinical_data = []
        self.__stratum_keys = []
        self._patient_id = np.array([key for key in list(dtset.keys()) if key not in to_exclude])

        for key in self._patient_id:
            imgs = {}
            for img_key in list(self._imgs_keys):
                imgs[img_key] = dtset[key][img_key][:]

            self._data.append(imgs)

            outcomes = {}
            for task in self._tasks:
                outcomes[task] = dtset[key].attrs[task]
            self._labels.append(outcomes)

            encoding_key = ""
            for strat_key in self.__strat_keys_name:
                encoding_key += '_' + str(dtset[key].attrs[strat_key])
            self.__stratum_keys.append(encoding_key)

            if self._with_clinical:
                attrs = dtset[key].attrs
                self._clinical_data.append([attrs[feat_name] for feat_name in self._features_name])

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self.__stratum_keys = np.array(self.__stratum_keys)
        self._clinical_data = np.array(self._clinical_data)
        f.close()

    def split(self,
              pop: bool = True,
              random_seed: int = 0,
              sample_size: float = 0.1,
              transform: Optional[Compose] = None,
              *args) -> RenalDataset:
        """
        Split the current dataset and return a stratified portion of it.

        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :param random_seed: The seed that will be used to split the data.
        :param sample_size: Proportion of the current set that will be used to create the new stratified set.
        :param transform: A function/transform that will be applied on the images and the ROI.
        :return: A RenalDataset that contain the splited data.
        """
        group_by = self.__stratified_stats()

        # Create a stratified group
        new_set = []
        random.seed(random_seed)
        for key, value in group_by.items():
            if len(value) > 1:
                _, split = train_test_split(value, test_size=sample_size, random_state=random_seed)
                new_set.extend(split)
            else:
                if random.random() > sample_size:
                    new_set.extend(value)

        data, labels, patient_id, stratum_keys, clin = self._extract_data(idx=new_set, pop=pop)

        new_dataset = RenalDataset(hdf5_filepath=None,
                                   tasks=self._tasks,
                                   imgs_keys=self._imgs_keys,
                                   clinical_features=self._features_name,
                                   split=None,
                                   stratification_keys=self.__strat_keys_name,
                                   transform=transform)
        new_dataset.add_data(data=data,
                             label=labels,
                             patient_id=patient_id,
                             stratum_keys=stratum_keys,
                             clinical_data=clin)
        return new_dataset

    def __stratified_stats(self) -> Dict[str, Sequence[int]]:
        """
        Group the data index by encoding key.

        :return: A dictionary of index list where the keys are the encoding key.
        """
        group_by = {}
        for i, key in enumerate(self.__stratum_keys):
            if key not in list(group_by.keys()):
                group_by[key] = [i]
            else:
                group_by[key].append(i)
        return group_by

    def stratified_stats(self) -> Dict[str, int]:
        """
        Calculate the number of data per encoding key.

        :return: A dictionary of int where the keys are the encoding key.
        """
        group_by = self.__stratified_stats()
        new_group_by = {}
        for key, value in group_by.items():
            new_group_by[key] = len(value)

        return new_group_by
