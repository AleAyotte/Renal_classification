"""
    @file:              DataManager.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       This file contain the RenalDataset class, which is used to load and preprocess both 2D and 3D
                        data to train a model. It also contain the split_trainset function which is used to create the
                        train/validation split.
"""
import h5py
from monai.transforms import Compose
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Tuple, Union


class RenalDataset(Dataset):
    """
    Renal classification dataset.

    ...
    Attributes
    ----------
    transform: Union[compose, None]
    __clinical_data: Union[np.array, None]
        If __with_clinical is True, then it will be a numpy array that contain the clinical of each patient in the
        dataset.
    __data: np.array
        A numpy array that contain the dataset medical images.
    __imgs_keys: Union[Sequence[string], string]
        A string or a list of string that indicate The images name in the hdf5 file that will be load in the dataset
        (Exemple: "t1").
    __labels : np.array
        A numpy array that contain the labels of each data for each task.
    __patient_id : np.array
        A list of string that indicate the patient id of each data in the dataset.
    __stratum_keys : np.array
        A list of string that indicate to which stratum the data is associated.
    __tasks : Sequence[string]
        A list of clinical_features that will be used has labels for tasks. (Default=['outcome'])
    __with_clinical: bool
        Indicate if the dataset should also store the clinical data.
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
    labels_bincount():
        Count the number of data per class for each task.
    """
    def __init__(self,
                 hdf5_filepath: str,
                 tasks: Sequence[str],
                 imgs_keys: Union[Sequence[str], str],
                 clinical_features: Union[Sequence[str], str] = None,
                 split: Optional[str] = "train",
                 stratification_keys: Optional[List[str]] = None,
                 transform: Optional[Compose] = None):
        """
        Create a dataset by loading the renal image at the given path.

        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param tasks: A list of clinical_features that will be used has labels for tasks.
        :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
        :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2). (Default="train")
        :param stratification_keys: The names of the attributes that will be used to execute stratification sampling.
        :param transform: A function/transform that will be applied on the images and the ROI.
        """
        assert split in ['train', 'test', None]
        self.transform = transform
        self.__imgs_keys = imgs_keys
        self.__tasks = tasks
        self.__with_clinical = clinical_features is not None

        self.__data = np.array([])
        self.__labels = np.array([])
        self.__patient_id = np.array([])
        self.__stratum_keys = np.array([])
        self.__clinical_data = None

        if clinical_features is not None:
            self.__clinical_data = np.empty(shape=(0, len(clinical_features)))

        if split is not None:
            if self.__with_clinical and type(clinical_features) is not list:
                clinical_features = [clinical_features]

            stratification_keys = [] if stratification_keys is None else stratification_keys
            self.__read_hdf5(hdf5_filepath, split, stratification_keys, clinical_features)

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
        self.__data = np.append(self.__data, data, 0)
        self.__labels = np.append(self.__labels, label, 0)
        self.__patient_id = np.append(self.__patient_id, patient_id, 0)
        self.__stratum_keys = np.append(self.__stratum_keys, stratum_keys, 0)
        if clinical_data is not None:
            self.__clinical_data = np.append(self.__clinical_data, clinical_data, 0)

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
        mask = np.ones(len(self.__data), dtype=bool)
        mask[idx] = False

        data = self.__data[~mask]
        labels = self.__labels[~mask]
        patient_id = self.__patient_id[~mask]
        stratum_keys = self.__stratum_keys[~mask]
        clin = self.__clinical_data[~mask] if self.__with_clinical else None

        if pop:
            self.__data = self.__data[mask]
            self.__labels = self.__labels[mask]
            self.__patient_id = self.__patient_id[mask]
            self.__stratum_keys = self.__stratum_keys[mask]
            self.__clinical_data = self.__clinical_data[mask] if self.__with_clinical else None

        return data, labels, patient_id, stratum_keys, clin

    def get_patient_id(self) -> Sequence[str]:
        """
        Return a list of patient id.

        :return: A numpy array of string that correspond to the list of patient id.
        """
        return self.__patient_id

    def labels_bincount(self) -> dict:
        """
        Count the number of data per class for each task

        :return: A list of np.array where each np.array represent the number of data per class.
                 The length of the list is equal to the number of task.
        """
        all_labels = {}
        for key in list(self.__labels[0].keys()):
            label_list = [int(label[key]) for label in self.__labels if label[key] >= 0]
            all_labels[key] = np.bincount(label_list)

            # If there are more than 2 classes, we only take the two last.
            if len(all_labels[key]) > 2:
                all_labels[key] = all_labels[key][1:]

        return all_labels

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
                    stratification_keys: List[str],
                    features_names: Sequence[str]) -> None:
        """
        Read the images and ROI from a given hdf5 filepath and store it in memory

        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        :param stratification_keys: The names of the attributes that will be used to execute stratification sampling.
        :param features_names: A list of string that indicate which clinical features will be used
                               to train the model.
        """
        f = h5py.File(filepath, 'r')
        dtset = f[split]
        self.__data = []
        self.__labels = []
        self.__clinical_data = []
        self.__stratum_keys = []
        self.__patient_id = np.array(list(dtset.keys()))

        for key in self.__patient_id:
            imgs = {}
            for img_key in list(self.__imgs_keys):
                imgs[img_key] = dtset[key][img_key][:]

            self.__data.append(imgs)

            outcomes = {}
            for task in self.__tasks:
                outcomes[task] = dtset[key].attrs[task]
            self.__labels.append(outcomes)

            encoding_key = ""
            for strat_key in stratification_keys:
                encoding_key += '_' + str(dtset[key].attrs[strat_key])
            self.__stratum_keys.append(encoding_key)

            if self.__with_clinical:
                attrs = dtset[key].attrs
                self.__clinical_data.append([attrs[feat_name] for feat_name in features_names])

        self.__data = np.array(self.__data)
        self.__labels = np.array(self.__labels)
        self.__stratum_keys = np.array(self.__stratum_keys)
        self.__clinical_data = np.array(self.__clinical_data)
        f.close()

    def remove_unlabeled_data(self) -> None:
        """
        Remove the data that has no label for every task. This should be done after the validation and test splits.
        """
        num_tasks = len(self.__tasks)
        unlabeled_idx = []
        for i in range(len(self.__labels)):
            unlabeled_task = 0
            for task in self.__tasks:
                unlabeled_task += 1 if self.__labels[i][task] == -1 else 0

            if unlabeled_task == num_tasks:
                unlabeled_idx.append(i)
        _, _, _, _, _ = self.extract_data(idx=unlabeled_idx, pop=True)

    def stratified_split(self,
                         pop: bool = True,
                         random_seed: int = 0,
                         sample_size: float = 0.1) -> Tuple[Sequence[dict],
                                                            Union[Sequence[dict], Sequence[int]],
                                                            Sequence[str],
                                                            Sequence[str],
                                                            Sequence[Sequence[int]]]:
        """
        Split the current dataset and return a stratified portion of it.

        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :param random_seed: The seed that will be used to split the data.
        :param sample_size: Proportion of the current set that will be used to create the new stratified set.
        :return: A tuple that contain the data (images), the labels, the encoding_keys, the patients id
                and the clinical data.
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

        return self.extract_data(idx=new_set, pop=pop)

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
        labels = {}
        for key in list(self.__labels[idx][0].keys()):
            labels[key] = torch.tensor([
                _dict[key] for _dict in self.__labels[idx]
            ]).long().squeeze()

        if self.__with_clinical:
            features = torch.tensor(self.__clinical_data[idx]).long().squeeze()
            return {"sample": sample, "labels": labels, "features": features}

        else:
            return {"sample": sample, "labels": labels}