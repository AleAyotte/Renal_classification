"""
    @file:              hdf5_dataset.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 09/2021

    @Description:       This file contain the BrainDataset class that inherit from HDF5Dataset. The BrainDataset will
                        be used to 1) load an HDF5 file that contain the BrainMets dataset, 2) split the dataset by
                        using a Sampler object (See data_manager/sampler.py) and 3) preprocess the images.
"""
from __future__ import annotations

import h5py
from monai.transforms import Compose
import numpy as np
from typing import Dict, Final, List, Optional, Sequence, Set, Tuple, Union

from constant import SplitName
from data_manager.hdf5_dataset import HDF5Dataset
from data_manager.sampler import Sampler

IMAGES: Final = "images"
MAX_ITER: Final = "max_iter"
TOL_DICT: Final = "tol_dict"


class BrainDataset(HDF5Dataset):
    """
    BrainMets classification dataset.

    ...
    Attributes
    ----------
    transform: Union[compose, None]
    _clinical_data: Union[np.array, None]
        If __with_clinical is True, then it will be a numpy array that contain the clinical of each patient in the
        dataset.
    _data: np.array
        A numpy array that contain the dataset medical images.
    __data_splitter : Sampler
        A sampler that will be used to split the dataset without changing too much the label ratio positive/negative.
    _imgs_keys: Union[Sequence[string], string]
        A string or a list of string that indicate The images name in the hdf5 file that will be load in the dataset
        (Exemple: "t1").
    __is_ready : bool
        A boolean that indicate if the dataset is ready to used in training.
    _labels : np.array
        A numpy array that contain the labels of each data for each task.
    __patients_data : dict
        A dictionary that contain the data per target and per patient. Will be remove when the method prepare_dataset
        will be call.
    _patient_id : np.array
        A list of string that indicate the patient id of each data in the dataset.
    __patients_labels : dict
        A dictionary that contain the label per target and per patient. Will be used by data_spliter to split the
        dataset Will be remove when the method prepare_dataset will be call.
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
    prepare_dataset()
        Transfer the data from self.__patients_data to the correct attribute. The attribute self.__patients_data and
        self.__patients_labels will be delete.
    remove_unlabeled_data():
        Remove the data that has no label for every task.
    split(tol_dict, max_iter, pop, random_seed, sample_size, transform):
        Split the current dataset and return a new dataset with the data.
    """

    def __init__(self,
                 imgs_keys: Union[Sequence[str], str],
                 tasks: Sequence[str],
                 clinical_features: Optional[Union[List[str], str]] = None,
                 exclude_list: Optional[List[str]] = None,
                 hdf5_filepath: Optional[str] = None,
                 split: Optional[str] = SplitName.TRAIN,
                 transform: Optional[Compose] = None) -> None:
        """
        Create a dataset by loading the renal image at the given path.

        :param imgs_keys: The images name in the hdf5 file that will be load in the dataset (Exemple: "t1").
        :param tasks: A list of clinical_features that will be used has labels for tasks.
        :param clinical_features: A list of string that indicate which clinical features will be used
                                  to train the model.
        :param exclude_list: A list of patient_id to exclude in this dataset.
        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Default=DatasetName.TRAIN)
        :param transform: A function/transform that will be applied on the images and the ROI.
        """
        super().__init__(tasks=tasks,
                         imgs_keys=imgs_keys,
                         clinical_features=clinical_features,
                         split=split,
                         transform=transform)
        self.__data_splitter = None
        self.__is_ready = False
        self.__patients_data = {}
        self.__patients_labels = {}

        if split is not None:
            to_exclude = set() if exclude_list is None else set(exclude_list)  # set because we do not care about order

            self._read_hdf5(to_exclude=to_exclude,
                            filepath=hdf5_filepath,
                            split=split.lower() if split is not None else split)

    def add_patient(self,
                    patients_data: dict,
                    patients_labels: dict) -> None:
        """
        Add a list of patient to the current dataset. WARNING: You cannot use this method after calling the
        prepare_dataset method, because no patient can be added after the data has been unpack.

        :param patients_data: A dictionary that contain the patients_data (images, features and labels) per target
                              and per patient.
        :param patients_labels: A dictionary that contain the patients labels per target and per patient.
        """
        assert not self.__is_ready, "Cannot add data when the dataset is ready for training. " \
                                    "The data has already been unpack"
        for pat_id in list(patients_data.keys()):
            self.__patients_data[pat_id] = patients_data[pat_id]
            self.__patients_labels[pat_id] = patients_labels[pat_id]

        self.__data_splitter = Sampler(data=self.__patients_labels, labels_name=self._tasks)

    def _extract_data(self,
                      idx: Sequence[int],
                      pop: bool = True) -> Tuple[Sequence[dict],
                                                 Union[Sequence[dict], Sequence[int]],
                                                 Sequence[str],
                                                 Sequence[Sequence[int]]]:
        """
        Extract data without applying transformation on the images.

        :param idx: The index of the data to extract.
        :param pop: If True, the extracted data are removed from the dataset. (Default: True)
        :return: A tuple that contain the data (images), the labels, the patients id and the clinical data.
        """
        mask = np.ones(len(self._data), dtype=bool)
        mask[idx] = False

        data = self._data[~mask]
        labels = self._labels[~mask]
        patient_id = self._patient_id[~mask]
        clin = self._clinical_data[~mask] if self._with_clinical else None

        if pop:
            self._data = self._data[mask]
            self._labels = self._labels[mask]
            self._patient_id = self._patient_id[mask]
            self._clinical_data = self._clinical_data[mask] if self._with_clinical else None

        return data, labels, patient_id, clin

    def extract_patient(self,
                        patients_list: Sequence[str],
                        pop: bool = True) -> Tuple[dict, dict]:
        """
        Extract data from patient that are present in a given list name.

        :param patients_list: A sequence of string that represent the patient name to extract.
        :param pop: If true those patient will be remove from the current dataset.
        :return: A tuple that contain a dictionary of patients_data (images, clinical_features and labels) and
                 a dictionary that contain the patients_labels per target and per patient.
        """
        assert not self.__is_ready, "Cannot extract data when the dataset is ready for training. " \
                                    "The data has already been unpack"

        patients_data = {}
        patients_labels = {}

        for patient in patients_list:
            patients_data[patient] = self.__patients_data[patient]
            patients_labels[patient] = self.__patients_labels[patient]

            if pop:
                del self.__patients_data[patient]
                del self.__patients_labels[patient]
                self.__data_splitter = Sampler(data=self.__patients_labels, labels_name=self._tasks)

        return patients_data, patients_labels

    @property
    def is_ready(self):
        return self.__is_ready

    def prepare_dataset(self) -> None:
        """
        Transfer the data from self.__patients_data to the correct attribute. The attribute self.__patients_data and
        self.__patients_labels will be delete.
        """
        assert not self.__is_ready, "The dataset has already been prepare."
        clinical_data = []
        data = []
        labels = []
        patients_id = []

        for pat_id, patient in list(self.__patients_data.items()):
            for target in list(patient.values()):
                data.append(target[IMAGES])
                labels.append({task: target[task] for task in self._tasks})
                patients_id.append(pat_id)
                if self._with_clinical:
                    clinical_data.append([target[clin] for clin in self.__features_name])

        # Remove the redundant data
        self.__patients_data = None
        self.__patients_labels = None

        self._data = np.array(data)
        self._labels = np.array(labels)
        self._patient_id = np.array(patients_id)
        if self._with_clinical:
            self._clinical_data = np.array(clinical_data)

        self.__is_ready = True
        self.__data_splitter = None
        self.remove_unlabeled_data()

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
        self._patient_id = np.array([key for key in list(dtset.keys()) if key not in to_exclude])
        self.__patients_data = {}
        self.__patients_labels = {}

        for pat_id in self._patient_id:
            patient = dtset[pat_id]
            self.__patients_data[pat_id] = {}
            self.__patients_labels[pat_id] = {}

            for target in list(patient.keys()):
                self.__patients_data[pat_id][target] = {}
                self.__patients_labels[pat_id][target] = {}

                imgs = {}
                for img_key in list(self._imgs_keys):
                    imgs[img_key] = patient[target][img_key][:]

                self.__patients_data[pat_id][target][IMAGES] = imgs

                for task in self._tasks:
                    self.__patients_data[pat_id][target][task] = patient[target].attrs[task]
                    self.__patients_labels[pat_id][target][task] = patient[target].attrs[task]

                if self._with_clinical:
                    for feat_name in self._features_name:
                        self.__patients_data[pat_id][target][feat_name] = patient[target].attrs[feat_name]
        f.close()

        self.__data_splitter = Sampler(data=self.__patients_labels, labels_name=self._tasks)

    def split(self,
              tol_dict: Dict[str, float],
              max_iter: int = 100,
              pop: bool = True,
              random_seed: Optional[int] = None,
              sample_size: float = 0.15,
              transform: Optional[Compose] = None,
              *args) -> BrainDataset:
        """
        Split the current dataset by sampling subset by using a sampler (see data_manager/sampler.py) and return
        a BrainDataset object that contain the subset of patient. The new BrainDataset is not ready and the method
        prepare_dataset must be called.

        :param tol_dict: A dictionary that indicate the tolerance factor (float) per label.
        :param max_iter: Maximum number of iteration.
        :param pop: If True, the extracted patient are removed from the dataset. (Default: True)
        :param random_seed: The seed that will be used to split to sample the set of patient.
        :param sample_size: A float that represent the proportion of data that will be use to create the test set.
        :param transform: A function/transform that will be applied on the images and the ROI.
        :return: A dataset that contain the splited data.
        """

        patient_list = self.__data_splitter.sample(tol_dict=tol_dict,
                                                   max_iter=max_iter,
                                                   seed=random_seed,
                                                   split_size=sample_size)
        patients_data, patients_labels = self.extract_patient(patient_list, pop)

        new_dataset = BrainDataset(hdf5_filepath=None,
                                   tasks=self._tasks,
                                   imgs_keys=self._imgs_keys,
                                   clinical_features=self._features_name,
                                   split=None,
                                   transform=transform)

        new_dataset.add_patient(patients_data=patients_data,
                                patients_labels=patients_labels)
        return new_dataset
