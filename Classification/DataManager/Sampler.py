"""
    @file:              Sampler.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 09/2021

    @Description:       This file contain the Sampler class. The sampler can split a dataset where there is multiple
                        data (ex: brain tumors) and labels per patient. This class will be used by the BrainDataset.
"""

import math
import numpy as np
from typing import Dict, Final, List, Optional, Tuple, Union

AIM_TARGET: Final = "aim_target"
AIM_RATE: Final = "aim_rate"
GREATER: Final = "greater_rate"
LOWER: Final = "lower_rate"
NUM_TARGET: Final = "num_target"
NUM_POS: Final = "num_pos"
POS_RATE: Final = "pos_rate"


class Sampler:
    """
    Renal classification dataset.

    ...
    Attributes
    ----------
    __data : Dict[str, Dict[str, Dict[str, int]]]
        A dictionary of patient where the item is a dictionary of target where the item is a dictionary of label.
    __labels : List[str]
        A list of string that contain the labels name.
    __data_stats : Dict[str, Dict[str, Union[float, int]]]
        A dictionary that contain, per label, the number of data, the number of positive data and the positive rate
        (num_pos/num_data).
    __patient_stats : Dict[str, Dict[str, Dict[str, Union[float, int]]]]
        A dictionary that indicate the following stats per patient and per label: number of data, number of positive
        data and positive rate (num_pos/num_data).
    __rate_status :  Dict[str, Dict[str, List[str]]]
        A dictionary that indicate per label, a list of patient that have a greater positive rate than the average and
        a list that indicate the patient that have a lower positive rate than the average.
    Methods
    -------
    get_split_stats(pat_list)
        Get statistic about a set of data per label. Those statistic include the number of data, the number of positive
        examples and the positive rate.
    sample(tol, seed, split_size)
        Sample from the list of patient a subset that can be used has a test set according to a tolerance factor per
        label.
    """
    def __init__(self,
                 data: dict,
                 labels_name: List[str]) -> None:
        """
        Create a sampler object with a given data dictionary that indicate the labels per target per patient and a
        given list of labels name.

        :param data: A dictionary that contain the labels per target and per patient. The dictionary should look like
                     this: {"PatientName1": {"Target1": {"Label1": value1, "Label2": value2, ...},
                                             "Target2": {"Label1": value1, "Label2": value2, ...},
                                             ...},
                            "PatientName2": {"Target1": ...},
                            ...}
                     The value of the label should one of those -1: No label, 0: Negative, 1: Positive.
        :param labels_name: A list that contain the name of the labels.
        """
        self.__data = data
        self.__labels = labels_name
        self.__data_stats, self.__patient_stats = self.__compute_data_stats()
        self.__rate_status = self.__compute_rate_status()

    def __compute_data_stats(self) -> Tuple[dict, dict]:
        """
        Compute the patient statistics (number of data, number of positive and positive rate, per label) and the
        data statistics (total number of data, total number of positive and global positive rate, per label).

        :return: A dictionary that contain the following stats per label: the number of data, the number of positive
                 data and the positive rate and a dictionary that contain the same stats by per patient and per label.
        """
        data_stats = {label: {NUM_TARGET: 0, NUM_POS: 0, POS_RATE: 0} for label in self.__labels}
        pat_stats = {pat_id: {label: {} for label in self.__labels}
                     for pat_id in list(self.__data.keys())}

        for label in self.__labels:
            for pat_id, target_list in list(self.__data.items()):
                num_target, num_pos = 0, 0
                for target in list(target_list.values()):
                    num_target += 1 if target[label] != -1 else 0
                    num_pos += 1 if target[label] == 1 else 0

                pat_stats[pat_id][label][NUM_TARGET] = num_target
                pat_stats[pat_id][label][NUM_POS] = num_pos
                pat_stats[pat_id][label][POS_RATE] = num_pos / num_target if num_target != 0 else 0

                data_stats[label][NUM_TARGET] += num_target
                data_stats[label][NUM_POS] += num_pos

            data_stats[label][POS_RATE] = data_stats[label][NUM_POS] / data_stats[label][NUM_TARGET]

        return data_stats, pat_stats

    def __compute_rate_status(self):
        """
        Create two list of patient. The first one indicate the patients with a greater positive rate then the
        average and the second one the patients with lower positive rate then the average.

        :return: A dictionary that contain per label a list with patients with positive rate greater than the average
                 and a list with patients with positive rate lower than the average.
        """
        rate_status = {label: {GREATER: [], LOWER: []} for label in self.__labels}
        for pat_id, pat_stat in list(self.__patient_stats.items()):
            for label in self.__labels:
                pos_rate = self.__data_stats[label][POS_RATE]
                if pat_stat[label][POS_RATE] >= pos_rate:
                    rate_status[label][GREATER].append(pat_id)
                else:
                    rate_status[label][LOWER].append(pat_id)
        return rate_status

    def __compute_threshold_limit(self,
                                  split_size: float,
                                  tol_dict: Dict[str, float]) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Calculate the threshold limit for the number of target and the positive that should be respect in the
        test set for each labels.

        :param split_size: A float that indicate the aimed proportion of data that should be use to create the test set.
        :param tol_dict: A dictionary of float that indicate the maximum deviation in percentage per labels. Ex:
                         if split_size=0.15 and tol_dict[labelA] = 0.02. Then the test set should contain between 13%
                         and 17% of the data.
        :return: A dictionary that indicate the threshold limit per label for the number of data (int) and for the
                 positive rate in percentage (float), but also the aimed number of data and the aimed positive rate.
        """
        thresh_dict = {}
        for label in self.__labels:
            stats = self.__data_stats[label]
            tol = tol_dict[label]
            thresh_dict[label] = {
                NUM_TARGET: np.floor((np.array([-tol, tol]) + split_size) * stats[NUM_TARGET]),
                POS_RATE: np.array([-tol, tol]) + stats[POS_RATE],
                AIM_TARGET: math.floor(split_size * stats[NUM_TARGET]),
                AIM_RATE: stats[POS_RATE]
            }

        return thresh_dict

    def get_split_stats(self,
                        pat_list: np.array) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Get statistic about a set of data per label. Those statistic include the number of data, the number of positive
        examples and the positive rate.

        :param pat_list: A numpy array that contain the patient_id of each patient in the set.
        :return: Return a dictionary of dictionary that contain the mentioned stats. The keys of the first dictionary
                 are the labels and the keys of the embedded dictionary is the name of the stats. The stats are the
                 number of data (int) (because one patient can attached to multiple data), the number of positive data
                 (int) and the positive rate (float).
        """
        stats = {label: {} for label in self.__labels}

        for label in self.__labels:
            stats[label][NUM_TARGET] = np.sum([self.__patient_stats[pat][label][NUM_TARGET]
                                               for pat in pat_list])
            stats[label][NUM_POS] = np.sum([self.__patient_stats[pat][label][NUM_POS]
                                            for pat in pat_list])
            stats[label][POS_RATE] = stats[label][NUM_POS] / stats[label][NUM_TARGET]

        return stats

    def __is_split_valid(self,
                         split_stats: Dict[str, Dict[str, Union[float, int]]],
                         thresh_dict: dict) -> Union[Tuple[bool, str], bool]:
        """
        Verify if a set of data, according to the given set statistics, respect a given set of threshold limit for each
        label.

        :param split_stats: A dictionary that give per label (keys of the first dictionary) the stats (keys of the
                            embedded dictionary) that will be used to verify if the dataset respect the minimum and
                            maximum number of target and positive rate.
        :param thresh_dict: A dictionary that give per label (keys of the first dictionary) the threshold limits for
                            the number of target and the positive rate (keys of the embedded dictionary).
        :return: A boolean that indicate if the set can be used has test set and, in the negative case, a string that
                 indicate the first label for which the conditions are not respect.
        """
        for label in self.__labels:
            lim = thresh_dict[label]
            stats = split_stats[label]
            if not lim[NUM_TARGET][0] <= stats[NUM_TARGET] <= lim[NUM_TARGET][1]:
                return False, label

            if not lim[POS_RATE][0] <= stats[POS_RATE] <= lim[POS_RATE][1]:
                return False, label

        return True, ""

    def __init_split(self,
                     split_size: float) -> Tuple[np.array, np.array]:
        """
        Randomly split the list of patient with no regard to the number of target per patient.

        :param split_size: A float that indicate the proportion of patient that will be present in the test set.
        :return: Two numpy array (string) that represent the list of patient for the train set and the test set.
        """
        pat_list = np.array(list(self.__data.keys()))
        np.random.shuffle(pat_list)
        num_patient = len(pat_list)
        num_test = math.floor(split_size*num_patient)
        return pat_list[num_test:], pat_list[:num_test]

    def sample(self,
               tol_dict: Dict[str, float],
               max_iter: int = 100,
               seed: Optional[int] = None,
               split_size: float = 0.15) -> np.array:
        """
        Sample from the list of patient a subset that can be used has a test set according to a tolerance factor per
        label.

        :param tol_dict: A dictionary that indicate the tolerance factor (float) per label.
        :param max_iter: Maximum number of iteration.
        :param seed: The seed that will be used to split to sample the set of patient.
        :param split_size: A float that represent the proportion of data that will be use to create the test set.
        :return: A list of patient name that represent the test set.
        """
        np.random.seed(seed) if seed is not None else None
        train_list, test_list = self.__init_split(split_size)
        thresh_dict = self.__compute_threshold_limit(split_size, tol_dict)
        split_stats = self.get_split_stats(test_list)
        is_valid, label = self.__is_split_valid(split_stats, thresh_dict)
        for _ in range(max_iter):
            if is_valid:
                return test_list

            label_stats = split_stats[label]
            tol_lim = thresh_dict[label]

            if label_stats[NUM_TARGET] < tol_lim[AIM_TARGET]:
                if label_stats[POS_RATE] < tol_lim[AIM_RATE]:
                    # get patient with greater rate
                    train_list, test_list = self.__transfer_patient(giver=train_list, receiver=test_list,
                                                                    label=label, greater_rate=True)
                else:
                    # get patient with lower rate
                    train_list, test_list = self.__transfer_patient(giver=train_list, receiver=test_list,
                                                                    label=label, greater_rate=False)
            else:
                if label_stats[POS_RATE] < tol_lim[AIM_RATE]:
                    # give patient with lower rate
                    test_list, train_list = self.__transfer_patient(giver=test_list, receiver=train_list,
                                                                    label=label, greater_rate=False)
                else:
                    # give patient with greater rate
                    test_list, train_list = self.__transfer_patient(giver=test_list, receiver=train_list,
                                                                    label=label, greater_rate=True)

            split_stats = self.get_split_stats(test_list)
            is_valid, label = self.__is_split_valid(split_stats, thresh_dict)
        else:
            raise Exception("The sampler has not been able to create a test set according the given tolerance factor.")

    def __transfer_patient(self,
                           giver: np.array,
                           greater_rate: bool,
                           label: str,
                           receiver: np.array) -> Tuple[np.array, np.array]:
        """
        Transfer a patient from giver set to receiver set. The patient is choose randomly from those that has a
        greater (lower) positive rate that the average for a given label.

        :param giver: A list of patient that represent the giver set.
        :param greater_rate: If true, a patient with greater positive rate than the average will be transfer to the
                             new set. Else, a patient with a lower positive rate will be transfer.
        :param label: A string that indicate the label for which we want a greater (or lower) positive rate
        :param receiver: A list of patient from which the new patient will be append.
        :return: Two numpy array, the first one represent the giver set with the removed patient and the second one
                 represent the receiver with the new patient.
        """
        candidate = set(self.__rate_status[label][GREATER if greater_rate else LOWER])
        candidate = np.array(list(candidate & set(giver)))
        np.random.shuffle(candidate)

        return np.delete(giver, np.argwhere(giver == candidate[-1])), np.append(receiver, candidate[-1])
