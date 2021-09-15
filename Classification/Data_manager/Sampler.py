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
    def __init__(self,
                 data: dict,
                 labels_name: List[str]):
        self.__data = data
        self.__labels = labels_name
        self.__data_stats, self.__patient_stats = self.__compute_data_stats()
        self.__rate_status = self.__compute_rate_status()

    def __compute_data_stats(self) -> Tuple[dict, dict]:
        data_stats = {label: {NUM_TARGET: 0, NUM_POS: 0, POS_RATE: 0} for label in self.__labels}
        pat_stats = {pat_id: {label: {} for label in self.__labels}
                     for pat_id in list(self.__data.keys())}

        for pat_id, target_list in list(self.__data.items()):
            for label in self.__labels:
                num_target, num_pos = 0, 0
                for target in target_list:
                    num_target += 1 if target[label] != -1 else 0
                    num_pos += 1 if target[label] == 1 else 0

                pat_stats[label][NUM_TARGET] = num_target
                pat_stats[label][NUM_POS] = num_pos
                pat_stats[label][POS_RATE] = num_pos / num_target

                data_stats[label][NUM_TARGET] += num_target
                data_stats[label][NUM_POS] += num_pos

        for label in self.__labels:
            data_stats[label][POS_RATE] = data_stats[label][NUM_POS] / data_stats[label][NUM_TARGET]
        return data_stats, pat_stats

    def __compute_rate_status(self):
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
                                  tol: float) -> Dict[str, Dict[str, Union[float, int]]]:
        tol_dict = {}
        for label in self.__labels:
            stats = self.__data_stats[label]

            tol_dict[label] = {
                NUM_TARGET: math.floor((np.array([-tol, tol]) + split_size) * stats[NUM_TARGET]),
                POS_RATE: np.array([-tol, tol]) + stats[POS_RATE],
                AIM_TARGET: math.floor(split_size * stats[NUM_TARGET]),
                AIM_RATE: stats[POS_RATE]
            }

        return tol_dict

    def get_split_stats(self,
                        pat_list: np.array) -> Dict[str, Dict[str, Union[float, int]]]:
        stats = {label: {} for label in self.__labels}

        for label in self.__labels:
            stats[label][NUM_TARGET] = np.sum([self.__patient_stats[pat][label][NUM_TARGET]
                                               for pat in pat_list])
            stats[label][NUM_POS] = np.sum([self.__patient_stats[pat][label][NUM_POS]
                                            for pat in pat_list])
            stats[label][POS_RATE] = stats[label][NUM_POS] / stats[label][NUM_TARGET]

        return stats

    def __is_split_valid(self,
                         split_stats: dict,
                         tol_dict: dict) -> Union[Tuple[bool, str], bool]:
        for label in self.__labels:
            lim = tol_dict[label]
            stats = split_stats[label]
            if not lim[NUM_TARGET][0] <= stats[NUM_TARGET] <= lim[NUM_TARGET][1]:
                return False, label

            if not lim[POS_RATE][0] <= stats[POS_RATE] <= lim[POS_RATE][1]:
                return False, label

        return True

    def __init_split(self,
                     split_size: float = 0.5) -> Tuple[np.array, np.array]:
        pat_list = np.array(list(self.__data.keys()))
        np.random.shuffle(pat_list)
        num_patient = len(pat_list)
        num_test = math.floor(split_size*num_patient)
        return pat_list[num_test:], pat_list[:num_test]

    def sample(self,
               tol: float,
               seed: Optional[int] = None,
               split_size: float = 0.15) -> List[str]:

        np.random.seed(seed) if seed is not None else None
        train_list, test_list = self.__init_split(split_size)
        tol_dict = self.__compute_threshold_limit(split_size, tol)
        split_stats = self.get_split_stats(test_list)
        is_valid, label = self.__is_split_valid(split_stats, tol_dict)

        while not is_valid:
            label_stats = split_stats[label]
            tol_lim = tol_dict[label]

            if label_stats[NUM_TARGET] < tol_lim[AIM_TARGET]:
                if label_stats[POS_RATE] < tol_lim[AIM_RATE]:
                    # get patient with greater rate
                    train_list, test_list = self.__transfer_patient(train_list, test_list, label, True)
                else:
                    # get patient with lower rate
                    train_list, test_list = self.__transfer_patient(train_list, test_list, label, False)
            else:
                if label_stats[POS_RATE] < tol_lim[AIM_RATE]:
                    # give patient with lower rate
                    test_list, train_list = self.__transfer_patient(test_list, train_list, label, False)
                else:
                    # give patient with greater rate
                    test_list, train_list = self.__transfer_patient(test_list, train_list, label, True)

            split_stats = self.get_split_stats(test_list)
            is_valid, label = self.__is_split_valid(split_stats, tol_dict)

        return test_list

    def __transfer_patient(self,
                           giver: np.array,
                           receiver: np.array,
                           label: str,
                           greater_rate: bool) -> Tuple[np.array, np.array]:

        candidate = set(self.__rate_status[label][GREATER if greater_rate else LOWER])
        candidate = np.array(list(candidate & set(giver)))
        np.random.shuffle(candidate)

        return np.delete(giver, -1), np.append(receiver, candidate[-1])
