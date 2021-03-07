"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain commun usefull function used by the main files.
"""

import numpy as np
from Trainer.Utils import compute_recall
from typing import Sequence


def print_data_distribution(dataset_name: str,
                            task_list: Sequence[str],
                            labels_bincount_list: Sequence[np.array]) -> None:
    """
    Print the number of data per class per task for a given dataset.

    :param dataset_name: A string that represent the name of the dataset on which the model has been assess.
    :param task_list: A list of string that represent the name of each task. This list should be in the same
                      order as labels_bincount_list.
    :param labels_bincount_list: A list of numpy array where each numpy array represent the number of data per class
                                 for a task.
    """
    print("**************************************")
    print("**{:^34s}**".format(dataset_name))
    print("**************************************")

    for i in range(len(task_list)):
        print("*"*20)
        print("**{:^16s}**".format(task_list[i]))
        print("*"*20)

        labels_bincount = labels_bincount_list[i]

        print("There is {} negative examples and {} positive examples\n".format(
            labels_bincount[0], labels_bincount[1]
        ))


def print_score(dataset_name: str,
                task_list: Sequence[str],
                conf_mat_list: Sequence[np.array],
                auc_list: Sequence[float]) -> None:
    """
    Print the recall and the AUC score per task for a given dataset on which the model has been assess.

    :param dataset_name: A string that represent the name of the dataset on which the model has been assess.
    :param task_list: A list of string that represent the name of each task. This list should be in the same
                      order as conf_mat_list and auc_list.
    :param conf_mat_list: A list of numpy array that represent the list of confusion matrix. There is one
                          confusion per task.
    :param auc_list: A list of float that represent the AUC score for each task.
    """
    print("**************************************")
    print("**{:^34s}**".format(dataset_name.upper() + "SCORE"))
    print("**************************************")

    for i in range(len(task_list)):
        auc = auc_list[i]
        recall = compute_recall(conf_mat_list[i])

        print("{} AUC: {}".format(task_list[i], auc))
        print("{} Recall: {}\n".format(task_list[i], recall))
