"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain commun usefull function used by the main files.
"""

from comet_ml import Experiment
import numpy as np
from Trainer.Utils import compute_recall
from typing import Sequence, Union


API_KEY_FILEPATH = "comet_api_key.txt"  # path to the file that contain the API KEY for comet.ml


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
        print("**{:^16s}**".format(task_list[i].capitalize()))
        print("*"*20)

        labels_bincount = labels_bincount_list[i]

        print("There is {} negative examples and {} positive examples\n".format(
            labels_bincount[0], labels_bincount[1]
        ))


def print_score(dataset_name: str,
                task_list: Sequence[str],
                conf_mat_list: Sequence[np.array],
                auc_list: Sequence[float],
                experiment: Union[Experiment, None] = None) -> None:
    """
    Print the recall and the AUC score per task for a given dataset on which the model has been assess.

    :param dataset_name: A string that represent the name of the dataset on which the model has been assess.
    :param task_list: A list of string that represent the name of each task. This list should be in the same
                      order as conf_mat_list and auc_list.
    :param conf_mat_list: A list of numpy array that represent the list of confusion matrix. There is one
                          confusion per task.
    :param auc_list: A list of float that represent the AUC score for each task.
    :param experiment: A comet experiment object. If gived, then the confusion matrix and the metrics will
                       saved online. (Default=None)
    """
    print("**************************************")
    print("**{:^34s}**".format(dataset_name.upper() + " SCORE"))
    print("**************************************")

    for i in range(len(task_list)):
        auc = auc_list[i]
        recall = compute_recall(conf_mat_list[i])
        recall.append(float("nan")) if len(recall) == 1 else None

        print("{} AUC: {}".format(task_list[i], auc))
        print("{} Recall: {}\n".format(task_list[i], recall))

        if experiment is not None:
            name = dataset_name + " " + task_list[i].capitalize()
            filename = name + ".json"
            experiment.log_confusion_matrix(matrix=conf_mat_list[i], title=name, file_name=filename)
            experiment.log_metric(name=name + " AUC", value=auc)
            experiment.log_metric(name=name + " Recall 0", value=recall[0])
            experiment.log_metric(name=name + " Recall 1", value=recall[1])


def read_api_key() -> str:
    """
    Read the api key needed for saving experimentation result on comet.ml.

    :return: The api key that is write in the file comet_api_key.txt.
    """
    f = open(API_KEY_FILEPATH)
    return str(f.read())


def save_hparam_on_comet(experiment: Experiment,
                         args_dict: dict) -> None:
    """
    Delete non hyperparameters arguments in the argparse and save the hyperparameters on comet.ml.

    :param experiment: The comet_ml experiment object. Will be used to save the hyperparameters online.
    :param args_dict: The argsparse dictionnary.
    """

    del args_dict["device"]
    del args_dict["pin_memory"]
    del args_dict["track_mode"]
    del args_dict["worker"]
    del args_dict["early_stopping"]
    experiment.log_parameters(args_dict)
