"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain commun usefull function used by the main files.
"""

from comet_ml import Experiment
from datetime import datetime
import numpy as np
import os
from Trainer.Utils import compute_recall, get_mean_accuracy
from typing import Optional, Sequence, Union

API_KEY_FILEPATH = "comet_api_key.txt"  # path to the file that contain the API KEY for comet.ml
CSV_PATH = "save/"


def get_predict_csv_path(model_name: str,
                         project_name: str,
                         testset_name: str,
                         task_name: Optional[str]):
    """
    Create the filepath that will be used to save the model prediction on each dataset.

    :param model_name: The name of the model that has been train
    :param project_name: The project name use in comet.ml
    :param testset_name: The testset that has been used
    :param task_name: The task name if it was single_task_learning.
    :return: The filepath of the prediction on the trainset, the valideset and the testset.
    """
    project_folder = os.path.join(CSV_PATH, project_name)
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    if task_name is not None:
        commun_path = str(os.path.join(project_folder, model_name + "_" + task_name + "_"))
    else:
        commun_path = str(os.path.join(project_folder, model_name + "_"))

    time = str(datetime.now())
    train_path = commun_path + "train_" + time + ".csv"
    valid_path = commun_path + "validation_" + time + ".csv"
    test_path = commun_path + testset_name + "_" + time + ".csv"

    return train_path, valid_path, test_path


def print_data_distribution(dataset_name: str,
                            task_list: Sequence[str],
                            labels_bincount_list: dict) -> None:
    """
    Print the number of data per class per task for a given dataset.

    :param dataset_name: A string that represent the name of the dataset on which the model has been assess.
    :param task_list: A list of string that represent the name of each task. This list should be in the same
                      order as labels_bincount_list.
    :param labels_bincount_list: A list of numpy array where each numpy array represent the number of data per class
                                 for a task.
    """

    line_sep = "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+"

    print(f"+{'-'*38}+")
    print(f"|{dataset_name:^38s}|")
    print(line_sep)
    print(f"|{'Task':^12s}|{'Negative':^12s}|{'Positive':^12s}|")
    print(line_sep)

    for i in range(len(task_list)):
        print("|{:^12s}|{:^12}|{:^12}|".format(task_list[i],
                                               labels_bincount_list[task_list[i]][0],
                                               labels_bincount_list[task_list[i]][1]))
        print(line_sep)


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

    dataset_label = dataset_name.upper() + " SCORE"
    line_sep = "+" + "-" * 20 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+"

    print(f"+{'-'*68}+")
    print(f"|{dataset_label:^68s}|")
    print(line_sep)
    print(f"|{'Task':^20s}|{'AUC':^11s}|{'Recall0':^11s}|{'Recall1':^11s}|{'Mean Recall':^11s}|")
    print(line_sep)

    for i in range(len(task_list)):
        auc = auc_list[i]
        recall = compute_recall(conf_mat_list[i])
        recall.append(float("nan")) if len(recall) == 1 else None
        mean_recall = get_mean_accuracy(recall, False)
        print(f"|{task_list[i]:^20s}|{auc:^11.2f}|{recall[0]:^11.2f}|{recall[1]:^11.2f}|{mean_recall:^11.2f}|")
        print(line_sep)

        if experiment is not None:
            if len(task_list) > 1:
                name = dataset_name + " " + task_list[i].capitalize()
            else:
                name = dataset_name
            filename = name + ".json"
            experiment.log_confusion_matrix(matrix=conf_mat_list[i], title=name, file_name=filename)
            experiment.log_metric(name=name + " AUC", value=auc)
            experiment.log_metric(name=name + " Recall 0", value=recall[0])
            experiment.log_metric(name=name + " Recall 1", value=recall[1])
            experiment.log_metric(name=name + " Mean Recall", value=mean_recall)


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
    del args_dict["track_mode"]
    del args_dict["worker"]
    experiment.log_parameters(args_dict)
