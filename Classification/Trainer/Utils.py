"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 02/2021

    @Description:       Contain some usefull function used by the Trainer. Those function are compute_recall,
                        get_mean_accuracy, to_one_hot. There is also the init_weight function used to initialize
                        the weight of the models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import torch
from torch import nn
from torch.autograd import Variable
from typing import List, Sequence, Union


def compute_recall(conf_matrix: np.array) -> List[float]:
    """
    Compute the recall of each class.

    :param conf_matrix: A numpy matrix that represent the confusion matrix.
    :return: A list of float that represent the recall of each class.
    """
    recalls = []

    for it in range(len(conf_matrix)):
        if conf_matrix[it].sum() > 0:
            recalls.append(conf_matrix[it, it] / conf_matrix[it].sum())
        else:
            recalls.append(float('nan'))
    return recalls


def find_optimal_cutoff(labels: Sequence[int], prediction: Sequence[float]):
    """
    Find the optimal threshold in a binary classification problem.
    From Manohar Swamynathan @https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

    :param labels: A sequence of integer that indicate the labels of each data.
    :param prediction: A sequence of float that indicate the prediction made by the model.
    :return: A float that represent the optimal threshold point that maximize the mean recall on the given dataset.
    """
    fpr, tpr, threshold = roc_curve(labels, prediction)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0]


def get_mean_accuracy(recalls: Sequence[float], 
                      geometric_mean: bool = True) -> float:
    """
    Compute the mean accuracy according to the recalls of each classes.

    :param recalls: A list of float that represent the recall of each classes.
    :param geometric_mean: If true, the geometric mean is used. Else the euclidian mean will be used.
    :return: The mean accuracy.
    """
    if geometric_mean:
        return np.prod(recalls) ** (1 / len(recalls))
    else:
        return float(np.mean(recalls))


def to_one_hot(inp: Union[torch.Tensor, Variable],
               num_classes: int, 
               device: str = "cuda:0") -> Variable:
    """
    Transform a logit ground truth to a one hot vector
    :param inp: The input vector to transform as a one hot vector
    :param num_classes: The number of classes in the dataset
    :param device: The device on which the result will be return. (Default="cuda:0", first GPU)

    :return: A one hot vector that represent the ground truth
    """
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)

    return Variable(y_onehot.to(device), requires_grad=False)
