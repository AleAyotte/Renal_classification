"""
    @file:              MultiTaskTrainer.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       Contain the class SingleTaskTrainer which inherit from the class Trainer. This class is used
                        to train the 2D/3D ResNet on one of the three task (malignancy, subtype and grade prediction).
"""

from Model.Module import MarginLoss
from monai.losses import FocalLoss
from monai.optimizers import Novograd
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from Trainer.Trainer import Trainer
from Trainer.Utils import to_one_hot, compute_recall, get_mean_accuracy
import torch
from torch import nn
from torch.cuda import amp
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Sequence, Tuple, Union


ALL_TASK = ["malignancy", "subtype", "grade", "ssign"]


class SingleTaskTrainer(Trainer):
    """
    The SingleTaskTrainer class inherit of the Trainer class. It handle the training and the assess of a given
    model on a single task.

    ...
    Attributes
    ----------
    _classes_weights : str
        The configuration of weights that will be applied on the loss during the training.
        Flat: All classes have the same weight during the training.
        Balanced: The weights are inversionaly proportional to the number of data of each classes in the training set.
        (Default="balanced")
    _loss : str
        The name of the loss that will be used during the training.
    __loss : torch.nn
        The loss function that will be used to train the model.
    _mixed_precision : bool
        If true, mixed_precision will be used during training and inferance.
    model : NeuralNet
        The neural network to train and evaluate.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    _tasks : list
        A list of string that contain the name of every task for which the model will be train.
    _track_mode : str
        Control the information that are registred by tensorboard. Options: all, low, none.
    _writer : SummaryWriter
        Use to keep track of the training with tensorboard.
    Methods
    -------
    fit():
        Train the model on the given dataset
    score(dt_loader: DataLoader, get_loss: bool = False):
        Compute the accuracy of the model on a given data loader.
    """
    def __init__(self,
                 loss: str = "ce",
                 tol: float = 0.01,
                 early_stopping: bool = False,
                 mixed_precision: bool = False,
                 pin_memory: bool = False,
                 num_classes: int = 2,
                 num_workers: int = 0,
                 classes_weights: str = "balanced",
                 save_path: str = "",
                 track_mode: str = "all",
                 task="malignancy"):
        """
        The constructor of the trainer class. 

        :param classes_weights: The configuration of weights that will be applied on the loss during the training.
                                Flat: All classes have the same weight during the training.
                                Balanced: The weights are inversionaly proportional to the number of data of each
                                          classes in the training set.
                                (Default="balanced")
        :param loss: The loss that will be use during mixup epoch. (Default="bce")
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param early_stopping: If true, the training will be stop after the third of the training if the model did
                               not achieve at least 50% validation accuracy for at least one epoch. (Default=False)
        :param mixed_precision: If true, mixed_precision will be used during training and inferance. (Default=False)
        :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will 
                           copied into the CUDA pinned memory. (Default=False)
        :param num_workers: Number of parallel process used for the preprocessing of the data. If 0, 
                            the main process will be used for the data augmentation. (Default=0)
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param track_mode: Control information that are registred by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default=all)
        """
        assert task.lower() in ALL_TASK, "Task should be one of those options: " \
                                         "'malignancy', 'subtype', 'grade', 'ssign'"

        super().__init__(tasks=[task],
                         num_classes={task: num_classes},
                         loss=loss,
                         tol=tol,
                         early_stopping=early_stopping,
                         mixed_precision=mixed_precision,
                         pin_memory=pin_memory,
                         num_workers=num_workers,
                         classes_weights=classes_weights,
                         save_path=save_path,
                         track_mode=track_mode)
        self.__loss = None
        self.__task = task

    def _init_loss(self, gamma: float) -> None:
        """
        Initialize the loss function by sending the classes weights on the appropriate device.

        :param gamma: Gamma parameter of the focal loss.
        """

        if self._classes_weights == "balanced":
            weight = torch.Tensor(self._weights[self._tasks[0]]).to(self._device)
        else:
            weight = None

        if self._loss == "ce":
            self.__loss = nn.CrossEntropyLoss(weight=weight)
        elif self._loss == "bce":
            self.__loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        elif self._loss == "focal":
            self.__loss = FocalLoss(gamma=gamma, weight=weight)
        else:
            self.__loss = MarginLoss()

    def _standard_epoch(self,
                        train_loader: DataLoader,
                        optimizers: Sequence[Union[torch.optim.Optimizer, Novograd]],
                        schedulers: Sequence[CosineAnnealingWarmRestarts],
                        grad_clip: float,
                        epoch: int) -> float:
        """
        Make a standard training epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizers: The torch optimizers that will used to train the model.
        :param schedulers: The learning rate schedulers that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss.
        """
        sum_loss = 0
        n_iters = len(train_loader)

        scaler = amp.grad_scaler.GradScaler() if self._mixed_precision else None
        for it, data in enumerate(train_loader, 0):
            # Extract the data
            images, labels = data["sample"].to(self._device), data["labels"][self._tasks[0]].to(self._device)
            features = None

            if "features" in list(data.keys()):
                features = data["features"].to(self._device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            # training step
            with amp.autocast(enabled=self._mixed_precision):
                pred = self.model(images) if features is None else self.model(images, features)
                loss = self.__loss(pred, labels)

            self._update_model(grad_clip, loss, optimizers, scaler, schedulers)
            sum_loss += loss

            if self._track_mode == "all":
                self._writer.add_scalars('Training/Loss', 
                                         {'Loss': loss.item()}, 
                                         it + epoch*n_iters)

        return sum_loss.item() / n_iters

    def _mixup_criterion(self,
                         pred: Sequence[torch.Tensor],
                         target: Sequence[Variable],
                         lamb: float,
                         permut: Sequence[int],
                         it: int) -> torch.FloatTensor:
        """
        Transform target into one hot vector and apply mixup on it

        :param pred: A matrix of the prediction of the model. 
        :param target: Vector of the ground truth.
        :param lamb: The mixing paramater that has been used to produce the mixup during the foward pass.
        :param permut: A numpy array that indicate which images has been shuffle during the foward pass.
        :return: The mixup loss as torch tensor.
        """
        pred, target = pred[0], target[0]

        if self.__loss.__class__.__name__ == "BCEWithLogitsLoss":
            # The last dimension length of the prediction correspond to the number of classes.
            hot_target = to_one_hot(target, pred.size()[-1], self._device)

            mixed_target = lamb*hot_target + (1-lamb)*hot_target[permut]

            loss = self.__loss(pred, mixed_target)

        else:
            loss = lamb*self.__loss(pred, target) + (1-lamb)*self.__loss(pred, target[permut])

        if self._track_mode == "all":
            self._writer.add_scalars('Training/Loss', 
                                     {'Loss': loss.item()}, 
                                     it)

        return loss

    def _mixup_epoch(self,
                     train_loader: DataLoader,
                     optimizers: Sequence[Union[torch.optim.Optimizer, Novograd]],
                     schedulers: Sequence[CosineAnnealingWarmRestarts],
                     grad_clip: float,
                     epoch: int) -> float:
        """
        Make a manifold mixup epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizers: The torch optimizers that will used to train the model.
        :param schedulers: The learning rate schedulers that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss.
        """
        sum_loss = 0
        n_iters = len(train_loader)

        scaler = amp.grad_scaler.GradScaler() if self._mixed_precision else None
        for it, data in enumerate(train_loader, 0):
            # Extract the data
            images, labels = data["sample"].to(self._device), data["labels"][self._tasks[0]].to(self._device)
            features = None

            if "features" in list(data.keys()):
                features = data["features"].to(self._device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            # Mixup activation
            mixup_key, lamb, permut = self.model.activate_mixup()

            # training step
            with amp.autocast(enabled=self._mixed_precision):
                pred = self.model(images) if features is None else self.model(images, features)
                loss = self._mixup_criterion([pred], 
                                             [labels], 
                                             lamb, 
                                             permut,
                                             it + epoch*n_iters)

            self._update_model(grad_clip, loss, optimizers, scaler, schedulers)
            sum_loss += loss

            self.model.disable_mixup(mixup_key)

        return sum_loss.item() / n_iters

    def _validation_step(self,
                         dt_loader: DataLoader,
                         epoch: int,
                         dataset_name: str = "Validation") -> Tuple[float, float]:
        """
        Execute the validation step and save the metrics with tensorboard.

        :param dt_loader: A torch data loader that contain test or validation data.
        :param epoch: The current epoch number.
        :param dataset_name: The name of the dataset will be used to save the metrics with tensorboard.
        :return: The accuracy as float and the loss as float.
        """

        with amp.autocast(enabled=self._mixed_precision):
            conf_mat, loss = self._get_conf_matrix(dt_loader=dt_loader, get_loss=True)
            conf_mat = conf_mat

            recalls = compute_recall(conf_mat)
            acc = get_mean_accuracy(recalls, geometric_mean=True)

        if self._track_mode != "none":
            self._writer.add_scalars('{}/Accuracy'.format(dataset_name), 
                                     {'Accuracy': acc}, 
                                     epoch)

            stats_dict = {}
            for i, recall in enumerate(recalls):
                stats_dict["Recall {}".format(i)] = recall

            self._writer.add_scalars('{}/Recall'.format(dataset_name), 
                                     stats_dict, 
                                     epoch)
        return acc, loss
    
    def _get_conf_matrix(self,
                         dt_loader: DataLoader,
                         get_loss: bool = False,
                         save_path: str = "",
                         use_optimal_threshold: bool = False) -> Union[Tuple[Sequence[np.array], float],
                                                                       Tuple[Sequence[np.array], Sequence[float]],
                                                                       Tuple[np.array, float]]:
        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data.
        :param get_loss: Return the loss instead of the auc score.
        :param save_path: The filepath of the csv that will be used to save the prediction.
        :param use_optimal_threshold: If true, use the optimal threshold to classify the data.
        :return: The confusion matrix and the average loss if get_loss == True.
        """

        outs, labels = self._predict(dt_loader=dt_loader)

        if save_path:
            patient_id = dt_loader.dataset.get_patient_id()
            self._save_prediction(outs, labels, patient_id, save_path)

        with torch.no_grad():
            outs = outs[self.__task]
            labels = labels[self.__task]

            threshold = self._optimal_threshold[self.__task] if use_optimal_threshold else 0.5
            # pred = torch.argmax(outs, dim=1)
            pred = torch.where(outs[:, 1] >= threshold, 1, 0)
            loss = self.__loss(outs, labels.to(self._device)) if get_loss else None

        conf_mat = confusion_matrix(labels.numpy(), pred.cpu().numpy())

        # Save the prediction if a filepath has been gived for the csv file.
        if get_loss:
            return conf_mat, loss.item()
        else:
            fpr, tpr, _ = roc_curve(y_true=labels.numpy(), y_score=outs[:, 1].cpu().numpy())
            auc_score = auc(fpr, tpr)
            return conf_mat, auc_score
