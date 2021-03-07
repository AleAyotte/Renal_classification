"""
    @file:              MultiTaskTrainer.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       Contain the class MultiTaskTrainer which inherit from the class Trainer. This class is used
                        to train the MultiLevelResNet and the SharedNet on the three task (malignancy, subtype and
                        grade prediction).
"""

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


class MultiTaskTrainer(Trainer):
    """
    The trainer class define an object that will be used to train and evaluate a given model. It handle the 
    mixed precision training, the mixup process and more.

    ...
    Attributes
    ----------
    _loss : str
        The name of the loss that will be used during the training.
    __m_loss : torch.nn
        The loss function of the malignant task.
    __s_loss : torch.nn
        The loss function of the subtype task.
    _mixed_precision : bool
        If true, mixed_precision will be used during training and inferance.
    model : NeuralNet
        The neural network to train and evaluate.
    __shared_net: bool
        If true, the model to train will be a SharedNet. In this we need to optimizer, one for the subnets and
        one for the sharing units and the Uncertainty loss parameters.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    _track_mode : str
        Control the information that are registred by tensorboard. Options: all, low, none (Default: all).
    __weights : Sequence[Sequence[int]]
        The list of weights that will be used to adjust the loss function
    _writer : SummaryWriter
        Use to keep track of the training with tensorboard.
    Methods
    -------
    fit():
        Train the model on the given dataset
    score(dt_loader: DataLoader, get_loss: bool = False):
        Compute the accuracy of the model on a given data loader.
    """
    def __init__(self, loss: str = "ce",
                 tol: float = 0.01,
                 mixed_precision: bool = False,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 classes_weights: str = "balanced",
                 shared_net: bool = False,
                 save_path: str = "",
                 track_mode: str = "all"):
        """
        The constructor of the trainer class. 

        :param loss: The loss that will be use during mixup epoch. (Default="ce")
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param mixed_precision: If true, mixed_precision will be used during training and inferance. (Default=False)
        :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will 
                           copied into the CUDA pinned memory. (Default=False)
        :param num_workers: Number of parallel process used for the preprocessing of the data. If 0, 
                            the main process will be used for the data augmentation. (Default=0)
        :param classes_weights: The configuration of weights that will be applied on the loss during the training.
                                Flat: All classes have the same weight during the training.
                                Balanced: The weights are inversionaly proportional to the number of data of each 
                                          classes in the training set.
                                Focused: Same as balanced but in the subtype and grade task, the total weights of the 
                                         two not none classes are 4 times higher than the weight class.
                                (Default="balanced")
        :param shared_net: If true, the model to train will be a SharedNet. In this we need to optimizer, one for the
                           subnets and one for the sharing units and the Uncertainty loss parameters. (Default=False)
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param track_mode: Control information that are registred by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default=all)
        """
        super().__init__(loss=loss,
                         tol=tol,
                         mixed_precision=mixed_precision,
                         pin_memory=pin_memory,
                         num_workers=num_workers,
                         shared_net=shared_net,
                         save_path=save_path,
                         track_mode=track_mode)

        assert classes_weights.lower() in ["flat", "balanced"], \
            "classes_weights should be one of those options: 'Flat' or 'Balanced'"
        weights = {"flat": [[1., 1.],
                            [1., 1.]],
                   "balanced": [[[1.3459, 0.7956]],
                                [2.0840, 0.6578]]}
        self.__weights = weights[classes_weights.lower()]
        self.__m_loss = None
        self.__s_loss = None
        
    def _init_loss(self, gamma: float) -> None:
        """
        Initialize the loss function by sending the classes weights on the appropriate device.

        :param gamma: Gamma parameter of the focal loss.
        """
        weight_0 = torch.Tensor(self.__weights[0]).to(self._device)
        weight_1 = torch.Tensor(self.__weights[1]).to(self._device)

        if self._loss == "ce":
            self.__m_loss = nn.CrossEntropyLoss(weight=weight_0)
            self.__s_loss = nn.CrossEntropyLoss(weight=weight_1)
        elif self._loss == "bce":
            self.__m_loss = nn.BCEWithLogitsLoss(pos_weight=weight_0)
            self.__s_loss = nn.BCEWithLogitsLoss(pos_weight=weight_1)
        elif self._loss == "focal":
            self.__m_loss = FocalLoss(gamma=gamma, weight=weight_0)
            self.__s_loss = FocalLoss(gamma=gamma, weight=weight_1)
        else:  # loss == "marg"
            raise NotImplementedError
    
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
            images, labels = data["sample"].to(self._device), data["labels"]
            images = Variable(images)

            m_labels = Variable(labels["malignant"].to(self._device))
            s_labels = Variable(labels["subtype"].to(self._device))

            features = None
            if "features" in list(data.keys()):
                features = Variable(data["features"].to(self._device))

            for optimizer in optimizers:
                optimizer.zero_grad()

            # training step
            with amp.autocast(enabled=self._mixed_precision):
                m_pred, s_pred = self.model(images) if features is None else self.model(images, features)

                s_mask = torch.where(s_labels > 0, 1, 0).bool()

                m_loss = self.__m_loss(m_pred, m_labels)
                s_loss = self.__s_loss(s_pred[s_mask], s_labels[s_mask] - 1)

                losses = torch.stack((m_loss, s_loss))
                loss = self.model.uncertainty_loss(losses)

            self._update_model(scaler, optimizers, schedulers, grad_clip, loss)
            sum_loss += loss

            if self._track_mode == "all":
                self._writer.add_scalars('Training/Loss', 
                                         {'Malignant': m_loss.item(),
                                          'Subtype': s_loss.item(),
                                          'Total': loss.item()}, 
                                         it + epoch*n_iters)

        return sum_loss.item() / n_iters

    def _mixup_criterion(self,
                         pred: Sequence[torch.Tensor],
                         labels: Sequence[Variable],
                         lamb: float,
                         permut: Sequence[int],
                         it: int) -> torch.FloatTensor:
        """
        Transform target into one hot vector and apply mixup on it

        :param pred: A matrix of the prediction of the model. 
        :param labels: Vector of the ground truth.
        :param lamb: The mixing paramater that has been used to produce the mixup during the foward pass.
        :param permut: A numpy array that indicate which images has been shuffle during the foward pass.
        :return: The mixup loss as torch tensor.
        """
        m_pred, s_pred = pred
        m_labels, s_labels = labels

        s_mask = torch.where(s_labels > 0, 1, 0).bool()

        s_labels = s_labels[s_mask] - 1

        if self.__m_loss.__class__.__name__ == "BCEWithLogitsLoss":
            m_hot_target = to_one_hot(m_labels, 2, self._device)
            s_hot_target = to_one_hot(s_labels, 2, self._device)

            m_mixed_target = lamb*m_hot_target + (1-lamb)*m_hot_target[permut]
            s_mixed_target = lamb*s_hot_target + (1-lamb)*s_hot_target[permut]

            m_loss = self.__m_loss(m_pred, m_mixed_target)
            s_loss = self.__s_loss(s_pred, s_mixed_target)

        else:
            m_loss = lamb*self.__m_loss(m_pred, m_labels) + (1-lamb)*self.__m_loss(m_pred, m_labels[permut])
            s_loss = lamb*self.__s_loss(s_pred, s_labels) + (1-lamb)*self.__s_loss(s_pred, s_labels[permut])

        losses = torch.stack((m_loss, s_loss))
        loss = self.model.uncertainty_loss(losses)

        if self._track_mode == "all":
            self._writer.add_scalars('Training/Loss', 
                                     {'Malignant': m_loss.item(),
                                      'Subtype': s_loss.item(),
                                      'Total': loss.item()}, 
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
            images, labels = data["sample"].to(self._device), data["labels"]
            images = Variable(images)

            m_labels = Variable(labels["malignant"].to(self._device))
            s_labels = Variable(labels["subtype"].to(self._device))

            features = None
            if "features" in list(data.keys()):
                features = Variable(data["features"].to(self._device))

            for optimizer in optimizers:
                optimizer.zero_grad()

            # Mixup activation
            mixup_key, lamb, permut = self.model.activate_mixup()

            # training step
            with amp.autocast(enabled=self._mixed_precision):
                m_pred, s_pred = self.model(images) if features is None else self.model(images, features)
                loss = self._mixup_criterion([m_pred, s_pred],
                                             [m_labels, s_labels],
                                             lamb,
                                             permut,
                                             it + epoch*n_iters)

            self._update_model(scaler, optimizers, schedulers, grad_clip, loss)
            self.model.disable_mixup(mixup_key)
            sum_loss += loss

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
        :return: The mean accuracy as float and the loss as float.
        """

        with amp.autocast(enabled=self._mixed_precision):
            conf_mat, loss = self._get_conf_matrix(dt_loader=dt_loader, get_loss=True)
            m_conf, s_conf = conf_mat

            m_recall = compute_recall(m_conf)
            s_recall = compute_recall(s_conf)
            
            m_acc = get_mean_accuracy(m_recall, geometric_mean=True)
            s_acc = get_mean_accuracy(s_recall, geometric_mean=True)

            mean_acc = get_mean_accuracy([m_acc, s_acc], geometric_mean=True)

        if self._track_mode != "none":
            self._writer.add_scalars('{}/Accuracy'.format(dataset_name), 
                                     {'Malignant': m_acc,
                                      'Subtype': s_acc},
                                     epoch)

            self._writer.add_scalars('{}/Recall/Malignant'.format(dataset_name), 
                                     {'Recall 0': m_recall[0],
                                      'Recall 1': m_recall[1]},
                                     epoch)
            
            self._writer.add_scalars('{}/Recall/Subtype'.format(dataset_name), 
                                     {'Recall 0': s_recall[0],
                                      'Recall 1': s_recall[1]},
                                     epoch)

            if dataset_name == "Validation":
                phi = self.model.uncertainty_loss.phi.detach().cpu().numpy()
                self._writer.add_scalars("Other/Uncertainty",
                                         {"phi_Malignant": phi[0],
                                          "phi_Subtype": phi[1]},
                                         epoch)
        return mean_acc, loss
    
    def _get_conf_matrix(self,
                         dt_loader: DataLoader,
                         get_loss: bool = False) -> Union[Tuple[Sequence[np.array], float],
                                                          Tuple[Sequence[np.array], Sequence[float]],
                                                          Tuple[np.array, float]]:
        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data.
        :param get_loss: Return also the loss if True.
        :return: The confusion matrix for each task. If get_loss is True then also return the average loss.
                 Otherwise, the AUC will be return for each task.
        """
        m_outs = torch.empty(0, 2).to(self._device)
        s_outs = torch.empty(0, 2).to(self._device)

        m_labels = torch.empty(0).long()
        s_labels = torch.empty(0).long()

        for data in dt_loader:
            images, labels = data["sample"].to(self._device), data["labels"]

            features = None
            if "features" in list(data.keys()):
                features = Variable(data["features"].to(self._device))
            with torch.no_grad():
                m_out, s_out = self.model(images) if features is None else self.model(images, features)

                m_outs = torch.cat([m_outs, m_out])
                s_outs = torch.cat([s_outs, s_out])

                m_labels = torch.cat([m_labels, labels["malignant"]])
                s_labels = torch.cat([s_labels, labels["subtype"]])
        
        with torch.no_grad():
            s_mask = torch.where(s_labels > 0, 1, 0).bool()

            s_labels = s_labels[s_mask] - 1
            s_outs = s_outs[s_mask]

            m_pred = torch.argmax(m_outs, dim=1)
            s_pred = torch.argmax(s_outs, dim=1)

            if self.__m_loss.__class__.__name__ == "BCEWithLogitsLoss":
                m_target = to_one_hot(m_labels, 2, self._device)
                s_target = to_one_hot(s_labels, 2, self._device)
            else:
                m_target, s_target = m_labels, s_labels

            m_loss = self.__m_loss(m_outs, m_target.to(self._device))
            s_loss = self.__s_loss(s_outs, s_target.to(self._device))

            losses = torch.stack((m_loss, s_loss))
            total_loss = self.model.uncertainty_loss(losses)

        m_conf = confusion_matrix(m_labels.numpy(), m_pred.cpu().numpy())
        s_conf = confusion_matrix(s_labels.numpy(), s_pred.cpu().numpy())

        if get_loss:
            return [m_conf, s_conf], total_loss.item()

        else:
            fpr, tpr, _ = roc_curve(y_true=m_labels.numpy(), y_score=m_outs[:, 1].cpu().numpy())
            m_auc_score = auc(fpr, tpr)
            fpr, tpr, _ = roc_curve(y_true=s_labels.numpy(), y_score=s_outs[:, 1].cpu().numpy())
            s_auc_score = auc(fpr, tpr)
            return [m_conf, s_conf], [m_auc_score, s_auc_score]
