"""
    @file:              multi_task_trainer.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 1/2021

    @Description:       Contain the class MultiTaskTrainer which inherit from the class trainer. This class is used
                        to train the MultiLevelResNet and the SharedNet on the three task (malignancy, subtype and
                        grade prediction).
"""

from copy import deepcopy
from monai.losses import FocalLoss
from monai.optimizers import Novograd
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
import torch
from torch import nn
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Sequence, Tuple, Union

from constant import ModelType, Tasks
from model.hard_shared_resnet import HardSharedResNet
from model.ltb_resnet import LTBResNet
from model.module import MarginLoss
from trainer.trainer import Trainer
from trainer.utils import to_one_hot, compute_recall, get_mean_accuracy


class MultiTaskTrainer(Trainer):
    """
    The MultiTaskTrainer class inherit of the trainer class. It handle the training and the assess of a given
    model on multiple tasks at the same time.

    ...
    Attributes
    ----------
    __cond_prob : Sequence[Sequence[str]]
        A list of pairs, where the pair A, B represent the name of task from which we want to compute the conditionnal
        probability P(A|B).
    _classes_weights : str
        The configuration of weights that will be applied on the loss during the training.
        Flat: All classes have the same weight during the training.
        Balanced: The weights are inversionaly proportional to the number of data of each classes in the training set.
        (Default="balanced")
    _loss : str
        The name of the loss that will be used during the training.
    _main_tasks : list
        A list of tasks that will be used to validate the model.
    _mixed_precision : bool
        If true, mixed_precision will be used during training and inferance.
    model : NeuralNet
        The neural network to train and evaluate.
    _model_type : ModelType
        Indicate the type of model that will be train. Used because some model need a particular training.
    _num_classes: dict
        A dictionnary that indicate the number of classes for each task. For a regression task, the number of classes
        should be equal to one.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    _tasks : list
        A list of string that contain the name of every task for which the model will be train.
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
    def __init__(self,
                 main_tasks: Sequence[str],
                 aux_tasks: Optional[Sequence[str]] = None,
                 classes_weights: str = "balanced",
                 conditional_prob: Optional[Sequence[Sequence[str]]] = None,
                 early_stopping: bool = False,
                 loss: str = "ce",
                 mixed_precision: bool = False,
                 model_type: ModelType = ModelType.STANDARD,
                 num_classes: Optional[Dict[str, int]] = None,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 save_path: str = "",
                 tol: float = 0.01,
                 track_mode: str = "all"):
        """
        The constructor of the trainer class.

        :param main_tasks: A sequence of string that indicate the name of the main tasks. The validation will be done
                           only on those tasks. In other words, the model will be optimized with the early stopping
                           criterion on the main tasks.
        :param aux_tasks: A sequence of string that indicate the name of the auxiliary tasks. The model will not be
                          validate on those tasks.
        :param classes_weights: The configuration of weights that will be applied on the loss during the training.
                                Flat: All classes have the same weight during the training.
                                Balanced: The weights are inversionaly proportional to the number of data of each
                                          classes in the training set.
                                (Default="balanced")
        :param conditional_prob: A list of pairs, where the pair A, B represent the name of task from which we want
                                 to compute the conditional probability P(A|B).
        :param early_stopping: If true, the training will be stop after the third of the training if the model did
                               not achieve at least 50% validation accuracy for at least one epoch. (Default=False)
        :param loss: The loss that will be use during mixup epoch. (Default="ce")
        :param mixed_precision: If true, mixed_precision will be used during training and inference. (Default=False)
        :param model_type: Indicate the type of NeuralNetwork that will be use. It will have an impact on optimizers
                           and the training. See ModelType in constant.py (Default=ModelType.STANDARD)
        :param num_classes: A dictionary that indicate the number of classes of each. For regression task, the number
                            of classes should be 1.
        :param num_workers: Number of parallel process used for the preprocessing of the data. If 0,
                            the main process will be used for the data augmentation. (Default=0)
        :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will
                           copied into the CUDA pinned memory. (Default=False)
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param track_mode: Control information that are registered by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default=all)
        """

        # We merge the main tasks with the auxiliary tasks.
        tasks = list(set(main_tasks).union(set(aux_tasks))) if aux_tasks is not None else deepcopy(main_tasks)
        self._aux_tasks = deepcopy(aux_tasks) if aux_tasks is not None else []
        self._main_tasks = deepcopy(main_tasks)

        # If num_classes has not been defined, then we assume that every task are binary classification.
        if num_classes is None:
            num_classes = {}
            for task in main_tasks:
                num_classes[task] = Tasks.CLASSIFICATION
            for task in aux_tasks:
                num_classes[task] = Tasks.REGRESSION

        # If num_classes has been defined for some tasks but not all, we assume that the remaining are regression task
        else:
            num_classes = deepcopy(num_classes)
            key_set = set(num_classes.keys())
            tasks_set = set(tasks)
            missing_tasks = tasks_set - key_set
            assert missing_tasks == (tasks_set ^ key_set), f"The following tasks are present in num_classes " \
                                                           "but not in tasks {}".format(key_set - tasks_set)
            for task in list(missing_tasks):
                num_classes[task] = 1

        # Define the number of classes for the tasks created by the conditionnal probability.
        self.__cond_prob = [] if conditional_prob is None else deepcopy(conditional_prob)
        for cond_tasks in self.__cond_prob:
            assert set(cond_tasks) <= set(self._main_tasks), "Tasks using in condition_pro should be part of " \
                                                             "the main tasks set."
            task1, task2 = cond_tasks
            task_name = task1 + "|" + task2
            num_classes[task_name] = num_classes[task1]

        super().__init__(classes_weights=classes_weights,
                         early_stopping=early_stopping,
                         loss=loss,
                         mixed_precision=mixed_precision,
                         num_classes=num_classes,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         save_path=save_path,
                         model_type=model_type,
                         tasks=tasks,
                         tol=tol,
                         track_mode=track_mode)

        self.__losses = torch.nn.ModuleDict()

    def _init_loss(self, gamma: float) -> None:
        """
        Initialize the loss function by sending the classes weights on the appropriate device.

        :param gamma: Gamma parameter of the focal loss.
        """
        # Define the classes weights
        weight = {}
        if self._classes_weights == "balanced":
            for task in self._classification_tasks:
                weight[task] = torch.Tensor(self._weights[task]).to(self._device)
        else:
            for task in self._tasks:
                weight[task] = None

        # Classification tasks
        if self._loss == "ce":
            for task in self._classification_tasks:
                self.__losses[task] = nn.CrossEntropyLoss(weight=weight[task])
        elif self._loss == "bce":
            for task in self._classification_tasks:
                self.__losses[task] = nn.BCEWithLogitsLoss(pos_weight=weight[task])
        elif self._loss == "focal":
            for task in self._classification_tasks:
                self.__losses[task] = FocalLoss(gamma=gamma, weight=weight[task])
        else:  # loss == "marg"
            for task in self._classification_tasks:
                self._losses[task] = MarginLoss()
        # Regression tasks
        for task in self._regression_tasks:
            self.__losses[task] = nn.MSELoss()

    def _standard_epoch(self,
                        epoch: int,
                        grad_clip: float,
                        optimizers: Sequence[Union[torch.optim.Optimizer, Novograd]],
                        schedulers: Sequence[CosineAnnealingWarmRestarts],
                        train_loader: DataLoader) -> float:
        """
        Make a standard training epoch

        :param epoch: The current epoch number. Will be used to save the result with tensorboard.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :param optimizers: The torch optimizers that will used to train the model.
        :param schedulers: The learning rate schedulers that will be used at each iteration.
        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :return: The average training loss.
        """
        sum_loss = 0
        n_iters = len(train_loader)

        for optimizer in optimizers:
            optimizer.zero_grad()

        scaler = amp.grad_scaler.GradScaler() if self._mixed_precision else None
        for it, data in enumerate(train_loader, 0):
            # Extract the data
            images, labels = data["sample"].to(self._device), data["labels"]

            features = None
            if "features" in list(data.keys()):
                features = data["features"].to(self._device)

            # training step
            with amp.autocast(enabled=self._mixed_precision):
                preds = self.model(images) if features is None else self.model(images, features)

                losses = []
                aux_losses = []
                metrics = {}
                for task in self._tasks:
                    labels[task] = labels[task].to(self._device)

                    # Compute the loss only where have labels (label != -1).
                    if task in self._classification_tasks:
                        mask = torch.where(labels[task] > -1, 1, 0).bool()
                        loss = self.__losses[task](preds[task][mask], labels[task][mask])
                    else:
                        loss = self.__losses[task](preds[task].squeeze(), labels[task])

                    losses.append(loss) if task in self._main_tasks else aux_losses.append(loss)
                    metrics[task] = loss.item()

                # Compute final loss
                if len(self._aux_tasks) > 0:
                    if isinstance(self.model, LTBResNet) or isinstance(self.model, HardSharedResNet):
                        loss = self.model.loss(torch.stack(losses), torch.stack(aux_losses))
                    else:
                        losses.extend(aux_losses)
                        loss = self.model.loss(torch.stack(losses))
                else:
                    loss = self.model.loss(torch.stack(losses))

            self._update_model(grad_clip, loss, optimizers, scaler, schedulers)
            sum_loss += loss

            if self._track_mode == "all":
                metrics["total"] = loss.item()
                self._writer.add_scalars('Training/Loss',
                                         metrics,
                                         it + epoch*n_iters)

        return sum_loss.item() / n_iters

    def _mixup_criterion(self,
                         it: int,
                         labels: Dict[str, torch.Tensor],
                         lamb: float,
                         permut: Sequence[int],
                         pred: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """
        Transform target into one hot vector and apply mixup on it

        :param it: The current iteration number. Will be used to save the training loss with tensorboard.
        :param labels: Vector of the ground truth.
        :param lamb: The mixing paramater that has been used to produce the mixup during the foward pass.
        :param permut: A numpy array that indicate which images has been shuffle during the foward pass.
        :param pred: A matrix of the prediction of the model.
        :return: The mixup loss as torch tensor.
        """
        raise NotImplementedError

    def _mixup_epoch(self,
                     epoch: int,
                     grad_clip: float,
                     optimizers: Sequence[Union[torch.optim.Optimizer, Novograd]],
                     schedulers: Sequence[CosineAnnealingWarmRestarts],
                     train_loader: DataLoader) -> float:
        """
        Make a manifold mixup epoch

        :param epoch: The current epoch number. Will be used to save the result with tensorboard.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :param optimizers: The torch optimizers that will used to train the model.
        :param schedulers: The learning rate schedulers that will be used at each iteration.
        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :return: The average training loss.
        """
        raise NotImplementedError

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

            recall = {}
            acc = {}
            all_acc = {}

            for task, matrix in conf_mat.items():
                rec = compute_recall(matrix)
                recall[task] = rec
                accuracy = get_mean_accuracy(rec, geometric_mean=True)

                all_acc[task] = accuracy  # Include accuracy of conditionnal probability related task
                if task in self._tasks:
                    acc[task] = accuracy

            mean_acc = get_mean_accuracy(list(acc.values()), geometric_mean=True)

        if self._track_mode != "none":
            self._writer.add_scalars(f'{dataset_name}/Accuracy',
                                     acc,
                                     epoch)

            for task, recalls in recall.items():
                rec = {}
                for i in range(self._num_classes[task]):
                    rec[f"Recall {i}"] = recalls[i]
                self._writer.add_scalars(f'{dataset_name}/Recall/{task.capitalize()}',
                                         rec,
                                         epoch)

            if dataset_name == "Validation" and self._model_type is ModelType.SHARED_NET:
                self.model.save_histogram_sharing_unit(epoch, self._writer)

        return mean_acc, loss

    @torch.no_grad()
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
        :param get_loss: Return also the loss if True.
        :param save_path: The filepath of the csv that will be used to save the prediction.
        :param use_optimal_threshold: If true, use the optimal threshold to classify the data.
        :return: The confusion matrix for each task. If get_loss is True then also return the average loss.
                 Otherwise, the AUC will be return for each task.
        """
        outs, labels = self._predict(dt_loader=dt_loader)

        if save_path:
            patient_id = dt_loader.dataset.get_patient_id()
            self._save_prediction(outs, labels, patient_id, save_path)

        auc_score = {}
        conf_mat = {}
        masks = {}
        preds = {}

        main_classification_tasks = list(set(self._main_tasks) & set(self._classification_tasks))
        main_regression_tasks = list(set(self._main_tasks) & set(self._regression_tasks))

        losses = []
        if get_loss:
            losses = [self.__losses[task](outs[task].squeeze(), labels[task]) for task in main_regression_tasks]

        # Compute the mask and the prediction
        for task in main_classification_tasks:
            masks[task] = torch.where(labels[task] > -1, 1, 0).bool()

            threshold = self._optimal_threshold[task] if use_optimal_threshold else 0.5
            preds[task] = torch.where(outs[task][:, 1] >= threshold, 1, 0).cpu()

        # Compute the confusion matrix and the loss for each task.
        for task in main_classification_tasks:
            task_labels = labels[task][masks[task]]
            conf_mat[task] = confusion_matrix(task_labels.numpy(),
                                              preds[task][masks[task]].numpy())
            if get_loss:
                if self._loss == "bce":
                    target = to_one_hot(task_labels, self._num_classes[task], self._device)
                else:
                    target = task_labels.to(self._device)
                losses.append(self.__losses[task](outs[task][masks[task]], target))

            # If get_loss is False, then we compute the auc score.
            else:
                task_outs = outs[task][masks[task].to(self._device)]
                fpr, tpr, _ = roc_curve(y_true=task_labels.numpy(),
                                        y_score=task_outs[:, 1].cpu().numpy())
                auc_score[task] = auc(fpr, tpr)

        self.__compute_conditional_prob(auc_score=auc_score,
                                        conf_mat=conf_mat,
                                        get_auc=not get_loss,
                                        labels=labels,
                                        masks=masks,
                                        preds=preds,
                                        outs=outs)

        total_loss = torch.sum(torch.stack(losses)) if get_loss else None

        return (conf_mat, total_loss) if get_loss else (conf_mat, auc_score)

    def _prepare_optim_and_schedulers(self,
                                      eta_min: float,
                                      eps: float,
                                      learning_rate: float,
                                      l2_coeff: float,
                                      mom: float,
                                      optim: str,
                                      t_0: int,
                                      **kwargs) -> Tuple[List[torch.optim.Optimizer],
                                                         List[CosineAnnealingWarmRestarts]]:
        """
        Initalize all optimizers and schedulers object.

        :param eta_min: Minimum value of the learning rate.
        :param eps: The epsilon parameter of the Adam Optimizer.
        :param learning_rate: Start learning rate of the optimizer.
        :param l2_coeff: L2 regularization coefficient.
        :param mom: The momentum parameter of the SGD Optimizer.
        :param optim: A string that indicate the optimizer that will be used for training.
        :param shared_eta_min: Ending learning rate value of the shared unit. if equal to 0, then shared_eta_min
                                will be equal to learning_rate*100. Only used when shared_net is True.
        :param shared_lr: Learning rate of the shared/branching unit. if equal to 0, then shared_lr will be
                          equal to learning_rate*100. Only used when shared_net is True.
        :param shared_l2: L2 coefficient of the sharing/branching unit. Only used when shared_net is True.
        :param t_0: Number of epoch before the first restart.
        :return: A list of optimizers and a list of learning rate schedulers
        """
        assert optim.lower() in ["adam", "sgd", "novograd"]
        eta_list = [eta_min]
        lr_list = [learning_rate]
        l2_list = [l2_coeff]
        parameters, loss_parameters = self.model.get_weights()

        shared_eta_min = kwargs["shared_eta_min"]
        shared_lr = kwargs["shared_lr"]
        shared_l2 = kwargs["shared_l2"]

        if self._model_type is not ModelType.STANDARD:
            eta_list.append(eta_min * DEFAULT_SHARED_LR_SCALE if shared_eta_min == 0 else shared_eta_min)
            lr_list.append(learning_rate * DEFAULT_SHARED_LR_SCALE if shared_lr == 0 else shared_lr)
            l2_list.append(shared_l2)

        if loss_parameters is not None:
            eta_list.append(eta_min)
            lr_list.append(learning_rate)
            l2_list.append(0)
            parameters.append(loss_parameters)

        return self._build_optim_and_schdulers(eta_list=eta_list,
                                               eps=eps,
                                               lr_list=lr_list,
                                               l2_list=l2_list,
                                               mom=mom,
                                               optim=optim,
                                               parameters_list=parameters,
                                               t_0=t_0)

    @torch.no_grad()
    def __compute_conditional_prob(self,
                                   auc_score: Dict[str, float],
                                   conf_mat: Dict[str, np.array],
                                   get_auc: bool,
                                   labels: Dict[str, torch.Tensor],
                                   masks: Dict[str, torch.Tensor],
                                   preds: Dict[str, torch.Tensor],
                                   outs: Dict[str, torch.Tensor]) -> None:
        """
        compute the conditional probability scores.

        :param auc_score: A dictionary of float that contains the auc per task.
        :param conf_mat: A dictionary of numpy array that contains the confusion matrix per task.
        :param get_auc: If true, the auc will also be compute.
        :param labels: A dictionary of torch.Tensor that contains the labels for each task.
        :param masks: A dictionary of torch.Tensor that indicate which data has a label for each task.
        :param preds: A dictionary of torch.Tensor that contains the predictions of the model per task.
        :param outs: A dictionary of torch.Tensor that contains the outputs of the model per task.
        """
        for task1, task2 in self.__cond_prob:
            task_name = task1 + "|" + task2

            mask = torch.where(
                torch.logical_and(
                    masks[task1],
                    torch.BoolTensor(preds[task2] == labels[task2])),
                1, 0
            ).bool()

            cond_pred = preds[task1][mask]
            if len(cond_pred) > 0:
                cond_conf = confusion_matrix(labels[task1][mask].numpy(),
                                             cond_pred.numpy(),
                                             labels=range(self._num_classes[task_name]))

                if get_auc:
                    task_outs = outs[task1][mask.to(self._device)]
                    fpr, tpr, _ = roc_curve(y_true=labels[task1][mask].numpy(),
                                            y_score=task_outs[:, 1].cpu().numpy())
                    auc_score[task_name] = auc(fpr, tpr)
            else:
                cond_conf = np.array([[float("nan"), float("nan")],
                                      [float("nan"), float("nan")]])
            conf_mat[task_name] = cond_conf
