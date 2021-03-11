"""
    @file:              Trainer.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       Contain the mother class Trainer from which the SingleTaskTrainer and MultiTaskTrainer will
                        inherit.
"""

from abc import ABC, abstractmethod
from Data_manager.DataManager import RenalDataset
from Model.NeuralNet import NeuralNet
from Model.ResNet_2D import ResNet2D
from monai.optimizers import Novograd
import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Sequence, Tuple, Union


DEFAULT_SHARED_LR_SCALE = 100  # Default rate between shared_lr and lr if shared_lr == 0
MINIMUM_ACCURACY = 0.5  # Minimum threshold of the accuracy used in the early stopping criterion
VALIDATION_BATCH_SIZE = 2  # Batch size used to create the validation dataloader and the test dataloader


class Trainer(ABC):
    """
    The trainer class define an object that will be used to train and evaluate a given model. It handle the 
    mixed precision training, the mixup process and more.

    ...
    Attributes
    ----------
    _classes_weights : str
        The configuration of weights that will be applied on the loss during the training.
        Flat: All classes have the same weight during the training.
        Balanced: The weights are inversionaly proportional to the number of data of each classes in the training set.
        (Default="balanced")
    __early_stopping : bool
        If true, the training will be stop after the third of the training if the model did not achieve at least 50%
        validation accuracy for at least one epoch.
    _loss : str
        The name of the loss that will be used during the training.
    _mixed_precision : bool
        If true, mixed_precision will be used during training and inferance.
    model : NeuralNet
        The neural network to train and evaluate.
    __num_work : int
        Number of parallel process used for the preprocessing of the data. If 0, the main process will 
        be used for the data augmentation.
    __pin_memory : bool
        The pin_memory option of the DataLoader. If true, the data tensor will copied into the CUDA pinned memory.
    __save_path : string
        Indicate where the weights of the network and the result will be saved.
    __shared_net: bool
        If true, the model to train will be a SharedNet. In this we need to optimizer, one for the subnets and
        one for the sharing units and the Uncertainty loss parameters.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    __tol : float
        Represent the tolerance factor. If the loss of a given epoch is below (1 - __tol) * best_loss, 
        then this is consider as an improvement.
    _track_mode : str
        Control the information that are registred by tensorboard. Options: all, low, none.
    _weights : dict
        A dictionnary of np.array that represent the balanced weights that would be used to adjust the loss function
        if __classes_weights == 'balanced'
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
                 classes_weights: str = "balanced",
                 early_stopping: bool = False,
                 loss: str = "ce",
                 mixed_precision: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 save_path: str = "",
                 shared_net: bool = False,
                 tol: float = 0.01,
                 track_mode: str = "all"):
        """
        The constructor of the trainer class. 

        :param classes_weights: The configuration of weights that will be applied on the loss during the training.
                                Flat: All classes have the same weight during the training.
                                Balanced: The weights are inversionaly proportional to the number of data of each
                                          classes in the training set.
                                (Default="balanced")
        :param early_stopping: If true, the training will be stop after the third of the training if the model did
                               not achieve at least 50% validation accuracy for at least one epoch. (Default=False)
        :param loss: The loss that will be use during mixup epoch. (Default="ce")
        :param mixed_precision: If true, mixed_precision will be used during training and inferance. (Default=False)
        :param num_workers: Number of parallel process used for the preprocessing of the data. If 0,
                            the main process will be used for the data augmentation. (Default=0)
        :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will
                           copied into the CUDA pinned memory. (Default=False)
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param shared_net: If true, the model to train will be a SharedNet. In this we need to optimizer, one for the
                           subnets and one for the sharing units and the Uncertainty loss parameters. (Default=False)
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param track_mode: Control information that are registred by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default=all)
        """
        super().__init__()

        assert loss.lower() in ["ce", "bce", "marg", "focal"], \
            "You can only choose one of the following loss ['ce', 'bce', 'marg']"
        assert track_mode.lower() in ["all", "low", "none"], \
            "Track mode should be one of those options: 'all', 'low' or 'none'"
        assert classes_weights.lower() in ["flat", "balanced"], \
            "classes_weights should be one of those options: 'Flat', 'Balanced'"

        self._classes_weights = classes_weights
        self.__cumulate_counter = 0
        self._device = None
        self.__early_stopping = early_stopping
        self.__experiment = None
        self._loss = loss.lower()
        self._mixed_precision = mixed_precision
        self.model = None
        self.__num_cumulated_batch = 1
        self.__num_work = num_workers
        self.__pin_memory = pin_memory
        self.__save_path = save_path
        self.__shared_net = shared_net
        self._soft = nn.Softmax(dim=-1)
        self.__tol = tol
        self._track_mode = track_mode.lower()
        self._weights = {}
        self._writer = None

    def fit(self,
            model: Union[NeuralNet, ResNet2D],
            trainset: RenalDataset,
            validset: RenalDataset,
            batch_size: int = 32,
            device: str = "cuda:0",
            eps: float = 1e-4,
            eta_min: float = 1e-4,
            gamma: float = 2.,
            grad_clip: float = 0,
            l2: float = 1e-4,
            learning_rate: float = 1e-3,
            mode: str = "standard",
            mom: float = 0.9,
            num_cumulated_batch: int = 1,
            num_epoch: int = 200,
            optim: str = "Adam",
            retrain: bool = False,
            shared_eta_min: float = 0,
            shared_lr: float = 0,
            t_0: int = 200,
            transfer_path: str = None,
            verbose: bool = True,
            warm_up_epoch: int = 0) -> None:
        """
        Train the model on the given dataset

        :param model: The model to train.
        :param trainset: The dataset that will be used to train the model.
        :param validset: The dataset that will be used to mesure the model performan
        :param batch_size: The batch size that will be used during the training. (Default=32)ce.
        :param device: The device on which the training will be done. (Default="cuda:0", first GPU)
        :param eps: The epsilon parameter of the Adam Optimizer. (Default=1e-4)
        :param eta_min: Minimum value of the learning rate. (Default=1e-4)
        :param gamma: Gamma parameter of the focal loss. (Default=2.0)
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient. (Default=0)
        :param l2: L2 regularization coefficient. (Default=1e-4)
        :param learning_rate: Start learning rate of the optimizer. (Default=1e-3)
        :param mode: The training type: Option: Standard training (No mixup) (Default)
                                                Mixup (Manifold mixup)
        :param mom: The momentum parameter of the SGD Optimizer. (Default=0.9)
        :param num_cumulated_batch: The number of batch that will be cumulated before updating the weight of the model.
        :param num_epoch: Maximum number of epoch during the training. (Default=200)
        :param optim: A string that indicate the optimizer that will be used for training. (Default='Adam')
        :param retrain: If false, the weights of the model will initialize. (Default=False)
        :param shared_eta_min: Ending learning rate value of the shared unit. if equal to 0, then shared_eta_min
                                will be equal to learning_rate*100. Only used when shared_net is True.
        :param shared_lr: Learning rate of the shared unit. if equal to 0, then shared_lr will be
                          equal to learning_rate*100. Only used when shared_net is True.
        :param t_0: Number of epoch before the first restart. (Default=200)
        :param transfer_path: If not None, initialize the model with transfer learning by loading the weight of
                              the model at the given path.
        :param verbose: If true, show the progress of the training. (Default=True)
        :param warm_up_epoch: Number of iteration before activating mixup. (Default=True)
        """
        # Indicator for early stopping
        best_accuracy = 0
        best_epoch = -1
        current_mode = "Standard"
        last_saved_loss = float("inf")
        early_stopping_epoch = int(num_epoch / 3) - 1
        self.__num_cumulated_batch = num_cumulated_batch
        self.__cumulate_counter = 0

        # Tensorboard writer
        self._writer = SummaryWriter()

        # Initialization of the model.
        self._device = device
        self.model = model.to(device)
        self.model.set_mixup(batch_size) if mode.lower() == "mixup" else None
        self.__init_weights(trainset.labels_bincount())
        self._init_loss(gamma=gamma)

        start_epoch = 0
        if retrain:
            start_epoch, last_saved_loss, best_accuracy = self.model.restore(self.__save_path)
        elif transfer_path is not None:
            _, _, _ = self.model.restore(transfer_path)

        # Initialization of the dataloader
        train_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  pin_memory=self.__pin_memory,
                                  num_workers=self.__num_work,
                                  shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(validset,
                                  batch_size=VALIDATION_BATCH_SIZE,
                                  pin_memory=self.__pin_memory,
                                  num_workers=self.__num_work,
                                  shuffle=False,
                                  drop_last=True)

        # Initialization of the optimizer and the scheduler
        assert optim.lower() in ["adam", "sgd", "novograd"]
        lr_list = [learning_rate]
        eta_list = [eta_min]
        parameters = [self.model.parameters()]

        if self.__shared_net:
            lr_list.append(learning_rate * DEFAULT_SHARED_LR_SCALE if shared_lr == 0 else shared_lr)
            eta_list.append(eta_min * DEFAULT_SHARED_LR_SCALE if shared_eta_min == 0 else shared_eta_min)

            parameters = [self.model.nets.parameters(),
                          list(self.model.sharing_units_dict.parameters()) +
                          list(self.model.uncertainty_loss.parameters())
                          ]
        optimizers = []

        if optim.lower() == "adam":
            for lr, param in zip(lr_list, parameters):
                optimizers.append(
                    torch.optim.Adam(param,
                                     lr=lr,
                                     weight_decay=l2,
                                     eps=eps)
                )
        elif optim.lower() == "sgd":
            for lr, param in zip(lr_list, parameters):
                optimizers.append(
                    torch.optim.SGD(param,
                                    lr=lr,
                                    weight_decay=l2,
                                    momentum=mom,
                                    nesterov=True)
                )
        else:
            for lr, param in zip(lr_list, parameters):
                optimizers.append(
                    Novograd(param,
                             lr=lr,
                             weight_decay=l2,
                             eps=eps)
                )

        n_iters = len(train_loader)
        schedulers = []
        for optimizer, eta in zip(optimizers, eta_list):
            schedulers.append(
                CosineAnnealingWarmRestarts(optimizer,
                                            T_0=t_0*n_iters,
                                            T_mult=1,
                                            eta_min=eta)
            )
        for scheduler in schedulers:
            scheduler.step(start_epoch*n_iters)

        # Go in training mode to activate mixup module
        self.model.train()

        with tqdm(total=num_epoch, initial=start_epoch, leave=True) as t:
            for epoch in range(start_epoch, num_epoch):

                current_mode = mode if warm_up_epoch <= epoch else current_mode

                # We make a training epoch
                if current_mode == "Mixup":
                    _ = self._mixup_epoch(train_loader, optimizers, schedulers, grad_clip, epoch)
                else:
                    _ = self._standard_epoch(train_loader, optimizers, schedulers, grad_clip, epoch)

                self.model.eval()

                val_acc, val_loss = self._validation_step(dt_loader=valid_loader, 
                                                          epoch=epoch)

                train_acc, train_loss = self._validation_step(dt_loader=train_loader, 
                                                              epoch=epoch,
                                                              dataset_name="Training")

                self._writer.add_scalars('Accuracy', 
                                         {'Training': train_acc,
                                          'Validation': val_acc}, 
                                         epoch)
                self._writer.add_scalars('Validation/Loss',
                                         {'loss': val_loss},
                                         epoch)
                self.model.train()

                # ------------------------------------------------------------------------------------------
                #                                   EARLY STOPPING PART
                # ------------------------------------------------------------------------------------------

                if (val_loss < last_saved_loss and val_acc >= best_accuracy) or \
                        (val_loss < last_saved_loss*(1+self.__tol) and val_acc > best_accuracy):
                    self.__save_checkpoint(epoch, val_loss, val_acc)
                    best_accuracy = val_acc
                    last_saved_loss = val_loss
                    best_epoch = epoch

                if verbose:
                    t.postfix = "train loss: {:.4f}, train acc {:.2f}%, val loss: {:.4f}, val acc: {:.2f}%, " \
                                "best acc: {:.2f}%, best epoch: {}, epoch type: {}".format(
                                    train_loss, train_acc * 100, val_loss, val_acc * 100, best_accuracy * 100,
                                    best_epoch + 1, current_mode
                                )
                t.update()

                if self.__early_stopping and epoch == early_stopping_epoch and best_accuracy < MINIMUM_ACCURACY:
                    break

        self._writer.close()
        self.model.restore(self.__save_path)

    def _update_model(self,
                      grad_clip: float,
                      loss: torch.FloatTensor,
                      optimizers: Sequence[Union[torch.optim.Optimizer, Novograd]],
                      scaler: Union[GradScaler, None],
                      schedulers: Sequence[CosineAnnealingWarmRestarts]) -> None:
        """
        Scale the loss if self._mixed_precision is True and update the weights of the model.

        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :param loss: The loss of the current epoch.
        :param optimizers: The torch optimizer that will used to train the model.
        :param scaler: The gradient scaler that will be used to scale the loss if self._mixed_precision is True.
        :param schedulers: The learning rate scheduler that will be used at each iteration.
        """
        self.__cumulate_counter += 1

        # Mixed precision enabled
        if self._mixed_precision:
            scaler.scale(loss).backward()

            if self.__cumulate_counter % self.__num_cumulated_batch == 0:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()

        # Mixed precision disabled
        else:
            loss.backward()

            if self.__cumulate_counter % self.__num_cumulated_batch == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                for optimizer in optimizers:
                    optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

    def __init_weights(self, labels_bincounts: dict) -> None:
        """
        Compute the balanced weight according to a given distribution of data.

        :param labels_bincounts: A dictionnary of numpy.array that give the distribution of data per class per task.
                                 The tasks name are used has key in this dictionnary.
        """

        for key, value in labels_bincounts.items():
            self._weights[key] = np.sum(value) / (2 * value)

    @abstractmethod
    def _get_conf_matrix(self,
                         dt_loader: DataLoader,
                         get_loss: bool = False) -> Union[Tuple[Sequence[np.array], float],
                                                          Tuple[Sequence[np.array], Sequence[float]],
                                                          Tuple[np.array, float]]:
        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data.
        :param get_loss: Return also the loss if True.
        :return: The confusion matrix for each classes and the average loss if get_loss == True.
        """
        raise NotImplementedError("Must override _get_conf_matrix.")

    @abstractmethod
    def _init_loss(self, gamma: float) -> None:
        """
        Initialize the loss function by sending the classes weights on the appropriate device.

        :param gamma: Gamma parameter of the focal loss.
        """
        raise NotImplementedError("Must override _init_loss")

    @abstractmethod
    def _mixup_criterion(self,
                         pred: torch.Tensor,
                         target: Variable,
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
        raise NotImplementedError("Must override _mixup_criterion.")

    @abstractmethod
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
        raise NotImplementedError("Must override _mixup_epoch.")

    def __save_checkpoint(self,
                          epoch: int,
                          loss: float,
                          accuracy: float) -> None:
        """
        Save the model and his at a the current state if the self.path is not None.

        :param epoch: Current epoch of the training
        :param loss: Current loss of the training
        :param accuracy: Current validation accuracy
        """

        if self.__save_path is not None:
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "loss": loss,
                        "accuracy": accuracy},
                       self.__save_path)

    def score(self,
              testset: RenalDataset) -> Union[Tuple[Sequence[np.array], float],
                                              Tuple[Sequence[np.array], Sequence[float]],
                                              Tuple[np.array, float]]:
        """
        Compute the accuracy of the model on a given test dataset

        :param testset: A torch dataset which contain our test data points and labels
        :return: The accuracy of the model.
        """
        test_loader = torch.utils.data.DataLoader(testset,
                                                  VALIDATION_BATCH_SIZE,
                                                  shuffle=False)

        self.model.eval()

        return self._get_conf_matrix(dt_loader=test_loader)

    @abstractmethod
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
        raise NotImplementedError("Must override _standard_epoch.")

    @abstractmethod
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
        raise NotImplementedError("Must override _validation_step.")
