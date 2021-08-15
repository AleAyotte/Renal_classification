"""
    @file:              Trainer.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 08/2021

    @Description:       Contain the mother class Trainer from which the SingleTaskTrainer and MultiTaskTrainer will
                        inherit.
"""

from abc import ABC, abstractmethod
import csv
from Data_manager.RenalDataset import RenalDataset
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
from Trainer.Utils import find_optimal_cutoff
from typing import Dict, List, Sequence, Tuple, Union


DEFAULT_SHARED_LR_SCALE = 10  # Default rate between shared_lr and lr if shared_lr == 0
MINIMUM_ACCURACY = 0.5  # Minimum threshold of the accuracy used in the early stopping criterion
VALIDATION_BATCH_SIZE = 2  # Batch size used to create the validation dataloader and the test dataloader
TEST_BATCH_SIZE = 1  # Batch size used to create the validation dataloader and the test dataloader


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
    _classification_tasks : List[str]
        A list that contain the name of every classification task on which the model will be train.
    __early_stopping : bool
        If true, the training will be stop after the third of the training if the model did not achieve at least 50%
        validation accuracy for at least one epoch.
    _loss : str
        The name of the loss that will be used during the training.
    _mixed_precision : bool
        If true, mixed_precision will be used during training and inferance.
    model : NeuralNet
        The neural network to train and evaluate.
    _num_classes: dict
        A dictionnary that indicate the number of classes for each task. For a regression task, the number of classes
        should be equal to one.
    __num_work : int
        Number of parallel process used for the preprocessing of the data. If 0, the main process will 
        be used for the data augmentation.
    __pin_memory : bool
        The pin_memory option of the DataLoader. If true, the data tensor will copied into the CUDA pinned memory.
    _regression_tasks : List[str]
        A list that contain the name of every regression task on which the model will be train.
    __save_path : string
        Indicate where the weights of the network and the result will be saved.
    __shared_net: bool
        If true, the model to train will be a SharedNet. In this we need to optimizer, one for the subnets and
        one for the sharing units and the Uncertainty loss parameters.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    _tasks : list
        A list of string that contain the name of every task for which the model will be train.
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
                 tasks: Sequence[str],
                 num_classes: Dict[str, int],
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

        :param tasks: A list of tasks on which the model will be train.
        :param num_classes: A dictionnary that indicate the number of classes of each. For regression task, the number
                            of classes should be 1.
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
        assert list(tasks).sort() == list(num_classes.keys()).sort(), \
            "The number of classes should be given for each task. " \
            "For a regression task num_classes should be equal to 1."

        self._classes_weights = classes_weights
        self._classification_tasks = [task for task in tasks if num_classes[task] > 1]
        self.__cumulate_counter = 0
        self._device = None
        self.__early_stopping = early_stopping
        self.__experiment = None
        self._loss = loss.lower()
        self._mixed_precision = mixed_precision
        self.model = None
        self._num_classes = num_classes
        self.__num_cumulated_batch = 1
        self.__num_work = num_workers
        self._optimal_threshold = {}
        self.__pin_memory = pin_memory
        self._regression_tasks = [task for task in tasks if num_classes[task] == 1]
        self.__save_path = save_path
        self.__shared_net = shared_net
        self._soft = nn.Softmax(dim=-1)
        self._tasks = tasks
        self.__tol = tol
        self._track_mode = track_mode.lower()
        self._weights = {}
        self._writer = None

        for task in self._classification_tasks:
            self._optimal_threshold[task] = 0.5

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
            learning_rate: float = 1e-3,
            l2: float = 1e-4,
            mode: str = "standard",
            mom: float = 0.9,
            num_cumulated_batch: int = 1,
            num_epoch: int = 200,
            optim: str = "Adam",
            retrain: bool = False,
            shared_eta_min: float = 0,
            shared_lr: float = 0,
            shared_l2: float = 0,
            t_0: int = 0,
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
        :param learning_rate: Start learning rate of the optimizer. (Default=1e-3)
        :param l2: L2 regularization coefficient. (Default=1e-4)
        :param mode: The training type: Option: Standard training (No mixup) (Default)
                                                Mixup (Manifold mixup)
        :param mom: The momentum parameter of the SGD Optimizer. (Default=0.9)
        :param num_cumulated_batch: The number of batch that will be cumulated before updating the weight of the model.
        :param num_epoch: Maximum number of epoch during the training. (Default=200)
        :param optim: A string that indicate the optimizer that will be used for training. (Default='Adam')
        :param retrain: If false, the weights of the model will initialize. (Default=False)
        :param shared_eta_min: Ending learning rate value of the sharing unit. if equal to 0, then shared_eta_min
                                will be equal to learning_rate*100. Only used when shared_net is True.
        :param shared_lr: Learning rate of the sharing unit. if equal to 0, then shared_lr will be
                          equal to learning_rate*100. Only used when shared_net is True.
        :param shared_l2: L2 coefficient of the sharing unit. Only used when shared_net is True.
        :param t_0: Number of epoch before the first restart. If equal to 0, then t_0 will be equal to num_epoch.
                    (Default=0)
        :param transfer_path: If not None, initialize the model with transfer learning by loading the weight of
                              the model at the given path.
        :param verbose: If true, show the progress of the training. (Default=True)
        :param warm_up_epoch: Number of iteration before activating mixup. (Default=True)
        """
        # Indicator for early stopping
        best_accuracy = 0
        best_epoch = 0
        current_mode = "Standard"
        early_stopping_epoch = int(num_epoch / 3) - 1
        last_saved_loss = float("inf")
        self.__cumulate_counter = 0
        self.__num_cumulated_batch = num_cumulated_batch
        self._writer = SummaryWriter()

        # Initialization of the model and the loss.
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
        t_0 = t_0 if t_0 <= 0 else num_epoch
        optimizers, schedulers = self.__prepare_optim_and_schedulers(eta_min=eta_min,
                                                                     eps=eps,
                                                                     learning_rate=learning_rate,
                                                                     l2_coeff=l2,
                                                                     mom=mom,
                                                                     optim=optim,
                                                                     shared_eta_min=shared_eta_min,
                                                                     shared_lr=shared_lr,
                                                                     shared_l2=shared_l2,
                                                                     t_0=t_0*len(train_loader))
        for scheduler in schedulers:
            scheduler.step(start_epoch*len(train_loader))

        with tqdm(total=num_epoch, initial=start_epoch, leave=True) as t:
            for epoch in range(start_epoch, num_epoch):
                current_mode = mode if warm_up_epoch <= epoch else current_mode

                # Training epoch
                self.model.train()
                if current_mode == "Mixup":
                    _ = self._mixup_epoch(epoch, grad_clip, optimizers, schedulers, train_loader)
                else:
                    _ = self._standard_epoch(epoch, grad_clip, optimizers, schedulers, train_loader)

                self.model.eval()
                val_acc, val_loss = self._validation_step(dt_loader=valid_loader, 
                                                          epoch=epoch)
                train_acc, train_loss = self._validation_step(dt_loader=train_loader, 
                                                              epoch=epoch,
                                                              dataset_name="Training")

                self._writer.add_scalars('Accuracy', 
                                         {'Training': train_acc, 'Validation': val_acc},
                                         epoch)
                self._writer.add_scalars('Validation/Loss',
                                         {'loss': val_loss},
                                         epoch)

                # -----------------------------------------------------------------
                #                         EARLY STOPPING
                # -----------------------------------------------------------------
                if (val_loss < last_saved_loss and val_acc >= best_accuracy) or \
                        (val_loss < last_saved_loss*(1+self.__tol) and val_acc > best_accuracy):
                    self.__save_checkpoint(epoch, val_loss, val_acc)
                    best_accuracy = val_acc
                    last_saved_loss = val_loss
                    best_epoch = epoch + 1

                if verbose:
                    t.postfix = f"{train_loss= :.4f}, {train_acc= :.2%}, {val_loss= :.4f}, {val_acc= :.2%}, " \
                                f"{best_accuracy = :.2%}, {best_epoch= }"
                t.update()

                if self.__early_stopping and epoch == early_stopping_epoch and best_accuracy < MINIMUM_ACCURACY:
                    break

        self._writer.close()
        self.model.restore(self.__save_path)

        # Compute the optimal threshold
        self.__get_threshold(train_loader)

    def score(self,
              testset: RenalDataset,
              save_path: str = "") -> Union[Tuple[Sequence[np.array], float],
                                            Tuple[Sequence[np.array], Sequence[float]],
                                            Tuple[np.array, float]]:
        """
        Compute the accuracy of the model on a given test dataset.

        :param testset: A torch dataset which contain our test data points and labels.
        :param save_path: The filepath of the csv that will be used to save the prediction.
        :return: The accuracy of the model.
        """
        test_loader = torch.utils.data.DataLoader(testset,
                                                  TEST_BATCH_SIZE,
                                                  shuffle=False)

        self.model.eval()

        return self._get_conf_matrix(dt_loader=test_loader, save_path=save_path, use_optimal_threshold=True)

    def _predict(self,
                 dt_loader: DataLoader) -> Tuple[Dict[str, torch.Tensor],
                                                 Dict[str, torch.Tensor]]:
        """
        Take a data loader and compute the prediction for every data in the dataloader.

        :param dt_loader: A torch.Dataloader that use a RenalDataset object.
        :return: A dictionnary that contain a list of prediction per task and a dictionnary that contain a
                 list of labels per task.
        """

        outs = {}
        labels = {}
        # Two dictionnary that contain a torch.Tensor associated to each task (keys)
        for task in self._tasks:
            labels[task] = torch.empty(0).long()
            outs[task] = torch.empty(0, self._num_classes[task]).to(self._device)

        for data in dt_loader:
            images, label = data["sample"].to(self._device), data["labels"]
            features = None

            if "features" in list(data.keys()):
                features = data["features"].to(self._device)

            with torch.no_grad():
                out = self.model(images) if features is None else self.model(images, features)

                for task in self._tasks:
                    labels[task] = torch.cat([labels[task], label[task]])
                    outs[task] = torch.cat([outs[task],
                                            out if len(self._tasks) == 1 else out[task]])

        return outs, labels

    def _save_prediction(self,
                         outs: Dict[str, torch.Tensor],
                         labels: dict,
                         patient_id: Sequence[str],
                         save_path: str) -> None:
        """
        Save the prediction made by the model on a given dataset.

        :param outs: A dictionnary of torch.tensor that represent the output of the model for each task.
        :param labels: A dictionnary of torch.tensor that represent the labels and where the keys are the name
                       of each task.
        :param patient_id: A list that contain the patient id in the same order has the labels and the outputs.
        :param save_path: The filepath of the csv that will be used to save the prediction.
        """

        preds = {}
        with torch.no_grad():
            for task in self._tasks:
                preds[task] = self._soft(outs[task]) if self._num_classes[task] > 1 else outs[task]

        with open(save_path, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            first_row = ["patient_id"]
            first_row.extend([f"{task} {word}" for task in self._tasks for word in ["labels", "prediction"]])

            csv_writer.writerow(first_row)

            for i in range(len(labels[self._tasks[0]])):
                row = [patient_id[i]]
                for task in self._tasks:
                    label = labels[task][i].item()
                    pred = preds[task][i] if self._num_classes[task] == 1 else preds[task][i][-1]
                    row.extend([label, pred.cpu().item()])

                csv_writer.writerow(row)

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

        scaler.scale(loss).backward() if self._mixed_precision else loss.backward()

        if self.__cumulate_counter % self.__num_cumulated_batch == 0:
            # Mixed precision enabled
            if self._mixed_precision:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()

            # Mixed precision disabled
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                for optimizer in optimizers:
                    optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

    def __prepare_optim_and_schedulers(self,
                                       eta_min: float,
                                       eps: float,
                                       learning_rate: float,
                                       l2_coeff: float,
                                       mom: float,
                                       optim: str,
                                       shared_eta_min: float,
                                       shared_lr: float,
                                       shared_l2: float,
                                       t_0: int) -> Tuple[List[torch.optim.Optimizer],
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
        :param shared_lr: Learning rate of the shared unit. if equal to 0, then shared_lr will be
                          equal to learning_rate*100. Only used when shared_net is True.
        :param shared_l2: L2 coefficient of the sharing unit. Only used when shared_net is True.
        :param t_0: Number of epoch before the first restart.
        :return: A list of optimizers and a list of learning rate schedulers
        """
        assert optim.lower() in ["adam", "sgd", "novograd"]
        eta_list = [eta_min]
        lr_list = [learning_rate]
        l2_list = [l2_coeff]
        parameters = [self.model.parameters()]

        if self.__shared_net:
            eta_list.append(eta_min * DEFAULT_SHARED_LR_SCALE if shared_eta_min == 0 else shared_eta_min)
            lr_list.append(learning_rate * DEFAULT_SHARED_LR_SCALE if shared_lr == 0 else shared_lr)
            l2_list.append(shared_l2)
            parameters = [self.model.nets.parameters(),
                          list(self.model.sharing_units_dict.parameters())]

            # If the Uncertainty loss is used, we need to add these parameters.
            if type(self.model.loss_module).__name__ == "UncertaintyLoss":
                parameters[1] += list(self.model.loss_module.parameters())

        optimizers = []

        if optim.lower() == "adam":
            for lr, l2, param in zip(lr_list, l2_list, parameters):
                optimizers.append(
                    torch.optim.Adam(param,
                                     lr=lr,
                                     weight_decay=l2,
                                     eps=eps)
                )
        elif optim.lower() == "sgd":
            for lr, l2, param in zip(lr_list, l2_list, parameters):
                optimizers.append(
                    torch.optim.SGD(param,
                                    lr=lr,
                                    weight_decay=l2,
                                    momentum=mom,
                                    nesterov=True)
                )
        else:
            for lr, l2, param in zip(lr_list, l2_list, parameters):
                optimizers.append(
                    Novograd(param,
                             lr=lr,
                             weight_decay=l2,
                             eps=eps)
                )

        schedulers = []
        for optimizer, eta in zip(optimizers, eta_list):
            schedulers.append(
                CosineAnnealingWarmRestarts(optimizer,
                                            T_0=t_0,
                                            T_mult=1,
                                            eta_min=eta)
            )
            return optimizers, schedulers

    def __get_threshold(self, train_loader: DataLoader) -> None:
        """
        Get the optimal threhold point based on the training set for all classification task.

        :param train_loader: The train dataloader
        """
        self.model.eval()
        outs, labels = self._predict(train_loader)

        for task in self._classification_tasks:
            mask = torch.where(labels[task] > -1, 1, 0).bool()
            self._optimal_threshold[task] = find_optimal_cutoff(labels[task][mask].numpy(),
                                                                outs[task][mask][:, 1].cpu().numpy())

    def __init_weights(self, labels_bincounts: dict) -> None:
        """
        Compute the balanced weight according to a given distribution of data.

        :param labels_bincounts: A dictionnary of numpy.array that give the distribution of data per class per task.
                                 The tasks name are used has key in this dictionnary.
        """

        for key, value in labels_bincounts.items():
            self._weights[key] = np.sum(value) / (2 * value)

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

    @abstractmethod
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
        :param use_optimal_threshold
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
        raise NotImplementedError("Must override _mixup_epoch.")

    @abstractmethod
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
