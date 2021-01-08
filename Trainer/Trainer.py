from abc import ABC, abstractmethod
from Data_manager.DataManager import RenalDataset, get_dataloader
from Model.NeuralNet import NeuralNet
from monai.optimizers import Novograd
import numpy as np
from Trainer.Utils import init_weights
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Sequence, Tuple, Union


class Trainer(ABC):
    """
    The trainer class define an object that will be used to train and evaluate a given model. It handle the 
    mixed precision training, the mixup process and more.

    ...
    Attributes
    ----------
    _loss : str
        The name of the loss that will be used during the training.
    model : NeuralNet
        The neural network to train and evaluate.
    __num_work : int
        Number of parallel process used for the preprocessing of the data. If 0, the main process will 
        be used for the data augmentation.
    __pin_memory : bool
        The pin_memory option of the DataLoader. If true, the data tensor will copied into the CUDA pinned memory.
    __save_path : string
        Indicate where the weights of the network and the result will be saved.
    _soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    __tol : float
        Represent the tolerance factor. If the loss of a given epoch is below (1 - __tol) * best_loss, 
        then this is consider as an improvement.
    _track_mode : str
        Control the information that are registred by tensorboard. Options: all, low, none (Default: all).
    __valid_split : float
        Percentage of the trainset that will be used to create the validation set.
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
                 valid_split: float = 0.2,
                 tol: float = 0.01,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 save_path: str = "",
                 track_mode: str = "all"):
        """
        The constructor of the trainer class. 

        :param loss: The loss that will be use during mixup epoch. (Default="bce")
        :param valid_split: Percentage of the trainset that will be used to create the validation set.
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will 
                           copied into the CUDA pinned memory. (Default=False)
        :param num_workers: Number of parallel process used for the preprocessing of the data. If 0, 
                            the main process will be used for the data augmentation. (Default: 0)
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param track_mode: Control information that are registred by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default: all)
        """
        super().__init__()

        assert loss.lower() in ["ce", "bce", "marg", "focal"], \
            "You can only choose one of the following loss ['ce', 'bce', 'marg']"
        assert track_mode.lower() in ["all", "low", "none"], \
            "Track mode should be one of those options: 'all', 'low' or 'none'"
        self.__tol = tol
        self.__valid_split = valid_split
        self.__pin_memory = pin_memory
        self.__num_work = num_workers
        self.__save_path = save_path
        self._track_mode = track_mode.lower()
        self.model = None
        self._device = None
        self._writer = None
        self._loss = loss.lower()
        self._soft = nn.Softmax(dim=-1)

    def fit(self,
            model: NeuralNet,
            trainset: RenalDataset,
            validset: RenalDataset,
            num_epoch: int = 200, 
            batch_size: int = 32,
            gamma: float = 2.,
            learning_rate: float = 1e-3, 
            eps: float = 1e-4, 
            l2: float = 1e-4, 
            t_0: int = 200,
            eta_min: float = 1e-4, 
            grad_clip: float = 0, 
            mode: str = "standard", 
            warm_up_epoch: int = 0,
            optim: str = "Adam", 
            retrain: bool = False, 
            device: str = "cuda:0", 
            verbose: bool = True) -> None:
        """
        Train the model on the given dataset

        :param model: The model to train.
        :param trainset: The dataset that will be used to train the model.
        :param validset: The dataset that will be used to mesure the model performance.
        :param num_epoch: Maximum number of epoch during the training. (Default=200)
        :param batch_size: The batch size that will be used during the training. (Default=150)
        :param gamma: Gamma parameter of the focal loss. (Default=2.0)
        :param learning_rate: Start learning rate of the Adam optimizer. (Default=0.1)
        :param eps: The epsilon parameter of the Adam Optimizer.
        :param l2: L2 regularization coefficient.
        :param t_0: Number of epoch before the first restart. (Default=200)
        :param eta_min: Minimum value of the learning rate. (Default=1e-4)
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient. (Default=0)
        :param mode: The training type: Option: Standard training (No mixup) (Default)
                                                Mixup (Manifold mixup)
        :param warm_up_epoch: Number of iteration before activating mixup. (Default=True)
        :param optim: A string that indicate the optimizer that will be used for training. (Default='Adam')
        :param retrain: If false, the weights of the model will initialize. (Default=False)
        :param device: The device on which the training will be done. (Default="cuda:0", first GPU)
        :param verbose: If true, show the progress of the training. (Default=True)
        """
        # Indicator for early stopping
        best_accuracy = 0
        best_epoch = -1
        current_mode = "Standard"
        last_saved_loss = float("inf")

        # Tensorboard writer
        self._writer = SummaryWriter()

        # Initialization of the model.
        self._device = device
        self.model = model.to(device)
        self.model.set_mixup(batch_size)
        self._init_loss(gamma=gamma)

        if retrain:
            start_epoch, last_saved_loss, best_accuracy = self.model.restore(self.save_path)
        else:
            self.model.apply(init_weights)
            start_epoch = 0

        # Initialization of the dataloader
        train_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  pin_memory=self.__pin_memory,
                                  num_workers=self.__num_work,
                                  shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(validset,
                                  batch_size=batch_size,
                                  pin_memory=self.__pin_memory,
                                  num_workers=self.__num_work,
                                  shuffle=False,
                                  drop_last=True)

        # Initialization of the optimizer and the scheduler
        assert optim.lower() in ["adam", "novograd"]
        if optim.lower() == "adam": 
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=l2,
                eps=eps
            )
        else:
            optimizer = Novograd(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=l2,
                eps=eps
            )

        n_iters = len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0*n_iters,
            T_mult=1,
            eta_min=eta_min
        )
        scheduler.step(start_epoch*n_iters)

        # Go in training mode to activate mixup module
        self.model.train()

        with tqdm(total=num_epoch, initial=start_epoch) as t:
            for epoch in range(start_epoch, num_epoch):

                _grad_clip = 0 if epoch > num_epoch / 2 else grad_clip
                current_mode = mode if warm_up_epoch <= epoch else current_mode

                # We make a training epoch
                if current_mode == "Mixup":
                    training_loss = self._mixup_epoch(train_loader, optimizer, scheduler, _grad_clip, epoch)
                else:
                    training_loss = self._standard_epoch(train_loader, optimizer, scheduler, _grad_clip, epoch)

                self.model.eval()

                val_acc, val_loss = self._validation_step(dt_loader=valid_loader, 
                                                          epoch=epoch)

                train_acc, train_loss = self._validation_step(dt_loader=train_loader, 
                                                              epoch=epoch,
                                                              dataset_name="training")

                self._writer.add_scalars('Accuracy', 
                                         {'Training': train_acc,
                                          'Validation': val_acc}, 
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

        self._writer.close()
        self.model.restore(self.__save_path)

    @abstractmethod
    def _init_loss(self, gamma: float) -> None:
        """
        Initialize the loss function by sending the classes weights on the appropriate device.

        :param gamma: Gamma parameter of the focal loss.
        """
        raise NotImplementedError("Must override _init_loss")
    
    @abstractmethod
    def _standard_epoch(self,
                        train_loader: DataLoader,
                        optimizer: Union[torch.optim.Adam, Novograd],
                        scheduler: CosineAnnealingWarmRestarts, 
                        grad_clip: float,
                        epoch: int) -> float:
        """
        Make a standard training epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizer: The torch optimizer that will used to train the model.
        :param scheduler: The learning rate scheduler that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss.
        """
        raise NotImplementedError("Must override _standard_epoch.")

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
                     optimizer: Union[torch.optim.Adam, Novograd],
                     scheduler: CosineAnnealingWarmRestarts,
                     grad_clip: float,
                     epoch: int) -> float:
        """
        Make a manifold mixup epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizer: The torch optimizer that will used to train the model.
        :param scheduler: The learning rate scheduler that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss.
        """
        raise NotImplementedError("Must override _mixup_epoch.")

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

    @abstractmethod
    def _get_conf_matrix(self,
                         dt_loader: DataLoader,
                         get_loss: bool = False) -> Union[Tuple[Sequence[np.array], float],
                                                          Tuple[np.array, float],
                                                          Sequence[np.array],
                                                          np.array]:
        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data.
        :param get_loss: Return also the loss if True.
        :return: The confusion matrix for each classes and the average loss if get_loss == True.
        """
        raise NotImplementedError("Must override _get_conf_matrix.")
    
    def score(self,
              testset: RenalDataset,
              batch_size: int = 150) -> Union[Tuple[Sequence[np.array], float],
                                              Tuple[np.array, float],
                                              Sequence[np.array],
                                              np.array]:
        """
        Compute the accuracy of the model on a given test dataset

        :param testset: A torch dataset which contain our test data points and labels
        :param batch_size: The batch_size that will be use to create the data loader. (Default=150)
        :return: The accuracy of the model.
        """
        test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)

        self.model.eval()

        return self._get_conf_matrix(dt_loader=test_loader)

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
