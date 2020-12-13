from Data_manager.DataManager import RenalDataset, get_dataloader
from Model.NeuralNet import NeuralNet
from torch import nn
from typing import Sequence, Union


class Trainer:
    """
    The trainer class define an object that will be used to train and evaluate a given model. It handle the 
    mixed precision training, the mixup process and more.

    ...
    Attributes
    ----------
    __bce_loss : torch.nn.BCELoss
        The binary cross entropy loss function.
    __ce_loss : torch.nn.CrossEntropyLoss
        The cross entropy loss function.
    __loss : string
        Indicate the loss that will be used during the training.
    __margin_loss : NotImplemented
        The margin loss as descrbed in "Dynamic Routing Between Capsules", Sabour et al (2017).
    model : NeuralNet
        The neural network to train and evaluate.
    __num_worker : int
        Number of parallel process used for the preprocessing of the data. If 0, the main process will 
        be used for the data augmentation.
    __pin_memory : bool
        The pin_memory option of the DataLoader. If true, the data tensor will copied into the CUDA pinned memory.
    __save_path : string
        Indicate where the weights of the network and the result will be saved.
    __soft : torch.nn.Softmax
        The softmax operation used to transform the last layer of a network into a probability.
    __tol : float
        Represent the tolerance factor. If the loss of a given epoch is below (1 - __tol) * best_loss, 
        then this is consider as an improvement.
    __valid_split : float
        Percentage of the trainset that will be used to create the validation set.
    Methods
    -------
    fit()
    """
    def __init__(self, loss: str = "ce",
                valid_split: float = 0.2, 
                tol: float = 0.01, 
                pin_memory: bool = False, 
                num_workers: int = 0, 
                save_path: str = ""):
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
        """
        self.__tol = tol
        self.__valid_split = valid_split
        self.__pin_memory = pin_memory
        self.__num_work = num_workers
        self.__save_path = save_path
        self.__model = None
        self.__device = None

        # ----------------------------------
        #              LOSS
        # ----------------------------------
        assert loss in ["ce", "bce", "marg"], \ 
            "You can only choose one of the following loss ['ce', 'bce', 'marg']"

        if loss == "marg":
            raise NotImplementedError

        self.__loss = loss
        self.__ce_loss = nn.CrossEntropyLoss()
        self.__bce_loss = nn.BCELoss()
        self.__margin_loss = None
        self.__soft = nn.Softmax(dim=-1)

    def fit(self, model: NeuralNet, 
            trainset: RenalDataset,
            seed: int = 0,
            num_epoch: int = 200, 
            batch_size: int = 150, 
            learning_rate: float = 0.01, 
            eps: float = 1e-4, 
            l2: float = 1e-4, 
            t_0: int = 200,
            eta_min: float = 1e-4, 
            grad_clip: float = 0, 
            mode: str = "standard", 
            warm_up_epoch: int = 0, 
            retrain: bool = False, 
            device: str = "cuda:0", 
            verbose: bool = True) -> None:
        """
        Train the model on the given dataset

        :param model: The model to train.
        :param trainset: The dataset that will be used to train the model.
        :param seed: The seed that will be used to split the trainset.
        :param num_epoch: Maximum number of epoch during the training. (Default=200)
        :param batch_size: The batch size that will be used during the training. (Default=150)
        :param learning_rate: Start learning rate of the Adam optimizer. (Default=0.1)
        :param eps: The epsilon parameter of the Adam Optimizer.
        :param l2: L2 regularization coefficient.
        :param t_0: Number of epoch before the first restart. (Default=200)
        :param eta_min: Minimum value of the learning rate. (Default=1e-4)
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient. (Default=0)
        :param mode: The training type: Option: Standard training (No mixup) (Default)
                                                Mixup (Standard manifold mixup)
        :param warm_up_epoch: Number of iteration before activating mixup. (Default=True)
        :param retrain: If false, the weights of the model will initialize. (Default=False)
        :param device: The device on which the training will be done. (Default="cuda:0", first GPU)
        :param verbose: If true, show the progress of the training. (Default=True)
        """
        # Indicator for early stopping
        best_accuracy = 0
        best_epoch = -1
        current_mode = "Standard"

        # We get the appropriate loss because mixup loss will always be bigger than standard loss.
        last_saved_loss = float("inf")

        # Initialization of the model.
        self.__device = device
        self.__model = model.to(device)
        self.__model.set_mixup(batch_size)

        if retrain:
            start_epoch, last_saved_loss, best_accuracy = self.model.restore(self.save_path)
        else:
            self.model.apply(init_weights)
            start_epoch = 0

        # Initialization of the dataloader
        train_loader, valid_loader = get_dataloader(trainset, batch_size, shuffle=True,
                                                    pin_memory=self.__pin_memory, 
                                                    num_workers=self.__num_work,
                                                    validation_split=self.__valid_split,
                                                    random_seed=seed)

        # Initialization of the optimizer and the scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2)

        n_iters = len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0*n_iters,
            T_mult=1,
            eta_min=eta_min)
        scheduler.step(start_epoch*n_iters)

        # Go in training mode to activate mixup module
        self.model.train()

        with tqdm(total=num_epoch, initial=start_epoch) as t:
            for epoch in range(start_epoch, num_epoch):

                _grad_clip = 0 if epoch > num_epoch / 2 else grad_clip
                current_mode = mode if warm_up_epoch <= epoch else current_mode

                # We make a training epoch
                if current_mode == "Mixup":
                    training_loss = self.mixup_epoch(train_loader, optimizer, scheduler, _grad_clip)
                else:
                    training_loss = self.standard_epoch(train_loader, optimizer, scheduler, _grad_clip)

                self.model.eval()

                with amp.autocast():
                    current_accuracy, val_loss = self.accuracy(dt_loader=valid_loader, get_loss=True)

                self.model.train()

                # ------------------------------------------------------------------------------------------
                #                                   EARLY STOPPING PART
                # ------------------------------------------------------------------------------------------

                if (val_loss < last_saved_loss and current_accuracy >= best_accuracy) or \
                        (val_loss < last_saved_loss*(1+self.__tol) and current_accuracy > best_accuracy):
                    self.save_checkpoint(epoch, val_loss, current_accuracy)
                    best_accuracy = current_accuracy
                    last_saved_loss = val_loss
                    best_epoch = epoch

                if verbose:
                    t.postfix = "train loss: {:.4f}, val loss: {:.4f}, val acc: {:.2f}%, best acc: {:.2f}%, " \
                                "best epoch: {}, epoch type: {}".format(
                                 training_loss, val_loss, current_accuracy * 100, best_accuracy * 100, best_epoch + 1,
                                 current_mode)
                t.update()
        self.model.restore(self.save_path)