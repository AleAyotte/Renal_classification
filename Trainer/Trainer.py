from Data_manager.DataManager import RenalDataset, get_dataloader
from Model.NeuralNet import NeuralNet
from monai.losses import FocalLoss
from monai.optimizers import Novograd
import numpy as np
from sklearn.metrics import confusion_matrix
from Trainer.Utils import init_weights, to_one_hot, compute_recall, get_mean_accuracy
import torch
from torch import nn
from torch.cuda import amp
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Sequence, Tuple, Union


class Trainer:
    """
    The trainer class define an object that will be used to train and evaluate a given model. It handle the 
    mixed precision training, the mixup process and more.

    ...
    Attributes
    ----------
    __m_loss : torch.nn
        The loss function of the malignant task.
    __s_loss : torch.nn
        The loss function of the subtype task.
    __g_loss : torch.nn
        The loss function of the grade task.
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
    __track_mode : str
        Control the information that are registred by tensorboard. Options: all, low, none (Default: all).
    __valid_split : float
        Percentage of the trainset that will be used to create the validation set.
    __writer : SummaryWriter
        Use to keep track of the training with tensorboard.
    Methods
    -------
    fit():
        Train the model on the given dataset
    score(dt_loader: DataLoader, get_loss: bool = False):
        Compute the accuracy of the model on a given data loader.
    """
    def __init__(self, loss: str = "ce",
                valid_split: float = 0.2, 
                tol: float = 0.01, 
                pin_memory: bool = False, 
                num_workers: int = 0,
                classes_weights: str = "balanced",
                gamma: float = 2.,
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
        :param classes_weights: The configuration of weights that will be applied on the loss during the training.
                                Flat: All classes have the same weight during the training.
                                balanced: The weights are inversionaly proportional to the number of data of each 
                                          classes in the training set.
                                focused: Same as balanced but in the subtype and grade task, the total weights of the 
                                         two not none classes are 4 times higher than the weight class.
        :param gamma: Gamma parameter of the focal loss
        :param save_path: Indicate where the weights of the network and the result will be saved.
        :param track_mode: Control information that are registred by tensorboard. none: no information will be saved.
                           low: Only accuracy will be saved at each epoch. All: Accuracy at each epoch and training
                           at each iteration. (Default: all)
        """
        self.__tol = tol
        self.__valid_split = valid_split
        self.__pin_memory = pin_memory
        self.__num_work = num_workers
        self.__save_path = save_path
        self.__track_mode = track_mode.lower()
        self.model = None
        self.__device = None
        self.__writer = None
        # ----------------------------------
        #              LOSS
        # ----------------------------------
        assert loss.lower() in ["ce", "bce", "marg", "focal"], "You can only choose one of the following loss ['ce', 'bce', 'marg']"
        assert track_mode.lower() in ["all", "low", "none"], "Track mode should be one of those options: 'all', 'low' or 'none'"
        weights = {"flat": [[1., 1.], 
                            [1., 1., 1.], 
                            [1., 1., 1.]],
                   "balanced": [[1/0.8, 1/1.2], 
                                [1/1.2, 1/0.414, 1/1.386], 
                                [1/1.2, 1/1.206, 1/0.594]],
                   "focused": [[1/0.8, 1/1.2], 
                               [1/(2 * 1.2), 1 / (0.75 * 0.414), 1 / (0.75 * 1.386)],
                               [1/(2 * 1.2), 1 / (0.75 * 1.206), 1 / (0.75 * 0.594)]]}

        weights = weights[classes_weights.lower()]

        #  TODO: Transfer in fit method where self.__device is available.
        if loss.lower() == "ce":
            self.__m_loss = nn.CrossEntropyLoss(weight=torch.Tensor(weights[0]).to("cuda:0"))
            self.__s_loss = nn.CrossEntropyLoss(weight=torch.Tensor(weights[1]).to("cuda:0"))
            self.__g_loss = nn.CrossEntropyLoss(weight=torch.Tensor(weights[2]).to("cuda:0"))
        elif loss.lower() == "bce":
            self.__m_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights[0]))
            self.__s_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights[1]))
            self.__g_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights[2]))
        elif loss.lower() == "focal":
            self.__m_loss = FocalLoss(gamma=gamma, weight=torch.Tensor(weights[0]))
            self.__s_loss = FocalLoss(gamma=gamma, weight=torch.Tensor(weights[1]))
            self.__g_loss = FocalLoss(gamma=gamma, weight=torch.Tensor(weights[2]))
        else:  # loss == "marg"
            raise NotImplementedError

        self.__soft = nn.Softmax(dim=-1)

    def fit(self, model: NeuralNet, 
            trainset: RenalDataset,
            seed: int = 0,
            num_epoch: int = 200, 
            batch_size: int = 32, 
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
        self.__writer = SummaryWriter()

        # Initialization of the model.
        self.__device = device
        self.model = model.to(device)
        self.model.set_mixup(batch_size)

        if retrain:
            start_epoch, last_saved_loss, best_accuracy = self.model.restore(self.save_path)
        else:
            self.model.apply(init_weights)
            start_epoch = 0

        # Initialization of the dataloader
        train_loader, valid_loader = get_dataloader(trainset, batch_size,
                                                    pin_memory=self.__pin_memory, 
                                                    num_workers=self.__num_work,
                                                    validation_split=self.__valid_split,
                                                    random_seed=seed)

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
                    training_loss = self.__mixup_epoch(train_loader, optimizer, scheduler, _grad_clip, epoch)
                else:
                    training_loss = self.__standard_epoch(train_loader, optimizer, scheduler, _grad_clip, epoch)

                self.model.eval()

                val_acc, val_loss = self.__validation_step(dt_loader=valid_loader, 
                                                                    epoch=epoch)

                train_acc, train_loss = self.__validation_step(dt_loader=train_loader, 
                                                               epoch=epoch, 
                                                               dataset_name="training")

                self.__writer.add_scalars('Accuracy', 
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

        self.__writer.close()
        self.model.restore(self.__save_path)

    def __standard_epoch(self, train_loader: DataLoader, 
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
        sum_loss = 0
        n_iters = len(train_loader)
        it = 0

        scaler = amp.grad_scaler.GradScaler()
        for step, data in enumerate(train_loader, 0):
            # Extract the data
            features, labels = data["sample"].to(self.__device), data["labels"]

            m_labels = labels["malignant"].to(self.__device)
            s_labels = labels["subtype"].to(self.__device)
            g_labels = labels["grade"].to(self.__device)

            features = Variable(features)
            m_labels, s_labels, g_labels = Variable(m_labels), Variable(s_labels), Variable(g_labels)

            optimizer.zero_grad()

            # training step
            with amp.autocast():
                m_pred, s_pred, g_pred = self.model(features)

                m_loss = self.__m_loss(m_pred, m_labels)
                s_loss = self.__s_loss(s_pred, s_labels)
                g_loss = self.__g_loss(g_pred, g_labels)

                loss = (m_loss + s_loss + g_loss) / 3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scheduler.step()

            scaler.update()

            # Save the loss
            sum_loss += loss

            if self.__track_mode == "all":
                self.__writer.add_scalars('Training/Loss', 
                                          {'Malignant': m_loss.item(),
                                           'Subtype': s_loss.item(),
                                           'Grade': g_loss.item(),
                                           'Total': loss.item()}, 
                                           it + epoch*n_iters)
            it += 1

        return sum_loss.item() / n_iters

    def __mixup_criterion(self, pred: torch.Tensor, 
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
        m_pred, s_pred, g_pred = pred
        m_target, s_target, g_target = target

        if self.__m_loss.__class__.__name__ == "BCEWithLogitsLoss":
            m_hot_target = to_one_hot(m_target, 2, self.__device)
            s_hot_target = to_one_hot(s_target, 3, self.__device)
            g_hot_target = to_one_hot(g_target, 3, self.__device)

            m_mixed_target = lamb*m_hot_target + (1-lamb)*m_hot_target[permut]
            s_mixed_target = lamb*s_hot_target + (1-lamb)*s_hot_target[permut]
            g_mixed_target = lamb*g_hot_target + (1-lamb)*g_hot_target[permut]

            m_loss = self.__m_loss(m_pred, m_mixed_target)
            s_loss = self.__m_loss(s_pred, s_mixed_target)
            g_loss = self.__m_loss(g_pred, g_mixed_target)

        else:
            m_loss = lamb*self.__m_loss(m_pred, m_target) + (1-lamb)*self.__m_loss(m_pred, m_target[permut])
            s_loss = lamb*self.__s_loss(s_pred, s_target) + (1-lamb)*self.__s_loss(s_pred, s_target[permut])
            g_loss = lamb*self.__g_loss(g_pred, g_target) + (1-lamb)*self.__g_loss(g_pred, g_target[permut])
        
        loss = (m_loss + s_loss + g_loss) / 3

        if self.__track_mode == "all":
            self.__writer.add_scalars('Training/Loss', 
                                      {'Malignant': m_loss.item(),
                                       'Subtype': s_loss.item(),
                                       'Grade': g_loss.item(),
                                       'Total': loss.item()}, 
                                       it)

        return loss


    def __mixup_epoch(self, train_loader: DataLoader, 
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
        sum_loss = 0
        n_iters = len(train_loader)
        it = 0

        scaler = amp.grad_scaler.GradScaler()
        for step, data in enumerate(train_loader, 0):
            features, labels = data["sample"].to(self.__device), data["labels"]

            m_labels = labels["malignant"].to(self.__device)
            s_labels = labels["subtype"].to(self.__device)
            g_labels = labels["grade"].to(self.__device)

            features = Variable(features)
            m_labels, s_labels, g_labels = Variable(m_labels), Variable(s_labels), Variable(g_labels)

            optimizer.zero_grad()

            # Mixup activation
            mixup_key, lamb, permut = self.model.activate_mixup()

            # training step
            with amp.autocast():
                m_pred, s_pred, g_pred = self.model(features)
                loss = self.__mixup_criterion([m_pred, s_pred, g_pred], 
                                              [m_labels, s_labels, g_labels], 
                                              lamb, 
                                              permut,
                                              it + epoch*n_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scheduler.step()

            scaler.update()

            # Save the loss
            sum_loss += loss
            it += 1

            self.model.disable_mixup(mixup_key)

        return sum_loss.item() / n_iters

    def __validation_step(self, dt_loader: DataLoader,
                          epoch: int, 
                          dataset_name: str = "Validation") -> Tuple[float, float]:
        """
        Execute the validation step and save the metrics with tensorboard.

        :param dt_loader: A torch data loader that contain test or validation data.
        :param epoch: The current epoch number.
        :param dataset_name: The name of the dataset will be used to save the metrics with tensorboard.
        :return: The mean accuracy as float and the loss as float.
        """

        with amp.autocast():
            m_conf, s_conf, g_conf, loss = self.__get_conf_matrix(dt_loader=dt_loader, get_loss=True)

            m_reccal = compute_recall(m_conf)
            s_reccal = compute_recall(s_conf)
            g_reccal = compute_recall(g_conf)
            
            m_acc = get_mean_accuracy(m_reccal, geometric_mean=True)
            s_acc = get_mean_accuracy(s_reccal[1:], geometric_mean=True)
            g_acc = get_mean_accuracy(g_reccal[1:], geometric_mean=True)
            
            mean_acc = get_mean_accuracy([m_acc, s_acc, g_acc], geometric_mean=False)

        if self.__track_mode != "none":
            self.__writer.add_scalars('{}/Accuracy'.format(dataset_name), 
                                        {'Malignant': m_acc,
                                        'Subtype': s_acc,
                                        'Grade': g_acc}, 
                                        epoch)

            self.__writer.add_scalars('{}/Recall/Malignant'.format(dataset_name), 
                                      {'Recall 0': m_reccal[0],
                                       'Recall 1': m_reccal[1]}, 
                                      epoch)
            
            self.__writer.add_scalars('{}/Recall/Subtype'.format(dataset_name), 
                                      {'Recall 0': s_reccal[0],
                                       'Recall 1': s_reccal[1],
                                       'Recall 2': s_reccal[2]}, 
                                      epoch)

            self.__writer.add_scalars('{}/Recall/Grade'.format(dataset_name), 
                                      {'Recall 0': g_reccal[0],
                                       'Recall 1': g_reccal[1],
                                       'Recall 2': g_reccal[2]}, 
                                      epoch)
        return mean_acc, loss
    
    def __get_conf_matrix(self, dt_loader: DataLoader, 
                          get_loss: bool = False) -> Union[Tuple[np.array, np.array, np.array, float],
                                                           Tuple[np.array, np.array, np.array]]:
        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data.
        :param get_loss: Return also the loss if True.
        :return: The confusion matrix for each classes and the average loss if get_loss == True.
        """
        m_outs = torch.empty(0, 2).to(self.__device)
        s_outs = torch.empty(0, 3).to(self.__device)
        g_outs = torch.empty(0, 3).to(self.__device)

        m_labels = torch.empty(0).long()
        s_labels = torch.empty(0).long()
        g_labels = torch.empty(0).long()

        for data in dt_loader:
            features, labels = data["sample"].to(self.__device), data["labels"]

            with torch.no_grad():
                m_labels = torch.cat([m_labels, labels["malignant"]])
                s_labels = torch.cat([s_labels, labels["subtype"]])
                g_labels = torch.cat([g_labels, labels["grade"]])

                m_out, s_out, g_out = self.model(features)

                m_outs = torch.cat([m_outs, m_out])
                s_outs = torch.cat([s_outs, s_out])
                g_outs = torch.cat([g_outs, g_out])
        
        with torch.no_grad():
            m_pred = torch.argmax(m_outs, dim=1)
            s_pred = torch.argmax(s_outs, dim=1)
            g_pred = torch.argmax(g_outs, dim=1)

            m_loss = self.__m_loss(m_outs, m_labels.to(self.__device))
            s_loss = self.__s_loss(s_outs, s_labels.to(self.__device))
            g_loss = self.__g_loss(g_outs, g_labels.to(self.__device))

            total_loss = (m_loss + s_loss + g_loss) / 3

        m_conf = confusion_matrix(m_labels.numpy(), m_pred.cpu().numpy())
        s_conf = confusion_matrix(s_labels.numpy(), s_pred.cpu().numpy())
        g_conf = confusion_matrix(g_labels.numpy(), g_pred.cpu().numpy())

        if get_loss:
            return m_conf, s_conf, g_conf, total_loss.item()
        else:
            return m_conf, s_conf, g_conf
    
    def score(self, testset: RenalDataset, 
                    batch_size: int = 150) -> Union[Tuple[np.array, np.array, np.array, float],
                                                    Tuple[np.array, np.array, np.array]]:
        """
        Compute the accuracy of the model on a given test dataset

        :param testset: A torch dataset which contain our test data points and labels
        :param batch_size: The batch_size that will be use to create the data loader. (Default=150)
        :return: The accuracy of the model.
        """
        test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)

        self.model.eval()

        return self.__get_conf_matrix(dt_loader=test_loader)

    def __save_checkpoint(self, epoch: int, 
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