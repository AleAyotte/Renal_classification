"""
    @file:              MultiTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 06/2021

    @Description:       Contain the main function to train a MultiLevel 3D ResNet for multitask learning.

    @Reference:         1) https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
"""
from ArgParser import argument_parser
from comet_ml import Experiment
from Constant import BlockType, DatasetName, DropType, Experimentation, Tasks
from Data_manager.DatasetBuilder import build_datasets
from Model.HardSharedResNet import HardSharedResNet
import torch
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from typing import Final
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MIN_NUM_TASKS: Final = 2  # Minimun number of tasks.
# MIXED_PRECISION: Final = True
MIXED_PRECISION: Final = False
MODEL_NAME: Final = "HardSharing"
PIN_MEMORY: Final = False
PRELU_L2: Final = 0  # L2 regularization should not be used when using PRELU activation as recommended by ref 1)
PROJECT_NAME: Final = "june-2021-pcgrade"
SAVE_PATH: Final = "save/HS_NET.pth"  # Save path of the Hard Sharing experiment
TOL: Final = 1.0  # The tolerance factor use by the trainer

if __name__ == "__main__":
    args = argument_parser(Experimentation.HARD_SHARING)

    # --------------------------------------------
    #                SETUP TASK
    # --------------------------------------------
    task_list = []
    num_classes = {}
    task_block = {}
    conditional_prob = []

    if args.malignancy:
        task_list.append(Tasks.MALIGNANCY)
        num_classes[Tasks.MALIGNANCY] = Tasks.CLASSIFICATION
        task_block[Tasks.MALIGNANCY] = BlockType.PREACT

    if args.subtype:
        task_list.append(Tasks.SUBTYPE)
        num_classes[Tasks.SUBTYPE] = Tasks.CLASSIFICATION
        task_block[Tasks.SUBTYPE] = BlockType.POSTACT
        if args.malignancy:
            conditional_prob.append([Tasks.SUBTYPE, Tasks.MALIGNANCY])

    if args.grade:
        task_list.append(Tasks.GRADE)
        num_classes[Tasks.GRADE] = Tasks.CLASSIFICATION
        task_block[Tasks.GRADE] = BlockType.PREACT
        if args.malignancy:
            conditional_prob.append([Tasks.GRADE, Tasks.MALIGNANCY])

    if len(task_list) < MIN_NUM_TASKS:
        raise Exception("You have to select at least two task.")

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(tasks=task_list,
                                                 testset_name=args.testset,
                                                 num_chan=args.num_chan_data)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = HardSharedResNet(tasks=task_list,
                           num_classes=num_classes,
                           depth=args.depth,
                           split_level=args.split_level,
                           in_shape=in_shape,
                           first_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_type=DropType[args.drop_type.upper()],
                           task_block=task_block,
                           act=args.activation).to(args.device)

    summary(net, tuple(trainset[0]["sample"].size()))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution(DatasetName.TRAIN,
                            task_list,
                            trainset.labels_bincount())
    print_data_distribution(DatasetName.VALIDATION,
                            task_list,
                            validset.labels_bincount())
    print_data_distribution(args.testset.upper(),
                            task_list,
                            testset.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(tasks=task_list,
                      num_classes=num_classes,
                      conditional_prob=conditional_prob,
                      early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
                      loss=args.loss,
                      tol=TOL,
                      num_workers=args.worker,
                      pin_memory=PIN_MEMORY,
                      classes_weights=args.weights,
                      track_mode=args.track_mode,
                      mixed_precision=MIXED_PRECISION)

    torch.backends.cudnn.benchmark = True

    trainer.fit(model=net, 
                trainset=trainset,
                validset=validset,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                grad_clip=args.grad_clip,
                eps=args.eps,
                batch_size=args.b_size,
                num_cumulated_batch=args.num_cumu_batch,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch,
                l2=PRELU_L2 if args.activation == "PReLU" else args.l2,
                retrain=args.retrain)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    if args.num_epoch > MIN_NUM_EPOCH:
        experiment = Experiment(api_key=read_api_key(),
                                project_name=PROJECT_NAME,
                                workspace="aleayotte",
                                log_env_details=False,
                                auto_metric_logging=False,
                                log_git_metadata=False,
                                auto_param_logging=False,
                                log_code=False)

        experiment.set_name("ResNet3D" + "_" + "MultiTask")
        experiment.log_code("ArgParser.py")
        experiment.log_code("MultiTaskMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/MultiTaskTrainer.py")
        experiment.log_code("Model/HardSharedResNet.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, "_".join(task_list))
        train_csv_path, valid_csv_path, test_csv_path = csv_path
    else:
        experiment = None
        train_csv_path = valid_csv_path = test_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name=DatasetName.TRAIN,
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name=DatasetName.VALIDATION,
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=test_csv_path)
    print_score(dataset_name=args.testset.upper(),
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
        experiment.log_parameter("Task", "_".join(task_list))
