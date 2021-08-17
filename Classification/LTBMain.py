"""
    @file:              LTBMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     08/2020
    @Last modification: 08/2021

    @Description:       Contain the main function to train a Learn-to-Bran ResNet (LTB-Resnet) for multitask learning.

    @Reference:         1) https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
"""
from ArgParser import argument_parser
from comet_ml import Experiment
from Constant import DatasetName, DropType, Experimentation, LTBConfig, ModelType, Tasks
from Data_manager.DatasetBuilder import build_datasets
from Model.LTBResNet import LTBResNet
import torch
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from typing import Final
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MIN_NUM_TASKS: Final = 2  # Minimun number of tasks.
MIXED_PRECISION: Final = True
MODEL_NAME: Final = "LTBResNet"
PIN_MEMORY: Final = False
PRELU_L2: Final = 0  # L2 regularization should not be used when using PRELU activation as recommended by ref 1)
PROJECT_NAME: Final = "aug-2021-ltb"
SAVE_PATH: Final = "save/LTB_NET.pth"  # Save path of the Hard Sharing experiment
TOL: Final = 1.0  # The tolerance factor use by the trainer

if __name__ == "__main__":
    args = argument_parser(Experimentation.LTB_RESNET)

    # --------------------------------------------
    #                SETUP TASK
    # --------------------------------------------
    task_list = []
    num_classes = {}
    conditional_prob = []

    if args.malignancy:
        task_list.append(Tasks.MALIGNANCY)
        num_classes[Tasks.MALIGNANCY] = Tasks.CLASSIFICATION

    if args.subtype:
        task_list.append(Tasks.SUBTYPE)
        num_classes[Tasks.SUBTYPE] = Tasks.CLASSIFICATION
        if args.malignancy:
            conditional_prob.append([Tasks.SUBTYPE, Tasks.MALIGNANCY])

    if args.grade:
        task_list.append(Tasks.GRADE)
        num_classes[Tasks.GRADE] = Tasks.CLASSIFICATION
        if args.malignancy:
            conditional_prob.append([Tasks.GRADE, Tasks.MALIGNANCY])

    if args.config == 1:
        block_config = LTBConfig.CONFIG1
    elif args.config == 2:
        block_config = LTBConfig.CONFIG2
    else:
        block_config = LTBConfig.CONFIG3

    if len(task_list) < MIN_NUM_TASKS:
        raise Exception("You have to select at least two task.")

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(tasks=task_list,
                                                 testset_name=args.testset,
                                                 num_chan=args.num_chan_data,
                                                 split_seed=args.seed)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    # Depth config

    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = LTBResNet(act=args.activation,
                    block_type_list=block_config,
                    block_width=args.width,
                    drop_rate=args.drop_rate,
                    drop_type=DropType[args.drop_type.upper()],
                    first_channels=args.in_channels,
                    in_shape=in_shape,
                    num_classes=num_classes,
                    tasks=task_list).to(args.device)

    # summary(net, tuple(trainset[0]["sample"].size()))

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
    trainer = Trainer(main_tasks=task_list,
                      num_classes=num_classes,
                      conditional_prob=conditional_prob,
                      early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
                      loss=args.loss,
                      tol=TOL,
                      model_type=ModelType.LTB_NET,
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
                shared_lr=args.lr/10,
                eta_min=args.eta_min,
                shared_eta_min=args.eta_min/10,
                grad_clip=args.grad_clip,
                eps=args.eps,
                batch_size=args.b_size,
                num_cumulated_batch=args.num_cumu_batch,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch,
                l2=PRELU_L2 if args.activation == "PReLU" else args.l2,
                retrain=args.retrain,
                warm_up_epoch=args.warm_up)

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

        experiment.set_name("LTBResNet" + "_" + "MultiTask")
        experiment.log_code("ArgParser.py")
        experiment.log_code("LTBMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/MultiTaskTrainer.py")
        experiment.log_code("Model/LTBResNet.py")

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
