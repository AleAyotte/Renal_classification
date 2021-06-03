"""
    @file:              MultiTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 06/2021

    @Description:       Contain the main function to train a MultiLevel 3D ResNet for multitask learning.
"""
from ArgParser import argument_parser, Experimentation, Tasks
from comet_ml import Experiment
from Data_manager.DatasetBuilder import build_datasets
from Model.HardSharedResNet import HardSharedResNet
import torch
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from typing import Final, List
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME: Final = "HardSharing"
PROJECT_NAME: Final = "june-2021-mal-sub"
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
        num_classes[Tasks.MALIGNANCY] = 2
        task_block[Tasks.MALIGNANCY] = "preact"

    if args.subtype:
        task_list.append(Tasks.SUBTYPE)
        num_classes[Tasks.SUBTYPE] = 2
        task_block[Tasks.SUBTYPE] = "postact"
        if args.malignancy:
            conditional_prob.append([Tasks.SUBTYPE, Tasks.MALIGNANCY])
    if args.grade:
        task_list.append(Tasks.GRADE)
        num_classes[Tasks.GRADE] = 2
        task_block[Tasks.GRADE] = "preact"
        if args.malignancy:
            conditional_prob.append([Tasks.GRADE, Tasks.MALIGNANCY])

    if len(task_list) < 2:
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
                           drop_type=args.drop_type,
                           task_block=task_block,
                           act=args.activation).to(args.device)

    summary(net, (args.num_chan_data, 96, 96, 32))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution("Training Set",
                            task_list,
                            trainset.labels_bincount())
    print_data_distribution("Validation Set",
                            task_list,
                            validset.labels_bincount())
    print_data_distribution(f"{args.testset.capitalize()} Set",
                            task_list,
                            testset.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(tasks=task_list,
                      num_classes=num_classes,
                      conditional_prob=[["subtype", "malignancy"]],
                      early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
                      loss=args.loss,
                      tol=TOL,
                      num_workers=args.worker,
                      pin_memory=False,
                      classes_weights=args.weights,
                      track_mode=args.track_mode,
                      mixed_precision=True)

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
                l2=0.009,
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

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, "HardShared_Net")
        train_csv_path, valid_csv_path, test_csv_path = csv_path
    else:
        experiment = None
        train_csv_path = valid_csv_path = test_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name="TRAIN",
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name="VALIDATION",
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=test_csv_path)
    print_score(dataset_name=f"{args.testset.upper()}",
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
        experiment.log_parameter("Task", "Hard_Shared_Net")
