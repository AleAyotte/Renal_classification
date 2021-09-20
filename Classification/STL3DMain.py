"""
    @file:              SingleTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 06/2021

    @Description:       Contain the main function to train a 3D ResNet on one of the three tasks
                        (malignancy, subtype and grade prediction).

    @Reference:         1) https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
"""
from comet_ml import Experiment
import os
import torch
from torchsummary import summary
from typing import Final

from ArgParser import argument_parser
from Constant import BlockType, DatasetName, DropType, Experimentation, SplitName, Tasks
from DataManager.DatasetBuilder import build_datasets
from Model.ResNet import ResNet
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet

MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME: Final = "STL3D_"
PRELU_L2: Final = 0  # L2 regularization should not be used when using PRELU activation as recommended by ref 1)
PROJECT_NAME: Final = "may-2021-hybrid"
SAVE_PATH: Final = "save/STL3D_NET"  # Save path of the single task learning with ResNet3D experiment
TOL: Final = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(experiment=Experimentation.SINGLE_TASK_3D)
    dataset_name = DatasetName.RCC if args.dataset == "rcc" else DatasetName.BMETS
    if dataset_name is DatasetName.RCC:
        assert args.task in [Tasks.GRADE, Tasks.MALIGNANCY, Tasks.SUBTYPE], "Incorrect task choice"
    else:
        assert args.task in [Tasks.ARE, Tasks.LRF], "Incorrect task choice"

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(dataset_name=dataset_name,
                                                 tasks=[args.task],
                                                 testset_name=args.testset,
                                                 num_chan=args.num_chan_data,
                                                 split_seed=args.seed)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(testset[0]["sample"].size()[1:])
    if args.config == 0:
        blocks_type = [BlockType.PREACT for _ in range(4)]
    elif args.config == 1:
        blocks_type = [BlockType.POSTACT for _ in range(4)]
    elif args.config == 2:
        blocks_type = [BlockType.POSTACT, BlockType.POSTACT,
                       BlockType.PREACT, BlockType.PREACT]
    elif args.config == 3:
        blocks_type = [BlockType.PREACT, BlockType.PREACT,
                       BlockType.POSTACT, BlockType.POSTACT]
    else:
        blocks_type = [BlockType.PREACT, BlockType.PREACT,
                       BlockType.PREACT, BlockType.POSTACT]

    net = ResNet(mixup=args.mixup,
                 depth=args.depth,
                 groups=args.groups,
                 in_shape=in_shape,
                 first_channels=args.in_channels,
                 num_in_chan=args.num_chan_data,
                 drop_rate=args.drop_rate,
                 drop_type=DropType[args.drop_type.upper()],
                 act=args.activation,
                 blocks_type=blocks_type).to(args.device)

    summary(net, tuple(trainset[0]["sample"].size()))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution(SplitName.TRAIN,
                            trainset.labels_bincount(),
                            [args.task])
    print_data_distribution(SplitName.VALIDATION,
                            validset.labels_bincount(),
                            [args.task])
    print_data_distribution(args.testset.upper(),
                            testset.labels_bincount(),
                            [args.task])
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    if args.seed is not None:
        dir_path = SAVE_PATH + f"/{args.seed}"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = dir_path + "/" + args.task + ".pth"
    else:
        save_path = SAVE_PATH + "_" + args.task + ".pth"

    trainer = Trainer(early_stopping=args.early_stopping,
                      save_path=save_path,
                      loss=args.loss,
                      tol=TOL,
                      num_workers=args.worker,
                      pin_memory=False,
                      classes_weights=args.weights,
                      task=args.task,
                      track_mode=args.track_mode,
                      mixed_precision=args.mixed_precision)

    torch.backends.cudnn.benchmark = True

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                mode=args.mode,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                grad_clip=args.grad_clip,
                warm_up_epoch=args.warm_up,
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
    if args.num_epoch >= MIN_NUM_EPOCH:
        experiment = Experiment(api_key=read_api_key(),
                                project_name=PROJECT_NAME,
                                workspace="aleayotte",
                                log_env_details=False,
                                auto_metric_logging=False,
                                log_git_metadata=False,
                                auto_param_logging=False,
                                log_code=False,)

        experiment.set_name("ResNet3D" + "_" + args.task)
        experiment.log_code("ArgParser.py")
        experiment.log_code("STL3DMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/SingleTaskTrainer.py")
        experiment.log_code("Model/NeuralNet.py")
        experiment.log_code("Model/ResNet.py")
        experiment.log_code("Model/Block.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME,  args.task, args.testset)
        train_csv_path, valid_csv_path, test_csv_path = csv_path

    else:
        experiment = None
        test_csv_path = ""
        valid_csv_path = ""
        train_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name=SplitName.TRAIN,
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name=SplitName.VALIDATION,
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=test_csv_path)
    print_score(dataset_name=args.testset.upper(),
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
