"""
    @file:              SharedNetMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 07/2021

    @Description:       Contain the main function to train a SharedMet for multitask learning.

    @Reference:         1) https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
"""

from comet_ml import Experiment
import torch
from torchsummary import summary
from typing import Final

from ArgParser import argument_parser
from Constant import BlockType, CS_CONFIG, DropType, Experimentation,\
    ModelType, SharingUnits, SplitName, SubNetDepth, Tasks
from DataManager.DatasetBuilder import build_datasets
from Model.ResNet import ResNet
from Model.SharedNet import SharedNet
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet

DEFAULT_SHARED_LR_SCALE = 100  # Default rate between shared_lr and lr if shared_lr == 0
LOAD_PATH: Final = "save/STL3D_NET/"  # Loading path of the single task model.
MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MIN_NUM_TASKS: Final = 2  # Minimum number of tasks
MODEL_NAME: Final = "SharedNet"
PRELU_L2: Final = 0  # L2 regularization should not be used when using PRELU activation as recommended by ref 1)
PROJECT_NAME: Final = "jul-2021-softsharing2"
SAVE_PATH: Final = "save/CS_Net.pth"  # Save path of the Cross-Stitch experiment
SUBSPACE: Final = [4, 8, 8, 0]
TOL: Final = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(Experimentation.SOFT_SHARING)

    # --------------------------------------------
    #                SETUP TASK
    # --------------------------------------------
    task_list = []
    num_classes = {}
    blocks_lists = {}
    conditional_prob = []

    # Depth config
    if args.depth_config == 1:
        config: Final = SubNetDepth.CONFIG1
    elif args.depth_config == 2:
        config: Final = SubNetDepth.CONFIG2
    else:
        config: Final = SubNetDepth.CONFIG3

    if args.malignancy:
        task_list.append(Tasks.MALIGNANCY)
        num_classes[Tasks.MALIGNANCY] = Tasks.CLASSIFICATION
        blocks_lists[Tasks.MALIGNANCY] = BlockType.PREACT

    if args.subtype:
        task_list.append(Tasks.SUBTYPE)
        num_classes[Tasks.SUBTYPE] = Tasks.CLASSIFICATION

        if config[Tasks.SUBTYPE] == 34:
            blocks_lists[Tasks.SUBTYPE] = BlockType.POSTACT
        else:
            blocks_lists[Tasks.SUBTYPE] = [BlockType.PREACT, BlockType.PREACT,
                                           BlockType.POSTACT, BlockType.POSTACT]
        if args.malignancy:
            conditional_prob.append([Tasks.SUBTYPE, Tasks.MALIGNANCY])

    if args.grade:
        task_list.append(Tasks.GRADE)
        num_classes[Tasks.GRADE] = Tasks.CLASSIFICATION
        blocks_lists[Tasks.GRADE] = BlockType.PREACT
        if args.malignancy:
            conditional_prob.append([Tasks.GRADE, Tasks.MALIGNANCY])

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
    sub_nets = torch.nn.ModuleDict()
    in_shape = tuple(trainset[0]["sample"].size()[1:])

    for task in task_list:
        sub_nets[task] = ResNet(
            blocks_type=blocks_lists[task],
            depth=config[task],
            groups=args.groups,
            in_shape=in_shape,
            first_channels=args.in_channels,
            num_in_chan=args.num_chan_data,
            drop_rate=args.drop_rate,
            drop_type=DropType[args.drop_type.upper()],
            act=args.activation,
        ).to(args.device)

        if args.pretrained:
            assert args.seed is not None, "You should specify the split seed to load pretrained model."
            load_path = LOAD_PATH + f"{args.seed}/" + task + ".pth"
            sub_nets[task].restore(load_path)

    net = SharedNet(sub_nets=sub_nets,
                    num_shared_channels=[args.in_channels * conf for conf in CS_CONFIG[args.cs_config]],
                    sharing_unit=SharingUnits[args.sharing_unit.upper()],
                    subspace_1={task: SUBSPACE[0] for task in task_list},
                    subspace_2={task: SUBSPACE[1] for task in task_list},
                    subspace_3={task: SUBSPACE[2] for task in task_list},
                    subspace_4={task: SUBSPACE[3] for task in task_list},
                    c=args.c,
                    spread=args.spread).to(args.device)

    summary(net, tuple(trainset[0]["sample"].size()))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution(SplitName.TRAIN,
                            task_list,
                            trainset.labels_bincount())
    print_data_distribution(SplitName.VALIDATION,
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
                      num_workers=args.worker,
                      pin_memory=False,
                      classes_weights=args.weights,
                      model_type=ModelType.SHARED_NET,
                      track_mode=args.track_mode,
                      mixed_precision=True)

    torch.backends.cudnn.benchmark = True

    shared_lr = args.lr * DEFAULT_SHARED_LR_SCALE if args.pretrained else args.lr
    shared_eta_min = args.eta_min * DEFAULT_SHARED_LR_SCALE if args.pretrained else args.eta_min

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                shared_l2=args.sharing_l2,
                shared_lr=shared_lr,
                shared_eta_min=shared_eta_min,
                grad_clip=args.grad_clip,
                eps=args.eps,
                batch_size=args.b_size,
                num_cumulated_batch=args.num_cumu_batch,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=max(args.num_epoch, 1),
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

        experiment.set_name("SharedNet" + "_" + "MultiTask")
        experiment.log_code("ArgParser.py")
        experiment.log_code("SharedNetMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/MultiTaskTrainer.py")
        experiment.log_code("Model/SharedNet.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, "_".join(task_list))
        train_csv_path, valid_csv_path, test_csv_path = csv_path
    else:
        experiment = None
        train_csv_path = valid_csv_path = test_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name=SplitName.TRAIN,
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name=SplitName.VALIDATION,
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
        hparam["shared_lr"] = shared_lr
        hparam["shared_eta_min"] = shared_eta_min
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
        experiment.log_parameter("Task", "_".join(task_list))
