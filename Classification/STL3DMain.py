"""
    @file:              SingleTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 04/2021

    @Description:       Contain the main function to train a 3D ResNet on one of the three tasks
                        (malignancy, subtype and grade prediction).
"""
from ArgParser import argument_parser, Experimentation
from comet_ml import Experiment
from Data_manager.DatasetBuilder import build_datasets
from Model.ResNet import ResNet
import torch
from torchsummary import summary
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


MIN_NUM_EPOCH = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME = "STL_3D"
# PROJECT_NAME = "renal-classification"
PROJECT_NAME = "may-2021-hybrid"
SAVE_PATH = "save/STL3D_NET.pth"  # Save path of the single task learning with ResNet3D experiment
TOL = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(experiment=Experimentation.SINGLE_TASK_3D)

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(tasks=[args.task],
                                                 testset_name=args.testset,
                                                 num_chan=args.num_chan_data)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(testset[0]["sample"].size()[1:])
    if args.config == 0:
        pre_act = [True, True, True, True]
    elif args.config == 1:
        pre_act = [False, False, False, False]
    elif args.config == 2:
        pre_act = [False, False, True, True]
    elif args.config == 3:
        pre_act = [True, True, False, False]
    else:
        pre_act = [True, True, True, False]

    net = ResNet(mixup=args.mixup,
                 depth=args.depth,
                 groups=args.groups,
                 in_shape=in_shape,
                 first_channels=args.in_channels,
                 num_in_chan=args.num_chan_data,
                 drop_rate=args.drop_rate,
                 drop_type=args.drop_type,
                 act=args.activation,
                 pre_act=pre_act).to(args.device)

    summary(net, (args.num_chan_data, 96, 96, 32))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution("Training Set",
                            [args.task],
                            trainset.labels_bincount())
    print_data_distribution("Validation Set",
                            [args.task],
                            validset.labels_bincount())
    print_data_distribution(f"{args.testset.capitalize()} Set",
                            [args.task],
                            testset.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
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
                l2=0 if args.activation == "PReLU" else 1e-4,
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
        experiment.log_code("SingleTaskMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/SingleTaskTrainer.py")
        experiment.log_code("Model/NeuralNet.py")
        experiment.log_code("Model/ResNet.py")
        experiment.log_code("Model/Block.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, args.task)
        train_csv_path, valid_csv_path, test_csv_path = csv_path

    else:
        experiment = None
        test_csv_path = ""
        valid_csv_path = ""
        train_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name="TRAIN",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name="VALIDATION",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=test_csv_path)
    print_score(dataset_name=f"{args.testset.upper()}",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
