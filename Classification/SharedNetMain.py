"""
    @file:              SharedNetMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 04/2021

    @Description:       Contain the main function to train a SharedMet for multitask learning.
"""
from ArgParser import argument_parser, Experimentation
from comet_ml import Experiment
from Data_manager.DatasetBuilder import build_datasets
from Model.ResNet import ResNet
from Model.SharedNet import SharedNet
import torch
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


FINAL_TASK_LIST = ["grade", "Subtype", "Subtype|Malignancy"]  # The list of task name on which the model is assess
LOAD_PATH = "save/"
MIN_NUM_EPOCH = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME = "SharedNet"
PROJECT_NAME = "renal-classification"
SAVE_PATH = "save/CS_Net.pth"  # Save path of the Cross-Stitch experiment
TASK_LIST = ["ssign", "subtype"]  # The list of attribute in the hdf5 file that will be used has labels.
TOL = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(Experimentation.SOFT_SHARING)

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(tasks=TASK_LIST,
                                                 testset_name=args.testset,
                                                 num_chan=args.num_chan_data)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(trainset[0]["sample"].size()[1:])
    mal_net = ResNet(depth=18,
                     first_channels=32,
                     in_shape=in_shape,
                     drop_rate=0.3,
                     drop_type="linear",
                     act="ReLU",
                     pre_act=True).to(args.device)

    sub_net = ResNet(depth=18,
                     first_channels=32,
                     in_shape=in_shape,
                     drop_rate=0.3,
                     drop_type="linear",
                     act="ReLU",
                     pre_act=[True, True, False, False]).to(args.device)

    if args.pretrained:
        mal_net.restore(LOAD_PATH + "STL3D_NET.pth")
        sub_net.restore(LOAD_PATH + "STL3D_NET.pth")

    sub_nets = torch.nn.ModuleDict()
    sub_nets["ssign"] = mal_net
    sub_nets["subtype"] = sub_net

    net = SharedNet(sub_nets=sub_nets,
                    num_shared_channels=[32, 64, 128, 256],
                    sharing_unit=args.sharing_unit,
                    subspace_1=[4, 3],
                    subspace_2=[8, 6],
                    subspace_3=[8, 6],
                    subspace_4=[4, 3],
                    c=0.85,
                    spread=0.1).to(args.device)
    summary(net, (3, 96, 96, 32))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution("Training Set",
                            TASK_LIST,
                            trainset.labels_bincount())
    print_data_distribution("Validation Set",
                            TASK_LIST,
                            validset.labels_bincount())
    print_data_distribution(f"{args.testset.capitalize()} Set",
                            TASK_LIST,
                            testset.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(tasks=TASK_LIST,
                      num_classes={"malignancy": 2, "subtype": 2},
                      conditional_prob=[["subtype", "malignancy"]],
                      early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
                      loss=args.loss,
                      tol=TOL,
                      num_workers=args.worker,
                      pin_memory=False,
                      classes_weights=args.weights,
                      shared_net=args.pretrained,
                      track_mode=args.track_mode,
                      mixed_precision=True)

    torch.backends.cudnn.benchmark = True

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                shared_lr=args.lr * 100 if args.pretrained else args.lr,
                shared_eta_min=args.eta_min * 100 if args.pretrained else args.eta_min,
                grad_clip=args.grad_clip,
                eps=args.eps,
                batch_size=args.b_size,
                num_cumulated_batch=args.num_cumu_batch,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=max(args.num_epoch, 1),
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

        experiment.set_name("SharedNet" + "_" + "MultiTask")
        experiment.log_code("SharedNetMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/MultiTaskTrainer.py")
        experiment.log_code("Model/SharedNet.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, "all")
        train_csv_path, valid_csv_path, test_csv_path = csv_path
    else:
        experiment = None
        train_csv_path = valid_csv_path = test_csv_path = ""

    _, _ = trainer.score(trainset, save_path=train_csv_path)

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
        experiment.log_parameter("Task", "SharedNet")
