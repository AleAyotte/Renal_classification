"""
    @file:              SharedNetMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 04/2021

    @Description:       Contain the main function to train a SharedMet for multitask learning.
"""
import argparse
from comet_ml import Experiment
from Data_manager.DatasetBuilder import build_datasets
from Model.ResNet import ResNet
from Model.SharedNet import SharedNet
import torch
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from Utils import print_score, print_data_distribution, read_api_key, save_hparam_on_comet


CSV_PATH = "save/SharedNet_"
DATA_PATH = "final_dtset/all.hdf5"
FINAL_TASK_LIST = ["grade", "Subtype", "Subtype|Malignancy"]  # The list of task name on which the model is assess
LOAD_PATH = "save/"
SAVE_PATH = "save/CS_Net.pth"  # Save path of the Cross-Stitch experiment
TASK_LIST = ["ssign", "subtype"]  # The list of attribute in the hdf5 file that will be used has labels.
TOL = 1.0  # The tolerance factor use by the trainer


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device on which the model will be trained.")
    parser.add_argument('--early_stopping', type=bool, default=False, nargs='?', const=True,
                        help="If true, the training will be stop after the third of the training if the model did not "
                             "achieve at least 50% validation accuracy for at least one epoch.")
    parser.add_argument('--eps', type=float, default=1e-3,
                        help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help="The minimal value of the learning rate.")
    parser.add_argument('--grad_clip', type=float, default=2.25,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
    parser.add_argument('--loss', type=str, default="ce",
                        help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                             "'bce' == binary cross entropoy, 'focal' = focal loss",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="The initial learning rate")
    parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 2],
                        help="The alpha parameter of each mixup module. Those alpha parameter are used to sample "
                             "the dristribution Beta(alpha, alpha).")
    parser.add_argument('--mode', type=str, default="standard",
                        help="If 'mode' == 'Mixup', the model will be train with manifold mixup. Else no mixup.",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                        help="The number of channels of the input images.")
    parser.add_argument('--num_cumu_batch', type=int, default=1,
                        help="The number of batch that will be cumulated before updating the weight of the model.")
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="The number of training epoch.")
    parser.add_argument('--optim', type=str, default="adam",
                        help="The optimizer that will be used to train the model.",
                        choices=["adam", "novograd", "sgd"])
    parser.add_argument('--pretrained', type=bool, default=False, nargs='?', const=True,
                        help="If True, then the SharedNet will be create with two subnet that has been pretrained on "
                             "their corresponding task. Also, the shared_lr will be equal to lr * 100 and "
                             "shared_eta_min will be equal to eta_min * 100.")
    parser.add_argument('--retrain', type=bool, default=False, nargs='?', const=True,
                        help="If true, load the last saved model and continue the training.")
    parser.add_argument('--sharing_unit', type=str, default="cross_stitch",
                        help="The sharing unit that will be used to create the SharedNet. The shared unit allow "
                             "information transfer between multiple subnets",
                        choices=["sluice", "cross_stitch"])
    parser.add_argument('--testset', type=str, default="test",
                        help="The name of the testset. If testset=='test' then a random stratified testset will be "
                             "sampled from the training set. Else if hold_out_set is choose, a predefined testset will"
                             "be loaded",
                        choices=["test", "hold_out_set"])
    parser.add_argument('--track_mode', type=str, default="all",
                        help="Determine the quantity of training statistics that will be saved with tensorboard."
                             "If low, the training loss will be saved only at each epoch and not at each iteration.",
                        choices=["all", "low", "none"])
    parser.add_argument('--warm_up', type=int, default=0,
                        help="Number of epoch before activating the mixup if 'mode' == mixup")
    parser.add_argument('--weights', type=str, default="balanced",
                        help="The weight that will be applied on each class in the training loss. If balanced, "
                             "the classes weights will be ajusted in the training.",
                        choices=["flat", "balanced", "focused"])
    parser.add_argument('--worker', type=int, default=0,
                        help="Number of worker that will be used to preprocess data.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    if args.mode == "Mixup":
        raise NotImplementedError

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
    mal_net = ResNet(mixup=args.mixup,
                     depth=18,
                     first_channels=32,
                     in_shape=in_shape,
                     drop_rate=0.3,
                     drop_type="linear",
                     act="LeakyReLU",
                     pre_act=True).to(args.device)

    sub_net = ResNet(mixup=args.mixup,
                     depth=18,
                     first_channels=32,
                     in_shape=in_shape,
                     drop_rate=0.3,
                     drop_type="linear",
                     act="LeakyReLU",
                     pre_act=True).to(args.device)

    if args.pretrained:
        mal_net.restore(LOAD_PATH + "STL3D_NET.pth")
        sub_net.restore(LOAD_PATH + "STL3D_NET.pth")

    sub_nets = torch.nn.ModuleDict()
    sub_nets["ssign"] = mal_net
    sub_nets["subtype"] = sub_net

    net = SharedNet(sub_nets=sub_nets,
                    mixup=args.mixup,
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
                      num_classes={"ssign": 2, "subtype": 2},
                      # conditional_prob=[["subtype", "malignancy"]],
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
                mode=args.mode,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                shared_lr=args.lr * 100 if args.pretrained else args.lr,
                shared_eta_min=args.eta_min * 100 if args.pretrained else args.eta_min,
                grad_clip=args.grad_clip,
                warm_up_epoch=args.warm_up,
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
    if args.num_epoch > 75:
        experiment = Experiment(api_key=read_api_key(),
                                project_name="renal-classification",
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

        csv_path = CSV_PATH + "_" + args.testset + ".csv"
    else:
        experiment = None
        csv_path = ""

    conf, auc = trainer.score(validset)
    print_score(dataset_name="VALIDATION",
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=csv_path)
    print_score(dataset_name=f"{args.testset.upper()}",
                task_list=list(auc.keys()),
                conf_mat_list=list(conf.values()),
                auc_list=list(auc.values()),
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
        experiment.log_parameter("Task", "SharedNet")
