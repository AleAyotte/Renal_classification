"""
    @file:              SingleTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 04/2021

    @Description:       Contain the main function to train a 3D ResNet on one of the three tasks
                        (malignancy, subtype and grade prediction).
"""
import argparse
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
# PROJECT_NAME = "april-2021-master"
PROJECT_NAME = "may-2021-post-act"
SAVE_PATH = "save/STL3D_NET.pth"  # Save path of the single task learning with ResNet3D experiment
TOL = 1.0  # The tolerance factor use by the trainer


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="The activation function use in the NeuralNet.",
                        choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                        help="The number of layer in the ResNet.")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device on which the model will be trained.")
    parser.add_argument('--drop_rate', type=float, default=0,
                        help="The drop rate hyperparameter used to configure the dropout layer. See drop_type")
    parser.add_argument('--drop_type', type=str, default="linear",
                        help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                             "Else if, drop_type == 'linear' the drop rate will grow linearly at each dropout layer "
                             "from 0 to 'drop_rate'.",
                        choices=["flat", "linear"])
    parser.add_argument('--early_stopping', type=bool, default=False, nargs='?', const=True,
                        help="If true, the training will be stop after the third of the training if the model did not "
                             "achieve at least 50% validation accuracy for at least one epoch.")
    parser.add_argument('--eps', type=float, default=1e-3,
                        help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help="The minimal value of the learning rate.")
    parser.add_argument('--grad_clip', type=float, default=1.25,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=16,
                        help="Number of channels after the first convolution.")
    parser.add_argument('--loss', type=str, default="ce",
                        help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                             "'bce' == binary cross entropoy, 'focal' = focal loss",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="The initial learning rate")
    parser.add_argument('--mixed_precision', type=bool, default=False, nargs='?', const=True,
                        help="If true, the model will be trained with mixed precision. "
                             "Mixed precision reduce memory consumption on GPU but reduce training speed.")
    parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 0],
                        help="The alpha parameter of each mixup module. Those alpha parameter are used to sample "
                             "the dristribution Beta(alpha, alpha).")
    parser.add_argument('--mode', type=str, default="Mixup",
                        help="If 'mode' == 'Mixup', the model will be train with manifold mixup. Else no mixup.",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                        help="The number of channels of the input images.")
    parser.add_argument('--num_cumu_batch', type=int, default=1,
                        help="The number of batch that will be cumulated before updating the weight of the model.")
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="The number of training epoch.")
    parser.add_argument('--optim', type=str, default="sgd",
                        help="The optimizer that will be used to train the model.",
                        choices=["adam", "novograd", "sgd"])
    parser.add_argument('--retrain', type=bool, default=False, nargs='?', const=True,
                        help="If true, load the last saved model and continue the training.")
    parser.add_argument('--task', type=str, default="malignancy",
                        help="The task on which the model will be train.",
                        choices=["malignancy", "subtype", "grade"])
    parser.add_argument('--testset', type=str, default="test",
                        help="The name of the testset. If testset=='test' then a random stratified testset will be "
                             "sampled from the training set. Else if hold_out_set is choose, a predefined testset will"
                             "be loaded",
                        choices=["test", "hold_out_set"])
    parser.add_argument('--track_mode', type=str, default="all",
                        help="Determine the quantity of training statistics that will be saved with tensorboard. "
                             "If low, the training loss will be saved only at each epoch and not at each iteration.",
                        choices=["all", "low", "none"])
    parser.add_argument('--warm_up', type=int, default=0,
                        help="Number of epoch before activating the mixup if 'mode' == mixup")
    parser.add_argument('--weights', type=str, default="balanced",
                        help="The weight that will be applied on each class in the training loss. If balanced, "
                             "The classes weights will be ajusted in the training.",
                        choices=["flat", "balanced"])
    parser.add_argument('--worker', type=int, default=0,
                        help="Number of worker that will be used to preprocess data.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

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
    net = ResNet(mixup=args.mixup,
                 depth=args.depth,
                 groups=args.groups,
                 in_shape=in_shape,
                 first_channels=args.in_channels,
                 num_in_chan=args.num_chan_data,
                 drop_rate=args.drop_rate,
                 drop_type=args.drop_type,
                 act=args.activation,
                 pre_act=False).to(args.device)

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
