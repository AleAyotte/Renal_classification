"""
    @file:              ArgParser.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 06/2021

    @Description:       Contain the argument parser that will be used by every Main file.
"""

import argparse
from Constant import Experimentation


def argument_parser(experiment: Experimentation) -> argparse.Namespace:
    """
    Get a list of argument for a experiment.

    :param experiment: An integer that represent the type of that will be execute.
    :return: An Namespace that contain the main argument for the experimentation.
    """

    # --------------------------------------------
    #             COMMUN PARAMETERS
    # --------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device on which the model will be trained.")
    parser.add_argument('--drop_rate', type=float, default=0,
                        help="The drop rate hyperparameter used to configure the dropout layer. See drop_type")
    parser.add_argument('--early_stopping', type=bool, default=False, nargs='?', const=True,
                        help="If true, the training will be stop after the third of the training if the model did not "
                             "achieve at least 50% validation accuracy for at least one epoch.")
    parser.add_argument('--eps', type=float, default=1e-3,
                        help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help="The minimal value of the learning rate.")
    parser.add_argument('--grad_clip', type=float, default=0,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
    parser.add_argument('--loss', type=str, default="ce",
                        help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                             "'bce' == binary cross entropoy, 'focal' = focal loss",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="The initial learning rate")
    parser.add_argument('--l2', type=float, default=1e-4,
                        help="The l2 regularization parameters.")
    parser.add_argument('--mixed_precision', type=bool, default=False, nargs='?', const=True,
                        help="If true, the model will be trained with mixed precision. "
                             "Mixed precision reduce memory consumption on GPU but reduce training speed.")
    parser.add_argument('--num_cumu_batch', type=int, default=1,
                        help="The number of batch that will be cumulated before updating the weight of the model.")
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="The number of training epoch.")
    parser.add_argument('--optim', type=str, default="adam",
                        help="The optimizer that will be used to train the model.",
                        choices=["adam", "novograd", "sgd"])
    parser.add_argument('--retrain', type=bool, default=False, nargs='?', const=True,
                        help="If true, load the last saved model and continue the training.")
    parser.add_argument('--seed', type=int, default=None,
                        help="The seed that will be used to split the data.")
    parser.add_argument('--testset', type=str, default="test",
                        help="The name of the testset. If testset=='test' then a random stratified testset will be "
                             "sampled from the training set. Else if hold_out_set is choose, a predefined testset will"
                             "be loaded",
                        choices=["test", "hold_out"])
    parser.add_argument('--track_mode', type=str, default="all",
                        help="Determine the quantity of training statistics that will be saved with tensorboard. "
                             "If low, the training loss will be saved only at each epoch and not at each iteration.",
                        choices=["all", "low", "none"])
    parser.add_argument('--weights', type=str, default="balanced",
                        help="The weight that will be applied on each class in the training loss. If balanced, "
                             "The classes weights will be ajusted in the training.",
                        choices=["flat", "balanced"])
    parser.add_argument('--worker', type=int, default=0,
                        help="Number of worker that will be used to preprocess data.")

    # --------------------------------------------
    #                SINGLE TASK 2D
    # --------------------------------------------
    if experiment is Experimentation.SINGLE_TASK_2D:
        parser.add_argument('--task', type=str, default="malignancy",
                            help="The task on which the model will be train.",
                            choices=["malignancy", "subtype", "grade"])

    # --------------------------------------------
    #                SINGLE TASK 3D
    # --------------------------------------------
    elif experiment is Experimentation.SINGLE_TASK_3D:
        parser.add_argument('--activation', type=str, default='ReLU',
                            help="The activation function use in the NeuralNet.",
                            choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
        parser.add_argument('--config', type=int, default=0, choices=[0, 1, 2, 3, 4])
        parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                            help="The number of layer in the ResNet.")
        parser.add_argument('--drop_type', type=str, default="linear",
                            help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                                 "Else if, drop_type == 'linear' the drop rate will grow linearly "
                                 "at each dropout layer from 0 to 'drop_rate'.",
                            choices=["flat", "linear"])
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--in_channels', type=int, default=16,
                            help="Number of channels after the first convolution.")
        parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 0],
                            help="The alpha parameter of each mixup module. Those alpha parameter are used to sample "
                                 "the dristribution Beta(alpha, alpha).")
        parser.add_argument('--mode', type=str, default="standard",
                            help="If 'mode' == 'Mixup', the model will be train with manifold mixup. Else no mixup.",
                            choices=["standard", "Mixup"])
        parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                            help="The number of channels of the input images.")
        parser.add_argument('--task', type=str, default="malignancy",
                            help="The task on which the model will be train.",
                            choices=["malignancy", "subtype", "grade"])
        parser.add_argument('--warm_up', type=int, default=0,
                            help="Number of epoch before activating the mixup if 'mode' == mixup")

    # --------------------------------------------
    #               HARD SHARED 3D
    # --------------------------------------------
    elif experiment is Experimentation.HARD_SHARING:
        parser.add_argument('--activation', type=str, default='ReLU',
                            help="The activation function use in the NeuralNet.",
                            choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
        parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                            help="The number of layer in the ResNet.")
        parser.add_argument('--drop_type', type=str, default="linear",
                            help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                                 "Else if, drop_type == 'linear' the drop rate will grow linearly  "
                                 "at each dropout layer from 0 to 'drop_rate'.",
                            choices=["flat", "linear"])
        parser.add_argument('--grade', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the grade task.")
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--in_channels', type=int, default=16,
                            help="Number of channels after the first convolution.")
        parser.add_argument('--malignancy', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the malignancy task.")
        parser.add_argument('--mtl_loss', type=str, default="uncertainty",
                            help="Indicate the multi-task loss that will be used to train the network."
                                 " Options = (uncertainty, uniform).",
                            choices=["uncertainty", "uniform"])
        parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                            help="The number of channels of the input images.")
        parser.add_argument('--split_level', type=int, default=4,
                            help="At which level the multi level resnet should split into sub net.\n"
                                 "1: After the first convolution, \n2: After the first residual level, \n"
                                 "3: After the second residual level, \n4: After the third residual level, \n"
                                 "5: After the last residual level so just before the fully connected layers.")
        parser.add_argument('--subtype', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the subtype task.")

    # --------------------------------------------
    #               SOFT SHARING 3D
    # --------------------------------------------
    elif experiment is Experimentation.SOFT_SHARING:
        parser.add_argument('--activation', type=str, default='ReLU',
                            help="The activation function use in the NeuralNet.",
                            choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
        parser.add_argument('--c', type=float, default=0.85,
                            help="The conservation parameter of the Cross-Stich Unit")
        parser.add_argument('--cs_config', type=int, default=1, choices=[0, 1, 2, 3],
                            help="The config used to the position of the cross-stitch module if sharing_unit = "
                                 "cross_stitch (see Constant.py CS_CONFIG).")
        parser.add_argument('--depth_config', type=int, default=1, choices=[1, 2, 3],
                            help="The config used to determine the depth of each sub-network "
                                 "(see Constant.py SubNetDepth).")
        parser.add_argument('--drop_type', type=str, default="linear",
                            help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                                 "Else if, drop_type == 'linear' the drop rate will grow linearly  "
                                 "at each dropout layer from 0 to 'drop_rate'.",
                            choices=["flat", "linear"])
        parser.add_argument('--grade', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the grade task.")
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--in_channels', type=int, default=16,
                            help="Number of channels after the first convolution.")
        parser.add_argument('--malignancy', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the malignancy task.")
        parser.add_argument('--mtl_loss', type=str, default="uncertainty",
                            help="Indicate the multi-task loss that will be used to train the network."
                                 " Options = (uncertainty, uniform).",
                            choices=["uncertainty", "uniform"])
        parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                            help="The number of channels of the input images.")
        parser.add_argument('--pretrained', type=bool, default=False, nargs='?', const=True,
                            help="If True, then the SharedNet will be create with two subnet that has been pretrained "
                                 "on their corresponding task. Also, the shared_lr will be equal to lr * 100 and "
                                 "shared_eta_min will be equal to eta_min * 100.")
        parser.add_argument('--sharing_l2', type=float, default=3e-6,
                            help="The l2 penalty coefficient applied to the shared module.")
        parser.add_argument('--sharing_unit', type=str, default="cross_stitch",
                            help="The sharing unit that will be used to create the SharedNet. The shared unit allow "
                                 "information transfer between multiple subnets",
                            choices=["sluice", "cross_stitch"])
        parser.add_argument('--spread', type=float, default=0.10,
                            help="The spread parameter of the Cross-Stitch Units.")
        parser.add_argument('--subtype', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the subtype task.")

    # --------------------------------------------
    #               HARD SHARED 3D
    # --------------------------------------------
    elif experiment is Experimentation.MTAN:
        parser.add_argument('--activation', type=str, default='ReLU',
                            help="The activation function use in the NeuralNet.",
                            choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
        parser.add_argument('--att_block', type=str, default="spatial",
                            help="Indicate the attention block type that will be used during training."
                                 " Options = (channel, spatialm cbam).",
                            choices=["channel", "spatial", "cbam"])
        parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                            help="The number of layer in the ResNet.")
        parser.add_argument('--drop_type', type=str, default="linear",
                            help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                                 "Else if, drop_type == 'linear' the drop rate will grow linearly  "
                                 "at each dropout layer from 0 to 'drop_rate'.",
                            choices=["flat", "linear"])
        parser.add_argument('--grade', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the grade task.")
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--in_channels', type=int, default=16,
                            help="Number of channels after the first convolution.")
        parser.add_argument('--malignancy', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the malignancy task.")
        parser.add_argument('--mtl_loss', type=str, default="uncertainty",
                            help="Indicate the multi-task loss that will be used to train the network."
                                 " Options = (uncertainty, uniform).",
                            choices=["uncertainty", "uniform"])
        parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                            help="The number of channels of the input images.")
        parser.add_argument('--subtype', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the subtype task.")

    else:
        raise Exception("This experimentation does not exist.")

    return parser.parse_args()
