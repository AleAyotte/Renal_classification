"""
    @file:              arg_parser.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/2021
    @Last modification: 02/2022

    @Description:       Contain the argument parser that will be used by every Main file.
"""

import argparse


def argument_parser() -> argparse.Namespace:
    """
    Get a list of argument for a experiment.

    :return: A Namespace that contain the main argument for the experimentation.
    """
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="experiment", dest="experiment",
                                      help="The experiment that will be done.", required=True)

    # --------------------------------------------
    #             COMMUN PARAMETERS
    # --------------------------------------------
    parent_parser = argparse.ArgumentParser(add_help=False)
    
    parent_parser.add_argument('--b_size', type=int, default=32,
                               help="The batch size.")
    parent_parser.add_argument('--device', type=str, default="cuda:0",
                               help="The device on which the model will be trained.")
    parent_parser.add_argument('--drop_rate', type=float, default=0,
                               help="The drop rate hyperparameter used to configure the dropout layer. See drop_type")
    parent_parser.add_argument('--early_stopping', type=bool, default=False, nargs='?', const=True,
                               help="If true, the training will be stop after the third of the training if the model"
                                    " did not achieve at least 50%% validation accuracy for at least one epoch.")
    parent_parser.add_argument('--eps', type=float, default=1e-3,
                               help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parent_parser.add_argument('--eta_min', type=float, default=1e-6,
                               help="The minimal value of the learning rate.")
    parent_parser.add_argument('--grad_clip', type=float, default=0,
                               help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient"
                                    " during the training.")
    parent_parser.add_argument('--loss', type=str, default="ce",
                               help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                                    "'bce' == binary cross entropoy, 'focal' = focal loss",
                               choices=["ce", "bce", "focal"])
    parent_parser.add_argument('--lr', type=float, default=1e-3,
                               help="The initial learning rate")
    parent_parser.add_argument('--l2', type=float, default=1e-4,
                               help="The l2 regularization parameters.")
    parent_parser.add_argument('--mixed_precision', type=bool, default=False, nargs='?', const=True,
                               help="If true, the model will be trained with mixed precision. "
                                    "Mixed precision reduce memory consumption on GPU but reduce training speed.")
    parent_parser.add_argument('--num_cumu_batch', type=int, default=1,
                               help="The number of batch that will be cumulated before updating the weight of "
                                    "the model.")
    parent_parser.add_argument('--num_epoch', type=int, default=100,
                               help="The number of training epoch.")
    parent_parser.add_argument('--optim', type=str, default="adam",
                               help="The optimizer that will be used to train the model.",
                               choices=["adam", "novograd", "sgd"])
    parent_parser.add_argument('--retrain', type=bool, default=False, nargs='?', const=True,
                               help="If true, load the last saved model and continue the training.")
    parent_parser.add_argument('--seed', type=int, default=None,
                               help="The seed that will be used to split the data.")
    parent_parser.add_argument('--testset', type=str, default="test",
                               help="The name of the testset. If testset=='test' then a random stratified testset "
                                    "will be sampled from the training set. Else if hold_out_set is choose, a "
                                    "predefined testset will be loaded",
                               choices=["test", "hold_out"])
    parent_parser.add_argument('--track_mode', type=str, default="all",
                               help="Determine the quantity of training statistics that will be saved with "
                                    "tensorboard. If low, the training loss will be saved only at each epoch and not "
                                    "at each iteration.",
                               choices=["all", "low", "none"])
    parent_parser.add_argument('--weights', type=str, default="balanced",
                               help="The weight that will be applied on each class in the training loss. If balanced, "
                                    "The classes weights will be ajusted in the training.",
                               choices=["flat", "balanced"])
    parent_parser.add_argument('--worker', type=int, default=0,
                               help="Number of worker that will be used to preprocess data.")

    # --------------------------------------------
    #                SINGLE TASK 2D
    # --------------------------------------------
    stl2d_parser = subparser.add_parser("SINGLE_TASK_2D", aliases=["stl_2d"],
                                        help="Parser of the STL 2D experimentation",
                                        parents=[parent_parser])

    stl2d_parser.add_argument('--task', type=str, default="malignancy",
                              help="The task on which the model will be train.",
                              choices=["malignancy", "subtype", "grade"])
    stl2d_parser.set_defaults(dataset="rcc", num_chan_data=3, warm_up=0)

    # --------------------------------------------
    #            3D COMMUN PARAMETERS
    # --------------------------------------------
    _3d_parser = argparse.ArgumentParser(add_help=False)

    _3d_parser.add_argument('--activation', type=str, default='ReLU',
                            help="The activation function use in the NeuralNet.",
                            choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
    _3d_parser.add_argument('--dataset', type=str, default="rcc", choices=["bmets", "rcc"],
                            help="The dataset that will be load.")
    _3d_parser.add_argument('--drop_type', type=str, default="linear",
                            help="If drop_type == 'flat' every dropout layer will have the same drop rate. "
                                 "Else if, drop_type == 'linear' the drop rate will grow linearly "
                                 "at each dropout layer from 0 to 'drop_rate'.",
                            choices=["flat", "linear"])
    _3d_parser.add_argument('--in_channels', type=int, default=16,
                            help="Number of channels after the first convolution.")
    _3d_parser.add_argument('--num_chan_data', type=int, default=4, choices=[3, 4],
                            help="The number of channels of the input images.")

    # --------------------------------------------
    #                SINGLE TASK 3D
    # --------------------------------------------
    stl3d_parser = subparser.add_parser("SINGLE_TASK_3D", aliases=["stl_3d"],
                                        help="Parser of the STL 3D experimentation",
                                        parents=[parent_parser, _3d_parser])

    stl3d_parser.add_argument('--config', type=int, default=0, choices=[0, 1, 2, 3, 4])
    stl3d_parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                              help="The number of layer in the ResNet.")
    stl3d_parser.add_argument('--groups', type=int, default=1)
    stl3d_parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 0],
                              help="The alpha parameter of each mixup module. Those alpha parameter are used to sample "
                                   "the dristribution Beta(alpha, alpha).")
    stl3d_parser.add_argument('--mode', type=str, default="standard",
                              help="If 'mode' == 'Mixup', the model will be train with manifold mixup. Else no mixup.",
                              choices=["standard", "Mixup"])
    stl3d_parser.add_argument('--task', type=str, default="malignancy",
                              help="The task on which the model will be train.",
                              choices=["are", "grade", "lrf", "malignancy", "subtype"])
    stl3d_parser.add_argument('--warm_up', type=int, default=0,
                              help="Number of epoch before activating the mixup if 'mode' == mixup")

    # --------------------------------------------
    #            MTL COMMUN PARAMETERS
    # --------------------------------------------
    mtl_parser = argparse.ArgumentParser(add_help=False)

    mtl_parser.add_argument('--are', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the are task.")
    mtl_parser.add_argument('--grade', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the grade task.")
    mtl_parser.add_argument('--lrf', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the lrf task.")
    mtl_parser.add_argument('--malignancy', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the malignancy task.")
    mtl_parser.add_argument('--mtl_loss', type=str, default="uncertainty",
                            help="Indicate the multi-task loss that will be used to train the network."
                                 " Options = (uncertainty, uniform).",
                            choices=["uncertainty", "uniform"])
    mtl_parser.add_argument('--subtype', type=bool, default=False, nargs='?', const=True,
                            help="Train the model on the subtype task.")
    mtl_parser.set_defaults(mode="standard")

    # --------------------------------------------
    #               HARD SHARED 3D
    # --------------------------------------------
    hs_parser = subparser.add_parser("HARD_SHARING", aliases=["hs"],
                                     help="Parser of the hard sharing experimentation",
                                     parents=[parent_parser, _3d_parser, mtl_parser])

    hs_parser.add_argument('--aux_coeff', type=float, default=0.25,
                           help="The coefficient that is applied to the losses of the auxiliary in the total loss.")
    hs_parser.add_argument('--aux_task_set', type=int, default=-1, choices=[-1, 0, 1, 2],
                           help="The set of auxiliary task that will be used in the experimentation."
                                "(see constant.py AuxTaskSet).")
    hs_parser.add_argument('--config', type=int, default=0, choices=[0, 1],
                           help="The config used to determine the block configuration of each sub-network. "
                                "0 will be the common configuration. 1 is PostActX2 - PreActX2.")
    hs_parser.add_argument('--depth_config', type=int, default=1, choices=[1, 2, 3],
                           help="The config used to determine the depth of each sub-network. The depth of the shared "
                                "layers is determined by the most common depth (see constant.py SubNetDepth).")
    hs_parser.add_argument('--split_level', type=int, default=4,
                           help="At which level the multi level resnet should split into sub net.\n"
                                "1: After the first convolution, \n2: After the first residual level, \n"
                                "3: After the second residual level, \n4: After the third residual level, \n"
                                "5: After the last residual level so just before the fully connected layers.")
    hs_parser.set_defaults(warm_up=0)

    # --------------------------------------------
    #               SOFT SHARING 3D
    # --------------------------------------------
    ss_parser = subparser.add_parser("SOFT_SHARING", aliases=["ss"],
                                     help="Parser of the soft_sharing experimentation",
                                     parents=[parent_parser, _3d_parser, mtl_parser])

    ss_parser.add_argument('--c', type=float, default=0.85,
                           help="The conservation parameter of the Cross-Stich Unit")
    ss_parser.add_argument('--cs_config', type=int, default=1, choices=[0, 1, 2, 3],
                           help="The config used to the position of the cross-stitch module if sharing_unit = "
                                "cross_stitch (see constant.py CS_CONFIG).")
    ss_parser.add_argument('--depth_config', type=int, default=1, choices=[1, 2, 3],
                           help="The config used to determine the depth of each sub-network "
                                "(see constant.py SubNetDepth).")
    ss_parser.add_argument('--groups', type=int, default=1)
    ss_parser.add_argument('--pretrained', type=bool, default=False, nargs='?', const=True,
                           help="If True, then the SharedNet will be create with two subnet that has been pretrained "
                                "on their corresponding task. Also, the shared_lr will be equal to lr * 100 and "
                                "shared_eta_min will be equal to eta_min * 100.")
    ss_parser.add_argument('--real_cs', type=bool, default=False, nargs='?', const=True,
                           help="If true, the real cross stitch implementation will be trained instead "
                                "of the SharedNet. By calling this, the sharing_unit parameter is ignored.")
    ss_parser.add_argument('--sharing_l2', type=float, default=3e-6,
                           help="The l2 penalty coefficient applied to the shared module.")
    ss_parser.add_argument('--sharing_unit', type=str, default="cross_stitch",
                           help="The sharing unit that will be used to create the SharedNet. The shared unit allow "
                                "information transfer between multiple subnets",
                           choices=["sluice", "cross_stitch"])
    ss_parser.add_argument('--spread', type=float, default=0.10,
                           help="The spread parameter of the Cross-Stitch Units.")
    ss_parser.set_defaults(warm_up=0)

    # --------------------------------------------
    #                   MTAN 3D
    # --------------------------------------------
    mtan_parser = subparser.add_parser("MTAN", aliases=["mtan"],
                                       help="Parser of the MTAN experimentation",
                                       parents=[parent_parser, _3d_parser, mtl_parser])

    mtan_parser.add_argument('--att_block', type=str, default="spatial",
                             help="Indicate the attention block type that will be used during training."
                                  " Options = (channel, spatialm cbam).",
                             choices=["channel", "spatial", "cbam"])
    mtan_parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                             help="The number of layer in the ResNet.")
    mtan_parser.set_defaults(warm_up=0)

    # --------------------------------------------
    #               LTB RESNET 3D
    # --------------------------------------------
    ltb_parser = subparser.add_parser("LTB", aliases=["ltb"],
                                      help="Parser of the ltb experimentation",
                                      parents=[parent_parser, _3d_parser, mtl_parser])
    ltb_parser.add_argument('--aux_coeff', type=float, default=0.25,
                            help="The coefficient that is applied to the losses of the auxiliary in the total loss.")
    ltb_parser.add_argument('--aux_task_set', type=int, default=1, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                            help="The set of auxiliary task that will be used in the experimentation."
                                 "(see constant.py AuxTaskSet).")
    ltb_parser.add_argument('--branch_eta', type=float, default=1e-6,
                            help="The final learning rate parameter of the gumbel softmax block.")
    ltb_parser.add_argument('--branch_lr', type=float, default=1e-4,
                            help="The learning rate parameter of the gumbel softmax block.")
    ltb_parser.add_argument('--branch_l2', type=float, default=0,
                            help="The l2 penalty of the gumbel softmax block.")
    ltb_parser.add_argument('--branch_num_epoch', type=int, default=200,
                            help="The number of training epoch that is use to find the optimal architecture.")
    ltb_parser.add_argument('--config', type=int, default=2, choices=[1, 2, 3],
                            help="The config used to determine the that will be used in the LTBResNet "
                                 "(see constant.py LTBConfig).")
    ltb_parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                            help="The number of layer in the ResNet.")
    ltb_parser.add_argument('--tau', type=float, default=5,
                            help="The tau parameter of the gumbel softmax block.")
    ltb_parser.add_argument('--warm_up', type=int, default=5,
                            help="Number of epoch before training the branching unit.")
    ltb_parser.add_argument('--width', type=int, default=2,
                            help="The number of parallel layers (possible path) in the Learn-To-Branch model.")

    # --------------------------------------------
    #           TASK-AFFINITY GROUPING
    # --------------------------------------------
    tag_parser = subparser.add_parser("TAG", aliases=["tag"],
                                      help="Parser of the task affinity grouping experimentation",
                                      parents=[parent_parser, _3d_parser, mtl_parser])

    tag_parser.add_argument('--aux_coeff', type=float, default=0.25,
                            help="The coefficient that is applied to the losses of the auxiliary in the total loss.")
    tag_parser.add_argument('--aux_task_set', type=int, default=1, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                            help="The set of auxiliary task that will be used in the experimentation."
                                 "(see constant.py AuxTaskSet).")
    tag_parser.add_argument('--branch_eta', type=float, default=1e-6,
                            help="The final learning rate parameter of the gumbel softmax block.")
    tag_parser.add_argument('--branch_lr', type=float, default=1e-4,
                            help="The learning rate parameter of the gumbel softmax block.")
    tag_parser.add_argument('--branch_l2', type=float, default=0,
                            help="The l2 penalty of the gumbel softmax block.")
    tag_parser.add_argument('--branch_num_epoch', type=int, default=200,
                            help="The number of training epoch that is use to find the optimal architecture.")
    tag_parser.add_argument('--config', type=int, default=2, choices=[1, 2, 3],
                            help="The config used to determine the that will be used in the LTBResNet "
                                 "(see constant.py LTBConfig).")
    tag_parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50],
                            help="The number of layer in the ResNet.")
    tag_parser.add_argument('--depth_config', type=int, default=1, choices=[1, 2, 3],
                            help="The config used to determine the depth of each sub-network. The depth of the shared "
                                 "layers is determined by the most commun depth (see constant.py SubNetDepth).")
    tag_parser.add_argument('--model', type=str, default="hs",
                            help="Indicate the type of model that will be train."
                                 " Options = (hard-sharing, ltb).",
                            choices=["hs", "ltb"])
    tag_parser.add_argument('--split_level', type=int, default=4,
                            help="At which level the multi level resnet should split into sub net.\n"
                                 "1: After the first convolution, \n2: After the first residual level, \n"
                                 "3: After the second residual level, \n4: After the third residual level, \n"
                                 "5: After the last residual level so just before the fully connected layers.")
    tag_parser.add_argument('--tag_freq', type=int, default=5,
                            help="Frequency at which the inter-task affinity will be computed")
    tag_parser.add_argument('--tau', type=float, default=5,
                            help="The tau parameter of the gumbel softmax block.")
    tag_parser.add_argument('--width', type=int, default=6,
                            help="The number of parallel layers (possible path) in the Learn-To-Branch model.")
    tag_parser.set_defaults(warm_up=0)
    return parser.parse_args()
