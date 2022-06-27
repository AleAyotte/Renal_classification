"""
    @file:              model_creation.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 02/2022

    @Description:       Contain the functions that will build the models for the different experiments.
"""
import argparse
import torch
from typing import Dict, Final, List, Tuple, Union

from constant import AttentionBlock, BlockType, CS_CONFIG, DropType, Experimentation, Loss,\
    LTBConfig, NetConfig, SharingUnits,  SubNetDepth, Tasks
from model.hard_shared_resnet import HardSharedResNet
from model.ltb_resnet import LTBResNet
from model.mtan import MTAN
from model.neural_net import NeuralNet
from model.resnet import ResNet
from model.resnet_2d import ResNet2D
from model.shared_net import SharedNet
from model.cross_stitch import CrossStitch

LOAD_PATH: Final = "save/STL3D_NET/"  # Loading path of the single task model.
MIN_NUM_TASKS: Final = 2
RESNET_NB_LEVEL: Final = 4
SUBSPACE: Final = [4, 8, 8, 0]


def build_hardshared(args: argparse.Namespace,
                     in_shape: Tuple[int, int, int],
                     num_classes: Dict[str, int],
                     tasks_list: List[str]) -> HardSharedResNet:
    """
    Build an Hard Shared ResNet

    :param args: A Namespace that contain the main argument for the experimentation.
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :param num_classes: A dictionary that indicate the number of classes for each task.
    :param tasks_list: A list of every task on which the model will train.
    :return: An HardSharedResNet that represent the network to train.
    """
    task_block = {}

    aux_tasks = [task for task in tasks_list if num_classes[task] is Tasks.REGRESSION]
    main_tasks = [task for task in tasks_list if num_classes[task] is Tasks.CLASSIFICATION]

    # Block configuration
    if args.config == 0:
        commun_block = BlockType.PREACT
        if Tasks.MALIGNANCY not in main_tasks and Tasks.GRADE not in main_tasks:
            if args.split_level == 4:
                commun_block = [BlockType.PREACT, BlockType.PREACT, BlockType.POSTACT]
            elif args.split_level == 5:
                commun_block = [BlockType.PREACT, BlockType.PREACT, BlockType.POSTACT, BlockType.POSTACT]

        for task in main_tasks:
            task_block[task] = BlockType.POSTACT if task is Tasks.SUBTYPE else BlockType.PREACT
        for task in aux_tasks:
            task_block[task] = BlockType.POSTACT if Tasks.SUBTYPE in main_tasks else BlockType.PREACT
    else:
        if args.split_level == 3:
            commun_block = BlockType.POSTACT
            for task in main_tasks + aux_tasks:
                task_block[task] = BlockType.PREACT
        elif args.split_level == 4:
            commun_block = [BlockType.POSTACT, BlockType.POSTACT, BlockType.PREACT]
            for task in main_tasks + aux_tasks:
                task_block[task] = BlockType.PREACT
        elif args.split_level == 5:
            commun_block = [BlockType.POSTACT, BlockType.POSTACT, BlockType.PREACT, BlockType]
            for task in main_tasks + aux_tasks:
                task_block[task] = BlockType.PREACT
        else:
            commun_block = BlockType.POSTACT
            for task in main_tasks + aux_tasks:
                task_block[task] = [BlockType.POSTACT, BlockType.PREACT, BlockType.PREACT]

    # Depth configuration
    if args.depth_config == 1:
        commun_depth = 18
        depth_config = {main_task: SubNetDepth.CONFIG1[main_task] for main_task in main_tasks}

    elif args.depth_config == 2:
        commun_depth = 34
        depth_config = {main_task: SubNetDepth.CONFIG2[main_task] for main_task in main_tasks}

    else:
        commun_depth = 18
        depth_config = {main_task: SubNetDepth.CONFIG3[main_task] for main_task in main_tasks}

    for task in aux_tasks:
        depth_config[task] = commun_depth
    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    net = HardSharedResNet(act=args.activation,
                           aux_tasks=aux_tasks,
                           aux_tasks_coeff=args.aux_coeff,
                           commun_block=commun_block,
                           commun_depth=commun_depth,
                           drop_rate=args.drop_rate,
                           drop_type=DropType[args.drop_type.upper()],
                           first_channels=args.in_channels,
                           in_shape=in_shape,
                           main_tasks=main_tasks,
                           num_classes=num_classes,
                           split_level=args.split_level,
                           task_block=task_block,
                           task_depth=depth_config).to(args.device)
    return net


def build_ltb(args: argparse.Namespace,
              in_shape: Tuple[int, int, int],
              num_classes: Dict[str, int],
              tasks_list: List[str]) -> LTBResNet:
    """
    Build a Learn-To-Branch ResNet

    :param args: A Namespace that contain the main argument for the experimentation.
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :param num_classes: A dictionary that indicate the number of classes for each task.
    :param tasks_list: A list of every task on which the model will be train.
    :return: A LTBResNET that represent the network to train.
    """
    aux_tasks = [task for task in tasks_list if num_classes[task] is Tasks.REGRESSION]
    main_tasks = [task for task in tasks_list if num_classes[task] is Tasks.CLASSIFICATION]

    net = LTBResNet(act=args.activation,
                    aux_tasks=aux_tasks,
                    aux_tasks_coeff=args.aux_coeff,
                    block_type_list=LTBConfig[f"CONFIG{args.config}"].value,
                    block_width=args.width,
                    drop_rate=args.drop_rate,
                    drop_type=DropType[args.drop_type.upper()],
                    first_channels=args.in_channels,
                    in_shape=in_shape,
                    main_tasks=main_tasks,
                    num_classes=num_classes).to(args.device)
    return net


def build_mtan(args: argparse.Namespace,
               in_shape: Tuple[int, int, int],
               num_classes: Dict[str, int],
               tasks_list: List[str]) -> MTAN:
    """
    Build a Multi-Task Attention Network

    :param args: A Namespace that contain the main argument for the experimentation.
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :param num_classes: A dictionary that indicate the number of classes for each task.
    :param tasks_list: A list of every task on which the model will be train.
    :return: a MTAN that represent the network to train.
    """
    net = MTAN(act=args.activation,
               att_type=AttentionBlock[args.att_block.upper()],
               blocks_type=NetConfig[f"CONFIG{args.config}"].value,
               depth=args.depth,
               drop_rate=args.drop_rate,
               drop_type=DropType[args.drop_type.upper()],
               first_channels=args.in_channels,
               in_shape=in_shape,
               loss=Loss[args.mtl_loss.upper()],
               num_classes=num_classes,
               tasks=tasks_list).to(args.device)
    return net


def build_resnet2d(args: argparse.Namespace,
                   num_clin_features: int) -> ResNet2D:
    """
    Load a pretrain ResNet2D and change the last layer for binary classification.

    :param args: A Namespace that contain the main argument for the experimentation.
    :param num_clin_features: The number of clinical features that will be used to classify the images.
    :return: A ResNet2D that represent the network to train.
    """
    return ResNet2D(drop_rate=args.drop_rate, nb_clinical_data=num_clin_features).to(args.device)


def build_resnet3d(args: argparse.Namespace,
                   in_shape: Tuple[int, int, int]) -> ResNet:
    """
    Build a ResNet 3D

    :param args: A Namespace that contain the main argument for the experimentation.
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :return: A ResNet3D that represent the network to train.
    """

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    net = ResNet(act=args.activation,
                 blocks_type=NetConfig[f"CONFIG{args.config}"].value,
                 depth=args.depth,
                 drop_rate=args.drop_rate,
                 drop_type=DropType[args.drop_type.upper()],
                 first_channels=args.in_channels,
                 groups=args.groups,
                 in_shape=in_shape,
                 mixup=args.mixup,
                 num_in_chan=args.num_chan_data).to(args.device)

    return net


def build_sharednet(args: argparse.Namespace,
                    in_shape: Tuple[int, int, int],
                    tasks_list: List[str]) -> SharedNet:
    """
    Build a Shared Net

    :param args: A Namespace that contain the main argument for the experimentation.
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :param tasks_list: A list of every task on which the model will be train.
    :return: A SharedNet that represent the network to train.
    """

    config = SubNetDepth[f"CONFIG{args.depth_config}"].value
    blocks_lists = {task: NetConfig.CONFIG2 for task in tasks_list}

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    sub_nets = torch.nn.ModuleDict()
    for task in tasks_list:
        sub_nets[task] = ResNet(
            act=args.activation,
            blocks_type=blocks_lists[task],
            depth=config[task],
            drop_rate=args.drop_rate,
            drop_type=DropType[args.drop_type.upper()],
            first_channels=args.in_channels,
            groups=args.groups,
            in_shape=in_shape,
            num_in_chan=args.num_chan_data,
        ).to(args.device)

        if args.pretrained:
            assert args.seed is not None, "You should specify the split seed to load pretrained model."
            load_path = LOAD_PATH + f"{args.seed}/" + task + ".pth"
            sub_nets[task].restore(load_path)

    if args.real_cs:
        net = CrossStitch(
            c=args.c,
            spread=args.spread,
            sub_nets=sub_nets,
            num_shared_channels=[args.in_channels * conf for conf in CS_CONFIG[args.cs_config]]
        ).to(args.device)

    else:
        net = SharedNet(
            c=args.c,
            sharing_unit=SharingUnits[args.sharing_unit.upper()],
            spread=args.spread,
            sub_nets=sub_nets,
            subspace_1={task: SUBSPACE[0] for task in tasks_list},
            subspace_2={task: SUBSPACE[1] for task in tasks_list},
            subspace_3={task: SUBSPACE[2] for task in tasks_list},
            subspace_4={task: SUBSPACE[3] for task in tasks_list},
            num_shared_channels=[args.in_channels * conf for conf in CS_CONFIG[args.cs_config]]
        ).to(args.device)

    return net


def create_model(args: argparse.Namespace,
                 experimentation: Experimentation,
                 in_shape: Tuple[int, int, int],
                 num_clin_features: int,
                 tasks_list: List[str]) -> Tuple[Union[NeuralNet, ResNet2D],
                                                 Dict[str, int],
                                                 List[List[str]]]:
    """
    Create the corresponding neural network according to the experimentation type and a given namespace of argument.

    :param args: A Namespace that contain the main argument for the experimentation.
    :param experimentation: Indicate the type of experimentation that will be run. (See constant.py)
    :param in_shape: A tuple that indicate the shape of an image tensor without the channel dimension.
    :param num_clin_features: The number of clinical features that will be used to classify the images.
    :param tasks_list: A list of every task on which the model will be train.
    :return: The NeuralNetwork to train, a dictionary that contain the number of classes for each task and a list
             of list that represent the conditional that we want to measure during the training.
    """
    num_classes = {}
    conditional_prob = []

    if experimentation in [Experimentation.STL_2D, Experimentation.STL_3D]:
        num_classes[args.task] = Tasks.CLASSIFICATION
    else:
        c_tasks = [Tasks.ARE, Tasks.GRADE, Tasks.LRF, Tasks.MALIGNANCY, Tasks.SUBTYPE]
        r_tasks = [task for task in tasks_list if task not in c_tasks]
        are_used = [args.are, args.grade, args.lrf, args.malignancy, args.subtype]

        for task, is_used in zip(c_tasks, are_used):
            if is_used:
                num_classes[task] = Tasks.CLASSIFICATION
                if args.malignancy and task in [Tasks.GRADE, Tasks.SUBTYPE]:
                    conditional_prob.append([task, Tasks.MALIGNANCY])

        for task in r_tasks:
            num_classes[task] = Tasks.REGRESSION
    if experimentation is Experimentation.TAG:
        if args.model == "hs":
            net = build_hardshared(args, in_shape=in_shape, num_classes=num_classes, tasks_list=tasks_list)
        else:
            net = build_ltb(args, in_shape=in_shape, num_classes=num_classes, tasks_list=tasks_list)

    elif experimentation is Experimentation.HARD_SHARING:
        net = build_hardshared(args, in_shape=in_shape, num_classes=num_classes, tasks_list=tasks_list)

    elif experimentation is Experimentation.LTB:
        net = build_ltb(args, in_shape=in_shape, num_classes=num_classes, tasks_list=tasks_list)

    elif experimentation is Experimentation.MTAN:
        net = build_mtan(args, in_shape=in_shape, num_classes=num_classes, tasks_list=tasks_list)

    elif experimentation is Experimentation.SOFT_SHARING:
        net = build_sharednet(args, in_shape=in_shape, tasks_list=tasks_list)

    elif experimentation is Experimentation.STL_2D:
        net = build_resnet2d(args, num_clin_features)

    elif experimentation is Experimentation.STL_3D:
        net = build_resnet3d(args, in_shape)

    else:
        raise NotImplementedError

    return net, num_classes, conditional_prob
