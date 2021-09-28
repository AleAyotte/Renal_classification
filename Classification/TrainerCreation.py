"""
    @file:              TrainerCreation.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 09/2021

    @Description:       Contain the functions that will build the trainers for the different experiments.
"""
import argparse
import os
from typing import Dict, Final, List, Optional, Union

from Constant import Experimentation, ModelType
from Trainer.SingleTaskTrainer import SingleTaskTrainer as STLTrainer
from Trainer.MultiTaskTrainer import MultiTaskTrainer as MTLTrainer

MIXED_PRECISION: Final = True
PIN_MEMORY: Final = False
SAVE_PATH: Final = "save/"  # Save path
TOL: Final = 1.0  # The tolerance factor use by the trainer


def create_trainer(args: argparse.Namespace,
                   experimentation: Experimentation,
                   model_type: ModelType,
                   num_classes: Dict[str, int],
                   tasks_list: List[str],
                   conditional_prob: Optional[List[List[str]]] = None) -> Union[STLTrainer, MTLTrainer]:
    """
    Create a SingleTaskTrainer or a MultiTaskTrainer according to the given argument and the experimentation type.

    :param args: A Namespace that contain the main argument for the experimentation.
    :param experimentation: Indicate the type of experimentation that will be run. (See Constant.py)
    :param model_type: The model type that will be train. (See Constant.py)
    :param num_classes: A dictionary that indicate the number of classes for each task.
    :param tasks_list: A list of every task on which the model will be train.
    :param conditional_prob: A list of pairs, where the pair A, B represent the name of task from which we want
                             to compute the conditional probability P(A|B).
    :return: A trainer that can be use to train a model.
    """
    # Single task experimentation
    if experimentation in [Experimentation.STL_2D, Experimentation.STL_3D]:

        if args.seed is not None:
            dir_path = SAVE_PATH + experimentation.name + f"/{args.seed}"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            save_path = dir_path + "/" + args.task + ".pth"
        else:
            save_path = SAVE_PATH + experimentation.name + "_" + args.task + ".pth"

        trainer = STLTrainer(classes_weights=args.weights,
                             early_stopping=args.early_stopping,
                             loss=args.loss,
                             mixed_precision=MIXED_PRECISION,
                             num_workers=args.worker,
                             pin_memory=PIN_MEMORY,
                             save_path=save_path,
                             task=tasks_list[0],
                             tol=TOL,
                             track_mode=args.track_mode)
    # Multi task experimentation
    else:
        save_path = SAVE_PATH + experimentation.name

        trainer = MTLTrainer(classes_weights=args.weights,
                             conditional_prob=conditional_prob,
                             early_stopping=args.early_stopping,
                             loss=args.loss,
                             main_tasks=tasks_list,
                             mixed_precision=MIXED_PRECISION,
                             model_type=model_type,
                             num_classes=num_classes,
                             num_workers=args.worker,
                             pin_memory=PIN_MEMORY,
                             save_path=save_path,
                             tol=TOL,
                             track_mode=args.track_mode)
    return trainer
