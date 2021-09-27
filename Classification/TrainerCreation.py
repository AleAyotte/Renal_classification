import argparse
import os
import torch
from typing import Dict, Final, List, Optional, Tuple, Union

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
