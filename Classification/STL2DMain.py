"""
    @file:              Main_ResNet2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 06/2021

    @Description:       Contain the main function to train a 2D ResNet on one of the three task
                        (malignancy, subtype and grade prediction).
"""

from comet_ml import Experiment
import torch
from typing import Final

from ArgParser import argument_parser
from Constant import DatasetName, Experimentation, SplitName
from DataManager.DatasetBuilder import build_datasets
from Model.ResNet_2D import ResNet2D
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet

MIN_NUM_EPOCH: Final = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME: Final = "STL_2D"
PROJECT_NAME: Final = "renal-classification"
SAVE_PATH: Final = "save/STL2D_NET.pth"  # Save path of the single task learning with ResNet2D experiment
TOL: Final = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(Experimentation.SINGLE_TASK_2D)
    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    if args.task in ["subtype", "grade"]:
        clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis", "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]
    else:
        clin_features = ["Age", "Sex", "size"]
    trainset, validset, testset = build_datasets(dataset_name=DatasetName.RCC,
                                                 tasks=[args.task],
                                                 testset_name=args.testset,
                                                 clin_features=clin_features,
                                                 num_dimension=2)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = ResNet2D(drop_rate=args.drop_rate,
                   nb_clinical_data=len(clin_features)).to(args.device)

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution(SplitName.TRAIN,
                            trainset.labels_bincount(),
                            [args.task])
    print_data_distribution(SplitName.VALIDATION,
                            validset.labels_bincount(),
                            [args.task])
    print_data_distribution(args.testset.upper(),
                            testset.labels_bincount(),
                            [args.task])
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

    torch.backends.cudnn.benchmark = not args.mixed_precision

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                grad_clip=args.grad_clip,
                eps=args.eps,
                batch_size=args.b_size,
                num_cumulated_batch=args.num_cumu_batch,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch,
                l2=0 if args.activation == "PReLU" else args.l2,
                retrain=args.retrain)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    if args.num_epoch > MIN_NUM_EPOCH:
        experiment = Experiment(api_key=read_api_key(),
                                project_name="renal-classification",
                                workspace="aleayotte",
                                log_env_details=False,
                                auto_metric_logging=False,
                                log_git_metadata=False,
                                auto_param_logging=False,
                                log_code=False)

        experiment.set_name("ResNet2D" + "_" + args.task)
        experiment.log_code("ArgParser.py")
        experiment.log_code("STL2DMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/SingleTaskTrainer.py")
        experiment.log_code("Model/ResNet_2D.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.task, args.testset)
        train_csv_path, valid_csv_path, test_csv_path = csv_path

    else:
        experiment = None
        test_csv_path = ""
        valid_csv_path = ""
        train_csv_path = ""

    conf, auc = trainer.score(trainset, save_path=train_csv_path)
    print_score(dataset_name=SplitName.TRAIN,
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(validset, save_path=valid_csv_path)
    print_score(dataset_name=SplitName.VALIDATION,
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(testset, save_path=test_csv_path)
    print_score(dataset_name=args.testset.upper(),
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
