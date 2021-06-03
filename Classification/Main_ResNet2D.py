"""
    @file:              Main_ResNet2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain the main function to train a 2D ResNet on one of the three task
                        (malignancy, subtype and grade prediction).
"""

from ArgParser import argument_parser, Experimentation
from comet_ml import Experiment
from Data_manager.DatasetBuilder import build_datasets
from Model.ResNet_2D import ResNet2D
import torch
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import get_predict_csv_path, print_score, print_data_distribution, read_api_key, save_hparam_on_comet


MIN_NUM_EPOCH = 75  # Minimum number of epoch to save the experiment with comet.ml
MODEL_NAME = "STL_2D"
PROJECT_NAME = "renal-classification"
SAVE_PATH = "save/STL2D_NET.pth"  # Save path of the single task learning with ResNet2D experiment
TOL = 1.0  # The tolerance factor use by the trainer


if __name__ == "__main__":
    args = argument_parser(Experimentation.SINGLE_TASK_2D)
    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    if args.task in ["subtype", "grade"]:
        clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis", "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]
    else:
        clin_features = ["Age", "Sex", "size"]
    trainset, validset, testset = build_datasets(tasks=[args.task],
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

    torch.backends.cudnn.benchmark = not args.mixed_precision

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                mode="standard",
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
        experiment.log_code("MultiTaskMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/SingleTaskTrainer.py")
        experiment.log_code("Model/ResNet_2D.py")

        csv_path = get_predict_csv_path(MODEL_NAME, PROJECT_NAME, args.testset, args.task)
        train_csv_path, valid_csv_path, test_csv_path = csv_path

    else:
        experiment = None
        test_csv_path = ""
        valid_csv_path = ""
        train_csv_path = ""

    _, _ = trainer.score(trainset, save_path=train_csv_path)

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
        del hparam["task"]
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
