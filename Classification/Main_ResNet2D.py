"""
    @file:              Main_ResNet2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain the main function to train a 2D ResNet on one of the three task
                        (malignancy, subtype and grade prediction).
"""

import argparse
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


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device on which the model will be trained.")
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help="The drop rate hyperparameter used to configure the dropout layer. See drop_type")
    parser.add_argument('--early_stopping', type=bool, default=False, nargs='?', const=True,
                        help="If true, the training will be stop after the third of the training if the model did not "
                             "achieve at least 50% validation accuracy for at least one epoch.")
    parser.add_argument('--eps', type=float, default=1e-3,
                        help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help="The minimal value of the learning rate.")
    parser.add_argument('--extra_data', type=bool, default=False, nargs='?', const=True,
                        help="If true, the second testest will be add to the training dataset. "
                             "The second dataset is determined with '--testset'.")
    parser.add_argument('--grad_clip', type=float, default=5,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
    parser.add_argument('--loss', type=str, default="ce",
                        help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                             "'bce' == binary cross entropoy, 'focal' = focal loss",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="The initial learning rate")
    parser.add_argument('--mixed_precision', type=bool, default=False, nargs='?', const=True,
                        help="If true, the model will be trained with mixed precision. "
                             "Mixed precision reduce memory consumption on GPU but reduce training speed.")
    parser.add_argument('--num_epoch', type=int, default=1000,
                        help="The number of training epoch.")
    parser.add_argument('--num_cumu_batch', type=int, default=1,
                        help="The number of batch that will be cumulated before updating the weight of the model.")
    parser.add_argument('--optim', type=str, default="adam",
                        help="The optimizer that will be used to train the model.",
                        choices=["adam", "novograd"])
    parser.add_argument('--retrain', type=bool, default=False, nargs='?', const=True,
                        help="If true, load the last saved model and continue the training.")
    parser.add_argument('--task', type=str, default="grade",
                        help="The task on which the model will be train.",
                        choices=["malignant", "subtype", "grade"]),
    parser.add_argument('--testset', type=str, default="stratified",
                        help="The name of the first testset. If 'testset'== stratified then the first testset will be "
                             "the stratified dataset and the independant will be the second and hence could be used as "
                             "extra data.",
                        choices=["stratified", "independant"])
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
                warm_up_epoch=args.warm_up,
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
