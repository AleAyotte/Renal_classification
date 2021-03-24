"""
    @file:              SingleTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       Contain the main function to train a 3D ResNet on one of the three tasks
                        (malignancy, subtype and grade prediction).
"""
import argparse
from comet_ml import Experiment
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet import ResNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
from random import randint
import torch
from torchsummary import summary
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import print_score, print_data_distribution, read_api_key, save_hparam_on_comet


SAVE_PATH = "save/STL3D_NET.pth"  # Save path of the single task learning with ResNet3D experiment


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
    parser.add_argument('--drop_type', type=str, default="flat",
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
    parser.add_argument('--extra_data', type=bool, default=False, nargs='?', const=True,
                        help="If true, the second testest will be add to the training dataset."
                             "The second dataset is determined with '--testset'.")
    parser.add_argument('--grad_clip', type=float, default=1.25,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
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
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="The number of training epoch.")
    parser.add_argument('--num_cumu_batch', type=int, default=1,
                        help="The number of batch that will be cumulated before updating the weight of the model.")
    parser.add_argument('--optim', type=str, default="sgd",
                        help="The optimizer that will be used to train the model.",
                        choices=["adam", "novograd", "sgd"])
    parser.add_argument('--pad_mode', type=str, default="constant",
                        help="How the image will be pad in the data augmentation.",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False, nargs='?', const=True,
                        help="The pin_memory parameter of the dataloader. If true, the data will be pinned in the gpu.")
    parser.add_argument('--task', type=str, default="malignant",
                        help="The task on which the model will be train.",
                        choices=["malignant", "subtype", "grade", "SSIGN"])
    parser.add_argument('--testset', type=str, default="stratified",
                        help="The name of the first testset. If 'testset'== stratified then the first testset will be "
                             "the stratified dataset and the independant will be the second and hence could be used as "
                             "extra data.",
                        choices=["stratified", "independent"])
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
    data_path = "final_dtset/{}.hdf5".format(args.task)

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    """
    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.1, prob=0.5),
        RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0.1, padding_mode="zeros"),
        RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
        RandZoomd(keys=["t1", "t2", "roi"], prob=0.5, min_zoom=1.00, max_zoom=1.05,
                  keep_size=False, mode="trilinear", align_corners=True),
        ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode=args.pad_mode),
        ToTensord(keys=["t1", "t2", "roi"])
    ])
    """
    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
        # RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
        RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 24], random_center=False),
        RandZoomd(keys=["t1", "t2", "roi"], prob=0.5, min_zoom=0.77, max_zoom=1.23,
                  keep_size=False, mode="trilinear", align_corners=True),
        ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode=args.pad_mode),
        ToTensord(keys=["t1", "t2", "roi"])
    ])
    test_transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        ToTensord(keys=["t1", "t2", "roi"])
    ])

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    testset_name = args.testset
    testset2_name = "independant" if args.testset == "stratified" else "stratified"

    trainset = RenalDataset(data_path, transform=transform,
                            imgs_keys=["t1", "t2", "roi"])
    validset = RenalDataset(data_path, transform=test_transform,
                            imgs_keys=["t1", "t2", "roi"],
                            split=None)
    testset = RenalDataset(data_path, transform=test_transform,
                           imgs_keys=["t1", "t2", "roi"],
                           split=testset_name)

    # If we want to use some extra data, then will we used the data of the second test set.
    if args.extra_data:
        testset2 = RenalDataset(data_path, transform=transform,
                                imgs_keys=["t1", "t2", "roi"],
                                split=testset2_name)
        data, label, _ = testset2.extract_data(np.arange(len(testset2)))
        trainset.add_data(data, label)
        del data
        del label
    # Else the second test set will be used to access the performance of the dataset at the end.
    else:
        testset2 = RenalDataset(data_path, transform=test_transform,
                                imgs_keys=["t1", "t2", "roi"],
                                split=testset2_name)
    seed = randint(0, 10000)
    trainset, validset = split_trainset(trainset, validset, validation_split=0.2, random_seed=seed)

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(testset[0]["sample"].size()[1:])
    net = ResNet(mixup=args.mixup,
                 depth=args.depth,
                 in_shape=in_shape,
                 first_channels=args.in_channels,
                 drop_rate=args.drop_rate,
                 drop_type=args.drop_type,
                 act=args.activation,
                 pre_act=True).to(args.device)

    summary(net, (3, 96, 96, 32))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution("Training Set",
                            ["outcome"],
                            trainset.labels_bincount())
    print_data_distribution("Validation Set",
                            ["outcome"],
                            validset.labels_bincount())
    print_data_distribution(f"{testset_name.capitalize()} Set",
                            ["outcome"],
                            testset.labels_bincount())
    if not args.extra_data:
        print_data_distribution(f"{testset2_name.capitalize()} Set",
                                ["outcome"],
                                testset2.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(early_stopping=args.early_stopping,
                      save_path=SAVE_PATH,
                      loss=args.loss,
                      tol=0.2,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
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
                retrain=True)

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

        experiment.set_name("ResNet3D" + "_" + args.task)
        experiment.log_code("SingleTaskMain.py")
        experiment.log_code("Trainer/Trainer.py")
        experiment.log_code("Trainer/SingleTaskTrainer.py")
        experiment.log_code("Model/ResNet.py")
    else:
        experiment = None

    conf, auc = trainer.score(validset)
    print_score(dataset_name="VALIDATION",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    conf, auc = trainer.score(testset)
    print_score(dataset_name=f"{testset_name.upper()} TEST",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc],
                experiment=experiment)

    if not args.extra_data:
        conf, auc = trainer.score(testset2)
        print_score(dataset_name=f"{testset2_name.upper()} TEST",
                    task_list=[args.task],
                    conf_mat_list=[conf],
                    auc_list=[auc],
                    experiment=experiment)

    if experiment is not None:
        hparam = vars(args)
        save_hparam_on_comet(experiment=experiment, args_dict=hparam)
        experiment.log_parameter("seed", seed)
