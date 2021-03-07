"""
    @file:              Main_ResNet2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 02/2021

    @Description:       Contain the main function to train a 2D ResNet on one of the three task
                        (malignancy, subtype and grade prediction).
"""

import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet_2D import ResNet2D
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
import torch
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Utils import print_score, print_data_distribution


SAVE_PATH = "save/STL2D_NET.pth"  # Save path of the single task learning with ResNet2D experiment


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
    parser.add_argument('--pad_mode', type=str, default="constant",
                        help="How the image will be pad in the data augmentation.",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False, nargs='?', const=True,
                        help="The pin_memory parameter of the dataloader. If true, the data will be pinned in the gpu.")
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
    data_path = "dataset_2D/Data_with_N4/{}.hdf5".format(args.task)

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        RandFlipd(keys=["t1", "t2"], spatial_axis=[0], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandAffined(keys=["t1", "t2"], prob=0.8, shear_range=0.5,
                    rotate_range=6.28, translate_range=0.1),
        RandZoomd(keys=["t1", "t2"], prob=0.5, min_zoom=0.95, max_zoom=1.05,
                  keep_size=False),
        ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=[224, 224], mode=args.pad_mode),
        ToTensord(keys=["t1", "t2"])
    ])

    test_transform = Compose([
        AddChanneld(keys=["t1", "t2"]),
        ToTensord(keys=["t1", "t2"])
    ])

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    # TODO: CHANGE THE testset name in the hdf5 file
    # "test" is the stratified test and test2 is the independent test.
    test1, test2 = ("test", "test2") if args.testset == "stratified" else ("test2", "test")
    testset_name = args.testset
    testset2_name = "independant" if args.testset == "stratified" else "stratified"

    if args.task in ["subtype", "grade"]:
        clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis", "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]
    else:
        clin_features = ["Age", "Sex", "size"]

    trainset = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2"],
                            clinical_features=clin_features)
    validset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2"], split=None,
                            clinical_features=clin_features)
    testset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2"], split=test1,
                           clinical_features=clin_features)

    # If we want to use some extra data, then will we used the data of the second test set.
    if args.extra_data:
        testset2 = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2"], split=test2,
                                clinical_features=clin_features)
        data, label, clin = testset2.extract_data(np.arange(len(testset2)))
        trainset.add_data(data, label, clin)
        del data
        del label
        del clin
    # Else the second test set will be used to access the performance of the dataset at the end.
    else:
        testset2 = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2"], split=test2,
                                clinical_features=clin_features)

    trainset, validset = split_trainset(trainset, validset, validation_split=0.2)

    # --------------------------------------------
    #             NORMALIZE THE DATA
    # --------------------------------------------
    mean, std = trainset.normalize_clin_data(get_norm_param=True)
    validset.normalize_clin_data(mean=mean, std=std)
    testset.normalize_clin_data(mean=mean, std=std)
    testset2.normalize_clin_data(mean=mean, std=std) if not args.extra_data else None

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
    print_data_distribution("{} Set".format(testset_name.capitalize()),
                            [args.task],
                            testset.labels_bincount())
    if not args.extra_data:
        print_data_distribution("{} Set".format(testset2_name.capitalize()),
                                [args.task],
                                testset2.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(early_stopping=args.early_stopping,
                      save_path="Check_moi_ca2.pth",
                      loss=args.loss,
                      tol=3.00,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
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
                t_0=args.num_epoch)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    conf, auc = trainer.score(validset)
    print_score(dataset_name="VALIDATION SCORE",
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc])

    conf, auc = trainer.score(testset)
    print_score(dataset_name="{} TEST SCORE".format(testset_name.upper()),
                task_list=[args.task],
                conf_mat_list=[conf],
                auc_list=[auc])

    if not args.extra_data:
        conf, auc = trainer.score(testset2)
        print_score(dataset_name="{} TEST SCORE".format(testset2_name.upper()),
                    task_list=[args.task],
                    conf_mat_list=[conf],
                    auc_list=[auc])
