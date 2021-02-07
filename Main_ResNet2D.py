import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet_2D import ResNet2D
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld, Rand2DElasticd
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
import torch
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--extra_data', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--mixup', type=int, action='store', nargs="*", default=[0, 2, 2, 2])
    parser.add_argument('--mode', type=str, default="standard",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--optim', type=str, default="adam",
                        choices=["adam", "novograd"])
    parser.add_argument('--pad_mode', type=str, default="constant",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--task', type=str, default="grade",
                        choices=["malignant", "subtype", "grade"]),
    parser.add_argument('--testset', type=str, default="stratified",
                        choices=["stratified", "independant"], help="The testset used to access the model")
    parser.add_argument('--track_mode', type=str, default="all",
                        choices=["all", "low", "none"])
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--weights', type=str, default="balanced",
                        choices=["flat", "balanced", "focused"])
    parser.add_argument('--worker', type=int, default=0)
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
        # Rand2DElasticd(keys=["t1", "t2"], spacing=10, magnitude_range=(0, 1), prob=1),
        # RandSpatialCropd(keys=["t1", "t2"], roi_size=[148, 148], random_center=False),
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
    # "test" is the stratified test and test2 is the independant test.
    test1, test2 = ("test", "test2") if args.testset == "stratified" else ("test2", "test")
    clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis", "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]

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

    # --------------------------------------------
    #                NEURAL NETWORK
    # --------------------------------------------
    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = ResNet2D(drop_rate=args.dropout,
                   nb_clinical_data=len(clin_features)).to(args.device)

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(save_path="Check_moi_ca2.pth",
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
                mode=args.mode,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                grad_clip=5,
                warm_up_epoch=args.warm_up,
                eps=args.eps,
                batch_size=args.b_size,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    print("**************************************")
    print("**{:^34s}**".format("VALIDATION SCORE"))
    print("**************************************")
    conf, auc = trainer.score(validset, 2)
    recall = compute_recall(conf)
    print("AUC: ", auc)
    print("Recall: ", recall)

    test1_label = "STRATIFIED TEST SCORE" if test1 == "test" else "INDEPENDANT TEST SCORE"
    print("**************************************")
    print("**{:^34s}**".format(test1_label))
    print("**************************************")
    conf, auc = trainer.score(testset, 2)
    recall = compute_recall(conf)
    print("AUC: ", auc)
    print("Recall: ", recall)

    if not args.extra_data:
        test2_label = "INDEPENDANT TEST SCORE" if test1 == "test" else "STRATIFIED TEST SCORE"
        print("**************************************")
        print("**{:^34s}**".format(test2_label))
        print("**************************************")
        conf, auc = trainer.score(testset2, 2)
        recall = compute_recall(conf)
        print("AUC: ", auc)
        print("Recall: ", recall)
