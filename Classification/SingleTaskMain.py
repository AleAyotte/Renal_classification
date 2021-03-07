"""
    @file:              SingleTaskMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 03/2021

    @Description:       Contain the main function to train a 3D ResNet on one of the three tasks
                        (malignancy, subtype and grade prediction).
"""
import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet import ResNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
import torch
from torchsummary import summary
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="The activation function use in the NeuralNet.",
                        choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--dataset', type=str, default='New_Option1',
                        help="The name of the folder that contain the dataset.",
                        choices=['Option1_with_N4', 'Option1_without_N4', 'New_Option1'])
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
                        choices=["malignant", "subtype", "grade"])
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
    data_path = "final_dtset/{}/{}.hdf5".format(args.dataset, args.task)

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
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

    test_transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        ToTensord(keys=["t1", "t2", "roi"])
    ])

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    # "test" is the stratified test and test2 is the independent test.
    test1, test2 = ("test", "test2") if args.testset == "stratified" else ("test2", "test")

    trainset = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2", "roi"])
    validset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2", "roi"], split=None)
    testset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2", "roi"], split=test1)

    # If we want to use some extra data, then will we used the data of the second test set.
    if args.extra_data:
        testset2 = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2", "roi"], split=test2)
        data, label, _ = testset2.extract_data(np.arange(len(testset2)))
        trainset.add_data(data, label)
        del data
        del label
    # Else the second test set will be used to access the performance of the dataset at the end.
    else:
        testset2 = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2", "roi"], split=test2)

    trainset, validset = split_trainset(trainset, validset, validation_split=0.2)

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
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(save_path="Check_moi_ca2.pth",
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
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch,
                retrain=False)

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

    test1_label = "STRATIFIED TEST SCORE" if test1 == "test" else "INDEPENDENT TEST SCORE"
    print("**************************************")
    print("**{:^34s}**".format(test1_label))
    print("**************************************")
    conf, auc = trainer.score(testset, 2)
    recall = compute_recall(conf)
    print("AUC: ", auc)
    print("Recall: ", recall)

    if not args.extra_data:
        test2_label = "INDEPENDENT TEST SCORE" if test1 == "test" else "STRATIFIED TEST SCORE"
        print("**************************************")
        print("**{:^34s}**".format(test2_label))
        print("**************************************")
        conf, auc = trainer.score(testset2, 2)
        recall = compute_recall(conf)
        print("AUC: ", auc)
        print("Recall: ", recall)
