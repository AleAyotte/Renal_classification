"""
    @file:              CrossStitchMain.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 03/2021

    @Description:       Contain the main function to train a SharedMet for multitask learning.
"""
import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet import ResNet
from Model.SharedNet import SharedNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from Utils import print_score, print_data_distribution


SAVE_PATH = "save/14_Fev_2020/"
TASK_LIST = ["Malignancy", "Subtype", "Subtype|Malignancy"]


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32,
                        help="The batch size.")
    parser.add_argument('--dataset', type=str, default='New_Option1',
                        help="The name of the folder that contain the dataset.",
                        choices=['Option1_with_N4', 'Option1_without_N4', "New_Option1"])
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device on which the model will be trained.")
    parser.add_argument('--eps', type=float, default=1e-3,
                        help="The epsilon hyperparameter of the Adam optimizer and the Novograd optimizer.")
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help="The minimal value of the learning rate.")
    parser.add_argument('--extra_data', type=bool, default=False, nargs='?', const=True,
                        help="If true, the second testest will be add to the training dataset. "
                             "The second dataset is determined with '--testset'.")
    parser.add_argument('--grad_clip', type=float, default=2.25,
                        help="The gradient clipping hyperparameter. Represent the maximal norm of the gradient during "
                             "the training.")
    parser.add_argument('--loss', type=str, default="ce",
                        help="The loss that will be use to train the model. 'ce' == cross entropy loss, "
                             "'bce' == binary cross entropoy, 'focal' = focal loss",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="The initial learning rate")
    parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 2],
                        help="The alpha parameter of each mixup module. Those alpha parameter are used to sample "
                             "the dristribution Beta(alpha, alpha).")
    parser.add_argument('--mode', type=str, default="standard",
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
    parser.add_argument('--pretrained', type=bool, default=False, nargs='?', const=True,
                        help="If True, then the SharedNet will be create with two subnet that has been pretrained on "
                             "their corresponding task. Also, the shared_lr will be equal to lr * 100 and "
                             "shared_eta_min will be equal to eta_min * 100.")
    parser.add_argument('--share_unit', type=str, default="sluice",
                        help="The sharing unit that will be used to create the SharedNet. The shared unit allow "
                             "information transfer between multiple subnets",
                        choices=["sluice", "cross_stitch"])
    parser.add_argument('--testset', type=str, default="stratified",
                        help="The name of the first testset. If 'testset'== stratified then the first testset will be "
                             "the stratified dataset and the independant will be the second and hence could be used as "
                             "extra data.",
                        choices=["stratified", "independant"])
    parser.add_argument('--track_mode', type=str, default="all",
                        help="Determine the quantity of training statistics that will be saved with tensorboard."
                             "If low, the training loss will be saved only at each epoch and not at each iteration.",
                        choices=["all", "low", "none"])
    parser.add_argument('--warm_up', type=int, default=0,
                        help="Number of epoch before activating the mixup if 'mode' == mixup")
    parser.add_argument('--weights', type=str, default="balanced",
                        help="The weight that will be applied on each class in the training loss. If balanced, "
                             "the classes weights will be ajusted in the training.",
                        choices=["flat", "balanced", "focused"])
    parser.add_argument('--worker', type=int, default=0,
                        help="Number of worker that will be used to preprocess data.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    if args.mode == "Mixup":
        raise NotImplementedError

    data_path = "final_dtset/{}/all.hdf5".format(args.dataset)
    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
        RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
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
    # TODO: CHANGE THE testset name in the hdf5 file
    # "test" is the stratified test and test2 is the independent test.
    test1, test2 = ("test", "test2") if args.testset == "stratified" else ("test2", "test")
    testset_name = args.testset
    testset2_name = "independant" if args.testset == "stratified" else "stratified"

    trainset = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2", "roi"])
    validset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2", "roi"], split=None)
    testset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2", "roi"], split=test1)

    # If we want to use some extra data, then will we used the data of the second test set.
    if args.extra_data:
        testset2 = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2", "roi"], split="test2")
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
    in_shape = tuple(trainset[0]["sample"].size()[1:])
    mal_net = ResNet(mixup=args.mixup,
                     depth=34,
                     first_channels=32,
                     in_shape=in_shape,
                     drop_rate=0.1,
                     drop_type="linear",
                     act="LeakyReLU",
                     pre_act=True).to(args.device)

    sub_net = ResNet(mixup=args.mixup,
                     depth=18,
                     first_channels=24,
                     in_shape=in_shape,
                     drop_rate=0.1,
                     drop_type="linear",
                     act="LeakyReLU",
                     pre_act=True).to(args.device)

    if args.pretrained:
        mal_net.restore(SAVE_PATH + "malignant.pth")
        sub_net.restore(SAVE_PATH + "subtype.pth")

    net = SharedNet(malignant_net=mal_net,
                    subtype_net=sub_net,
                    mixup=args.mixup,
                    subspace_1=[4, 3],
                    subspace_2=[8, 6],
                    subspace_3=[8, 6],
                    subspace_4=[4, 3],
                    c=0.85,
                    spread=0.1).to(args.device)
    summary(net, (3, 96, 96, 32))

    # --------------------------------------------
    #                SANITY CHECK
    # --------------------------------------------
    print_data_distribution("Training Set",
                            TASK_LIST,
                            trainset.labels_bincount())
    print_data_distribution("Validation Set",
                            TASK_LIST,
                            validset.labels_bincount())
    print_data_distribution("{} Set".format(testset_name.capitalize()),
                            TASK_LIST,
                            testset.labels_bincount())
    if not args.extra_data:
        print_data_distribution("{} Set".format(testset2_name.capitalize()),
                                TASK_LIST,
                                testset2.labels_bincount())
    print("\n")

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    trainer = Trainer(save_path="Check_moi_ca2.pth",
                      loss=args.loss,
                      tol=1.00,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
                      classes_weights=args.weights,
                      shared_net=False,
                      track_mode=args.track_mode,
                      mixed_precision=True)

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                mode=args.mode,
                learning_rate=args.lr,
                eta_min=args.eta_min,
                shared_lr=args.lr * 100 if args.pretrained else args.lr,
                shared_eta_min=args.eta_min * 100 if args.pretrained else args.eta_min,
                grad_clip=args.grad_clip,
                warm_up_epoch=args.warm_up,
                eps=args.eps,
                batch_size=args.b_size,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch,
                l2=0.009)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    conf, auc = trainer.score(validset)
    print_score(dataset_name="VALIDATION SCORE",
                task_list=TASK_LIST,
                conf_mat_list=conf,
                auc_list=auc)

    conf, auc = trainer.score(testset)
    print_score(dataset_name="{} TEST SCORE".format(testset_name.upper()),
                task_list=TASK_LIST,
                conf_mat_list=conf,
                auc_list=auc)

    if not args.extra_data:
        conf, auc = trainer.score(testset2)
        print_score(dataset_name="{} TEST SCORE".format(testset2_name.upper()),
                    task_list=TASK_LIST,
                    conf_mat_list=conf,
                    auc_list=auc)
