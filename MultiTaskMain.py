import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet import MultiLevelResNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd, Rand3DElasticd
import numpy as np
from torchsummary import summary
from Trainer.MultiTaskTrainer import MultiTaskTrainer as Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
    parser.add_argument('--b_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='Option1_without_N4',
                        choices=['Option1_with_N4', 'Option1_without_N4',
                                 'Option2_with_N4', 'Option2_without_N4'])
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_type', type=str, default="flat",
                        choices=["flat", "linear"])
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--extra_data', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixup', type=float, action='store', nargs="*", default=[0, 2, 2, 2])
    parser.add_argument('--mode', type=str, default="Mixup",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--optim', type=str, default="adam",
                        choices=["adam", "novograd"])
    parser.add_argument('--pad_mode', type=str, default="constant",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--split_level', type=int, default=4)
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
    device = args.device

    data_path = "final_dtset/{}/new_all.hdf5".format(args.dataset)

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.1, prob=0.5),
        # Rand3DElasticd(keys=["t1", "t2", "roi"], sigma_range=(3, 3), magnitude_range=(15, 35), prob=0.5),
        RandAffined(keys=["t1", "t2", "roi"], prob=0.5, shear_range=[0.4, 0.4, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0, padding_mode="zeros"),
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
    # "test" is the stratified test and test2 is the independant test.
    test1, test2 = ("test", "test2") if args.testset == "stratified" else ("test2", "test")

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
    net = MultiLevelResNet(mixup=args.mixup,
                           depth=args.depth,
                           split_level=args.split_level,
                           in_shape=in_shape,
                           first_channels=args.in_channels,
                           drop_rate=args.dropout,
                           drop_type=args.drop_type,
                           act=args.activation).to(args.device)

    # --------------------------------------------
    #                   TRAINER
    # --------------------------------------------
    summary(net, (3, 96, 96, 32))
    trainer = Trainer(save_path="Check_moi_ca2.pth",
                      loss=args.loss,
                      tol=1.00,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
                      classes_weights=args.weights,
                      track_mode=args.track_mode,
                      mixed_precision=True)

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
                t_0=args.num_epoch,
                l2=1e-6)

    # --------------------------------------------
    #                    SCORE
    # --------------------------------------------
    print("**************************************")
    print("**{:^34s}**".format("VALIDATION SCORE"))
    print("**************************************")
    m_conf, s_conf, g_conf = trainer.score(validset, 32)

    m_acc = compute_recall(m_conf)
    s_acc = compute_recall(s_conf)
    g_acc = compute_recall(g_conf)

    print("m_acc: ", m_acc)
    print("s_acc: ", s_acc)
    print("g_acc: ", g_acc)

    test1_label = "STRATIFIED TEST SCORE" if test1 == "test" else "INDEPENDANT TEST SCORE"
    print("**************************************")
    print("**{:^34s}**".format(test1_label))
    print("**************************************")
    m_conf, s_conf, g_conf = trainer.score(testset, 32)

    m_acc = compute_recall(m_conf)
    s_acc = compute_recall(s_conf)
    g_acc = compute_recall(g_conf)

    print("m_acc: ", m_acc)
    print("s_acc: ", s_acc)
    print("g_acc: ", g_acc)

    if not args.extra_data:
        test2_label = "INDEPENDANT TEST SCORE" if test1 == "test" else "STRATIFIED TEST SCORE"
        print("**************************************")
        print("**{:^34s}**".format(test2_label))
        print("**************************************")
        m_conf, s_conf, g_conf = trainer.score(testset2, 32)

        m_acc = compute_recall(m_conf)
        s_acc = compute_recall(s_conf)
        g_acc = compute_recall(g_conf)

        print("m_acc: ", m_acc)
        print("s_acc: ", s_acc)
        print("g_acc: ", g_acc)
