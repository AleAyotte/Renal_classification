import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from matplotlib import pyplot as plt
from Model.ResNet import ResNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld, RandGaussianSharpend
from monai.transforms import RandSpatialCropd, SpatialPadd
import numpy as np
from torchsummary import summary
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'PReLU', 'LeakyReLU', 'Swish', 'ELU'])
    parser.add_argument('--b_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='Option1_with_N4',
                        choices=['Option1_with_N4', 'Option1_without_N4',
                                 'Option2_with_N4', 'Option2_without_N4'])
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_type', type=str, default="flat",
                        choices=["flat", "linear"])
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--extra_data', type=bool, default=False)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixup', type=int, action='store', nargs="*", default=[0, 2, 2, 2])
    parser.add_argument('--mode', type=str, default="Mixup",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--optim', type=str, default="adam",
                        choices=["adam", "novograd"])
    parser.add_argument('--pad_mode', type=str, default="constant",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--task', type=str, default="malignant",
                        choices=["malignant", "subtype", "grade"])
    parser.add_argument('--track_mode', type=str, default="all",
                        choices=["all", "low", "none"])
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--worker', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    device = args.device

    data_path = "final_dtset/{}/{}.hdf5".format(args.dataset, args.task)

    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0, 1], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
        SpatialPadd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode=args.pad_mode),
        ToTensord(keys=["t1", "t2", "roi"])
    ])

    test_transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        ToTensord(keys=["t1", "t2", "roi"])
    ])

    trainset = RenalDataset(data_path, transform=transform)
    validset = RenalDataset(data_path, transform=test_transform, split=None)
    testset = RenalDataset(data_path, transform=test_transform, split="test")

    if args.extra_data:
        trainset2 = RenalDataset(data_path, transform=transform, split="test2")
        data, label, _ = trainset2.extract_data(np.arange(len(trainset2)))
        trainset.add_data(data, label)
        del data
        del label

    trainset, validset = split_trainset(trainset, validset, validation_split=0.1)

    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = ResNet(mixup=args.mixup,
                 depth=args.depth,
                 in_shape=in_shape,
                 first_channels=args.in_channels,
                 drop_rate=args.dropout,
                 drop_type=args.drop_type,
                 act=args.activation,
                 pre_act=True).to(args.device)

    summary(net, (3, 96, 96, 32))
    # print(net)

    trainer = Trainer(save_path="Check_moi_ca2.pth",
                      loss=args.loss,
                      tol=0.05,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
                      task=args.task,
                      track_mode=args.track_mode)

    trainer.fit(model=net,
                trainset=trainset,
                validset=validset,
                mode=args.mode,
                learning_rate=args.lr,
                grad_clip=5,
                warm_up_epoch=args.warm_up,
                eps=args.eps,
                batch_size=args.b_size,
                device=args.device,
                optim=args.optim,
                num_epoch=args.num_epoch,
                t_0=args.num_epoch)

    conf = trainer.score(testset, 32)
    recall = compute_recall(conf)

    print("Recall: ", recall)
