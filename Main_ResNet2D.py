import argparse
from Data_manager.DataManager import RenalDataset, split_trainset
from Model.ResNet_2D import ResNet2D
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd
import numpy as np
import torch
from torchsummary import summary
from Trainer.SingleTaskTrainer import SingleTaskTrainer as Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, default=32)
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_type', type=str, default="flat",
                        choices=["flat", "linear"])
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--extra_data', type=bool, default=False)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', type=bool, default=False)
    parser.add_argument('--mixup', type=int, action='store', nargs="*", default=[0, 2, 2, 2])
    parser.add_argument('--mode', type=str, default="standard",
                        choices=["standard", "Mixup"])
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--optim', type=str, default="adam",
                        choices=["adam", "novograd"])
    parser.add_argument('--pad_mode', type=str, default="constant",
                        choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--task', type=str, default="grade",
                        choices=["malignant", "subtype", "grade"])
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

    data_path = "dataset_2D/Data_without_N4/{}.hdf5".format(args.task)

    transform = Compose([
        AddChanneld(keys=["t1", "t2"]),
        RandFlipd(keys=["t1", "t2"], spatial_axis=[0, 1], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandAffined(keys=["t1", "t2"], prob=0.5, shear_range=10,
                    rotate_range=6.28, translate_range=0.1),
        # RandSpatialCropd(keys=["t1", "t2"], roi_size=86, random_center=False),
        RandZoomd(keys=["t1", "t2"], prob=0.5, min_zoom=0.95, max_zoom=1.05,
                  keep_size=True),
        # ResizeWithPadOrCropd(keys=["1", "t2"], spatial_size=[128, 128], mode=args.pad_mode),
        ToTensord(keys=["t1", "t2"])
    ])

    test_transform = Compose([
        AddChanneld(keys=["t1", "t2"]),
        ToTensord(keys=["t1", "t2"])
    ])
    
    clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis", "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]
    trainset = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2"],
                            clinical_features=clin_features)
    validset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2"], split=None,
                            clinical_features=clin_features)
    testset = RenalDataset(data_path, transform=test_transform, imgs_keys=["t1", "t2"], split="test",
                           clinical_features=clin_features)

    if args.extra_data:
        trainset2 = RenalDataset(data_path, transform=transform, imgs_keys=["t1", "t2"], split="test2",
                                 clinical_features=clin_features)
        data, label, _ = trainset2.extract_data(np.arange(len(trainset2)))
        trainset.add_data(data, label)
        del data
        del label

    trainset, validset = split_trainset(trainset, validset, validation_split=0.2)

    in_shape = tuple(trainset[0]["sample"].size()[1:])
    net = ResNet2D(drop_rate=0.5,
                   nb_clinical_data=len(clin_features)).to(args.device)

    # summary(net, (3, 96, 96, 32))

    trainer = Trainer(save_path="Check_moi_ca2.pth",
                      loss=args.loss,
                      tol=0.05,
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

    conf = trainer.score(testset, 32)
    recall = compute_recall(conf)

    print("Recall: ", recall)
