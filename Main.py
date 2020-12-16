import argparse
from Data_manager.DataManager import RenalDataset
from matplotlib import pyplot as plt
from Model.ResNet import MultiLevelResNet
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld, RandGaussianSharpend
from torchsummary import summary
from Trainer.Trainer import Trainer
from Trainer.Utils import compute_recall


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--mode', type=str, default="Mixup",
                        choices=["standard", "Mixup"])
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", "bce", "focal"])
    parser.add_argument('--weights', type=str, default="balanced",
                        choices=["flat", "balanced", "focused"])                        
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--optim', type=str, default="adam",
                        choices=["adam", "novograd"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_type', type=str, default="flat",
                        choices=["flat", "linear"])
    parser.add_argument('--track_mode', type=str, default="all",
                        choices=["all", "low", "none"])
    parser.add_argument('--b_size', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--pin_memory', type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    device = args.device

    data_path = "final_dtset/Option1_with_N4/all.hdf5"

    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0, 1], prob=0.5),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        # RandGaussianSharpend(keys=["t1", "t2"], prob=0.3),
        ToTensord(keys=["t1", "t2", "roi"])
        ])

    trainset = RenalDataset(data_path, transform=transform)
    testset = RenalDataset(data_path, split="test")

    net = MultiLevelResNet(mixup=[0., 2., 2., 2.],
                           in_shape=(96, 96, 32),
                           first_channels=args.in_channels,
                           drop_rate=args.dropout,
                           drop_type=args.drop_type).to(args.device)

    summary(net, (3, 96, 96, 32))

    trainer = Trainer(save_path="Check_moi_ca.pth", 
                      loss=args.loss,
                      num_workers=args.worker,
                      pin_memory=args.pin_memory,
                      classes_weights=args.weights,
                      track_mode=args.track_mode)

    trainer.fit(model=net, 
                trainset=trainset, 
                mode=args.mode,
                learning_rate=args.lr, 
                grad_clip=5,
                warm_up_epoch=0,
                eps=args.eps,
                batch_size=args.b_size,
                device=args.device,
                optim=args.optim)

    m_conf, s_conf, g_conf = trainer.score(testset, 32)

    m_acc = compute_recall(m_conf)
    s_acc = compute_recall(s_conf)
    g_acc = compute_recall(g_conf)
    
    print("m_acc: ", m_acc)
    print("s_acc: ", s_acc)
    print("g_acc: ", g_acc)
