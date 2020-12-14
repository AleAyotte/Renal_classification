from Data_manager.DataManager import RenalDataset
from matplotlib import pyplot as plt
from Model.ResNet import MultiLevelResNet
from monai.transforms import RandFlipd, ScaleIntensityd, ToTensord, Compose, AddChanneld
from torchsummary import summary
from Trainer.Trainer import Trainer


data_path = "final_dtset/Option1_with_N4/all.hdf5"

transform = Compose([
    AddChanneld(keys=["t1", "t2", "roi"]),
    RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0, 1], prob=0.5),
    ScaleIntensityd(keys=["t1", "t2"]),
    ToTensord(keys=["t1", "t2", "roi"])
    ])

trainset = RenalDataset(data_path, transform=transform)
testset = RenalDataset(data_path, split="test")

net = MultiLevelResNet(mixup=[0., 2., 2., 2.],
                       in_shape=(96, 96, 32)).to("cuda:0")

summary(net, (3, 96, 96, 32))

trainer = Trainer(save_path="Check_moi_ca.pt")
trainer.fit(model=net, trainset=trainset, mode="mixup", grad_clip=5)