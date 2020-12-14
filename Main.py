from Data_manager.DataManager import RenalDataset
from Model.ResNet import MultiLevelResNet
from monai.transforms import RandFlipd, ScaleIntensityd, ToTensord, Compose
from Trainer.Trainer import Trainer

data_path = "/home/local/USHERBROOKE/ayoa2402/Maitrise/Renal_classification/final_dtset/Option2_without_N4/all.hdf5"

transform = Compose([
    RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[1, 2]),
    ScaleIntensityd(keys=["t1", "t2"]),
    ToTensord(keys=["t1", "t2", "roi"])
    ])

trainset = RenalDataset(data_path, transform=transform)
testset = RenalDataset(data_path, split=test)

data = trainset[8]
sample, labels = data["sample"], data["labels"]
print(type(sample))
print(type(labels))
# Net = MultiLevelResNet(mixup=[0., 2., 2., 2.])
