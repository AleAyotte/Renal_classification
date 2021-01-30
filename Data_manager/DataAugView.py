from DataManager import RenalDataset
from matplotlib import pyplot as plt
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd, Rand2DElasticd


if __name__ == "__main__":
    DATA_PATH = "dataset_2D/Data_with_N4/grade.hdf5"
    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        RandFlipd(keys=["t1", "t2"], spatial_axis=[0, 1], prob=0),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0.5),
        RandAffined(keys=["t1", "t2"], prob=0, shear_range=0.5,
                    rotate_range=6.28, translate_range=0.1),
        RandSpatialCropd(keys=["t1", "t2"], roi_size=[148, 148], random_center=False),
        Rand2DElasticd(keys=["t1", "t2"], spacing=7, magnitude_range=(0, 1), prob=1),
        RandZoomd(keys=["t1", "t2"], prob=0, min_zoom=1, max_zoom=1.05,
                  keep_size=False),
        ResizeWithPadOrCropd(keys=["t1", "t2"], spatial_size=[224, 224], mode="constant"),
        ToTensord(keys=["t1", "t2"])
    ])

    test_transform = Compose([
        AddChanneld(keys=["t1", "t2"]),
        ToTensord(keys=["t1", "t2"])
    ])

    trainset = RenalDataset(DATA_PATH, transform=transform, imgs_keys=["t1", "t2"])
    testset = RenalDataset(DATA_PATH, transform=test_transform, imgs_keys=["t1", "t2"])

    for i in range(len(trainset)):
        fig = plt.figure(figsize=(24, 30))
        trans_images = trainset[i]["sample"][i % 2, :, :, :].numpy()
        ori_images = testset[i]["sample"][i % 2, :, :, :].numpy()

        for k in range(3):
            fig.add_subplot(2, 3, 1 + k, title="Transformed image")
            plt.set_cmap(plt.gray())
            plt.imshow(trans_images[k, :, :])

        for k in range(3):
            fig.add_subplot(2, 3, 4 + k, title="Original image")
            plt.set_cmap(plt.gray())
            plt.imshow(ori_images[k, :, :])

        plt.show()
        continu = input("Continuer: ")
        if continu == "n":
            break
