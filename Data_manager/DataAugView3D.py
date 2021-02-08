from DataManager import RenalDataset
from matplotlib import pyplot as plt
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd, Rand3DElasticd


if __name__ == "__main__":
    DATA_PATH = "final_dtset/Option1_without_N4/malignant.hdf5"

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[0], prob=0),
        RandFlipd(keys=["t1", "t2", "roi"], spatial_axis=[1], prob=0),
        RandScaleIntensityd(keys=["t1", "t2"], factors=0.2, prob=0),
        # Rand3DElasticd(keys=["t1", "t2", "roi"], sigma_range=(5, 8), magnitude_range=(100, 200), prob=1),
        Rand3DElasticd(keys=["t1", "t2", "roi"], sigma_range=(3, 3), magnitude_range=(15, 35), prob=1),
        RandAffined(keys=["t1", "t2", "roi"], prob=1, shear_range=[0.3, 0.3, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0, padding_mode="zeros"),
        RandSpatialCropd(keys=["t1", "t2", "roi"], roi_size=[64, 64, 16], random_center=False),
        RandZoomd(keys=["t1", "t2", "roi"], prob=1, min_zoom=1, max_zoom=1.10,
                  keep_size=False),
        ResizeWithPadOrCropd(keys=["t1", "t2", "roi"], spatial_size=[96, 96, 32], mode="constant"),
        ToTensord(keys=["t1", "t2", "roi"])
    ])
    test_transform = Compose([
        AddChanneld(keys=["t1", "t2", "roi"]),
        ToTensord(keys=["t1", "t2", "roi"])
    ])

    trainset = RenalDataset(DATA_PATH, transform=transform, imgs_keys=["t1", "t2", "roi"])
    testset = RenalDataset(DATA_PATH, transform=test_transform, imgs_keys=["t1", "t2", "roi"])

    for i in range(len(trainset)):
        trans_images = trainset[i]["sample"][:, :, :, :].numpy()
        ori_images = testset[i]["sample"][:, :, :, :].numpy()

        slice_view = [[48, slice(None), slice(None)],
                      [slice(None), 48, slice(None)],
                      [slice(None), slice(None), 16]]

        # Compare images t1, t2 and roi in axial view
        fig0 = plt.figure(figsize=(24, 30))
        for k in range(3):
            fig0.add_subplot(2, 3, 1 + k, title="Transformed image")
            plt.set_cmap(plt.gray())
            plt.imshow(trans_images[k, :, :, 16])

        for k in range(3):
            fig0.add_subplot(2, 3, 4 + k, title="Original image")
            plt.set_cmap(plt.gray())
            plt.imshow(ori_images[k, :, :, 16])
        plt.show()

        # Show sagittal, coronal and axial slice.
        fig1 = plt.figure(figsize=(24, 30))
        for k in range(3):
            fig1.add_subplot(2, 3, 1 + k, title="Transformed image")
            plt.set_cmap(plt.gray())
            plt.imshow(trans_images[i % 2, slice_view[k][0], slice_view[k][1], slice_view[k][2]])

        for k in range(3):
            fig1.add_subplot(2, 3, 4 + k, title="Original image")
            plt.set_cmap(plt.gray())
            plt.imshow(ori_images[i % 2, slice_view[k][0], slice_view[k][1], slice_view[k][2]])
        plt.show()

        continu = input("Continuer: ")
        if continu == "n":
            break