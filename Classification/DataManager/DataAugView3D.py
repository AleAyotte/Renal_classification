"""
    @file:              DataAugView3D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     02/2021
    @Last modification: 09/2021

    @Description:       This file is used to visualize the transformation on the 3D data. Usefull to the determine the
                        DataAugmentation hyperparameter that should be used during the training.
"""

from matplotlib import pyplot as plt
from monai.transforms import RandFlipd, RandScaleIntensityd, ToTensord, Compose, AddChanneld
from monai.transforms import RandSpatialCropd, RandZoomd, RandAffined, ResizeWithPadOrCropd, Rand3DElasticd

from Constant import DatasetName
from DataManager.Constant import *
from DataManager.BrainDataset import BrainDataset
from DataManager.RenalDataset import RenalDataset


if __name__ == "__main__":

    dataset = DatasetName.BMETS
    if dataset == DatasetName.RCC:
        crop_size = [64, 64, 16]
        data_path = RCC_4CHAN
        imgs_key_all = ["t1", "t2", "roi"]
        imgs_key_partial = ["t1", "t2"]
        pad_size = [96, 96, 32]
        slic = RCC_SLICE
    else:
        crop_size = [48, 48, 16]
        data_path = BMETS_B
        imgs_key_all = ["t1ce", "dose", "roi"]
        imgs_key_partial = ["t1ce"]
        pad_size = [64, 64, 32]
        slic = BMETS_SLICE

    # --------------------------------------------
    #              DATA AUGMENTATION
    # --------------------------------------------
    transform = Compose([
        AddChanneld(keys=imgs_key_all),
        RandFlipd(keys=imgs_key_all, spatial_axis=[0], prob=1),
        RandScaleIntensityd(keys=imgs_key_partial, factors=0.1, prob=1),
        # Rand3DElasticd(keys=imgs_key_all, sigma_range=(5, 8), magnitude_range=(100, 200), prob=1),
        # Rand3DElasticd(keys=imgs_key_all, sigma_range=(3, 3), magnitude_range=(15, 35), prob=1),
        RandAffined(keys=imgs_key_all, prob=1, shear_range=[0.46, 0.46, 0],
                    rotate_range=[0, 0, 6.28], translate_range=0.66, padding_mode="zeros"),
        RandSpatialCropd(keys=imgs_key_all, roi_size=crop_size, random_center=False),
        RandZoomd(keys=imgs_key_all, prob=1, min_zoom=0.77, max_zoom=1.23,
                  keep_size=False),
        ResizeWithPadOrCropd(keys=imgs_key_all, spatial_size=pad_size, mode="constant"),
        ToTensord(keys=imgs_key_all)
    ])
    test_transform = Compose([
        AddChanneld(keys=imgs_key_all),
        ToTensord(keys=imgs_key_all)
    ])

    if dataset == "rcc":
        trainset = RenalDataset(hdf5_filepath=data_path, transform=transform,
                                imgs_keys=imgs_key_all, tasks=["malignancy"])
        testset = RenalDataset(hdf5_filepath=data_path, transform=test_transform,
                               imgs_keys=imgs_key_all, tasks=["malignancy"])
    else:
        trainset = BrainDataset(hdf5_filepath=data_path, transform=transform,
                                imgs_keys=imgs_key_all, tasks=["are"])
        testset = BrainDataset(hdf5_filepath=data_path, transform=test_transform,
                               imgs_keys=imgs_key_all, tasks=["are"])
        trainset.prepare_dataset()
        testset.prepare_dataset()

    for i in range(len(trainset)):
        trans_images = trainset[i]["sample"][:, :, :, :].numpy()
        ori_images = testset[i]["sample"][:, :, :, :].numpy()

        slice_view = [[slic[0], slice(None), slice(None)],
                      [slice(None), slic[1], slice(None)],
                      [slice(None), slice(None), slic[2]]]

        # Compare images t1, t2 and roi in axial view
        fig0 = plt.figure(figsize=(24, 30))
        for k in range(3):
            fig0.add_subplot(2, 3, 1 + k, title="Transformed image")
            plt.set_cmap(plt.gray())
            plt.imshow(trans_images[k, :, :, slic[2]])

        for k in range(3):
            fig0.add_subplot(2, 3, 4 + k, title="Original image")
            plt.set_cmap(plt.gray())
            plt.imshow(ori_images[k, :, :, slic[2]])
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
