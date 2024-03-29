"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte

    @Creation Date:     10/2020
    @Last modification: 01//2021

    @Description:       This file contain several usefull function for the normalization that cannot be add to a class.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Sequence, Tuple, Union
from skimage.transform import resize


def convert_3d_to_2d(imgs: Union[Sequence[np.array], np.array],
                     masks: Union[Sequence[np.array], np.array],
                     reshape_size: Sequence[int] = None,
                     reshape: bool = False,
                     apply_mask_on_img: bool = True) -> Sequence[np.array]:
    """
    Converte one or many single channel 3D images into 3 channels 2.5D images where the 3 channels represent the
    axial, corronal and sagittal view. Apply the same transformation on the maks.

    :param imgs: An image or a list of image to convert in 2.5D.
    :param masks:A mask or a list of mask to convert in 2.5D
    :param reshape_size: The size of the reshaped 2.5D image (Default: [256, 256]).
    :param reshape: A boolean that indicate if the images must be reshape.
    :param apply_mask_on_img: If true, the pixel of image will 0 where the pixel of the mask are 0.
    :return: A list of images.
    """
    imgs = [imgs] if type(imgs) is not list else imgs
    masks = [masks] if type(masks) is not list else masks

    if len(imgs) == 2*len(masks):
        masks = [m for m in masks for _ in range(2)]
    else:
        assert len(imgs) == len(masks), "There should be as many or twice more images as masks."\
                                       "\n nb imgs: {}, nb masks {}".format(len(imgs), len(masks))

    new_imgs = []
    new_masks = []

    for img, mask in zip(imgs, masks):

        # Verify the shape of the images
        img, mask = np.squeeze(img), np.squeeze(mask)
        img_shape = np.array(np.shape(img))
        assert len(img_shape) == 3, "One or many image are not in 3D with one channel"

        # Apply the mask on the image
        if apply_mask_on_img:
            img = np.where(mask > 0, img, 0)

        # Extract the middle slice
        mid_slice = np.floor(img_shape / 2).astype(int)

        if reshape:
            new_img = np.array(
                [resize(img[mid_slice[0], :, :], output_shape=reshape_size, preserve_range=True),
                 resize(img[:, mid_slice[1], :], output_shape=reshape_size, preserve_range=True),
                 resize(img[:, :, mid_slice[2]], output_shape=reshape_size, preserve_range=True)]
            )

            new_mask = np.array(
                [resize(mask[mid_slice[0], :, :], output_shape=reshape_size, preserve_range=True),
                 resize(mask[:, mid_slice[1], :], output_shape=reshape_size, preserve_range=True),
                 resize(mask[:, :, mid_slice[2]], output_shape=reshape_size, preserve_range=True)]
            )
        else:
            new_img = np.array([img[mid_slice[0], :, :],
                                img[:, mid_slice[1], :],
                                img[:, :, mid_slice[2]]])

            new_mask = np.array([mask[mid_slice[0], :, :],
                                mask[:, mid_slice[1], :],
                                mask[:, :, mid_slice[2]]])

        new_mask = np.where(new_mask > 0.9, 1, 0)

        new_imgs.append(new_img)
        new_masks.append(new_mask)

    return new_imgs, new_masks


def get_temporary_files(folder_path: str, extension: str = "nii.gz") -> Sequence[str]:
    """
    List all files with a given extension in a given directory. Can be used to list the temporary
    files created by antspy.

    :param folder_path: The path to the temporary files.
    :param extension: Only the files that finish with this extension will be return.
    :return: A list of string that correspond to a list of filepath.
    """
    filelist = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            filelist.append(str(os.path.join(folder_path, file)))

    return filelist


def get_group(parent: Union[h5py.File, h5py.Group],
              key: str) -> h5py.Group:
    """
    Return a group in a given h5py file or group. If the group doesn't exist, it will be create.

    :param parent: The parent h5py file or group of the seeked group.
    :param key: The name of the group.
    :return: A h5py group
    """
    if key in list(parent.keys()):
        return parent[key]
    else:
        return parent.create_group(key)


def read_metadata(filename: str,
                  group: str) -> dict:
    """
    Read the attributes of all patient that are member of a given group in a given hdf5 file.

    :param filename:The path of the hdf5 file.
    :param group:The name of the group.
    :return: A dictionnary of dictonnary where keys are the PatientID.
    """
    f = h5py.File(filename, 'r')
    group = f[group]
    patient_list = list(group.keys())

    metadata = {}
    for patient in patient_list:
        patient_data = {}
        for attr, value in group[patient].attrs.items():
            patient_data[attr] = value
        metadata[patient] = patient_data

    return metadata


def rotate_and_compare(img: np.array,
                       ref_img: np.array,
                       rotation_set: int = 0,
                       plot: bool = False) -> Tuple[np.array, bool]:
    """
    Apply a set of rotation to an image and verify if the image fit with a given image of reference.

    :param img: A numpy array that represent the image to rotate.
    :param ref_img: A numpy array that represent the image of reference to match.
    :param rotation_set: A integer that indicate the set of rotation to use on the image.
    :param plot: if true, plot the images.
    :return: The rotated image as np.array and a boolean that indicate if the rotated image fit with the
             image of reference.
    """
    if rotation_set == 0:
        rotated_img = np.swapaxes(img, 0, 2)
        rotated_img = np.rot90(rotated_img, 1, (0, 2))
        rotated_img = np.flip(rotated_img, 0)

    elif rotation_set == 1:
        rotated_img = np.swapaxes(img, 1, 2)
        rotated_img = np.flip(rotated_img, 1)

    elif rotation_set == 2:
        rotated_img = img

    else:
        raise NotImplementedError

    if np.shape(rotated_img) != np.shape(ref_img):
        return rotated_img, False

    else:
        diff = np.abs(rotated_img - ref_img) / max(np.max(rotated_img), np.max(ref_img))

        if plot:
            print(np.mean(rotated_img))
            print(np.max(rotated_img))
            print(np.min(rotated_img))

            print("\n", np.mean(ref_img))
            print(np.max(ref_img))
            print(np.min(ref_img))

            print("\n", np.mean(diff))
            print(np.max(diff))
            print(np.unravel_index(np.argmax(diff), diff.shape))

            fig = plt.figure(figsize=(24, 30))
            plt.set_cmap(plt.gray())

            fig.add_subplot(1, 2, 1, title="Image 1")
            plt.imshow(rotated_img[:, :, 10])

            fig.add_subplot(1, 2, 2, title="Image 2")
            plt.imshow(ref_img[:, :, 10])

            plt.show()

        return rotated_img, (diff < 1e-2).all()


def update_dataset(group: h5py.Group,
                   key: str,
                   new_data: np.array) -> None:
    """
    Update the value of a dataset in a given group. If the dataset doesn't exist, it will be create.

    :param group: The group that own the dataset.
    :param key: The name of the dataset.
    :param new_data: A numpy array that represent the new dataset.
    """
    if key in list(group.keys()):
        group[key][:] = new_data
    else:
        group.create_dataset(key, data=new_data)


def update_attribute(group: h5py.Group,
                     key: str,
                     new_data) -> None:
    """
    Update the value of a attribute in a given group. If the attribute doesn't exist, it will be create.

    :param group: The group that own the dataset.
    :param key: The name of the attribute.
    :param new_data: The new value of the attribute to update.
    """
    if key in list(group.attrs.keys()):
        group.attrs[key] = new_data
    else:
        group.attrs.create(key, new_data)
