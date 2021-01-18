import h5py
from typing import Sequence, Tuple, Union
from Utils import convert_3d_to_2d, get_group, update_dataset, update_attribute
import numpy as np


def transform_file(filepath: str,
                   new_filepath: str,
                   reshape_size: Sequence[int] = None,
                   reshape: bool = True,
                   apply_mask: bool = True) -> None:
    """
    Convert a hdf5 dataset of 3D images into a dataset of 2.5D images.

    :param filepath: The name of the hdf5 file.
    :param new_filepath: The name of the converted hdf5 file.
    :param reshape_size: The size of the reshaped 2.5D image (Default: [256, 256]).
    :param reshape: A boolean that indicate if the images must be reshape.
    :param apply_mask: If true, the pixel of image will 0 where the pixel of the mask are 0.
    """
    reshape_size = [256, 256] if reshape_size is None else reshape_size

    split = ["train", "test", "test2"]
    f = h5py.File(filepath, 'r')
    f2 = h5py.File(new_filepath, 'a')

    for split in split:
        dtset = f[split]
        new_dtset = get_group(f2, split)

        for key in list(dtset.keys()):
            t1, t2, roi = dtset[key]["t1"][:], dtset[key]["t2"][:], dtset[key]["roi"][:]
            new_imgs, _ = convert_3d_to_2d([t1, t2],
                                           [roi],
                                           apply_mask_on_img=apply_mask,
                                           reshape_size=reshape_size,
                                           reshape=reshape)

            # If the patient already exist in the new dataset
            patient = get_group(new_dtset, key)
            update_dataset(patient, "t1", new_imgs[0])
            update_dataset(patient, "t2", new_imgs[1])

            for attr, value in dtset[key].attrs.items():
                update_attribute(patient, attr, value)

    f.close()
    f2.close()


file = ["grade.hdf5", "malignant.hdf5", "subtype.hdf5"]
newfile = ["new_grade.hdf5", "new_malignant.hdf5", "new_subtype.hdf5"]

for file, newfile in zip(file, newfile):
    transform_file(file, newfile)
