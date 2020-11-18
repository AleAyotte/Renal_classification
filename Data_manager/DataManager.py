import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RenalDataset(Dataset):
    """
    Renal classification dataset.
    """
    def __init__(self, hdf5_filepath: str, split: str = "train", intensity_transform=None,
                 morph_transform=None, load_clinical: bool = False):
        """


        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        :param intensity_transform: A function/transform that will be applied on the images only.
        :param morph_transform: A function/transform that will be applied on the images and the ROI.
        :param load_clinical: A boolean that indicate if dataset will also contain the clinical data.
        """

        self.intens_trans = intensity_transform
        self.morph_trans = morph_transform
        self.__with_clinical = load_clinical
        self.__data = None
        self.__clinical_data = None
        self.__label = []

        self.__read_hdf5(hdf5_filepath, split)

    def __read_hdf5(self, filepath, split):
        """


        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        """
        f = h5py.File(filepath, 'r')
        dtset = f[split]
        data = []
        clinical_data = []

        for key in list(dtset.keys()):
            data.append([dtset[key]["t1"][:], dtset[key]["t2"][:], dtset[key]["roi"][:]])
            self.__label.append(dtset[key].attrs["outcome"])

            if self.__with_clinical:
                attrs = dtset[key].attrs
                clinical_data.append([attrs[key] for key in list(attrs.keys) if key != "outcome"])

        self.__data = np.array(data)
        self.__clinical_data = np.array(data)
        f.close()

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seed = np.random.randint(low=0)

        data = []
        for i in range(3):
            torch.manual_seed(seed)
            temp = self.morph_trans(self.__data[idx, i])
            data.append(temp if i == 2 else self.intens_trans(temp))

        sample = torch.stack(data)
        return sample.transpose if len(sample.shape) == 5 else sample
