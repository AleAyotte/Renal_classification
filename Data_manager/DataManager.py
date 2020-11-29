import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class RenalDataset(Dataset):
    """
    Renal classification dataset.
    """
    def __init__(self, hdf5_filepath: str, split: str = "train", transform=None, load_clinical: bool = False):
        """
        Create a dataset by loading the renal image at the given path.

        :param hdf5_filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        :param transform: A function/transform that will be applied on the images and the ROI.
        :param load_clinical: A boolean that indicate if dataset will also contain the clinical data.
        """

        self.transform = transform
        self.__with_clinical = load_clinical
        self.__data = []
        self.__clinical_data = []
        self.__label = []

        self.__read_hdf5(hdf5_filepath, split)

    def __read_hdf5(self, filepath, split):
        """
        Read the images and ROI from a given hdf5 filepath and store it in memory

        :param filepath: The filepath of the hdf5 file where the data has been stored.
        :param split: A string that indicate which subset will be load. (Option: train, test, test2).
        """
        f = h5py.File(filepath, 'r')
        dtset = f[split]
        data = []
        clinical_data = []

        for key in list(dtset.keys()):
            self.__data.append({"t1": dtset[key]["t1"][:],
                                "t2": dtset[key]["t2"][:],
                                "roi": dtset[key]["roi"][:]})
            self.__label.append(dtset[key].attrs["outcome"])

            if self.__with_clinical:
                attrs = dtset[key].attrs
                clinical_data.append([attrs[key] for key in list(attrs.keys) if key != "outcome"])

        self.__clinical_data = np.array(data)
        f.close()

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.transform(self.__data[idx])

        if type(idx) == int:
            if len(sample["t1"].size()) == 4:
                sample = torch.cat((sample["t1"], sample["t2"], sample["roi"]), 0)
            else:
                sample = torch.stack([sample["t1"], sample["t2"], sample["roi"]])
        else:
            temp = []
            for samp in sample:
                if len(samp["t1"].size()) == 4:
                    temp.append(torch.cat((samp["t1"], samp["t2"], samp["roi"]), 0))
                else:
                    temp.append(torch.stack([samp["t1"], samp["t2"], samp["roi"]]))

            sample = torch.stack(temp)

        return sample, torch.tensor(self.__label[idx]).long()


def get_dataloader(dataset: RenalDataset, bsize: int = 32,
                   validation_split: float = 0.2, random_seed: int = 0):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                               sampler=valid_sampler)

    return train_loader, valid_loader
