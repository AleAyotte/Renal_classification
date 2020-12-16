import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Sequence, Tuple


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

            if "outcome" in list(dtset[key].attrs.keys()):
                self.__label.append(dtset[key].attrs["outcome"])
            else:
                self.__label.append({"malignant": dtset[key].attrs["malignant"],
                                     "subtype": dtset[key].attrs["subtype"],
                                     "grade": dtset[key].attrs["grade"]})
            if self.__with_clinical:
                attrs = dtset[key].attrs
                clinical_data.append([attrs[key] for key in list(attrs.keys) if key != "outcome"])

        self.__clinical_data = np.array(data)
        f.close()

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.transform(self.__data[idx])

        if type(idx) == int:
            # Stack the images into torch tensor
            if len(sample["t1"].size()) == 4:
                sample = torch.cat((sample["t1"], sample["t2"], sample["roi"]), 0)
            else:
                sample = torch.stack([sample["t1"], sample["t2"], sample["roi"]])

            # Transfer the label in a dictionnary of torch tensor
            if type(self.__label[idx]) == dict:
                labels = {}
                for key, value in self.__label[idx].items():
                    labels[key] = torch.tensor(value).long()

            # Transfer the label in a torch tensor
            else:
                labels = torch.tensor(self.__label[idx]).long()

        else:
            # Stack the images into torch tensor
            temp = []
            for samp in sample:
                if len(samp["t1"].size()) == 4:
                    temp.append(torch.cat((samp["t1"], samp["t2"], samp["roi"]), 0))
                else:
                    temp.append(torch.stack([samp["t1"], samp["t2"], samp["roi"]]))
            sample = torch.stack(temp)

            # Extract and stack the labels in a dictionnary of torch tensor
            if type(self.__label[idx][0]) == dict:
                labels = {}
                for key in list(self.__label[idx][0].keys()):
                    labels[key] = torch.tensor([
                        _dict[key] for _dict in self.__label[idx]
                    ]).long

            # Extract and stack the labels in a torch tensor
            else:
                labels = torch.tensor(self.__label[idx]).long()
        return {"sample": sample, "labels": labels}


def get_dataloader(dataset: RenalDataset, 
                   bsize: int = 32,
                   pin_memory: bool = False,
                   num_workers: int = 0,
                   validation_split: float = 0.2, 
                   random_seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Transform a dataset into a training dataloader and a validation dataloader.

    :param dataset: The dataset to split.
    :param bsize: The batch size.
    :param pin_memory: The pin_memory option of the DataLoader. If true, the data tensor will 
                       copied into the CUDA pinned memory. (Default=False)
    :param num_workers: Number of parallel process used for the preprocessing of the data. If 0, 
                        the main process will be used for the data augmentation. (Default: 0)
    :param validation_split: A float that indicate the percentage of the dataset that will be used to create the 
                             validation dataloader.
    :param random_seed: The seed that will be used to randomly split the dataset.
    :return: Two dataloader, one for training and one for the validation.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=bsize,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       sampler=train_sampler,
                                       drop_last=True)
    valid_loader = DataLoader(dataset, batch_size=bsize,
                                       sampler=valid_sampler)

    return train_loader, valid_loader
