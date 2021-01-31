"""
    @file:              Normalize2D.py
    @Author:            Alexandre Ayotte

    @Creation Date:     10/2020
    @Last modification: 01/2021

    @Description:       This script used to be the pipeline to normalize the images, transform them in 2.5D and save
                        then in a hdf5 file by using the Patient class.
"""
from itertools import product
from math import ceil
import os
from Patient import Patient
import ray
from tqdm import tqdm
from typing import Tuple, Union
from Utils import get_temporary_files, read_metadata
ray.init(include_dashboard=False)


# path_images = "/home/alex/Data/Corrected/"
path_images = "/home/alex/Data/n4_temp/"
temp_path = "/home/alex/Data/Temp/"
temp_save_path = "/home/alex/Maitrise/Renal_classification/Renal_classification/Normalization/temp_images"
exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357", 'Kidney-Penn-086',
           'Kidney-Penn-115', 'Kidney-Penn-125', 'Kidney-Penn-169', 'Kidney-Penn-329', 'Kidney-Penn-556',
           'Kidney-Penn-587', 'Kidney-Penn-722', 'Kidney-Penn-745', 'Kidney-Penn-753', 'Kidney-Penn-765',
           'Kidney-Penn-774', 'Kidney-Penn-788', 'Kidney-Penn-797', 'Kidney-TCGA-048', 'Kidney-TCGA-054']
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]

folder = "Option1_with_N4/"
voxel_size = [0.44, 0.43, 0.41]
crop_shape = [224, 224, 224]

save_path = "Option1_with_N4"
dtset = ["train", "test", "test2"]
task_list = ["malignant", "subtype", "grade"]

file_list = ["grade.hdf5", "malignant.hdf5", "subtype.hdf5"]
newfile_list = ["new_grade.hdf5", "new_malignant.hdf5", "new_subtype.hdf5"]

batch_size = 4


@ray.remote
def normalize_patient(pat_id: str) -> Union[Tuple[bool, str],
                                            Tuple[bool, str, str]]:
    """
    Normalize the images related to a given patient id.

    :param pat_id: The patient id as string. Used to create the patient object.
    :return: A boolean that indicate if the images has been succesfully normalized and saved. Also,
             the patient id is returned.
    """
    try:
        pat_ = Patient(pat_id, path_images)

        # pat_.apply_n4(save=False)

        pat_.resample_and_crop(resample_params=voxel_size,
                               crop_shape=crop_shape,
                               interp_type=0,
                               ponderate_center=False,
                               save=True,
                               save_path=temp_save_path)

        del pat_
        return True, pat_id

    except Exception as e:
        del pat_
        return False, pat_id, str(e)


def save_in_hdf5(pat_id: str,
                 pat_: Patient,
                 metadata_: dict) -> None:
    """
    Save a patient in the hdf5 files.

    :param pat_id: A string that represent the patient id. Used to acced the patient metadataé
    :param pat_: The patient object that will be used to save the images and metadata in hdf5.
    :param metadata_: The dictionnary of metadata.
    """
    for (t, savefile), dts in product(zip(task_list, newfile_list), dtset):
        if pat_id in list(metadata_[t][dts].keys()):
            pat_.save_in_hdf5(savefile, dts,
                              merge_roi=False,
                              apply_roi=True,
                              convert_in_2_5d=True,
                              save_roi=False,
                              metadata=metadata_[t][dts][pat_id])


tempfiles = set(get_temporary_files(folder_path="/tmp"))
# -------------------------------------
#            Read metadata
# -------------------------------------
metadata = {}

for task, file in zip(task_list, file_list):
    temp_dict = {}
    for split in dtset:
        temp_dict[split] = read_metadata(file, split)

    metadata[task] = temp_dict

# -------------------------------------
#        Generate patient list
# -------------------------------------
patient_list = []
for i in range(len(institution)):
    for j in range(1, nb_patient[i]+1):
        if j < 10:
            _nb = "00" + str(j)
        elif j < 100:
            _nb = "0" + str(j)
        else:
            _nb = str(j)

        patient_id = institution[i] + "-" + _nb

        patient_list.append(patient_id) if patient_id not in exclude else None

nb_left = len(patient_list)

with tqdm(total=ceil(len(patient_list) / batch_size)) as t:
    while nb_left > 0:
        cur_batch = min(batch_size, nb_left)

        # Normalize the images
        workers = [normalize_patient.remote(patient_list.pop(0)) for i in range(cur_batch)]

        for _ in range(cur_batch):
            # Collect the result
            worker, workers = ray.wait(workers, num_returns=1)
            result = ray.get(worker)[0]

            if result[0]:
                # Save the images in the hdf5 file.
                pat = Patient(result[1], temp_save_path)
                save_in_hdf5(result[1], pat, metadata)
            else:
                print(result[1] + " Problem: " + result[2])

        t.update()
        nb_left -= cur_batch
        # -------------------------------------
        #           Delete temp files
        # -------------------------------------
        new_tempfiles = set(get_temporary_files(folder_path="/tmp"))
        for tempfile in list(new_tempfiles.difference(tempfiles)):
            os.remove(tempfile)

        for tempfile in list(get_temporary_files(folder_path=temp_save_path)):
            os.remove(tempfile)
