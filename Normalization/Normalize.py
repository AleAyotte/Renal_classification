"""
    @file:              Normalize.py
    @Author:            Alexandre Ayotte

    @Creation Date:     10/2020
    @Last modification: 04/2021

    @Description:       This script has been use has a pipeline to normalize a set of 3D images by using the Patient
                        class. Take note that the images were then saved in hdf5 with Transfer_in_hdf5.py.
"""
import os
from Patient import Patient
from tqdm import trange
from Utils import get_temporary_files


PATH = "/home/alex/Data/Corrected/"
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
NB_PATIENT = [25, 833, 112, 56, 118, 50]

VOXEL_SIZE = [1.03, 1.00, 2.90]
CROP_SHAPE = [96, 96, 32]
SAVE_PATH = "temp_images"

tempfiles = set(get_temporary_files(folder_path="/tmp"))

# *************************************************
#        Resample, crop and normalize image
# *************************************************

for i in range(len(institution)):
    for j in trange(1, NB_PATIENT[i]+1):
        patient_id = f"{institution[i]}-{j:03d}"
        try:
            pat = Patient(patient_id, PATH)

            pat.resample_and_crop(resample_params=VOXEL_SIZE,
                                  crop_shape=CROP_SHAPE,
                                  interp_type=0,
                                  threshold=100,
                                  save=False,
                                  register=False,
                                  ponderate_center=False,
                                  save_path=SAVE_PATH)
            pat.apply_znorm(save=False, save_path=SAVE_PATH)
            pat.save_images(save_path=SAVE_PATH, with_roi=True)

        except Exception as e:
            print(f"Problem with : {patient_id}. {e}")

# *************************************************
#                 Delete temp files
# *************************************************
new_tempfiles = set(get_temporary_files(folder_path="/tmp"))
for tempfile in list(new_tempfiles.difference(tempfiles)):
    os.remove(tempfile)