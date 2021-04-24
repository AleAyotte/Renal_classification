"""
    @file:              Transfer_in_hdf5.py
    @Author:            Alexandre Ayotte

    @Creation Date:     12/2020
    @Last modification: 01//2021

    @Description:       This script has been used to read the clinical data of each Patient and saving those data with
                        the images of each Patient in a hdf5 file by using the class Patient.
"""
import pandas as pd
from Patient import Patient
from tqdm.auto import tqdm


CSV_FILE = "RenalDataset2.csv"
PATH_IMAGES = "temp_images/"
DTSET = "train"


data = pd.read_csv(CSV_FILE)
other_col = list(data.columns)[2:]

# We create the patient list with their metadata
patient_list = {}
for index, row in data.iterrows():
    pat_id = row["PatientID"]

    patient_list[pat_id] = {}
    for col in other_col:
        patient_list[pat_id][col] = row[col]

# We save the images and the metadata into a single hdf5 file
for pat_id in tqdm(list(patient_list.keys()), leave=False):
    try:
        pat = Patient(pat_id, PATH_IMAGES)
        pat.save_in_hdf5("dataset.hdf5", DTSET,
                         metadata=patient_list[pat_id],
                         merge_roi=False)
    except Exception as e:
        continue
