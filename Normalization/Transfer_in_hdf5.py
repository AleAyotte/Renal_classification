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


path_images = "E:/WORKSPACE_RadiomicsComputation/Kidney/"
path_csv = "C:/Users/Alexandre/Desktop/Maitrise/Renal_classification/CSV/"
folder = ["Option1_with_N4/", "Option2_with_N4/", "Option1_without_N4/", "Option2_without_N4/"]
# folder = ["Option1_with_N4/", "Option1_without_N4/"]
task = ["malignant", "subtype", "grade"]
dtset = ["train", "test", "test2"]

"""
for f in tqdm(folder):
    for t in tqdm(task, leave=False):
        for d in tqdm(dtset, leave=False):

            csv_file = d + "_" + t + "_info.csv"
            data = pd.read_csv(path_csv + d + "/" + csv_file)
            other_col = list(data.columns)[2:]

            for index, row in tqdm(data.iterrows(), total=data.shape[0], leave=False):
                patient_id, inst = row["PatientID"], row["Institution"]

                try:
                    metadata = {}
                    for col in other_col:
                        metadata[col] = row[col]

                    metadata[t] = metadata.pop("outcome")
                    pat = Patient(patient_id, path_images + f, inst, d)
                    pat.save_in_hdf5(path_images + "final_dtset/" + f + "all.hdf5", metadata=metadata)

                except Exception as e:
                    continue
"""
"""
for f in tqdm(folder):
    for d in tqdm(dtset, leave=False):
        patient_list = {}
        for t in task:

            csv_file = d + "_" + t + "_info.csv"
            data = pd.read_csv(path_csv + d + "/" + csv_file)
            other_col = list(data.columns)[2:]

            for index, row in data.iterrows():
                pat_id, inst = row["PatientID"], row["Institution"]

                if t == "malignant":
                    patient_list[pat_id] = {"subtype": 0, "grade": 0}
                try:
                    for col in other_col:
                        patient_list[pat_id][col] = row[col]

                    patient_list[pat_id][t] = patient_list[pat_id].pop("outcome")
                    if t != "malignant":
                        patient_list[pat_id][t] += 1

                except Exception as e:
                    print(list(patient_list.keys()))
                    print(pat_id)
                    print(t, d)
                    raise NotImplementedError

        for pat_id in tqdm(list(patient_list.keys()), leave=False):
            try:
                pat = Patient(pat_id, path_images + f, "NA", d)
                pat.save_in_hdf5(path_images + "final_dtset/" + f + "all.hdf5", metadata=patient_list[pat_id])
            except Exception as e:
                continue
"""

for f in tqdm(folder):
    for d in tqdm(dtset, leave=False):
        patient_list = {}

        # csv_file = d + "_" + t + "_info.csv"
        csv_file = d + "_set.csv"
        data = pd.read_csv(path_csv + d + "/" + csv_file)
        other_col = list(data.columns)[2:]

        for index, row in data.iterrows():
            pat_id = row["PatientID"]

            patient_list[pat_id] = {}
            try:
                for col in other_col:
                    patient_list[pat_id][col] = row[col]

            except Exception as e:
                print(list(patient_list.keys()))
                print(pat_id)
                print(t, d)
                raise NotImplementedError

        for pat_id in tqdm(list(patient_list.keys()), leave=False):
            try:
                pat = Patient(pat_id, path_images + f, )
                pat.save_in_hdf5(path_images + "final_dtset/" + f + "all.hdf5", d, metadata=patient_list[pat_id])
            except Exception as e:
                continue