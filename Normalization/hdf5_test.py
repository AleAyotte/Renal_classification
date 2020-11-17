import numpy as np
import pandas as pd
from Patient import Patient
from tqdm.auto import tqdm


path_images = "E:/WORKSPACE_RadiomicsComputation/Kidney/"
path_csv = "C:/Users/Alexandre/Desktop/Maitrise/Renal_classification/CSV/"
folder = ["Option1_with_N4/", "Option2_with_N4/", "Option1_without_N4/", "Option2_without_N4/"]
task = ["malignant", "subtype", "grade"]
dtset = ["train", "test", "test2"]

for f in tqdm(folder):
    for t in tqdm(task, leave=False):
        # print("\nfolder: {}, task: {}".format(f, t))
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

                    pat = Patient(patient_id, path_images + f, inst, d)
                    pat.save_in_hdf5(path_images + "final_dtset/" + f + t + ".hdf5", metadata=metadata)
                except Exception as e:
                    continue

