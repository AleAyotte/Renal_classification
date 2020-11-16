import numpy as np
import pandas as pd
from Patient import Patient


path_images = "E:/WORKSPACE_RadiomicsComputation/Kidney/Option1_with_N4/"
path_csv = "C:/Users/Alexandre/Desktop/Maitrise/Renal_classification/CSV/"
task = ["malignant", "subtype", "grade"]
exception = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357",
             "Kidney-CH-075", "Kidney-Penn-775", "Kidney-Penn-788"]
dtset = ["train", "test", "test2"]

"""
conne = {"ma valeur": 12345,
         "ma valeur 2": "asdf"}

pat = Patient("Kidney-Penn-005", path_images, "CH", dtset)
pat.save_in_hdf5("malignant_train.hdf5", metadata=conne)
"""

for t in task:
    for d in dtset:

        csv_file = d + "_" + t + "_info.csv"
        data = pd.read_csv(path_csv + d + "/" + csv_file)
        other_col = list(data.columns)[2:]
        print(t, d)
        for index, row in data.iterrows():
            patient_id, inst = row["PatientID"], row["Institution"]

            if patient_id not in exception:
                metadata = {}
                for col in other_col:
                    metadata[col] = row[col]

                pat = Patient(patient_id, path_images, inst, d)
                pat.save_in_hdf5(t + ".hdf5", metadata=metadata)
