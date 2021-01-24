from itertools import product
import numpy as np
from Patient import Patient
from tqdm import trange
from Utils import read_metadata

path_images = "/home/alex/Data/Corrected/"
temp_path = "/home/alex/Data/Temp/"
exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357", 'Kidney-Penn-086',
           'Kidney-Penn-115', 'Kidney-Penn-125', 'Kidney-Penn-169', 'Kidney-Penn-329', 'Kidney-Penn-556',
           'Kidney-Penn-587', 'Kidney-Penn-722', 'Kidney-Penn-745', 'Kidney-Penn-753', 'Kidney-Penn-765',
           'Kidney-Penn-774', 'Kidney-Penn-788', 'Kidney-Penn-797', 'Kidney-TCGA-048', 'Kidney-TCGA-054']
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]

folder = "Option1_without_N4/"
voxel_size = [0.77, 0.75, 0.72]
crop_shape = [128, 128, 128]

save_path = "Option1_without_N4"
dtset = ["train", "test", "test2"]
task_list = ["malignant", "subtype", "grade"]

file_list = ["grade.hdf5", "malignant.hdf5", "subtype.hdf5"]
newfile_list = ["new_grade.hdf5", "new_malignant.hdf5", "new_subtype.hdf5"]

# -----------------------------
#        Read metadata
# -----------------------------
metadata = {}

for task, file in zip(task_list, file_list):
    temp_dict = {}
    for split in dtset:
        temp_dict[split] = read_metadata(file, split)

    metadata[task] = temp_dict

for i in range(len(institution)):
    for j in trange(1, nb_patient[i]+1):
        if j < 10:
            _nb = "00" + str(j)
        elif j < 100:
            _nb = "0" + str(j)
        else:
            _nb = str(j)

        patient_id = institution[i] + "-" + _nb

        if patient_id not in exclude:
            try:
                pat = Patient(patient_id, path_images)

                pat.resample_and_crop(resample_params=voxel_size,
                                      crop_shape=crop_shape,
                                      interp_type=0,
                                      ponderate_center=False,
                                      save=False,
                                      save_path=save_path)

                for (t, savefile), dts in product(zip(task_list, newfile_list), dtset):
                    if patient_id in list(metadata[t][dts].keys()):
                        pat.save_in_hdf5(savefile, dts,
                                         merge_roi=False,
                                         apply_roi=True,
                                         convert_in_2_5d=True,
                                         save_roi=False,
                                         metadata=metadata[t][dts][patient_id])
            except Exception as e:
                print(patient_id, " problem during saving in hdf5: ", e)
                continue
