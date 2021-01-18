from Patient import Patient
from tqdm import trange


_path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Corrected"
temp_path = "n4_temp"
exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357", 'Kidney-Penn-086',
           'Kidney-Penn-115', 'Kidney-Penn-125', 'Kidney-Penn-169', 'Kidney-Penn-329', 'Kidney-Penn-556',
           'Kidney-Penn-587', 'Kidney-Penn-722', 'Kidney-Penn-745', 'Kidney-Penn-753', 'Kidney-Penn-765',
           'Kidney-Penn-774', 'Kidney-Penn-788', 'Kidney-Penn-797', 'Kidney-TCGA-048', 'Kidney-TCGA-054']
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]

voxel_size = [1.1, 1.1, 5.0]
crop_shape = [[96, 96, 32], [96, 96, 32]]
read_path = [temp_path, temp_path, _path, _path]
save_path = ["Option1_with_N4", "Option1_without_N4"]

# *************************************************
#              Apply n4 bias on image
# *************************************************
for i in range(len(institution)):
    for j in trange(1, nb_patient[i] + 1):
        if j < 10:
            _nb = "00" + str(j)
        elif j < 100:
            _nb = "0" + str(j)
        else:
            _nb = str(j)

        patient_id = institution[i] + "-" + _nb

        if patient_id not in exclude:
            pat = Patient(patient_id, _path)
            pat.apply_n4(save=True, save_path=temp_path)


# *************************************************
#        Resample, crop and normalize image
# *************************************************
for k in range(4):
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
                pat = Patient(patient_id, read_path[k])

                pat.resample_and_crop(resample_params=voxel_size,
                                      crop_shape=crop_shape[k],
                                      interp_type=0,
                                      save=False,
                                      save_path=save_path[k])
                pat.apply_znorm(save=True, save_path=save_path[k])
