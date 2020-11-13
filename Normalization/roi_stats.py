import numpy as np
from Patient import Patient
from scipy import stats
from tqdm import trange


_path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Corrected"
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]

exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357"]

roi_length = np.empty((0, 3), float)
roi_shape = np.empty((0, 3), int)
img_shape = np.empty((0, 3), int)
voxel_spacing = np.empty((0, 3), float)

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
            pat = Patient(patient_id, _path, institution[i], "Train")

            measures = pat.get_measure()

            roi_length = np.append(roi_length, [measures['roi_size']], axis=0)
            roi_shape = np.append(roi_shape, [measures['t1_roi_shape']], axis=0)
            roi_shape = np.append(roi_shape, [measures['t2_roi_shape']], axis=0)
            img_shape = np.append(img_shape, [measures['t1_shape']], axis=0)
            img_shape = np.append(img_shape, [measures['t2_shape']], axis=0)
            voxel_spacing = np.append(voxel_spacing, [measures['t1_voxel_spacing']], axis=0)
            voxel_spacing = np.append(voxel_spacing, [measures['t2_voxel_spacing']], axis=0)

print("\n##########################################")
print("\n              95 percentile               ")
print("\n##########################################")
print("\nROI LENGTH:")
print("per_5: ", np.percentile(roi_length, 5, axis=0))
print("per_95:", np.percentile(roi_length, 95, axis=0))
print("trimmed mean: ", stats.trim_mean(roi_length, 0.05, axis=0))

print("\nROI SHAPE:")
print("per_5: ", np.percentile(roi_shape, 5, axis=0))
print("per_95:", np.percentile(roi_shape, 95, axis=0))
print("trimmed mean: ", stats.trim_mean(roi_shape, 0.05, axis=0))

print("\nIMAGE SHAPE:")
print("per_5: ", np.percentile(img_shape, 5, axis=0))
print("per_95:", np.percentile(img_shape, 95, axis=0))
print("trimmed mean: ", stats.trim_mean(img_shape, 0.05, axis=0))

print("\nVOXEL SPACING:")
print("per_5: ", np.percentile(voxel_spacing, 5, axis=0))
print("per_95:", np.percentile(voxel_spacing, 95, axis=0))
print("trimmed mean: ", stats.trim_mean(voxel_spacing, 0.05, axis=0))

print("\n\n")
print("##########################################")
print("              90 percentile               ")
print("##########################################")
print("\nROI LENGTH:")
print("per_10: ", np.percentile(roi_length, 10, axis=0))
print("per_90:", np.percentile(roi_length, 90, axis=0))
print("trimmed mean: ", stats.trim_mean(roi_length, 0.10, axis=0))

print("\nROI SHAPE:")
print("per_10: ", np.percentile(roi_shape, 10, axis=0))
print("per_90:", np.percentile(roi_shape, 90, axis=0))
print("trimmed mean: ", stats.trim_mean(roi_shape, 0.10, axis=0))

print("\nIMAGE SHAPE:")
print("per_10: ", np.percentile(img_shape, 10, axis=0))
print("per_90", np.percentile(img_shape, 90, axis=0))
print("trimmed mean: ", stats.trim_mean(img_shape, 0.10, axis=0))

print("\nVOXEL SPACING:")
print("per_10: ", np.percentile(voxel_spacing, 10, axis=0))
print("per_90", np.percentile(voxel_spacing, 90, axis=0))
print("trimmed mean: ", stats.trim_mean(voxel_spacing, 0.10, axis=0))
