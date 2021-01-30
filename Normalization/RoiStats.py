"""
    @file:              RoiStats.py
    @Author:            Alexandre Ayotte

    @Creation Date:     10/2020
    @Last modification: 01//2021

    @Description:       This script has been use to compute pertinent statistics about the region of interest (ROI) of
                        our dataset by using the Patient class. These statistics are about the size of the ROI along
                        each dimension, the image lenght along each dimension, the voxel size along each dimension and
                        the distance between the center of mass of the t1 and t2 ROIs.
"""
import numpy as np
from Patient import Patient
from scipy import stats
from tqdm import trange
from matplotlib import pyplot as plt


_path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Corrected"
institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]

exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357"]

roi_length = np.empty((0, 3), float)
roi_shape = np.empty((0, 3), int)
img_shape = np.empty((0, 3), int)
voxel_spacing = np.empty((0, 3), float)
distance = []

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

            distance.append(np.linalg.norm(measures["roi_distance"]))

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

print("\nDISTANCE:")
print("per_5: ", np.percentile(distance, 5, axis=0))
print("per_95:", np.percentile(distance, 95, axis=0))
print("trimmed mean: ", stats.trim_mean(distance, 0.05, axis=0))

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

print("\nDISTANCE:")
print("per_10: ", np.percentile(distance, 10, axis=0))
print("per_90", np.percentile(distance, 90, axis=0))
print("trimmed mean: ", stats.trim_mean(distance, 0.10, axis=0))

ok = np.where(np.array(distance) > 50, 1, 0).sum()
ok2 = np.where(np.array(distance) == 50, 1, 0).sum()

print("\n", ok)
print(ok2)
n, bins, patches = plt.hist(distance, 20, facecolor='green')
# n1, bins1 = np.histogram(distance, 20)
# plt.plot(bins1[1:-1], n1)
plt.show()

n, bins, patches = plt.hist(distance, 50, facecolor='green')
# n1, bins1 = np.histogram(distance, 20)
# plt.plot(bins1[1:-1], n1)
plt.show()
n, bins, patches = plt.hist(distance, 100, facecolor='green')
# n1, bins1 = np.histogram(distance, 20)
# plt.plot(bins1[1:-1], n1)
plt.show()
# print(histo)