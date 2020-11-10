from MRI_image import MRIimage
import numpy as np


path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Test"
patient_id = "Kidney-TCGA-008"

path_image = str(path + "/" + patient_id) + "__T1C"

my_image = MRIimage(path_image=path_image, path_roi=path_image + "_ROI", modality="T1C", keep_mem=True)

print(my_image.get_roi_measure())
my_image.plot([45, 130, 90], axis="all")
my_image.plot([45, 130, 90], axis="all", roi=True)

my_image.crop([61, 61, 21], save=True, save_path="norme")

my_image.plot([10, 30, 30], axis="all")
my_image.plot([10, 30, 30], axis="all", roi=True)

my_image.apply_znorm(save=True, save_path="norme")
my_image.plot([10, 30, 30], axis="all")

"""
my_image = MRIimage(path_image=path_image, path_roi=path_image + "_ROI", modality="T1C", keep_mem=True)

my_image.crop([301, 301, 301], save=True, save_path="norme")

my_image.plot([150, 150, 150], axis="all")
my_image.plot([150, 150, 150], axis="all", roi=True)
"""

# my_image.apply_znorm(save=True, save_path="norme")
# my_image.plot([20, 100, 100], axis="all")

