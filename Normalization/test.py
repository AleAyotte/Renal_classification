from MRI_image import MRIimage
import numpy as np


path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Test"
patient_id = "Kidney-TCGA-008"

path_image = str(path + "/" + patient_id) + "__T1C"

my_image = MRIimage(path_image=path_image, path_roi=path_image + "_ROI", modality="T1C", keep_mem=True)

my_image.plot([20, 100, 100], axis="all")
my_image.apply_znorm(save=True, save_path="norme")
my_image.plot([20, 100, 100], axis="all")

img = my_image.get_img()
