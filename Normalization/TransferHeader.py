"""
    @file:              TransferHeader.py
    @Author:            Alexandre Ayotte

    @Creation Date:     10/2020
    @Last modification: 01//2021

    @Description:       This script has been use to transfer header from a first image save in a NIFTI file to another
                        images and its corresponding ROI. This has been done by finding the pair of image that math
                        together according to their name and the value of the voxel. Most of the job has been done by
                        using the MRIimage class.
"""
from MRI_image import MRIimage
from tqdm import trange


path_npy = "E:/WORKSPACE_RadiomicsComputation/Kidney/DATA"
path_nifti = "E:/WORKSPACE_RadiomicsComputation/Kidney/NIFTI"
pythonCodePATH = 'C:/Users/Alexandre/Desktop/Maitrise/MEDomicsLab-develop/Code'
path = "E:/WORKSPACE_RadiomicsComputation/Kidney/renal_nifti"
save_path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Corrected"

institution = ["Kidney-XY2", "Kidney-Penn", "Kidney-CH", "Kidney-TCGA", "Kidney-Mayo", "Kidney-HP"]
nb_patient = [25, 833, 112, 56, 118, 50]
modality = ["T1C", "T2WI"]
exclude = ["Kidney-Penn-238", "Kidney-Penn-254", "Kidney-Penn-337", "Kidney-Penn-357"]

for i in range(len(institution)):
    for j in trange(1, nb_patient[i]+1):
        if j < 10:
            _nb = "00" + str(j)
        elif j < 100:
            _nb = "0" + str(j)
        else:
            _nb = str(j)

        patient_id = institution[i] + "-" + _nb

        for moda in modality:
            if patient_id not in exclude:
                try:
                    path_image = str(path + "/" + patient_id) + "__" + moda
                    my_image = MRIimage(modality=moda,
                                        path_image=path_image + ".MRscan__VOL",
                                        path_roi=path_image + ".MRscan__ROI")
                    my_image.transfer_header(npy_dir=path_npy,
                                             nifti_dir=path_nifti,
                                             medomics_code_path=pythonCodePATH,
                                             save=True,
                                             save_path=save_path)
                except Exception as e:
                    print("problem with patient {}, {}, {}, \n {}".format(institution[i], j, moda, e))
                    continue
