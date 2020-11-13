import numpy as np
from MRI_image import MRIimage
from os import path


class Patient:
    """
    Create two MRIimage object that correspond to the image T1 and T2 of the giving and measure statistics about
    the patient. Offer the same preprosseing of MRIimage and give the possibility to merge the ROI if the region of
    interest have the same shape.

    ...
    Attributes
    ----------
    __dataset : string
        Indicate the patient is part of which dataset. {Option: Train, Test1, Test2}
    __id : string
        The patient identificator. (Exemple: "Kidney-TCGA-008")
    __inst : string
        Indicate the is part of which institution.
    __keep_mem : bool
        Indicate if the image and its ROI is keep in memory.
    __measure: Dict
        A dictionnary that contain usefull statistic about the images and their ROI.
            roi_size: List[float]
                A list that indicate the maximum dimension of the two ROI in mm for each axis.
            t1_shape: List[int]
                A list that indicate the dimension of the T1C image in term of number of voxel.
            t2_shape: List[int]
                A list that indicate the dimension of the T2WI image in term of number of voxel.
            t1_roi_shape: List[int]
                A list that indicate the shape of the T1C ROI in term of number of voxel.
            t2_roi_shape: List[int]
                A list that indicate the shape of the T2WI ROI in term of number of voxel.
    __path : string
        The directory name that contain the images and their ROI.
    __roi_merged : bool
        A boolean that indicate if the ROI has been merged.
    __t1 : MRIimage
        The T1C image and its ROI as a MRIimage
    __t2 : MRIimage
        The T2WI image and its ROI as a MRIimage
    Methods
    -------
    apply_n4(save: bool= False, save_path="")
        Apply the n4_bias_feild_correction on the image with Antspy and save it if requested.
    apply_znorm(save: bool= False, save_path="")
        Apply the z normalization on the image and save it if requested.
    resample_and_crop(resample_params, crop_shape, interp_type: int = 1, merge_roi: bool = False,
                      save: bool = False, save_path: str = "")
         Resample both images and their ROI, crop them and if requested merge the ROI together.
    detach()
        Release memory taken by the images and their ROI.
    save_images(path: str = "", with_roi: bool = False)
        Save one or both images with their ROI if requested.
    set_roi_merged()
        Change the state of the attribute self.roi_merged to True. Usefull if the user know that
        the ROI has been merge in another
    """
    def __init__(self, patient_id: str, _path: str, institution: str, dataset: str, keep_mem: bool = True):
        self.__id = patient_id
        self.__path = _path
        self.__inst = institution
        self.__dataset = dataset
        self.__keep_mem = keep_mem
        self.__t1 = MRIimage(modality="T1C",
                             path_image=path.join(self.__path, patient_id + "__T1C" + ".MRscan__VOL"),
                             path_roi=path.join(self.__path, patient_id + "__T1C" + ".MRscan__ROI"),
                             keep_mem=True)
        self.__t2 = MRIimage(modality="T2WI",
                             path_image=path.join(self.__path, patient_id + "__T2WI" + ".MRscan__VOL"),
                             path_roi=path.join(self.__path, patient_id + "__T2WI" + ".MRscan__ROI"),
                             keep_mem=True)
        self.__measure = {"roi_size": [],
                          "t1_shape": [],
                          "t2_shape": [],
                          "t1_roi_shape": [],
                          "t2_roi_shape": []}
        self.__roi_merged = False

    def apply_n4(self, save: bool = False, save_path=""):
        """
        Apply the n4_bias_feild_correction on the image with Antspy and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        self.__t1.apply_n4(save=save, save_path=save_path)
        self.__t2.apply_n4(save=save, save_path=save_path)

    def apply_znorm(self, save: bool = False, save_path=""):
        """
        Apply the z normalization on the image and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        self.__t1.apply_znorm(save=save, save_path=save_path)
        self.__t2.apply_znorm(save=save, save_path=save_path)

    def resample_and_crop(self,  resample_params, crop_shape, interp_type: int = 1, merge_roi: bool = False,
                          save: bool = False, save_path: str = ""):
        """
        Resample both images and their ROI, crop them and if requested merge the ROI together.

        :param resample_params: List or tuple that indicate the new voxel dimension in mm.
        :param crop_shape: The dimension of the region to crop in term of number of voxel.
        :param interp_type: The interpolation algorithm that will be used to resample the image.
                            (0: Linear, 1: nearest neighbor, 2: gaussian, 3: windowed sinc, 4: bspline)
        :param merge_roi: A boolean that indicate if the ROI will be merge at the end of the process.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """

        self.__t1.resample(resample_params=resample_params, interp_type=interp_type, save=False)
        self.__t2.resample(resample_params=resample_params, interp_type=interp_type, save=False)

        self.__t1.crop(crop_shape=crop_shape, save=False if merge_roi else save, save_path=save_path)
        self.__t2.crop(crop_shape=crop_shape, save=False if merge_roi else save, save_path=save_path)

        if merge_roi:
            self.merge_roi(save=save, save_path=save_path)

    def detach(self):
        """
        Release memory taken by the images and their ROI.
        """
        self.__t1.detach()
        self.__t2.detach()

    def get_t1(self) -> MRIimage:
        return self.__t1

    def get_t2(self) -> MRIimage:
        return self.__t2

    def get_measure(self):
        if len(self.__measure["roi_size"]) == 0:
            self.__read_measure()
        return self.__measure

    def merge_roi(self, save: bool = False, save_path: str = ""):
        """
        Merge the ROI of the t1 image and the t2 image and set the new ROI to the images T1 and T2.

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save.
        """
        _ = self.get_measure()
        assert (self.__measure["t1_roi_shape"] == self.__measure["t2_roi_shape"]).all(), \
            "The ROI do not have the same shape. You should resample and crop the two ROI before merging them."

        roi_t1 = self.__t1.get_roi()
        roi_t2 = self.__t2.get_roi()

        new_roi = np.where(roi_t1 > 0.9 or roi_t2 > 0.9, 1, 0)
        self.__t1.update_roi(new_roi=new_roi, save=save, save_path=save_path)
        self.__t2.update_roi(new_roi=new_roi, save=save, save_path=save_path)
        self.__roi_merged = True

    def __read_measure(self):
        """
        Compute usefull measure about the ROI of each image.
        """
        t1_roi_measure = self.__t1.get_roi_measure()
        t2_roi_measure = self.__t2.get_roi_measure()

        tumor_max_size = [max(t1_roi_measure["length_mm"][0], t2_roi_measure["length_mm"][0]),
                          max(t1_roi_measure["length_mm"][1], t2_roi_measure["length_mm"][1]),
                          max(t1_roi_measure["length_mm"][2], t2_roi_measure["length_mm"][2])]

        self.__measure["roi_size"] = tumor_max_size
        self.__measure["t1_roi_shape"] = t1_roi_measure["lenght_voxel"]
        self.__measure["t2_roi_shape"] = t2_roi_measure["lenght_voxel"]
        self.__measure['t1_shape'] = self.__t1.get_metadata["img_shape"]
        self.__measure['t2_shape'] = self.__t2.get_metadata["img_shape"]

    def save_images(self, modality: str = "both", save_path: str = "", with_roi: bool = False):
        """
        Save the image (ROI) in a NIFTI file and update reading path.

        :param modality: Indicate which image will saved. (Option: T1/T1C, T2/T2WI, both).
        :param save_path: The path to the folder were the image (ROI) will be saved. (Ex: "Documents/Data").
                      If no value are give, the last path used to read the image will be used.
        :param with_roi: A boolean that indicate if the ROI should also be save.
        """
        if modality.lower() in ["t1", "t1c", "both"]:
            self.__t1.save_image(path=save_path, with_roi=with_roi)

        if modality.lower() in ["t2", "t2wi", "both"]:
            self.__t2.save_image(path=save_path, with_roi=with_roi)

    def set_roi_merged(self):
        """
        Change the state of the attribute self.roi_merged to True. Usefull if the user know that
        the ROI has been merge in another
        """
        self.__roi_merged = True
