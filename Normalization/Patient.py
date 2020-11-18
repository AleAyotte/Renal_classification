import h5py
from matplotlib import pyplot as plt
from MRI_image import MRIimage
import numpy as np
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
    __measure : Dict
        A dictionnary that contain usefull statistic about the images and their ROI.
            roi_size : List[float]
                A list that indicate the maximum dimension of the two ROI in mm for each axis.
            t1_shape : List[int]
                A list that indicate the dimension of the T1C image in term of number of voxel.
            t2_shape : List[int]
                A list that indicate the dimension of the T2WI image in term of number of voxel.
            t1_roi_shape : List[int]
                A list that indicate the shape of the T1C ROI in term of number of voxel.
            t2_roi_shape : List[int]
                A list that indicate the shape of the T2WI ROI in term of number of voxel.
            t1_voxel_spacing : List[int]
                The voxels dimensions in mm of the T1 image.
            t2_voxel_spacing : List[int]
                The voxels dimensions in mm of the T2 image.
            roi_distance : List[float]
                The distance between the two center of mass.
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
    detach()
        Release memory taken by the images and their ROI.
    merge_roi(save: bool = False, save_path: str = "")
        Merge the ROI of the t1 image and the t2 image and set the new ROI to the images T1 and T2.
    plot_image_and_roi(slice_t1: int = -1, slice_t2: int = -1)
        Plot the images and their corresponding ROI in axial view.
    resample_and_crop(resample_params, crop_shape, interp_type: int = 1, merge_roi: bool = False,
                      save: bool = False, save_path: str = "")
         Resample both images and their ROI, crop them and if requested merge the ROI together.
    save_images(path: str = "", with_roi: bool = False)
        Save one or both images with their ROI if requested.
    save_in_hdf5(filepath: str, modality: str = "both", merged_roi: bool = True, metadata: dict = None):
        Save the images and their merged ROI into an hdf5 file with the clinical data
    set_roi_merged()
        Change the state of the attribute self.roi_merged to True. Usefull if the user know that
        the ROI has been merge in another
    """
    def __init__(self, patient_id: str, _path: str, institution: str, dataset: str):
        self.__id = patient_id
        self.__path = _path
        self.__inst = institution
        self.__dataset = dataset
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
                          "t2_roi_shape": [],
                          "t1_voxel_spacing": [],
                          "t2_voxel_spacing": [],
                          "roi_distance": []}
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

    def get_measure(self) -> dict:
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
        assert (self.__measure["t1_shape"] == self.__measure["t2_shape"]).all(), \
            "The ROI do not have the same shape. You should resample and crop the two ROI before merging them." \
            "\n{}, {}".format(self.__measure["t1_shape"], self.__measure["t2_shape"])

        roi_t1 = self.__t1.get_roi()
        roi_t2 = self.__t2.get_roi()
        roi = roi_t1 + roi_t2

        new_roi = np.where(roi > 0.9, 1, 0)
        self.__t1.update_roi(new_roi=new_roi, save=save, save_path=save_path)
        self.__t2.update_roi(new_roi=new_roi, save=save, save_path=save_path)
        self.__roi_merged = True

    def plot_image_and_roi(self, slice_t1: int = -1, slice_t2: int = -1):
        """
        Plot the images and their corresponding ROI in axial view.

        :param slice_t1: A positive integer that represent the axial slice to visualize for the T1C image and its ROI.
                         (If = -1 the selected slice will be choose according the center of the ROI.)
        :param slice_t2: A positive integer that represent the axial slice to visualize for the T2WI image and its ROI.
                         (If = -1 the selected slice will be choose according the center of the ROI.)
        """
        assert slice_t1 >= -1 and slice_t2 >= -1, "The slices parameters should be a positive integer"

        slice_t1 = slice_t1 if slice_t1 > -1 else self.__t1.get_roi_measure()["center_voxel"][2]
        slice_t2 = slice_t2 if slice_t2 > -1 else self.__t2.get_roi_measure()["center_voxel"][2]

        slices = [slice_t1, slice_t1, slice_t2, slice_t2]
        titles = ["Image T1C", "ROI T1C", "Image T2WI", "ROI T2WI"]
        imgs = [self.__t1.get_img(),
                self.__t1.get_roi(),
                self.__t2.get_img(),
                self.__t2.get_roi()]

        fig = plt.figure(figsize=(24, 30))

        for i in range(len(imgs)):
            fig.add_subplot(2, 2, i + 1, title=titles[i])
            plt.set_cmap(plt.gray())
            plt.imshow(imgs[i][:, :, slices[i]])

        plt.show()

    def __get_ponderate_center(self) -> np.array:
        """
        Measure the ponderate mean of the center of mass of both ROI modality.

        :return: A numpy array that represent the ponderate center of mass in mm.
        """
        t1_roi_measure = self.__t1.get_roi_measure()
        t2_roi_measure = self.__t2.get_roi_measure()

        roi_center_t1 = np.array(t1_roi_measure["center_mm"])
        roi_center_t2 = np.array(t2_roi_measure["center_mm"])
        t1_weight = self.__t1.get_roi().sum()
        t2_weight = self.__t2.get_roi().sum()

        roi_center = ((roi_center_t1 * t1_weight) + (roi_center_t2 * t2_weight)) / (t1_weight + t2_weight)

        return roi_center

    def resample_and_crop(self,  resample_params, crop_shape, interp_type: int = 1, threshold: float = 50,
                          merge_roi: bool = False, save: bool = False, save_path: str = ""):
        """
        Resample both images and their ROI, crop them and if requested merge the ROI together.

        :param resample_params: List or tuple that indicate the new voxel dimension in mm.
        :param crop_shape: The dimension of the region to crop in term of number of voxel.
        :param interp_type: The interpolation algorithm that will be used to resample the image.
                            (0: Linear, 1: nearest neighbor, 2: gaussian, 3: windowed sinc, 4: bspline)
        :param threshold: Maximum distance between the two center of mass before cropping.
        :param merge_roi: A boolean that indicate if the ROI will be merge at the end of the process.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """

        self.__t1.resample(resample_params=resample_params, interp_type=interp_type, save=False)
        self.__t2.resample(resample_params=resample_params, interp_type=interp_type, save=False)

        self.__t1.to_canonical(save=False)
        self.__t2.to_canonical(save=False)

        self.__read_measure()

        distance = np.linalg.norm(self.__measure["roi_distance"])
        if distance > threshold:
            raise Exception("The distance between the two center of mass is too high.".format(distance))

        roi_center = self.__get_ponderate_center()
        center_t1 = self.__t1.spatial_to_voxel(roi_center).astype(int)
        center_t2 = self.__t2.spatial_to_voxel(roi_center).astype(int)

        self.__t1.crop(crop_shape=crop_shape,
                       center=center_t1,
                       save=False if merge_roi else save,
                       save_path=save_path)

        self.__t2.crop(crop_shape=crop_shape,
                       center=center_t2,
                       save=False if merge_roi else save,
                       save_path=save_path)

        if merge_roi:
            self.merge_roi(save=save, save_path=save_path)

    def __read_measure(self):
        """
        Compute usefull measure about the ROI of each image.
        """
        t1_roi_measure = self.__t1.get_roi_measure()
        t2_roi_measure = self.__t2.get_roi_measure()

        tumor_max_size = [max(t1_roi_measure["length_mm"][0], t2_roi_measure["length_mm"][0]),
                          max(t1_roi_measure["length_mm"][1], t2_roi_measure["length_mm"][1]),
                          max(t1_roi_measure["length_mm"][2], t2_roi_measure["length_mm"][2])]

        distance = t1_roi_measure["center_mm"] - t2_roi_measure["center_mm"]

        self.__measure["roi_distance"] = distance
        self.__measure["roi_size"] = tumor_max_size
        self.__measure["t1_roi_shape"] = t1_roi_measure["length_voxel"]
        self.__measure["t2_roi_shape"] = t2_roi_measure["length_voxel"]
        self.__measure['t1_shape'] = self.__t1.get_metadata()["img_shape"]
        self.__measure['t2_shape'] = self.__t2.get_metadata()["img_shape"]
        self.__measure['t1_voxel_spacing'] = self.__t1.get_metadata()["voxel_spacing"]
        self.__measure['t2_voxel_spacing'] = self.__t2.get_metadata()["voxel_spacing"]

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

    def save_in_hdf5(self, filepath: str, modality: str = "both", merged_roi: bool = True, metadata: dict = None):
        """
        Save the images and their merged ROI into an hdf5 file with the clinical data

        :param filepath: The filepath of the hdf5 file.
        :param modality: The modality to save into the hdf5 file.
        :param merged_roi: If true, save the merged ROI instead of the 2 ROI.
        :param metadata: A dictionnary that contain metadata to save in hdf5 file.
        """

        if merged_roi is False:
            raise NotImplementedError

        f = h5py.File(filepath, 'a')
        if self.__dataset in list(f.keys()):
            dts = f[self.__dataset]

            if self.__id in list(dts.keys()):
                pat = dts[self.__id]
            else:
                pat = dts.create_group(self.__id)
        else:
            dts = f.create_group(self.__dataset)
            pat = dts.create_group(self.__id)

        if modality.lower() in ["t1", "t1c", "both"]:
            if "t1" in list(pat.keys()):
                pat["t1"][:] = self.__t1.get_img()
            else:
                pat.create_dataset("t1", data=self.__t1.get_img())

        if modality.lower() in ["t2", "t2wi", "both"]:
            if "t2" in list(pat.keys()):
                pat["t2"][:] = self.__t2.get_img()
            else:
                pat.create_dataset("t2", data=self.__t2.get_img())

        self.merge_roi()

        if "roi" in list(pat.keys()):
            pat["roi"][:] = self.__t1.get_roi()
        else:
            pat.create_dataset("roi", data=self.__t1.get_roi())

        if metadata is not None:
            for key, value in metadata.items():
                if key in list(pat.attrs.keys()):
                    pat.attrs[key] = value
                else:
                    pat.attrs.create(key, value)
        f.close()

    def set_roi_merged(self):
        """
        Change the state of the attribute self.roi_merged to True. Usefull if the user know that
        the ROI has been merge in another
        """
        self.__roi_merged = True
