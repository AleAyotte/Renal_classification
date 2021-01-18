import h5py
from matplotlib import pyplot as plt
from MRI_image import MRIimage
import numpy as np
from os import path
import os
from Utils import convert_3d_to_2d, get_group, update_dataset, update_attribute


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
    register(t1_fixed: bool = True, type_of_transform: str = "Translation", focus_mask: bool = False,
             save: bool = False, save_path: str = "")
        Register the image T2 on the image T1 (Or the opposite if t1_fixed is False) and adjust the ROI.
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
    def __init__(self,
                 patient_id: str,
                 _path: str,
                 institution: str,
                 dataset: str):
        self.__id = patient_id
        self.__path = _path
        self.__inst = institution
        self.__dataset = dataset
        self.__t1 = MRIimage(modality="T1C",
                             path_image=path.join(self.__path, patient_id + "__T1C" + ".MRscan__VOL"),
                             path_roi=path.join(self.__path, patient_id + "__T1C" + ".MRscan__ROI"))
        self.__t2 = MRIimage(modality="T2WI",
                             path_image=path.join(self.__path, patient_id + "__T2WI" + ".MRscan__VOL"),
                             path_roi=path.join(self.__path, patient_id + "__T2WI" + ".MRscan__ROI"))
        self.__measure = {"roi_size": [],
                          "t1_shape": [],
                          "t2_shape": [],
                          "t1_roi_shape": [],
                          "t2_roi_shape": [],
                          "t1_voxel_spacing": [],
                          "t2_voxel_spacing": [],
                          "roi_distance": []}
        self.__roi_merged = False

    def apply_n4(self,
                 save: bool = False,
                 save_path="") -> None:
        """
        Apply the n4_bias_feild_correction on the image with Antspy and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        self.__t1.apply_n4(save=save, save_path=save_path)
        self.__t2.apply_n4(save=save, save_path=save_path)

    def apply_znorm(self,
                    save: bool = False,
                    save_path="") -> None:
        """
        Apply the z normalization on the image and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        self.__t1.apply_znorm(save=save, save_path=save_path)
        self.__t2.apply_znorm(save=save, save_path=save_path)

    def get_measure(self) -> dict:
        if len(self.__measure["roi_size"]) == 0:
            self.__read_measure()
        return self.__measure

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

    def get_t1(self) -> MRIimage:
        return self.__t1

    def get_t2(self) -> MRIimage:
        return self.__t2

    def merge_roi(self,
                  save: bool = False,
                  save_path: str = "") -> None:
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

    def plot_image_and_roi(self,
                           slice_t1: int = -1,
                           slice_t2: int = -1,
                           slice_orientation: str = "axial") -> None:
        """
        Plot the images and their corresponding ROI in axial view.

        :param slice_t1: A positive integer that represent the axial slice to visualize for the T1C image and its ROI.
                         (If = -1 the selected slice will be choose according the center of the ROI.)
        :param slice_t2: A positive integer that represent the axial slice to visualize for the T2WI image and its ROI.
                         (If = -1 the selected slice will be choose according the center of the ROI.)
        :param slice_orientation: A string that indicate the slice orientation. (Options: axial, coronal and sagittal)
        """
        _SLICE_ORIENTATION = ['sagittal', 'coronal', 'axial']
        assert slice_orientation.lower() in _SLICE_ORIENTATION, "The slices orientation could only be 'axial', " \
                                                                "'coronal' or 'sagittal'."
        assert slice_t1 >= -1 and slice_t2 >= -1, "The slices parameters should be a positive integer."

        ind = _SLICE_ORIENTATION.index(slice_orientation.lower())
        slice_t1 = slice_t1 if slice_t1 > -1 else self.__t1.get_roi_measure()["center_voxel"][ind]
        slice_t2 = slice_t2 if slice_t2 > -1 else self.__t2.get_roi_measure()["center_voxel"][ind]

        slices_t1 = [slice_t1 if x == ind else slice(None) for x in range(3)]
        slices_t2 = [slice_t2 if x == ind else slice(None) for x in range(3)]

        slices_list = [slices_t1, slices_t1, slices_t2, slices_t2]
        titles = ["Image T1C", "ROI T1C", "Image T2WI", "ROI T2WI"]
        imgs = [self.__t1.get_img(),
                self.__t1.get_roi(),
                self.__t2.get_img(),
                self.__t2.get_roi()]

        fig = plt.figure(figsize=(24, 30))

        for i in range(len(imgs)):
            fig.add_subplot(2, 2, i + 1, title=titles[i])
            plt.set_cmap(plt.gray())
            plt.imshow(imgs[i][slices_list[i][0], slices_list[i][1], slices_list[i][2]])

        plt.show()

    def __read_measure(self) -> None:
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

    def register(self,
                 t1_fixed: bool = True,
                 type_of_transform: str = "Translation",
                 focus_mask: bool = False,
                 save: bool = False,
                 save_path: str = "") -> None:
        """
        Register the image T2 on the image T1 (Or the opposite if t1_fixed is False) and adjust the ROI.

        :param t1_fixed: If true, the image T2 will be register on the image T1. Otherwise, it will be the opposite.
                         (Default: True)
        :param type_of_transform: Type of transformation that will be used for the registration.
                                  (Default: Translation)
        :param focus_mask: If true, the fixed mask will be used to focus the registration.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        from ants.registration import registration, apply_transforms
        from ants.utils.convert_nibabel import to_nibabel, from_nibabel

        t1 = self.__t1.get_nifti()
        t2 = self.__t2.get_nifti()
        roi_t1 = self.__t1.get_nifti(roi=True)
        roi_t2 = self.__t2.get_nifti(roi=True)

        fixed_img = from_nibabel(t1) if t1_fixed else from_nibabel(t2)
        moving_img = from_nibabel(t2) if t1_fixed else from_nibabel(t1)
        fixed_roi = from_nibabel(roi_t1) if t1_fixed else from_nibabel(roi_t2)
        moving_roi = from_nibabel(roi_t2) if t1_fixed else from_nibabel(roi_t1)

        result = registration(fixed=fixed_img, moving=moving_img,
                              type_of_transform=type_of_transform,
                              mask=fixed_roi if focus_mask else None)
        new_ants_roi = apply_transforms(fixed=fixed_roi, moving=moving_roi,
                                        transformlist=result['fwdtransforms'])

        new_img = to_nibabel(result['warpedmovout'])
        new_roi = np.where(new_ants_roi.numpy() >= 0.5, 1, 0)

        if t1_fixed:
            self.__t2.update_nifti(new_img, new_roi, save, save_path)
        else:
            self.__t1.update_nifti(new_img, new_roi, save, save_path)

        # Delete temporary files.
        for file in result['fwdtransforms']:
            if path.exists(file):
                os.remove(file)

    def resample_and_crop(self,
                          resample_params,
                          crop_shape,
                          interp_type: int = 1,
                          threshold: float = 50,
                          register: bool = False,
                          register_mode: str = "threshold",
                          merge_roi: bool = False,
                          save: bool = False,
                          save_path: str = "") -> None:
        """
        Resample both images and their ROI, crop them and if requested merge the ROI together.

        :param resample_params: List or tuple that indicate the new voxel dimension in mm.
        :param crop_shape: The dimension of the region to crop in term of number of voxel.
        :param interp_type: The interpolation algorithm that will be used to resample the image.
                            (0: Linear, 1: nearest neighbor, 2: gaussian, 3: windowed sinc, 4: bspline)
        :param threshold: Maximum distance between the two center of mass before cropping.
        :param register: If true, the image "t2" will be register on the image "t1"
        :param register_mode: A string that indicate when the registration is applied. (Option: 'threshold', 'always')
                              If threashold then the image t1 is register on image t2 if the distance between
                              the two center of mass is too high.
        :param merge_roi: A boolean that indicate if the ROI will be merge at the end of the process.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        is_t1_axial = (self.__t1.get_metadata()['orientation'] == "axial")

        self.__t1.resample(resample_params=resample_params, interp_type=interp_type, save=False, save_path=save_path,
                           reorient=True)
        self.__t2.resample(resample_params=resample_params, interp_type=interp_type, save=False, save_path=save_path,
                           reorient=True)

        if register and register_mode.lower() == "always":
            self.register(is_t1_axial)

        self.__read_measure()
        distance = np.linalg.norm(self.__measure["roi_distance"])

        # If the distance between the center of mass is too high, then we register the images and the ROI.
        if distance > threshold:
            if register is True and register_mode == "threshold":
                self.register(is_t1_axial)
                self.__read_measure()
                distance = np.linalg.norm(self.__measure["roi_distance"])

                if distance > threshold:
                    self.register(is_t1_axial, focus_mask=True)
                    self.__read_measure()
                    distance = np.linalg.norm(self.__measure["roi_distance"])

            if distance > threshold:
                raise Exception("The distance between the two center of mass is too high.".format(distance))

        roi_center = self.__get_ponderate_center()

        self.__t1.crop(crop_shape=crop_shape,
                       center=roi_center,
                       save=False if merge_roi else save,
                       save_path=save_path)

        self.__t2.crop(crop_shape=crop_shape,
                       center=roi_center,
                       save=False if merge_roi else save,
                       save_path=save_path)

        if merge_roi:
            self.merge_roi(save=save, save_path=save_path)

    def save_images(self,
                    modality: str = "both",
                    save_path: str = "",
                    with_roi: bool = False) -> None:
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

    def save_in_hdf5(self,
                     filepath: str,
                     modality: str = "both",
                     merge_roi: bool = True,
                     apply_roi: bool = False,
                     save_roi: bool = True,
                     convert_in_2_5d: bool = False,
                     metadata: dict = None) -> None:
        """
        Save the images and their merged ROI into an hdf5 file with the clinical data

        :param filepath: The filepath of the hdf5 file.
        :param modality: The modality to save into the hdf5 file.
        :param merge_roi: If true, save the merged ROI instead of the 2 ROI.
        :param apply_roi: If true, the pixel of image will be 0 where the pixel of the roi are 0.
        :param save_roi: If true, the ROI is save in the dataset.
        :param convert_in_2_5d: If true, the images will convert in 2.5D before been saved.
        :param metadata: A dictionnary that contain metadata to save in hdf5 file.
        """

        self.merge_roi() if merge_roi else None

        f = h5py.File(filepath, 'a')

        # Create the patient
        dts = get_group(f, self.__dataset)
        pat = get_group(dts, self.__id)

        # Convert the image in 2.5D
        if convert_in_2_5d:
            imgs, rois = convert_3d_to_2d([self.__t1.get_img(), self.__t2.get_img()],
                                          [self.__t1.get_roi(), self.__t2.get_roi()],
                                          reshape=False,
                                          apply_mask_on_img=apply_roi)
            t1, t2 = imgs[0], imgs[1]
            roi_t1, roi_t2 = rois[0], rois[1]
        else:
            t1, t2 = self.__t1.get_img(), self.__t2.get_img()
            roi_t1, roi_t2 = self.__t1.get_roi(), self.__t2.get_roi()

        # Save the images
        if modality.lower() in ["t1", "t1c", "both"]:
            update_dataset(pat, "t1", t1)

        if modality.lower() in ["t2", "t2wi", "both"]:
            update_dataset(pat, "t2", t2)

        if save_roi:
            if merge_roi:
                update_dataset(pat, "roi", roi_t1)
            else:
                update_dataset(pat, "roi_t1", roi_t1)
                update_dataset(pat, "roi_t2", roi_t2)

        # Save the metadata
        if metadata is not None:
            for key, value in metadata.items():
                update_attribute(pat, key, value)

        f.close()

    def set_roi_merged(self):
        """
        Change the state of the attribute self.roi_merged to True. Usefull if the user know that
        the ROI has been merge in another
        """
        self.__roi_merged = True
