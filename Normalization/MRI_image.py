import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os.path
from scipy.ndimage import center_of_mass
import sys
from Utils import rotate_and_compare


class MRIimage:
    """
    Read two NIFTI files that correspond to a MRI image and it region of interest (ROI) respectively and perform
    different normalization and other preprocessing method. Morever, its also compute some usefull measure about
    image and ROI that can be used to determine the best normalization and correction process.

    ...
    Attributes
    ----------
    __filename : string
        The filename of the nifti image. (Exemple: "Kidney-TCGA-008__T1C")
    __img : nib.nifti1.Nifti1Image
        The nifti image.
    __metadata : Dict
        A dictionnary that contain the major metadata that will be used to normalize the image.
            img_shape : List(int)
                Number of voxel along each dimension.
            img_spacing : List(float)
                The images dimensions in mm.
            orientation : str
                Indicate the orientation of the original DICOM file. (Option: "axial", "coronal", "sagittal")
            voxel_spacing : List(float)
                The voxels dimensions in mm.
    __modality: string
        Indicate the modality of the image. (Option: T1C, T2WI)
    __path : string
        The directory name that contain the image and its ROI.
    __roi : nib.nifti1.Nifti1Image
        The region of interest (ROI) of the image.
    __roi_measure: Dict
        A dictionnary that contain usefull statistic about the ROI.
            length_mm: List[float]
                A list that indicate the dimension of the ROI in mm.
            length_voxel: List[int]
                A list that indicate the dimension of the ROI in term of number of voxel.
            center_mm: List[float]
                A list that indicate the coordinate of the ROI center's in mm.
            center_voxel: List[int]
                A list that indicate the coordinate of the ROI center's in term of number of voxel.
    Methods
    -------
    apply_n4(save: bool= False, save_path="")
        Apply the n4_bias_feild_correction on the image with Antspy and save it if requested.
    apply_znorm(save: bool= False, save_path="")
        Apply the z normalization on the image and save it if requested.
    crop(crop_shape, save: bool = False, save_path: str = "")
        Crop a part of the image and the ROI and save the image and the ROI if requested.
    plot(_slice, axis: str = "all", roi: bool = False)
        Load and plot the image (ROI) along a given axis or along each axis.
    resample(resample_params, interp_type: int = 1, save: bool = False, save_path: str = "")
        Resample an image using antspy according to the new voxel dimension given.
    save_image(path: str = "", with_roi: bool = False)
        Save the image (ROI) in a NIFTI file and update reading path.
    spatial_to_voxel(spatial_pos: list) : np.array
        Convert spatial position into voxel position
    voxel_to_spatial(voxel_pos: list) : np.array
        Convert voxel position into spatial position.
    to_canonical(save: bool = False, save_path: str = "")
        Modify the orientation of the image and the ROI to be as closest canonical possible. So the orientation
        will be set to RAS (Left to Right, Posterior to Anterior, Inferior to Superior).
    transfer_header(npy_dir: str, nifti_dir: str, medomics_code_path: str, save: bool = False, save_path: str = "")
        Transfer the header of a nifti file created with MRIcroGL to the NIFTI image and ROI
        generated by the medomics code.
    update_roi(new_roi: np.array, save: bool = False, save_path: str = "")
        Change the dataobj of the ROI and update the roi_measure.
    """

    def __init__(self, modality: str, path_image: str, path_roi: str):

        self.__path = os.path.dirname(path_image)
        self.__path_roi = os.path.dirname(path_roi)
        self.__filename = os.path.splitext(os.path.basename(path_image))[0]
        self.__filename_roi = os.path.splitext(os.path.basename(path_roi))[0]
        self.__modality = modality
        self.__img = None
        self.__roi = None
        self.__metadata = {'img_shape': [],
                           'voxel_spacing': [],
                           'img_spacing': [],
                           'orientation': ""}
        self.__roi_measure = {'length_mm': [],
                              'length_voxel': [],
                              'center_mm': [],
                              'center_voxel': []}
        self.__read_metadata()

    def apply_n4(self, save: bool = False, save_path=""):
        """
        Apply the n4_bias_feild_correction on the image with Antspy and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        import ants
        from ants.utils.bias_correction import n4_bias_field_correction as n4
        from ants.utils.convert_nibabel import to_nibabel, from_nibabel

        if self.__img is None:
            ants_img = ants.image_read(self._get_path(), reorient=True)
        else:
            ants_img = from_nibabel(self.__img)
        corrected_img = to_nibabel(n4(ants_img))

        self.__img = corrected_img

        if save:
            self.save_image(path=save_path)

    def apply_znorm(self, save: bool = False, save_path=""):
        """
        Apply the z normalization on the image and save it if requested

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        self.__img = self.get_nifti()  # Ensure that the image is loaded in memory
        img = self.get_img()
        img = (img - np.mean(img)) / np.std(img)

        self.__img = nib.Nifti1Image(img, affine=self.__img.affine, header=self.__img.header)

        if save:
            self.save_image(path=save_path)

    def __compute_roi_measure(self):
        """
        Compute some usefull statistics and measure about
        """
        roi = self.get_roi()
        roi = np.rint(roi)

        # We sum all voxel intensity in each sagittal, coronal, axial slice
        slice_lists = [roi.sum(axis=(1, 2)), roi.sum(axis=(0, 2)), roi.sum(axis=(0, 1))]
        coord = []

        # We detect the first and the last slice that have at least one voxel with non zero intensity along each axis.
        for _list in slice_lists:
            non_zero_slices = np.where(_list >= 0.9)
            coord.append([np.min(non_zero_slices), np.max(non_zero_slices)])

        # compute the lenght of roi along each axis in term of mm and number of voxel
        self.__roi_measure['length_voxel'] = [x[1] - x[0] + 1 for x in coord]
        self.__roi_measure['length_mm'] = [self.__metadata['voxel_spacing'][i] * self.__roi_measure['length_voxel'][i]
                                           for i in range(3)]

        # compute the roi center coordinate along each axis in term of mm and number of voxel
        center_voxel = list(center_of_mass(roi))
        self.__roi_measure['center_voxel'] = list(np.array(center_voxel).astype(int))
        self.__roi_measure['center_mm'] = self.voxel_to_spatial(center_voxel)

    def crop(self, crop_shape, center=None, save: bool = False, save_path: str = ""):
        """
        Crop a part of the image and the ROI and save the image and the ROI if requested.

        :param crop_shape: The dimension of the region to crop in term of number of voxel.
        :param center: A list that indicate the center of the cropping box in term of spatial position.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        assert np.sum(np.array(crop_shape) % 2) == 0, "All element of crop_shape should be even number."

        self.__img = self.get_nifti()  # Ensure that the image is loaded in memory
        self.__compute_roi_measure()

        radius = [int(x / 2) - 1 for x in crop_shape]
        center = self.__roi_measure['center_mm'] if center is None else center
        center_min = self.spatial_to_voxel(np.floor(center)).astype(int)
        center_max = self.spatial_to_voxel(np.ceil(center)).astype(int)
        img_shape = self.__metadata['img_shape']

        # Pad the image and the ROI if its necessary
        padding = []
        for rad, cent_min, cent_max, shape in zip(radius, center_min, center_max, img_shape):
            padding.append(
                [abs(min(cent_min - rad, 0)), max(cent_max + rad + 1 - shape, 0)]
            )

        img = self.get_img()
        roi = self.get_roi()

        img = np.pad(img, tuple([tuple(x) for x in padding]))
        roi = np.pad(roi, tuple([tuple(x) for x in padding]))
        center_min = [center_min[i] + padding[i][0] for i in range(3)]
        center_max = [center_max[i] + padding[i][0] for i in range(3)]

        # Crop the image
        img = img[center_min[0]-radius[0]:center_max[0]+radius[0] + 1,
                  center_min[1]-radius[1]:center_max[1]+radius[1] + 1,
                  center_min[2]-radius[2]:center_max[2]+radius[2] + 1]
        roi = roi[center_min[0] - radius[0]:center_max[0] + radius[0] + 1,
                  center_min[1] - radius[1]:center_max[1] + radius[1] + 1,
                  center_min[2] - radius[2]:center_max[2] + radius[2] + 1]

        # Update self.__metadata and self.__roi_measure
        self.__metadata['img_shape'] = crop_shape
        self.__metadata['img_spacing'] = [crop_shape[i] * self.__metadata['voxel_spacing'][i]
                                          for i in range(3)]

        # Update the image and the ROI
        self.__img = nib.Nifti1Image(img, affine=self.__img.affine, header=self.__img.header)
        self.update_roi(new_roi=roi, save=save, save_path=save_path)

    def __find_rotation(self, ref_img: np.array):
        """
        Try every set of rotation to find which one can match the original image with the image of reference.

        :param ref_img: A numpy array that represent the image of reference to match.
        :return: A boolean that indicate if a match has been found, the rotated image and the rotated ROI as two numpy
                 array
        """
        img = self.get_img()
        roi = self.get_roi()

        for i in range(3):
            rotated_img, is_match = rotate_and_compare(img, np.array(ref_img.dataobj),
                                                       rotation_set=i, plot=False)
            if is_match:
                rotated_roi, _ = rotate_and_compare(roi, np.array(ref_img.dataobj),
                                                    rotation_set=i, plot=False)
                return True, rotated_img, rotated_roi
        else:
            return False, None, None

    def get_nifti(self, roi: bool = False) -> nib.nifti1.Nifti1Image:
        if (self.__img is None and not roi) or (self.__roi is None and roi):
            return nib.load(self._get_path(roi))
        else:
            return self.__roi if roi else self.__img

    def get_img(self) -> np.array:
        return np.array(self.get_nifti().dataobj)

    def get_roi(self) -> np.array:
        return np.array(self.get_nifti(roi=True).dataobj)

    def _get_path(self, roi: bool = False):
        """
        Return the complete path (directory + filename) were the image (ROI) is (or will be) saved.

        :param roi: A boolean that indicate if we want the path of ROI instead of the image path.
        :return: A string that represent the path.
        """
        if not roi:
            file_path = os.path.join(self.__path, self.__filename) + ".MRscan__VOL.nii.gz"
        else:
            file_path = os.path.join(self.__path_roi, self.__filename_roi) + ".MRscan__ROI.nii.gz"
        return file_path

    def get_metadata(self) -> dict:
        return self.__metadata

    def get_roi_measure(self) -> dict:
        if len(self.__roi_measure['length_mm']) == 0:
            self.__compute_roi_measure()
        return self.__roi_measure

    def plot(self, _slice, axis: str = "all", roi: bool = False):
        """
        Load and plot the image (ROI) along a given axis or along each axis.

        :param _slice: An integer or a list of 3 integer that indicate which slice will be visualized for the axial,
                       coronal and sagittal if axis = all. If axis != all, you should only specify an integer.
        :param axis:   A string that indicate which axis will be visualized. {Option: Axial, Coronal, Sagittal or all).
        :param roi:    A boolean that indiate if we want to visualize the region of interest instead of the image.
        """
        assert axis.lower() in ["axial", "coronal", "sagittal", "all"]
        assert len(_slice) == 3 if type(_slice) == list else True
        img = self.get_roi() if roi else self.get_img()

        imgs = []
        titles = []
        _slice = [_slice for _ in range(3)] if not type(_slice) == list else _slice

        if axis.lower() == "axial" or axis.lower() == "all":
            imgs.append(img[:, :, _slice[0]])
            titles.append("axial")
        if axis.lower() == "coronal" or axis.lower() == "all":
            imgs.append(img[:, _slice[1], :])
            titles.append("coronal")
        if axis.lower() == "sagittal" or axis.lower() == "all":
            imgs.append(img[_slice[2], :, :])
            titles.append("sagittal")

        fig = plt.figure(figsize=(24, 30))

        for i in range(len(imgs)):
            fig.add_subplot(1, len(imgs), i + 1, title=titles[i])
            plt.set_cmap(plt.gray())
            plt.imshow(imgs[i])

        plt.show()

    def __read_metadata(self):
        """
        Read the header of the NIFTI image and get some major metadata about the image like voxel dimension.
        """
        AXIAL_ORIENTATION = [('R', 'A', 'S'), ('L', 'A', 'S')]
        CORONAL_ORIENTATION = [('L', 'S', 'P')]

        self.__img = self.get_nifti()
        header = self.__img.header

        self.__metadata['img_shape'] = header['dim'][1:4]
        self.__metadata['voxel_spacing'] = header['pixdim'][1:4]
        self.__metadata['img_spacing'] = [
            header['dim'][i]*header['pixdim'][i] for i in range(1, 4)
        ]

        if nib.aff2axcodes(self.__img.affine) in AXIAL_ORIENTATION:
            self.__metadata['orientation'] = "axial"
        elif nib.aff2axcodes(self.__img.affine) in CORONAL_ORIENTATION:
            self.__metadata['orientation'] = "coronal"
        else:
            raise Exception("ERROR for patient {}, unknow orientation {}".format(self.__filename,
                                                                                 nib.aff2axcodes(self.__img.affine)))

    def __read_study_time(self, npy_dir: str, medomics_code_path: str) -> str:
        """
        Read the npy file and return the study date concatened with the study time.

        :param npy_dir: The directory where the npy files are stored.
        :param medomics_code_path: The path directory of the medomics code.
        :return: The concatenate study_date and study_time
        """
        sys.path.append(medomics_code_path)

        file_path = str(os.path.join(npy_dir, self.__filename))
        file_path += ".MRscan.npy"
        sData = np.load(file_path, allow_pickle=True)
        sData = np.concatenate(sData)

        return sData[2][-1].StudyDate + sData[2][-1].StudyTime

    def resample(self, resample_params, interp_type: int = 1, save: bool = False, save_path: str = "",
                 reorient: bool = True):
        """
        Resample an image using antspy according to the new voxel dimension given.

        :param resample_params: List or tuple that indicate the new voxel dimension in mm.
                                (In orientation ('L', 'A', 'S') or ('R', 'A', 'S'))
        :param interp_type: The interpolation algorithm that will be used to resample the image.
                            (0: Linear, 1: nearest neighbor, 2: gaussian, 3: windowed sinc, 4: bspline)
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        :param reorient: If True, the file will be read with antspy and reoriented with it instead of
                         convert from nibabel. Not saved modification will be lost.
        """
        import ants
        from ants.registration import resample_image
        from ants.utils.convert_nibabel import to_nibabel, from_nibabel

        if self.__img is None or reoriented:
            ants_img = ants.image_read(self._get_path(), reorient=reorient)
        else:
            ants_img = from_nibabel(self.__img)

        if self.__roi is None or reoriented:
            ants_roi = ants.image_read(self._get_path(roi=True), reorient=reorient)
        else:
            ants_roi = from_nibabel(self.__roi)

        # Check the orientation and change the order of the resample parameters if needed.
        if self.__metadata["orientation"] == "coronal" and reorient is False:
            resample_params = [resample_params[0], resample_params[2], resample_params[1]]

        corrected_img = to_nibabel(resample_image(ants_img, resample_params, False, interp_type))
        corrected_roi = to_nibabel(resample_image(ants_roi, resample_params, False, 0))

        new_roi = np.where(np.array(corrected_roi.dataobj) >= 0.5, 1, 0)

        self.__img = corrected_img
        self.__read_metadata()
        self.update_roi(new_roi=new_roi, save=save, save_path=save_path)

    def save_image(self, path: str = "", with_roi: bool = False):
        """
        Save the image (ROI) in a NIFTI file and update reading path.

        :param path: The path to the folder were the image (ROI) will be saved. (Ex: "Documents/Data").
                     If no value are give, the last path used to read the image will be used.
        :param with_roi: A boolean that indicate if the ROI should also be save.
        """
        assert self.__img is not None, "The image not loaded in memory"

        self.__path = self.__path if not path else path  # Update the path
        self.__img.to_filename(self._get_path())

        if with_roi:
            self.__path_roi = self.__path_roi if not path else path  # Update the roi path
            self.__roi.to_filename(self._get_path(roi=True))

    def spatial_to_voxel(self, spatial_pos: list) -> np.array:
        """
        Convert spatial position into voxel position

        :param spatial_pos: A list that correspond to the spatial location in mm.
        :return: A numpy array that correspond to the position in the voxel.
        """
        affine = self.get_nifti().affine
        affine = npl.inv(affine)

        m = affine[:3, :3]
        translation = affine[:3, 3]
        return m.dot(spatial_pos) + translation

    def voxel_to_spatial(self, voxel_pos: list) -> np.array:
        """
        Convert voxel position into spatial position.

        :param voxel_pos: A list that correspond to the location in voxel.
        :return: A numpy array that correspond to the spatial position in mm.
        """
        affine = self.get_nifti().affine
        m = affine[:3, :3]
        translation = affine[:3, 3]
        return m.dot(voxel_pos) + translation

    def to_canonical(self, save: bool = False, save_path: str = ""):
        """
        Modify the orientation of the image and the ROI to be as closest canonical possible. So the orientation
        will be set to RAS (Left to Right, Posterior to Anterior, Inferior to Superior).

        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        img = self.get_nifti()
        roi = self.get_nifti(roi=True)

        self.__img = nib.as_closest_canonical(img)
        roi = nib.as_closest_canonical(roi)

        self.__read_metadata()
        self.update_roi(new_roi=np.array(roi.dataobj), save=save, save_path=save_path)

    def transfer_header(self, npy_dir: str, nifti_dir: str, medomics_code_path: str,
                        save: bool = False, save_path: str = ""):
        """
        Transfer the header of a nifti file created with MRIcroGL to the NIFTI image and ROI
        generated by the medomics code.

        :param npy_dir: The directory where the npy files are stored.
        :param nifti_dir: The directory where the nifti files with the correct header are stored.
        :param medomics_code_path: The path directory of the medomics code.
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """
        study_time = self.__read_study_time(npy_dir=npy_dir, medomics_code_path=medomics_code_path)
        path_begin = os.path.join(nifti_dir, "DICOM_" + self.__modality + "_" + study_time)
        filenames = glob.glob(str(path_begin) + '*.nii.gz')

        for file in filenames:
            ref_img = nib.load(file)
            is_match, new_img, new_roi = self.__find_rotation(ref_img)

            if is_match:
                self.__img = nib.Nifti1Image(new_img, affine=ref_img.affine, header=ref_img.header)
                self.__read_metadata()
                self.update_roi(new_roi, save=save, save_path=save_path)
                break
        else:
            raise Exception("No corresponding image find for patient {}".format(
                self.__filename
            ))

    def update_roi(self, new_roi: np.array, save: bool = False, save_path: str = ""):
        """
        Change the dataobj of the ROI and update the roi_measure.

        :param new_roi: A numpy array that represent the new region of interest
        :param save: A boolean that indicate if we need to save the image after this operation.
                     If keep memory is false, than the image will be saved either if save is true or false.
        :param save_path: A string that indicate the path where the images will be save
        """

        assert (np.array(new_roi.shape) == self.__metadata['img_shape']).all(), \
            "The new ROI do not have same shape as the image." "New roi shape: {}, image shape: {}".format(
                new_roi.shape, self.__metadata['img_shape'])

        self.__roi = nib.Nifti1Image(new_roi, affine=self.__img.affine, header=self.__img.header)
        self.__compute_roi_measure()
        if save:
            self.save_image(path=save_path, with_roi=True)
