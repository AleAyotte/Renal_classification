import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os.path


class MRIimage:
    """
    Read two NIFTI files that correspond to a MRI image and it region of interest (ROI) respectively and perform
    different normalization and other preprocessing method. Morever, its also compute some usefull measure about
    image and ROI that can be used to determine the best normalization and correction process.

    ...
    Attributes
    ----------
    __path : string
        The directory name that contain the image and its ROI.
    __filename : string
        The filename of the nifti image. (Exemple: "Kidney-TCGA-008__T1C")
    __keep_mem : bool
        Indicate if the image and its ROI is keep in memory.
    __modality : string
        Modality of the image. (Exemple: T1C or T2WI)
    __img : nib.nifti1.Nifti1Image
        The nifti image.
    __roi : nib.nifti1.Nifti1Image
        The region of interest (ROI) of the image.
    __metadata : Dict
        A dictionnary that contain the major metadata that will be used to normalize the image.
            img_shape : List(int)
                Number of voxel along each dimension.
            voxel_spacing : List(float)
                The voxels dimensions in mm.
            image_spacing : List(float)
                The images dimensions in mm.
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
    detach()
        Release memory taken by the image and its ROI.
    load_image()
        Load the image and its ROI in memory.
    plot(_slice, axis: str = "all", roi: bool = False)
        Load and plot the image (ROI) along a given axis or along each axis.
    save_image(path: str = "", with_roi: bool = False)
        Save the image (ROI) in a NIFTI file and update reading path.
    """
    def __init__(self, path_image: str, path_roi: str, modality: str, keep_mem: bool = True):

        self.__path = os.path.dirname(path_image)
        self.__path_roi = os.path.dirname(path_roi)
        self.__filename = os.path.splitext(os.path.basename(path_image))[0]
        self.__filename_roi = os.path.splitext(os.path.basename(path_roi))[0]
        self.__keep_mem = keep_mem
        self.__modality = modality
        self.__img = None
        self.__roi = None
        self.__metadata = {'img_shape': [],
                           'voxel_spacing': [],
                           'image_spacing': []}
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
            ants_img = ants.image_read(self._get_path())
        else:
            ants_img = from_nibabel(self.__img)
        corrected_img = to_nibabel(n4(ants_img))

        self.__img = corrected_img

        if save or not self.__keep_mem:
            self.save_image(path=save_path)

        if not self.__keep_mem:
            self.detach()

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

        if save or not self.__keep_mem:
            self.save_image(path=save_path)

        if not self.__keep_mem:
            self.detach()

    def detach(self):
        """
        Release memory taken by the image and its ROI.
        """
        self.__keep_mem = False
        self.__img = None
        self.__roi = None

    def __compute_roi_measure(self):
        """
        Compute some usefull statistics and measure about
        """
        roi = self.get_roi()

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
        self.__roi_measure['center_voxel'] = [round((x[1] + x[0])/2) for x in coord]
        self.__roi_measure['center_mm'] = [self.__metadata['voxel_spacing'][i] * (coord[i][1] + coord[i][0])/2
                                           for i in range(3)]

    def get_nifti(self, roi: bool = False):
        if (self.__img is None and not roi) or (self.__roi is None and roi):
            return self._read_image(roi)
        else:
            return self.__roi if roi else self.__img

    def get_img(self):
        return np.array(self.get_nifti().dataobj)

    def get_roi(self):
        return np.array(self.get_nifti(roi=True).dataobj)

    def _get_path(self, roi: bool = False):
        """
        Return the complete path (directory + filename) were the image (ROI) is (or will be) saved.

        :param roi: A boolean that indicate if we want the path of ROI instead of the image path.
        :return: A string that represent the path.
        """
        if not roi:
            file_path = os.path.join(self.__path, self.__filename)
        else:
            file_path = os.path.join(self.__path_roi, self.__filename_roi)
        return file_path + ".nii.gz"

    def get_metadata(self):
        return self.__metadata

    def get_roi_measure(self):
        if len(self.__roi_measure['length_mm']) == 0:
            self.__compute_roi_measure()
        return self.__roi_measure

    def load_img(self):
        """
        Load the nifti image and its ROI in memory
        """
        self.__keep_mem = True
        self.__img = self._read_image()
        self.__roi = self._read_image(roi=True)

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

    def _read_image(self, roi: bool = False):
        """
        Read the image (ROI) NIFTI file and adjust the image orientation according to the NIFTI standard.

        :param roi: A boolean that indicate if we want to read the ROI nifti file instead of the image.
        :return: A nib.nifti1.Nifti1Image object that reprensent the image (ROI) with its metadata.
        """
        img = nib.load(self._get_path(roi))

        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
            img = nib.as_closest_canonical(img)

        return img

    def __read_metadata(self):
        """
        Read the header of the NIFTI image and get some major metadata about the image like voxel dimension.
        """
        img = self._read_image()
        header = img.header

        self.__metadata['img_shape'] = header['dim'][1:4]
        self.__metadata['voxel_spacing'] = header['pixdim'][1:4]
        self.__metadata['image_spacing'] = [
            header['dim'][i]*header['pixdim'][i] for i in range(1, 4)
        ]

        if self.__keep_mem:
            self.load_img()

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
