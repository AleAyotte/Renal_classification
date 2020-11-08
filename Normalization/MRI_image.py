import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os.path
import string


class MRIimage:
    """

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
    Methods
    -------
    detach()
        Release memory taken by the image and its ROI.
    load_image()
        Load the image and its ROI in memory.

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

        self.__read_metadata()

    def apply_n4(self, save: bool = False, save_path=""):
        """

        :param save:
        :param save_path:
        :return:
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
        :return:
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

        :param roi:
        :return:
        """
        if not roi:
            file_path = os.path.join(self.__path, self.__filename)
        else:
            file_path = os.path.join(self.__path_roi, self.__filename_roi)
        return file_path + ".nii.gz"

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

        :param _slice:
        :param axis:
        :param roi:
        :return:
        """
        assert axis.lower() in ["axial", "coronal", "sagittal", "all"]
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

        :param roi:
        :return:
        """
        img = nib.load(self._get_path(roi))

        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
            img = nib.as_closest_canonical(img)

        return img

    def __read_metadata(self):
        """

        :return:
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

        :param path:
        :param with_roi:
        :return:
        """
        assert self.__img is not None, "The image not loaded in memory"

        self.__path = self.__path if not path else path  # Update the path
        self.__img.to_filename(self._get_path())

        if with_roi:
            self.__path_roi = self.__path_roi if not path else path  # Update the roi path
            self.__roi.to_filename(self._get_path(roi=True))
