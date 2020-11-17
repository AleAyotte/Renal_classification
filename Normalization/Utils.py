import numpy as np
import matplotlib.pyplot as plt


def rotate_and_compare(img: np.array, ref_img: np.array, rotation_set: int = 0, plot: bool = False):
    """
    Apply a set of rotation to an image and verify if the image fit with a given image of reference.

    :param img: A numpy array that represent the image to rotate.
    :param ref_img: A numpy array that represent the image of reference to match.
    :param rotation_set: A integer that indicate the set of rotation to use on the image.
    :param plot: if true, plot the images.
    :return: The rotated image as np.array and a boolean that indicate if the rotated image fit with the
             image of reference.
    """
    if rotation_set == 0:
        rotated_img = np.swapaxes(img, 0, 2)
        rotated_img = np.rot90(rotated_img, 1, (0, 2))
        rotated_img = np.flip(rotated_img, 0)

    elif rotation_set == 1:
        rotated_img = np.swapaxes(img, 1, 2)
        rotated_img = np.flip(rotated_img, 1)

    elif rotation_set == 2:
        rotated_img = img

    else:
        raise NotImplementedError

    if np.shape(rotated_img) != np.shape(ref_img):
        return rotated_img, False

    else:
        diff = np.abs(rotated_img - ref_img) / max(np.max(rotated_img), np.max(ref_img))

        if plot:
            print(np.mean(rotated_img))
            print(np.max(rotated_img))
            print(np.min(rotated_img))

            print("\n", np.mean(ref_img))
            print(np.max(ref_img))
            print(np.min(ref_img))

            print("\n", np.mean(diff))
            print(np.max(diff))
            print(np.unravel_index(np.argmax(diff), diff.shape))

            fig = plt.figure(figsize=(24, 30))
            plt.set_cmap(plt.gray())

            fig.add_subplot(1, 2, 1, title="Image 1")
            plt.imshow(rotated_img[:, :, 10])

            fig.add_subplot(1, 2, 2, title="Image 2")
            plt.imshow(ref_img[:, :, 10])

            plt.show()

        return rotated_img, (diff < 1e-2).all()