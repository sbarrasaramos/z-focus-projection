import collections.abc
import math
from typing import Tuple

import cv2
import more_itertools
import numpy as np

DEFAULT_MOVING_AVG_WINDOW = (150, 150)
DEFAULT_CANNY_THRESHOLDS = (10, 100)  # 10, 20


def _apply_canny(
    image: np.ndarray,
    threshold_1: int = DEFAULT_CANNY_THRESHOLDS[0],
    threshold_2: int = DEFAULT_CANNY_THRESHOLDS[1],
) -> np.ndarray:
    return cv2.Canny(image, threshold_1, threshold_2)


def _matrix_blur(
    matrix: np.ndarray,
    window_x: int = DEFAULT_MOVING_AVG_WINDOW[0],
    window_y: int = DEFAULT_MOVING_AVG_WINDOW[1],
) -> np.ndarray:
    """
    Returns the 2D moving average of the input matrix over the specified window
    :param matrix: Input array/image
    :param window_x: Size of column subsets for the moving average
    :param window_y: Size of row subsets for the moving average
    :return: Output array/image
    """
    return cv2.blur(matrix, (window_x, window_y))


class Stack(collections.abc.MutableSequence):
    """
    Class containing a one-dimensional (in space or time) stack of images
    """

    def __init__(self, image_vector=None):
        if image_vector is None:
            image_vector = []
        self.images = image_vector

    def __setitem__(self, i: int, o: np.ndarray) -> None:
        self.images.__setitem__(i, o)

    def insert(self, index: int, value: np.ndarray) -> None:
        self.images.insert(index, value)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        return self.images.__getitem__(i)

    def __delitem__(self, key):
        del self.images[key]

    def normalize(self, normalization_factor=None):
        if normalization_factor is None:
            normalization_factor = int(255 / max(np.max(image) for image in self))
        self.images = [image * normalization_factor for image in self]

    @classmethod
    def frompath(cls, file_path):
        return cls(cv2.imreadmulti(file_path)[1])


class ZStack(Stack):
    """
    Class containing a one-dimensional stack of images
    in the direction perpendicular to them
    """

    def row_sharp_projection(
        self,
        canny_threshold_1=DEFAULT_CANNY_THRESHOLDS[0],
        canny_threshold_2=DEFAULT_CANNY_THRESHOLDS[1],
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Returns a single image containing the sharpest focus of the stack for every row,
        by calculating the row-wise average of the result of an edge detection algorithm
        and picking the highest value
        :param canny_threshold_1: Edge detection lower threshold
        :param canny_threshold_2: Edge detection upper threshold
        :return:
        """

        # Implement and insert tuning_canny_parameters

        canny_xwise_means = [
            np.mean(
                _apply_canny(image, canny_threshold_1, canny_threshold_2).astype(
                    np.float32
                )
                / 255,
                axis=1,
            )
            for image in self
        ]
        sharpest_row_indexes = np.argmax(np.stack(canny_xwise_means, axis=-1), axis=1)
        smooth_sharpest_row_indexes = np.round(
            _matrix_blur(sharpest_row_indexes).ravel()
        ).astype(np.int32)

        # Implement and insert tuning_moving_average

        # Take the line of the image corresponding to the index
        # and stack it into a new image
        sharp_image_matrix = np.stack(
            [
                self[sharp_im][line, :]
                for line, sharp_im in enumerate(smooth_sharpest_row_indexes)
            ],
            axis=0,
        )

        return sharp_image_matrix, smooth_sharpest_row_indexes

    def roi_sharp_projection(self, roi_size) -> Tuple[np.ndarray, np.ndarray]:

        """
        Returns a single image containing the sharpest focus of the stack for every ROI,
        by calculating the ROI-wise average of the result of an edge detection algorithm
        and picking the highest value
        :param roi_size: Number of pixels in the x direction of the ROI
        :return:
        """

        edges_stack = ZStack([_apply_canny(image) for image in self])

        # Implement and insert tuning_canny_parameters

        piece_number = int(math.ceil(edges_stack.images[0].shape[1] / roi_size))
        canny_xwise_means = [
            [
                np.mean(_apply_canny(piece).astype(np.float32) / 255, axis=1)
                for piece in np.array_split(image, piece_number, axis=1)
            ]
            for image in edges_stack
        ]
        sharpest_piece_indexes = np.argmax(canny_xwise_means, axis=0).T

        # Build whole index matrix and then the image: allows for moving average
        sharpest_pixel_indexes = []
        for line, sharp_indexes in enumerate(sharpest_piece_indexes):
            index_line = []
            for piece, sharp_index in enumerate(sharp_indexes):
                index_chunk = [sharp_index] * roi_size
                index_line = [*index_line, *index_chunk]
            sharpest_pixel_indexes.append(index_line)
        sharpest_indexes_array = np.array(sharpest_pixel_indexes)
        smooth_sharpest_indexes_array = np.round(_matrix_blur(sharpest_indexes_array))

        # Implement and insert tuning_moving_average

        sharp_image_matrix = sum(
            [
                np.multiply(image, smooth_sharpest_indexes_array == z_coord)
                for z_coord, image in enumerate(self.images)
            ]
        )

        return sharp_image_matrix, smooth_sharpest_indexes_array


class HyperStack(collections.abc.Sequence):
    """
    Class containing a two-dimensional stack of images:
    one of the directions is perpendicular to them
    """

    def __len__(self) -> int:
        return len(self.z_stacks)

    def __getitem__(self, i: int):
        return self.z_stacks[i]

    def __init__(self, z_size: int, image_vector):
        self.images = image_vector
        self.z_stacks = [
            ZStack(image_vector=z_stack)
            for z_stack in more_itertools.chunked(self.images, z_size)
        ]
        self.size = (z_size, len(self.z_stacks))

    @classmethod
    def frompath(cls, z_size, file_path):
        return cls(z_size, cv2.imreadmulti(file_path)[1])

    def row_sharp_projection(self):
        """
        Applies row_sharp_projection to every ZStack in the Hyperstack
        :return: Image Stack in the direction perpendicular to z
        """
        sharp_stack = Stack(
            [z_stack.row_sharp_projection()[0] for z_stack in self.z_stacks]
        )
        return sharp_stack, "foo"

    def roi_sharp_projection(self, piece_size):
        """
        Applies roi_sharp_projection to every ZStack in the Hyperstack
        :return: Image Stack in the direction perpendicular to z
        """
        new_stack = Stack(
            [z_stack.roi_sharp_projection(piece_size)[0] for z_stack in self.z_stacks]
        )
        return new_stack, "foo"


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Option 1: using the algorithm on a simple ZStack
    # cv2.imwritemulti(
    #     'D:\\Nucleus\\16-04-2021\\MediumHighDensity\\SharpDIA_MediumHighDensity_16042021.tif',
    #     sharpest_images(
    #         Stack(
    #             11,
    #             file_path='D:\\Nucleus\\16-04-2021\\MediumHighDensity\\DIA_MediumHighDensity_16042021.tif')
    #     )
    # )

    # Option 2: using the algorithm on a 2D Hyperstack (Z & t, or Z & x,...)
    my_hyperstack = HyperStack.frompath(
        31,
        file_path=r"C:\Users\sbarr\Documents\PhD\Experiments\
        Nucleus\03-05-2021\Green03052021.tif",
    )
    sharp_stack, _ = my_hyperstack.roi_sharp_projection(256)
    # sharp_stack, _ = my_hyperstack.row_sharp_projection()
    sharp_stack.normalize()
    cv2.imwritemulti(
        r"C:\Users\sbarr\Documents\PhD\Experiments\Nucleus\
        03-05-2021\SharpV02_Green03052021.tif",
        sharp_stack.images,
    )
