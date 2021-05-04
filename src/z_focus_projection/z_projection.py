import collections.abc
import math

import cv2
import matplotlib.pyplot as plt
import more_itertools
import numpy as np

DEFAULT_MOVING_AVG_WINDOW = (150, 150)
DEFAULT_CANNY_THRESHOLDS = (10, 100)  # 10, 20
DEFAULT_PLOT_SIZE = (1000, 600)


def apply_canny(
    image: np.ndarray,
    threshold_1: int = DEFAULT_CANNY_THRESHOLDS[0],
    threshold_2: int = DEFAULT_CANNY_THRESHOLDS[1],
) -> np.ndarray:
    return cv2.Canny(image, threshold_1, threshold_2)


def moving_average(
    image: np.ndarray,
    window_x: int = DEFAULT_MOVING_AVG_WINDOW[0],
    window_y: int = DEFAULT_MOVING_AVG_WINDOW[1],
) -> np.ndarray:
    return cv2.blur(image, (window_x, window_y))


class Stack(collections.abc.MutableSequence):
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
    def row_sharp_projection(self):

        # We need to plot the canny edges results of one or more images
        # for the user to calibrate on the go
        # edges_stack = ZStack([apply_canny(image) for image in self])
        # for image in edges_stack.images:
        #     plot_images(image)

        canny_xwise_means = [
            np.mean(apply_canny(image).astype(np.float32) / 255, axis=1)
            for image in self
        ]
        sharpest_row_indexes = np.argmax(np.stack(canny_xwise_means, axis=-1), axis=1)
        smooth_sharpest_row_indexes = np.round(
            moving_average(sharpest_row_indexes).ravel()
        ).astype(np.int32)

        # I think we should plot this to help the user... --> Parameter calibration
        plt.plot(smooth_sharpest_row_indexes)
        plt.show()

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

    def piece_sharp_projection(self, piece_size):
        edges_stack = ZStack([apply_canny(image) for image in self])

        # We need to plot the canny edges results of one or more images
        # for the user to calibrate on the go

        piece_number = int(math.ceil(edges_stack.images[0].shape[1] / piece_size))
        canny_xwise_means = [
            [
                np.mean(apply_canny(piece).astype(np.float32) / 255, axis=1)
                for piece in np.array_split(image, piece_number, axis=1)
            ]
            for image in edges_stack
        ]
        sharpest_piece_indexes = np.argmax(canny_xwise_means, axis=0).T

        # Option 1: build image matrix from sharp_indexes
        # new_matrix = []
        # for line, sharp_indexes in enumerate(sharpest_piece_indexes):
        #     index_line = []
        #     for piece, sharp_index in enumerate(sharp_indexes):
        #         index_chunk = self[sharp_index][
        #           line, piece*piece_size:(piece+1)*piece_size
        #         ].tolist()
        #         index_line = [*index_line, *index_chunk]
        #     new_matrix.append(index_line)
        # sharp_image_matrix = np.array(new_matrix)

        # Option 2: build index matrix and then the image: allows for moving average
        sharpest_pixel_indexes = []
        for line, sharp_indexes in enumerate(sharpest_piece_indexes):
            index_line = []
            for piece, sharp_index in enumerate(sharp_indexes):
                index_chunk = [sharp_index] * piece_size
                index_line = [*index_line, *index_chunk]
            sharpest_pixel_indexes.append(index_line)
        sharpest_indexes_array = np.array(sharpest_pixel_indexes)
        smooth_sharpest_indexes_array = np.round(moving_average(sharpest_indexes_array))

        # I think we should plot this to help the user and implement
        # something to be able to select the parameters on the go
        plt.plot(sharpest_piece_indexes)
        plt.show()

        sharp_image_matrix = sum(
            [
                np.multiply(image, smooth_sharpest_indexes_array == z_coord)
                for z_coord, image in enumerate(self.images)
            ]
        )

        return sharp_image_matrix, "foo"


class HyperStack(collections.abc.Sequence):
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
        new_stack = Stack(
            [z_stack.row_sharp_projection()[0] for z_stack in self.z_stacks]
        )
        return new_stack, "foo"

    def piece_sharp_projection(self, piece_size):
        new_stack = Stack(
            [z_stack.piece_sharp_projection(piece_size)[0] for z_stack in self.z_stacks]
        )
        return new_stack, "foo"


def plot_line(line: np.ndarray) -> None:
    plt.plot(line)
    plt.show()


def plot_images(*images: np.ndarray) -> None:
    for idx, img in enumerate(images):
        img = img.astype(np.uint8)
        if images[0].shape > DEFAULT_PLOT_SIZE:
            img = cv2.resize(img, DEFAULT_PLOT_SIZE)
        cv2.imshow(f"image {idx}", img)
    cv2.waitKey()


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
    sharp_stack, _ = my_hyperstack.piece_sharp_projection(256)
    # sharp_stack, _ = my_hyperstack.row_sharp_projection()
    sharp_stack.normalize()
    cv2.imwritemulti(
        r"C:\Users\sbarr\Documents\PhD\Experiments\Nucleus\
        03-05-2021\SharpV02_Green03052021.tif",
        sharp_stack.images,
    )
