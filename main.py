import numpy as np
import cv2
import collections.abc
from scipy import stats
import matplotlib.pyplot as plt


class Stack(collections.abc.Sequence):

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        return self.images[i]

    def __init__(self, z_size, *, image_vector=None, file_path=None):
        if file_path:
            self.images = cv2.imreadmulti(file_path)[1]
        else:
            self.images = image_vector
        self.size = [z_size, int(len(self.images) / z_size)]


def get_blur_value(image):
    canny = cv2.Canny(image, 50, 250)
    return np.nanmean(canny)


def sharpest_images(image_stack: Stack):
    blur_list = np.zeros(image_stack.size)
    for i, image in enumerate(image_stack):
        blur_list[i % image_stack.size[0], int(np.floor(i / image_stack.size[0]))] = get_blur_value(image)
    sharpest_z_list = np.nanargmax(blur_list, axis=0)
    image_vector = [image_stack[t * image_stack.size[0] + z] for t, z in enumerate(sharpest_z_list)]
    sharp_stack = Stack(1, image_vector=image_vector)
    return sharp_stack


def function(z_size, image_path):
    my_stack = Stack(z_size, file_path=image_path)
    cannys_ = []
    cannys_x_means_ = []
    for image in my_stack:
        cannys_.append(canny_img := apply_canny(image).astype(np.float32) / 255)
        cannys_x_means_.append(np.mean(canny_img, axis=1))
    matrix_col_means = np.stack(cannys_x_means_, axis=-1)
    max_col_idx = np.argmax(matrix_col_means, axis=1)
    max_col_idx = np.round(cv2.blur(max_col_idx, (1,100))).astype(np.int32).ravel()

    new_image_list = []
    for line, sharp_im in enumerate(max_col_idx):
        new_image_list.append(my_stack[sharp_im][line, :])
    new_image_matrix = np.stack(new_image_list, axis=0)
    print(new_image_matrix.shape)




def apply_canny(image):
    return cv2.Canny(image, 10, 20)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # cv2.imwritemulti(
    #     'D:\\Nucleus\\16-04-2021\\MediumHighDensity\\SharpDIA_MediumHighDensity_16042021.tif',
    #     sharpest_images(
    #         Stack(
    #             11,
    #             file_path='D:\\Nucleus\\16-04-2021\\MediumHighDensity\\DIA_MediumHighDensity_16042021.tif')
    #     )
    # )
    function(6, r'D:\Nucleus\16-04-2021\MediumLowDensity\Actin\Corner.tif')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
