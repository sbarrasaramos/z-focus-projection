import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PLOT_SIZE = (1000, 600)


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
