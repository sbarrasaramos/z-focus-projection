from typing import Union

from plotting import plot_images
import typer
from z_projection import _apply_canny
from z_projection import DEFAULT_CANNY_THRESHOLDS
from z_projection import HyperStack
from z_projection import Stack


def tuning_canny_parameters(image_set: Union[Stack, HyperStack]):
    lower_threshold, upper_threshold = DEFAULT_CANNY_THRESHOLDS
    for image in image_set.images:
        while True:
            plot_images(_apply_canny(image, lower_threshold, upper_threshold))
            change = typer.confirm(
                f"Edge detection thresholds are {lower_threshold} "
                f"and {upper_threshold}. "
                f"Do you want to change them? [y/n]"
            )
            if change:
                lower_threshold = typer.prompt(
                    "Please, input the new lower threshold: "
                )
                upper_threshold = typer.prompt(
                    "Please, input the new upper threshold: "
                )
            else:
                break
            next_image = typer.confirm(
                "Do you want to proceed to the next image? [y/n]"
            )
            if next_image:
                break
            else:
                continue

    return lower_threshold, upper_threshold


def tuning_moving_average():
    pass
