from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import List, Optional, TextIO, Union

import imagesize

from laia.data.image_from_list_dataset import _get_img_ids_and_filepaths
from laia.data.text_image_from_text_table_dataset import (
    _get_images_and_texts_from_text_table,
)


class Split(Enum):
    """
    Split names
    """

    train = "train"
    val = "val"
    test = "test"


class ImageLabelsStats:
    """
    Compute statistics on the dataset

    Args:
        stage: String indicating the stage of the processing, either "test" or "fit"
        tr_txt_table: Path to the train text table (train mode)
        va_txt_table: Path to the validation text table (train mode)
        img_list: Path to the list of test images (test mode)
        img_dirs: Path to images
    """

    def __init__(
        self,
        stage: str,
        tr_txt_table: Optional[Union[TextIO, str, List[str]]] = None,
        va_txt_table: Optional[Union[TextIO, str, List[str]]] = None,
        te_txt_table: Optional[Union[TextIO, str, List[str]]] = None,
        img_list: Optional[Union[TextIO, str, List[str]]] = None,
        img_dirs: Optional[Union[List[str], str, List[Path], Path]] = None,
    ):
        assert stage in ["fit", "test", "train", "val"]

        if stage == "fit":
            assert tr_txt_table and va_txt_table
            filenames = _get_images_and_texts_from_text_table(tr_txt_table, img_dirs)[1]
            filenames += _get_images_and_texts_from_text_table(va_txt_table, img_dirs)[
                1
            ]

            labels = _get_images_and_texts_from_text_table(tr_txt_table, img_dirs)[2]
            labels += _get_images_and_texts_from_text_table(va_txt_table, img_dirs)[2]

        elif stage == "test":
            assert img_list or te_txt_table
            if te_txt_table:
                _, filenames, labels = _get_images_and_texts_from_text_table(
                    te_txt_table, img_dirs
                )
            else:
                filenames = _get_img_ids_and_filepaths(img_list, img_dirs)[1]
                labels = []

        elif stage == "train":
            assert tr_txt_table
            _, filenames, labels = _get_images_and_texts_from_text_table(
                tr_txt_table, img_dirs
            )

        elif stage == "val":
            assert va_txt_table
            _, filenames, labels = _get_images_and_texts_from_text_table(
                va_txt_table, img_dirs
            )

        sizes = list(map(imagesize.get, filenames))
        self.widths, self.heights = zip(*sizes)
        self.labels = [x.split() for x in labels]
        self.filenames = filenames

    @cached_property
    def character_set(self) -> int:
        """
        Get the set of characters
        """
        return set([char for line in self.labels for char in line])

    @cached_property
    def min_width(self) -> int:
        """
        Compute the minimum width of images
        """
        return min(self.widths)

    @cached_property
    def max_width(self) -> int:
        """
        Compute the maximum width of images
        """
        return max(self.widths)

    @cached_property
    def min_height(self) -> bool:
        """
        Compute the minimum height of images
        """
        return min(self.heights)

    @cached_property
    def max_height(self) -> bool:
        """
        Compute the maximum height of images
        """
        return max(self.heights)

    @cached_property
    def is_fixed_height(self) -> bool:
        """
        Check if all images have the same height
        """
        return self.max_height == self.min_height

    def get_invalid_images_height(self, expected_height: int) -> List[str]:
        """
        List images with invalid height
        """
        return [
            filename
            for filename, height in zip(self.filenames, self.heights)
            if height != expected_height
        ]
