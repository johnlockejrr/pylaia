from __future__ import absolute_import

import io
from os import listdir

from os.path import isfile, join, splitext
from torch._six import string_classes

import laia.logging as log
from laia.data.text_image_dataset import TextImageDataset

IMAGE_EXTENSIONS = '.jpg', '.png', '.jpeg', '.pbm', '.pgm', '.ppm', '.bmp'

_logger = log.get_logger(__name__)


class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(self, txt_table, imgs_dir, img_transform=None,
                 txt_transform=None, img_extensions=IMAGE_EXTENSIONS,
                 encoding='utf8'):
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, imgs_dir, img_extensions, encoding=encoding)
        # Prepare dataset using the previous image filenames and transcripts.
        super(TextImageFromTextTableDataset, self).__init__(
            imgs, txts, img_transform, txt_transform)

    def __getitem__(self, index):
        """Returns the ID of the example, the image and its transcript from
        the dataset.

        Args:
          index (int): Index of the item to return.

        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img') and
            the transcript ('txt') of the image.
        """
        out = super(TextImageFromTextTableDataset, self).__getitem__(index)
        out['id'] = self._ids[index]
        return out


def _get_valid_image_filenames_from_dir(imgs_dir, img_extensions):
    img_extensions = set(img_extensions)
    valid_image_filenames = {}
    for fname in listdir(imgs_dir):
        bname, ext = splitext(fname)
        fname = join(imgs_dir, fname)
        if isfile(fname) and ext.lower() in img_extensions:
            valid_image_filenames[bname] = fname
    return valid_image_filenames


def find_image_filename_from_id(img_id, img_dir, img_extensions):
    extensions = set(ext.lower() for ext in img_extensions)
    extensions.update(ext.upper() for ext in img_extensions)
    for ext in extensions:
        fname = join(img_dir, img_id if img_id.endswith(ext) else img_id + ext)
        if isfile(fname):
            return fname
    return None


def _load_text_table_from_file(table_file, encoding='utf8'):
    if isinstance(table_file, string_classes):
        with io.open(table_file, 'r', encoding=encoding) as f:
            for n, line in enumerate((l.split() for l in f), 1):
                # Skip empty lines and lines starting with #
                if not len(line) or line[0].startswith('#'):
                    continue
                yield n, line[0], line[1:]


def _get_images_and_texts_from_text_table(table_file, imgs_dir, img_extensions, encoding='utf8'):
    ids, imgs, txts = [], [], []
    for _, imgid, txt in _load_text_table_from_file(table_file, encoding=encoding):
        fname = find_image_filename_from_id(imgid, imgs_dir, img_extensions)
        if fname is None:
            _logger.warning('No image file was found in folder "{}" for image '
                            'ID "{}", ignoring example...', imgs_dir, imgid)
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
            txts.append(txt)

    return ids, imgs, txts
