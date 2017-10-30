import logging

from os import listdir
from os.path import isfile, join, splitext

from .text_image_dataset import TextImageDataset

_VALID_IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.pbm', '.pgm', '.ppm', '.bmp')

class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(self, txt_table, imgs_dir, img_transform=None,
                 txt_transform=None, valid_extensions=_VALID_IMAGE_EXTENSIONS):
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, imgs_dir, valid_extensions)
        # Prepare dataset using the previous image filenames and transcripts.
        super(TextImageFromTextTableDataset, super).__init__(
            imgs, txts, img_transform, txt_transform)

    def __getitem__(self, index):
        img, txt = super(TextImageFromTextTableDataset, super).__getitem__(index)
        return self._ids[index], img, txt

def _get_valid_image_filenames_from_dir(imgs_dir, valid_extensions):
    valid_extensions = set(valid_extensions)
    valid_image_filenames = {}
    for fname in listdir(imgs_dir):
        bname, ext = splitext(fname)
        fname = join(imgs_dir, fname)
        if isfile(fname) and ext.lower() in valid_extensions:
            valid_image_filenames[bname] = fname
    return valid_image_filenames

def _load_text_table_from_file(table_file):
    if isinstance(txt_table, (str, unicode)):
        txt_table = open(txt_table, 'r')

    for n, line in enumerate(txt_table, 1):
        line = line.split()
        # Skip empty lines and lines starting with #
        if len(line) == 0 or line[0][0] == '#':
            continue
        yield n, line[0], line[1:]

    txt_table.close()

def _get_images_and_texts_from_text_table(table_file, imgs_dir, valid_exts):
    imgid2fname = _get_valid_image_filenames_from_dir(imgs_dir, valid_exts)
    ids, imgs, txts = [], [], []
    for _, imgid, txt in _load_text_table_from_file(table_table):
        fname = imgid2fname.get(imgid)
        if fname is None:
            logging.warning('No image file was found in folder "%s" for image '
                            'ID "%s", ignoring example...', imgs_dir, imgid)
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
            txts.append(txt)

    return ids, imgs, txts


