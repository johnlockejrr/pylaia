import pytest
from PIL import Image

from laia.utils.stats import ImageLabelsStats

TR_TXT_TABLE = """
tmp-0 a a e b c
tmp-1 a a b b c
tmp-2 d a b b c
"""

VA_TXT_TABLE = """
tmp-3 a a a a c
tmp-4 b b f b c
tmp-5 b a b b c
"""

TE_TXT_TABLE = """
tmp-6 a a a a c
tmp-7 b b b b c
tmp-8 b a b b c
"""

IMG_LIST = """
tmp-6
tmp-7
tmp-8
"""


def prepare_images(data_dir, image_ids, sizes):
    for image_id, size in zip(image_ids, sizes):
        im = Image.new(mode="L", size=size)
        im.save(str(data_dir / f"{image_id}.jpg"))


def prepare_training_data(data_dir, split, image_ids, sizes):
    prepare_images(data_dir, image_ids, sizes)
    txt_file = "tr.txt" if split == "train" else "va.txt"
    txt_table = data_dir / txt_file
    txt_table.write_text(TR_TXT_TABLE if split == "train" else VA_TXT_TABLE, "utf-8")
    return [data_dir], str(txt_table)


def prepare_test_data(data_dir, sizes):
    prepare_images(data_dir, IMG_LIST.split(), sizes)
    img_list = data_dir / "img_list.txt"
    img_list.write_text(IMG_LIST, "utf-8")
    return [data_dir], str(img_list)


@pytest.mark.parametrize(
    "train_image_ids, val_image_ids, train_sizes, val_sizes, expected_max_width, expected_is_fixed_height, expected_charset, expected_height",
    [
        (
            ["tmp-0", "tmp-1", "tmp-2"],
            ["tmp-3", "tmp-4", "tmp-5"],
            [(556, 100)],
            [(556, 100)],
            556,
            True,
            {"a", "b", "c", "e"},
            100,
        ),
        (
            ["tmp-0", "tmp-1", "tmp-2"],
            ["tmp-3", "tmp-4", "tmp-5"],
            [(556, 100), (1150, 100), (1200, 100)],
            [(556, 100), (600, 100), (1200, 100)],
            1200,
            True,
            {"a", "b", "c", "d", "e", "f"},
            100,
        ),
        (
            ["tmp-0", "tmp-1", "tmp-2"],
            ["tmp-3", "tmp-4", "tmp-5"],
            [(556, 100), (600, 110), (1200, 100)],
            [(600, 110), (1150, 100), (1200, 100)],
            1200,
            False,
            {"a", "b", "c", "d", "e", "f"},
            100,
        ),
    ],
)
def test_img_stats_fit_stage(
    tmpdir,
    train_image_ids,
    val_image_ids,
    train_sizes,
    val_sizes,
    expected_max_width,
    expected_is_fixed_height,
    expected_charset,
    expected_height,
):
    img_dirs, tr_txt_table = prepare_training_data(
        tmpdir, "train", train_image_ids, train_sizes
    )
    va_txt_table = prepare_training_data(tmpdir, "val", val_image_ids, val_sizes)[1]
    img_stats = ImageLabelsStats(
        stage="fit",
        tr_txt_table=tr_txt_table,
        va_txt_table=va_txt_table,
        img_dirs=img_dirs,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height
    assert img_stats.character_set == expected_charset
    if img_stats.is_fixed_height:
        img_stats.min_height == expected_height


@pytest.mark.parametrize(
    "sizes, expected_max_width, expected_is_fixed_height, expected_height",
    [
        ([(546, 100)], 546, True, 100),
        ([(250, 100), (395, 100), (1150, 100)], 1150, True, 100),
        ([(556, 128), (600, 110), (1200, 100)], 1200, False, None),
    ],
)
def test_img_stats_test_stage(
    tmpdir, sizes, expected_max_width, expected_is_fixed_height, expected_height
):
    img_dirs, img_list = prepare_test_data(tmpdir, sizes)
    img_stats = ImageLabelsStats(
        stage="test",
        img_list=img_list,
        img_dirs=img_dirs,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height
    if img_stats.is_fixed_height:
        img_stats.min_height == expected_height
    assert img_stats.labels == []


@pytest.mark.parametrize(
    "split, image_ids, sizes, expected_max_width, expected_is_fixed_height, expected_charset, expected_height",
    [
        ("train", ["tmp-0"], [(556, 100)], 556, True, {"b", "a", "e", "c"}, 100),
        (
            "train",
            ["tmp-0", "tmp-1", "tmp-2"],
            [(556, 100), (600, 100), (1150, 100)],
            1150,
            True,
            {"a", "b", "c", "d", "e"},
            100,
        ),
        (
            "train",
            ["tmp-0", "tmp-1", "tmp-2"],
            [(556, 100), (600, 110), (1150, 100)],
            1150,
            False,
            {"a", "b", "c", "d", "e"},
            None,
        ),
        ("val", ["tmp-3"], [(556, 100)], 556, True, {"a", "c"}, 100),
        (
            "val",
            ["tmp-3", "tmp-4", "tmp-5"],
            [(556, 100), (600, 100), (1150, 100)],
            1150,
            True,
            {"a", "b", "c", "f"},
            100,
        ),
        (
            "val",
            ["tmp-3", "tmp-4", "tmp-5"],
            [(556, 100), (2000, 110), (1150, 100)],
            2000,
            False,
            {"a", "b", "c", "f"},
            None,
        ),
    ],
)
def test_img_stats_train_val_stage(
    tmpdir,
    split,
    image_ids,
    sizes,
    expected_max_width,
    expected_is_fixed_height,
    expected_charset,
    expected_height,
):
    img_dirs, txt_table = prepare_training_data(tmpdir, split, image_ids, sizes)
    img_stats = ImageLabelsStats(
        stage=split,
        tr_txt_table=txt_table if split == "train" else None,
        va_txt_table=txt_table if split == "val" else None,
        img_dirs=img_dirs,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height
    assert img_stats.character_set == expected_charset
    if img_stats.is_fixed_height:
        img_stats.min_height == expected_height
