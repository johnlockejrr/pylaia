#!/usr/bin/env python3
from typing import Any, Dict, List, Optional

import jsonargparse
from jsonargparse.typing import NonNegativeInt

import laia.common.logging as log
from laia.common.arguments import (
    CommonArgs,
    TrainArgs,
)
from laia.common.loader import ModelLoader
from laia.scripts.htr import common_main
from laia.utils import ImageLabelsStats, Statistics, SymbolsTable


def run(
    syms: str,
    img_dirs: List[str],
    tr_txt_table: str,
    va_txt_table: str,
    te_txt_table: str,
    fixed_input_height: NonNegativeInt,
    common: CommonArgs = CommonArgs(),
    train: TrainArgs = TrainArgs(),
):
    # Check model
    loader = ModelLoader(
        common.train_path, filename=common.model_filename, device="cpu"
    )
    model = loader.load()
    assert (
        model is not None
    ), "Could not find the model. Have you run pylaia-htr-create-model?"

    # Prepare the symbols
    syms = SymbolsTable(syms)
    for d in train.delimiters:
        assert d in syms, f'The delimiter "{d}" is not available in the symbols file'

    outname = "statistics.md"
    mdfile = Statistics(outname)

    for split in ["train", "val", "test"]:
        # Check for missing image
        dataset_stats = ImageLabelsStats(
            stage=split,
            tr_txt_table=tr_txt_table,
            va_txt_table=va_txt_table,
            te_txt_table=te_txt_table,
            img_dirs=img_dirs,
        )

        # Check if images have variable height
        if fixed_input_height > 0:
            assert dataset_stats.is_fixed_height, f"Found images with variable heights in {split} set: {dataset_stats.get_invalid_images_height(fixed_input_height)}."

        # Check if characters are in syms
        for char in dataset_stats.character_set:
            assert (
                char in syms
            ), f'The character "{char}" is not in the symbols file but appears in the {split} set'

        # Check if images are too small
        min_valid_width = model.get_min_valid_image_size(dataset_stats.max_width)
        if dataset_stats.min_width < min_valid_width:
            log.warning(
                f"Found some images too small for convolutions (width<{min_valid_width}). They will be padded during training."
            )

        # Write markdown section
        mdfile.create_split_section(
            split,
            dataset_stats.widths,
            dataset_stats.heights,
            dataset_stats.labels,
            train.delimiters,
        )

    log.info("Dataset is correct")

    # Write markdown statistics file
    mdfile.document.create_md_file()
    log.info(f"Statistics written to {outname}")


def get_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--config", action=jsonargparse.ActionConfigFile, help="Configuration file"
    )
    parser.add_argument(
        "syms",
        type=str,
        help=(
            "Mapping from strings to integers. "
            "The CTC symbol must be mapped to integer 0"
        ),
    )
    parser.add_argument(
        "img_dirs",
        type=List[str],
        default=[],
        help="Directories containing segmented line images",
    )
    parser.add_argument(
        "tr_txt_table",
        type=str,
        help="Train labels",
    )
    parser.add_argument(
        "va_txt_table",
        type=str,
        help="Val labels",
    )
    parser.add_argument(
        "te_txt_table",
        type=str,
        help="Test labels",
    )
    parser.add_argument(
        "--fixed_input_height",
        type=NonNegativeInt,
        default=0,
        help=(
            "Height of the input images. If 0, a variable height model "
            "will be used (see `adaptive_pooling`). This will be used to compute the "
            "model output height at the end of the convolutional layers"
        ),
    )
    parser.add_class_arguments(CommonArgs, "common")
    parser.add_class_arguments(TrainArgs, "train")
    parser.add_function_arguments(log.config, "logging")

    args = parser.parse_args(argv, with_meta=False).as_dict()
    args["common"] = CommonArgs(**args["common"])
    args["train"] = TrainArgs(**args["train"])
    return args


def main() -> None:
    args = get_args()
    args = common_main(args)
    run(**args)


if __name__ == "__main__":
    main()
