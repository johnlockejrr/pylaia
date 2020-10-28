import argparse
from typing import Any, Union

import pytorch_lightning as pl

from laia.common.arguments_types import (
    NumberInClosedRange,
    NumberInOpenRange,
    str2loglevel,
)

default_args = {
    "batch_size": (
        ("--batch_size",),
        {
            "type": NumberInClosedRange(type=int, vmin=1),
            "default": 8,
            "help": "Batch size (must be >= 1)",
        },
    ),
    "learning_rate": (
        ("--learning_rate",),
        {
            "type": NumberInOpenRange(type=float, vmin=0),
            "default": 0.0005,
            "help": "Learning rate (must be > 0)",
        },
    ),
    "momentum": (
        ("--momentum",),
        {
            "type": NumberInClosedRange(type=float, vmin=0),
            "default": 0.0,
            "help": "Momentum (must be >= 0)",
        },
    ),
    "weight_l2_penalty": (
        ("--weight_l2_penalty",),
        {
            "default": 0.0,
            "type": NumberInClosedRange(type=float, vmin=0),
            "help": "Apply this L2 weight penalty to the loss function",
        },
    ),
    "nesterov": (
        ("--nesterov",),
        {
            "type": pl.utilities.parsing.str_to_bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "Whether to use Nesterov momentum",
        },
    ),
    "seed": (
        ("--seed",),
        {
            "type": lambda x: int(x, 0),
            "default": 0x12345,
            "help": "Seed for random number generators",
        },
    ),
    "color_mode": (
        ("--color_mode",),
        {
            "type": str,
            "default": "L",
            "choices": ["L", "RGB", "RGBA"],
            "help": "L (grayscale): 1 channel, RGB: 3 channels, RGBA: 4 channels",
        },
    ),
    "use_distortions": (
        ("--use_distortions",),
        {
            "type": pl.utilities.parsing.str_to_bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "Whether to use dynamic distortions to augment the training data",
        },
    ),
    "train_path": (
        ("--train_path",),
        {"type": str, "default": "", "help": "Save any files in this location"},
    ),
    "monitor": (
        ("--monitor",),
        {
            "type": str,
            "default": "va_cer",
            "choices": [f"va_{m}" for m in ("loss", "cer", "wer")],
            "help": "Metric to monitor for early stopping and checkpointing",
        },
    ),
    "checkpoint": (
        ("--checkpoint",),
        {
            "type": str,
            "default": "",
            "help": (
                "Any of: 1 - Nothing: load the best checkpoint with respect to the "
                "--monitor, checkpoints will be searched in the --experiment_dirname "
                "directory. 2 - A filepath: the filepath to the checkpoint (e.g. "
                '"/tmp/model.ckpt"). 3 - A filename: the filename of the checkpoint to '
                'load inside the --experiment_dirname directory (e.g. "model.ckpt"). '
                '4 - A glob pattern: e.g. "epoch=*.ckpt" will load the checkpoint '
                "with the highest epoch number, globbing will be done inside the "
                "--experiment_dirname directory"
            ),
        },
    ),
    "checkpoint_k": (
        ("--checkpoint_k",),
        {
            "type": int,
            "default": 3,
            "help": (
                "-1: all models are saved. "
                "0: no models are saved. "
                "k: the best k models will be saved"
            ),
        },
    ),
    "model_filename": (
        ("--model_filename",),
        {"type": str, "default": "model", "help": "File name of the model"},
    ),
    "experiment_dirname": (
        ("--experiment_dirname",),
        {
            "type": str,
            "default": "experiment",
            "help": (
                "Directory name of the experiment. "
                "It will be created inside of --train_path"
            ),
        },
    ),
    "logging_also_to_stderr": (
        ("--logging_also_to_stderr",),
        {
            "default": "ERROR",
            "type": str2loglevel,
            "help": (
                "If you are logging to a file, use this level for logging "
                "also to stderr (use any of: debug, info, warning, error, "
                "critical)"
            ),
        },
    ),
    "logging_level": (
        ("--logging_level",),
        {
            "default": "INFO",
            "type": str2loglevel,
            "help": (
                "Use this level for logging (use any of: debug, info, "
                "warning, error, critical)"
            ),
        },
    ),
    "logging_file": (
        ("--logging_file",),
        {"type": str, "help": "Write the logs to this file"},
    ),
    "logging_overwrite": (
        ("--logging_overwrite",),
        {
            "type": pl.utilities.parsing.str_to_bool,
            "nargs": "?",
            "const": True,
            "default": False,
            "help": "If true, overwrite the logfile instead of appending it",
        },
    ),
    "print_args": (
        ("--print_args",),
        {
            "type": pl.utilities.parsing.str_to_bool,
            "nargs": "?",
            "const": True,
            "default": True,
            "help": "If true, log to INFO the arguments passed to the program",
        },
    ),
}


def get_key(mapping: Union[dict, argparse.Namespace], key: Any) -> Any:
    default = default_args[key][1]["default"]
    if isinstance(mapping, argparse.Namespace):
        return getattr(mapping, key, default)
    return mapping.get(key, default)


class LaiaParser:
    parser = None

    def get(self) -> "LaiaParser":
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                conflict_handler="resolve",
            )
            self.add_defaults(
                "logging_also_to_stderr",
                "logging_file",
                "logging_level",
                "logging_overwrite",
                "print_args",
            )
        return self

    def add_defaults(self, *args, **kwargs) -> "LaiaParser":
        for arg in args:
            args_, kwargs_ = default_args[arg]
            self.add_argument(*args_, **kwargs_)
        for arg, default_value in kwargs.items():
            args_, kwargs_ = default_args[arg]
            kwargs_["default"] = default_value
            self.add_argument(*args_, **kwargs_)
        return self

    def add_argument(self, *args, **kwargs) -> "LaiaParser":
        self.get().parser.add_argument(*args, **kwargs)
        return self

    def parse_args(
        self, *args, should_log: bool = True, **kwargs
    ) -> argparse.Namespace:
        ns = self.get().parser.parse_args(*args, **kwargs)
        if should_log and ns.print_args:
            import laia.common.logging as log

            log.config_from_args(ns)
            log.get_logger(__name__).info(str(vars(ns)))
        return ns


def group_to_namespace(
    args: argparse.Namespace, group: argparse._ArgumentGroup, dest: str
):
    """Moves arguments in a group to their own namespace"""
    ns = argparse.Namespace()
    for a in group._group_actions:
        setattr(ns, a.dest, getattr(args, a.dest, None))
        delattr(args, a.dest)
    setattr(args, dest, ns)
    return args


# TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/3076
# TODO: upstream blocklist
def add_lightning_args(parser, blocklist=None):
    import pytorch_lightning.utilities.parsing as parsing
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.utilities.argparse_utils import (
        _gpus_allowed_type,
        _gpus_arg_default,
        _int_or_float_type,
        get_init_arguments_and_types,
        parse_args_from_docstring,
    )

    cls = Trainer

    if blocklist is None:
        blocklist = []
    blocklist += ["kwargs"]
    depr_arg_names = cls.get_deprecated_arg_names() + blocklist

    allowed_types = (str, int, float, bool)

    args_help = parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__)
    for arg, arg_types, arg_default in (
        at for at in get_init_arguments_and_types(cls) if at[0] not in depr_arg_names
    ):
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = parsing.str_to_bool
            # if only two args (str, bool)
            elif len(arg_types) == 2 and set(arg_types) == {str, bool}:
                use_type = parsing.str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == "gpus" or arg == "tpu_cores":
            use_type = _gpus_allowed_type
            arg_default = _gpus_arg_default

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == "track_grad_norm":
            use_type = float

        parser.add_argument(
            f"--{arg}",
            dest=arg,
            default=arg_default,
            type=use_type,
            help=args_help.get(arg),
            **arg_kwargs,
        )

    return parser
