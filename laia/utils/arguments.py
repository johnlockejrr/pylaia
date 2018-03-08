import argparse

from laia.utils.arguments_types import str2bool, NumberInClosedRange, \
    NumberInOpenRange

_parser = None
_default_args = {
    'batch_size': (
        ('--batch_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'default': 8,
            'help': 'Batch size (must be >= 1)'
        }),
    'learning_rate': (
        ('--learning_rate',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'default': 0.0005,
            'help': 'Learning rate (must be > 0)'
        }),
    'momentum': (
        ('--momentum',),
        {
            'type': NumberInClosedRange(type=float, vmin=0),
            'default': 0,
            'help': 'Momentum (must be >= 0)'
        }),
    'gpu': (
        ('--gpu',),
        {
            'type': int,
            'default': 1,
            'help': 'Use this GPU (starting from 1)'
        }),
    'seed': (
        ('--seed',),
        {
            'type': int,
            'default': 0x12345,
            'help': 'Seed for random number generators'
        }),
    'final_fixed_height': (
        ('--final_fixed_height',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'default': 20,
            'help': 'Final height for the pseudo-images after the convolutions '
                    '(must be >= 1)'
        }),
    'max_epochs': (
        ('--max_epochs',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Maximum number of training epochs'
        }),
    'min_epochs': (
        ('--min_epochs',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Minimum number of training epochs'
        }),
    'valid_cer_std_window_size': (
        ('--valid_cer_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the validation CER (must be >= 2)'
        }),
    'valid_cer_std_threshold': (
        ('--valid_cer_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the validation '
                    'CER is below this threshold (must be > 0)'
        }),
    'valid_map_std_window_size': (
        ('--valid_map_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the validation Mean Average Precision (mAP) ' 
                    '(must be >= 2)'
        }),
    'valid_map_std_threshold': (
        ('--valid_map_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the validation '
                    'Mean Average Precision (mAP) is below this threshold ' 
                    '(must be > 0)'
        }),
    'show_progress_bar': (
        ('--show_progress_bar',),
        {
            'type': str2bool,
            'nargs': '?',
            'const': True,
            'default': False,
            'help': 'Whether or not to show a progress bar for each epoch'
        }),
    'use_distortions': (
        ('--use_distortions',),
        {
            'type': str2bool,
            'nargs': '?',
            'const': True,
            'default': True,
            'help': 'Whether or not to use dynamic distortions to augment the '
                    'training data'
        }),
    'train_loss_std_threshold': (
        ('--train_loss_std_threshold',),
        {
            'type': NumberInOpenRange(type=float, vmin=0),
            'help': 'Stop training if the standard deviation of the training '
                    'loss is below this threshold (must be > 0)'
        }),
    'train_loss_std_window_size': (
        ('--train_loss_std_window_size',),
        {
            'type': NumberInClosedRange(type=int, vmin=2),
            'help': 'Use this number of epochs to compute the standard '
                    'deviation of the training loss (must be >= 2)'
        }),
    'num_samples_per_epoch': (
        ('--num_samples_per_epoch',),
        {
            'type': NumberInClosedRange(type=int, vmin=1),
            'help': 'Use this number of training examples randomly sampled '
                    'from the dataset in each epoch'
        }),
    'num_iterations_per_update': (
        ('--num_iterations_per_update',),
        {
            'default': 1,
            'type': NumberInClosedRange(type=int, vmin=1),
            'metavar': 'N',
            'help': 'Update parameters every N iterations'
        }),
    'weight_l2_penalty': (
        ('--weight_l2_penalty',),
        {
            'default': 0.0,
            'type': NumberInClosedRange(type=float, vmin=0),
            'help': 'Apply this L2 weight penalty to the loss function'
        })
}


def _get_parser():
    global _parser
    if not _parser:
        _parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return _parser


def add_defaults(*args, **kwargs):
    # If called without arguments add all
    if len(args) == 0 and len(kwargs) == 0:
        for args_, kwargs_ in _default_args.values():
            add_argument(*args_, **kwargs_)
    # Otherwise add only those given
    else:
        for arg in args:
            args_, kwargs_ = _default_args[arg]
            add_argument(*args_, **kwargs_)
        for arg, default_value in kwargs.items():
            args_, kwargs_ = _default_args[arg]
            kwargs_['default'] = default_value
            add_argument(*args_, **kwargs_)
    return _parser


def add_argument(*args, **kwargs):
    _get_parser().add_argument(*args, **kwargs)


def args():
    return _get_parser().parse_args()
