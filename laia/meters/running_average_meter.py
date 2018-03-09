import math
import numpy as np
import torch

from laia.meters.meter import Meter

from typing import Union


class RunningAverageMeter(Meter):
    """Computes the running average and standard deviation of a set of values.

    Some examples:

    >>> RunningAverageMeter().add(5).value
    (5.0, 0.0)

    >>> RunningAverageMeter().add(5).add(-5).value
    (0.0, 5.0)
    """

    def __init__(self):
        self._n = 0.0
        self._s = 0.0
        self._s2 = 0.0

    def reset(self):
        """Reset the running average and standard deviation.

        Returns:
            the :obj:`.RunningAverageMeter` (`self`)
        """
        self._n = 0.0
        self._s = 0.0
        self._s2 = 0.0
        return self

    def add(self, v):
        """Add a new value (or set of values) to the running average.

        If the value is list or tuple, a PyTorch tensor or variable, or a Numpy
        Ndarray, each of its elements will be added.

        Arguments:
            v : value or set of values.
        Returns:
            the :obj:`.RunningAverageMeter` (`self`)
        """
        if torch.is_tensor(v):
            self._n += v.numel()
            self._s += torch.sum(v)
            self._s2 += torch.sum(v * v)
        elif isinstance(v, torch.autograd.Variable) and torch.is_tensor(v.data):
            self._n += v.data.numel()
            self._s += torch.sum(v.data)
            self._s2 += torch.sum(v.data * v.data)
        elif isinstance(v, np.ndarray):
            self._n += v.size
            self._s += np.sum(v)
            self._s2 += np.sum(v * v)
        elif isinstance(v, (list, tuple)):
            self._n += len(v)
            self._s += sum(v)
            self._s2 += sum(map(lambda x: x * x, v))
        else:
            self._n += 1
            self._s += v
            self._s2 += v * v
        return self

    @property
    def value(self):
        avg = float(self._s) / float(self._n)
        # Note: The max is to avoid precision issues.
        var = max(0.0, float(self._s2) / float(self._n) - avg * avg)
        return avg, math.sqrt(var)
