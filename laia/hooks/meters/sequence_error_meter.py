from __future__ import absolute_import

import editdistance

from laia.hooks.meters import Meter


class SequenceErrorMeter(Meter):
    def __init__(self, exceptions_threshold=5):
        super(SequenceErrorMeter, self).__init__(exceptions_threshold)
        self._num_errors = 0
        self._ref_length = 0

    def reset(self):
        self._num_errors = 0
        self._ref_length = 0
        return self

    def add(self, refs, hyps):
        assert hasattr(refs, '__iter__') or hasattr(refs, '__getitem__')
        assert hasattr(hyps, '__iter__') or hasattr(hyps, '__getitem__')
        assert hasattr(refs, '__len__') and hasattr(hyps, '__len__')
        assert len(refs) == len(hyps)
        for ref, hyp in zip(refs, hyps):
            self._num_errors += editdistance.eval(ref, hyp)
            self._ref_length += len(ref)
        return self

    @property
    def value(self):
        return float(self._num_errors) / float(self._ref_length)
