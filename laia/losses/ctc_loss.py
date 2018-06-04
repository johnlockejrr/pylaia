from __future__ import absolute_import

import itertools
from typing import List, Union, Sequence

import torch
from torch.autograd import Variable, Function
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

try:
    from warpctc_pytorch import _CTC
except ImportError:
    import warnings

    warnings.warn("Missing CTC loss function library")


def transform_output(output):
    # Size: T x N x D
    if isinstance(output, PackedSequence):
        acts, act_lens = pad_packed_sequence(output)
    elif torch.is_tensor(output):
        acts, act_lens = output, [output.size(0)] * output.size(1)
    else:
        raise NotImplementedError("Not implemented for type {}".format(type(output)))
    if isinstance(acts, Variable):
        acts = acts.data
    return acts, act_lens


def copy_valid_indices(
    acts,  # type: Union[torch.Tensor, Variable]
    labels,  # type: List[List[int]]
    act_lens,  # type: List[int]
    label_lens,  # type: List[int]
    valid_indices,  # type: List[int]
):
    # type: (...) -> (torch.Tensor, List[List[int]], List[int], List[int])
    """Copy the CTC inputs without the erroneous samples"""
    if len(valid_indices) == 0:
        return None, [], [], []
    # Note: The batch size must be in the second dimension
    seq_length, _, output_dim = acts.size()
    acts_copy = acts.new(seq_length, len(valid_indices), output_dim)
    for new_idx, ok_idx in enumerate(valid_indices):
        acts_copy[:, new_idx, :] = acts[:, ok_idx, :]
    return (
        acts_copy,
        [labels[i] for i in valid_indices],
        [act_lens[i] for i in valid_indices],
        [label_lens[i] for i in valid_indices],
    )


def set_zeros_in_errors(size, input, valid_indices):
    # type: (Union[torch.Size, Sequence[int]], torch.Tensor, Sequence[int]) -> torch.Tensor
    """Copy the tensor with zeros in the erroneous indices"""
    # Note: The batch size must be in the second dimension
    return torch.zeros(size).index_copy_(1, torch.LongTensor(valid_indices), input)


def get_valids_and_errors(act_lens, label_lens):
    # type: (List[int], List[int]) -> (List[int], List[int])
    """Check for errors by comparing the size of the
    output against the size of the target."""
    assert len(act_lens) == len(label_lens)
    check = [act_lens[i] > 2 * label_lens[i] + 1 for i in range(len(act_lens))]
    return (
        # Indices of OK samples
        [i for i, valid in enumerate(check) if valid],
        # Indices of the samples with errors regarding their ref length
        [i for i, valid in enumerate(check) if not valid],
    )


class CTC(Function):

    @staticmethod
    def forward(ctx, output, target):
        # type: (torch.Tensor, List[List[int]]) -> (torch.Tensor, torch.IntTensor * 4)
        acts, act_lens = transform_output(output)
        assert act_lens[0] == acts.size(0), "Maximum length does not match"
        assert len(target) == acts.size(1), "Batch size does not match"

        label_lens = [len(y) for y in target]
        valid_indices, err_indices = get_valids_and_errors(act_lens, label_lens)

        acts, labels, act_lens, label_lens = (
            copy_valid_indices(acts, target, act_lens, label_lens, valid_indices)
            if err_indices
            else (acts, target, act_lens, label_lens)
        )

        # Prepare tensors of the correct type
        act_lens = torch.IntTensor(act_lens)
        labels = torch.IntTensor(list(itertools.chain.from_iterable(labels)))
        label_lens = torch.IntTensor(label_lens)
        err_indices = torch.IntTensor(err_indices)

        ctx.saved = valid_indices, err_indices, acts.size()
        return (acts, labels, act_lens, label_lens, err_indices)

    @staticmethod
    def backward(ctx, grad_output, *_):
        if isinstance(grad_output, Variable):
            grad_output = grad_output.data
        valid_indices, err_indices, original_size = ctx.saved
        return (
            set_zeros_in_errors(original_size, grad_output, valid_indices)
            if err_indices
            else grad_output,
            None,
        )


class CTCLoss(Module):
    """
    Attributes:
        size_average (optional): normalize the loss by the batch size
            (default: `True`)
        length_average (optional): normalize the loss by the total number of frames
            in the batch. If `True`, supersedes `size_average`
            (default: `False`)
    """

    def __init__(self, size_average=True, length_average=False):
        # type: (bool, bool) -> None
        super(CTCLoss, self).__init__()
        self._size_average = size_average
        self._length_average = length_average

    def forward(self, output, target):
        # type: (torch.Tensor, List[List[int]]) -> (Union[float, torch.FloatTensor], List[int])
        """
        Arguments:
            output: Size seqLength x outputDim, contains
                the output from the network as well as a list of size
                (seqLength) containing batch sizes of the sequence
            target: Contains the size of each output
                sequence from the network. Size batchSize
        """
        acts, labels, act_lens, label_lens, err_indices = CTC.apply(output, target)
        """
        acts: Contains the output from the network.
            Size seqLength x batchSize x outputDim
        labels: Contains all the targets of the batch
            in one sequence. 1 dimensional
        act_lens: Contains the size of each output
            sequence from the network. Size batchSize
        label_lens: Contains the label length of each
            sample. Size batchSize
        """
        return (
            _CTC.apply(
                acts,
                labels,
                act_lens,
                label_lens,
                self._size_average,
                self._length_average,
            ),
            err_indices.data.tolist(),
        )
