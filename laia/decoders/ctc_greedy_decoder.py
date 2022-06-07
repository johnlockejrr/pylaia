from typing import Any, Dict, List

import torch

from laia.losses.ctc_loss import transform_batch


class CTCGreedyDecoder:
    def __call__(self, x: Any, segmentation: bool = False) -> Dict[str, List]:
        x, xs = transform_batch(x)
        x = x.detach()

        # Apply softmax to have log-probabilities
        x = torch.nn.functional.log_softmax(x, dim=-1)

        # Transform to list where len(list) = batch_size
        x = [x[: xs[i], i, :] for i in range(len(xs))]
        x = [x_n.max(dim=1) for x_n in x]

        out = {}
        if segmentation:
            out["prob-segmentation"] = [x_n.values.exp() for x_n in x]
            out["segmentation"] = [
                CTCGreedyDecoder.compute_segmentation(x_n.indices.tolist()) for x_n in x
            ]

        probs = [x_n.values.exp() for x_n in x]
        x = [x_n.indices for x_n in x]

        # Remove repeated symbols
        zero_tensor = torch.tensor([0]).cuda() if x[0].is_cuda else torch.tensor([0])
        counts = [
            torch.unique_consecutive(x_n, return_counts=True)[1] for x_n in x
        ]  # counts of consecutive symbols [0, 0, 0, 1, 2, 2] => [3, 1, 2]
        idxs = [
            torch.cat((zero_tensor, count.cumsum(0)[:-1])) for count in counts
        ]  # compute index to keep [0, 3, 4] (always keep the first index, then use cumulative sum of counts tensor)
        x = [x[i][idxs[i]] for i in range(len(x))]  # keep only non consecutive symbols
        probs = [
            probs[i][idxs[i]] for i in range(len(x))
        ]  # keep only probabilities associated to non consecutives symbols

        # Remove CTC blank symbol
        idxs = [
            torch.nonzero(x_n, as_tuple=True) for x_n in x
        ]  # get index for non blank symbols
        x = [x[i][idxs[i]] for i in range(len(x))]  # keep only non blank symbols
        probs = [
            probs[i][idxs[i]] for i in range(len(x))
        ]  # keep only probabilities associated to non blank symbols

        # Save results
        out["hyp"] = [x_n.tolist() for x_n in x]

        # Return char-based probability and overall probability
        out["prob-htr-char"] = [
            prob.tolist() for prob in probs
        ]  # to compute word-based probability later
        return out

    @staticmethod
    def compute_segmentation(x: List[int]) -> List[int]:
        if not x:
            return []
        return [0] + [i for i in range(1, len(x)) if x[i] != x[i - 1] != 0] + [len(x)]
