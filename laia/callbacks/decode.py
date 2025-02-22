from typing import Callable, Optional, Union

import pytorch_lightning as pl
from bidi import get_display
from tqdm.auto import tqdm

import laia.common.logging as log
from laia.common.arguments import ReadingOrder
from laia.data.transforms.text import tokenize, untokenize
from laia.decoders import CTCGreedyDecoder
from laia.utils import SymbolsTable

_logger = log.get_logger(__name__)


def compute_word_prob(symbols, hyp, prob, input_separator):
    """
    Compute confidence score for each word.
    Returns a list of word-level confidence scores.
    """
    space_id = symbols._sym2val[input_separator]
    word_prob_list = []
    word_prob, word_chars = [], ""
    for value, prob in zip(hyp, prob):
        char = symbols[value]
        if value != space_id:
            word_prob.append(prob)
            word_chars += char
        elif word_chars:
            word_prob_list.append(sum(word_prob) / len(word_prob))
            word_prob, word_chars = [], ""
    if word_chars:
        word_prob_list.append(sum(word_prob) / len(word_prob))
    return word_prob_list


class Decode(pl.Callback):
    def __init__(
        self,
        decoder: Optional[Callable] = CTCGreedyDecoder(),
        syms: Optional[Union[dict, SymbolsTable]] = None,
        use_symbols: bool = False,
        input_space: str = "<space>",
        output_space: str = " ",
        convert_spaces: bool = False,
        join_string: Optional[str] = None,
        separator: str = " ",
        include_img_ids: bool = True,
        print_line_confidence_scores: bool = False,
        print_word_confidence_scores: bool = False,
        reading_order: str = "LTR",
    ):
        super().__init__()
        self.decoder = decoder
        self.syms = syms
        self.use_symbols = use_symbols
        if use_symbols:
            assert syms is not None
        self.input_space = input_space
        self.output_space = output_space
        self.convert_spaces = convert_spaces
        if convert_spaces:
            assert use_symbols
        self.join_string = join_string
        self.separator = separator
        self.include_img_ids = include_img_ids
        self.print_line_confidence_scores = print_line_confidence_scores
        self.print_word_confidence_scores = print_word_confidence_scores
        self.reading_order = reading_order

        if self.reading_order == ReadingOrder.RTL and not convert_spaces:
            # cannot use get_display() if convert_spaces is False
            _logger.warn(
                "The reading order is set to RTL (Right-to-Left), but '--decode.convert_spaces' is set to False. "
                "This will cause LTR (Left-to-Right) text predictions because PyLaia uses the bidi algorithm to handle RTL text. "
                "For predictions in RTL order, consider setting '--decode.convert_spaces true'."
            )

    @property
    def print_confidence_scores(self):
        return self.print_word_confidence_scores or self.print_line_confidence_scores

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args)
        img_ids = pl_module.batch_id_fn(batch)
        hyps = self.decoder(outputs)["hyp"]

        line_probs = (
            self.decoder(outputs)["prob-htr"]
            if self.print_line_confidence_scores
            else []
        )

        if self.print_word_confidence_scores:
            probs = self.decoder(outputs)["prob-htr-char"]
            word_probs = [
                compute_word_prob(self.syms, hyp, prob, self.input_space)
                for hyp, prob in zip(hyps, probs)
            ]

        else:
            probs = []
            word_probs = []

        for i, (img_id, hyp) in enumerate(zip(img_ids, hyps)):
            if self.use_symbols:
                hyp = [self.syms[v] for v in hyp]
                if self.reading_order == ReadingOrder.RTL:
                    hyp = get_display(untokenize(" ".join(hyp)))
                    hyp = tokenize(hyp).split(" ")

                if self.convert_spaces:
                    hyp = [
                        self.output_space if sym == self.input_space else sym
                        for sym in hyp
                    ]
            if self.join_string is not None:
                hyp = self.join_string.join(str(x) for x in hyp).strip()

            if self.print_confidence_scores:
                if self.print_word_confidence_scores:
                    word_prob = [f"{prob:.2f}" for prob in word_probs[i]]
                    if self.reading_order == ReadingOrder.RTL:
                        word_prob.reverse()
                    self.write(
                        f"{img_id}{self.separator}{word_prob}{self.separator}{hyp}"
                        if self.include_img_ids
                        else f"{word_prob}{self.separator}{hyp}"
                    )

                else:
                    line_prob = line_probs[i]
                    self.write(
                        f"{img_id}{self.separator}{line_prob:.2f}{self.separator}{hyp}"
                        if self.include_img_ids
                        else f"{line_prob:.2f}{self.separator}{hyp}"
                    )

            else:
                self.write(
                    f"{img_id}{self.separator}{hyp}"
                    if self.include_img_ids
                    else str(hyp)
                )

    def write(self, value):
        # no idea why adding the line break is necessary. in distributed mode with
        # >1 gpus some lines will not include it. also happens with print() so it's
        # not a tqdm issue. couldn't reproduce it on toy examples but it does
        # happen in the iam-htr example
        return tqdm.write(value + "\n", end="")
