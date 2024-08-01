import re
from functools import partial
from typing import Dict, List, Union

from bidi.algorithm import get_display

import laia.common.logging as log
from laia.utils.symbols_table import SymbolsTable

_logger = log.get_logger(__name__)


def rtl_word(word: str, pattern: re.Pattern) -> str:
    return "".join(reversed(list(filter(None, pattern.split(word)))))


def rtl(sentence: str, tokens: set) -> str:
    pattern = re.compile(rf"{'|'.join(f'({re.escape(word)})' for word in tokens)}")
    return " <space> ".join(
        reversed(
            list(map(partial(rtl_word, pattern=pattern), sentence.split(" <space> ")))
        )
    )


def untokenize(sentence: str) -> str:
    return "".join(sentence.split()).replace("<space>", " ")


def tokenize(sentence: str) -> str:
    return " ".join([token if token != " " else "<space>" for token in list(sentence)])


class ToTensor:
    def __init__(
        self, syms: Union[Dict, SymbolsTable], reading_order: str = "LTR"
    ) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms
        self.reading_order = reading_order

    def __call__(self, x: str) -> List[int]:
        if self.reading_order == "RTL":
            x = untokenize(x)
            x = get_display(x)
            x = tokenize(x)
        values = []
        for c in x.split():
            v = (
                self._syms.get(c, None)
                if isinstance(self._syms, Dict)
                else self._syms[c]
            )
            if v is None:
                _logger.error('Could not find "{}" in the symbols table', c)
            values.append(v)
        return values

    def __repr__(self) -> str:
        return f"text.{self.__class__.__name__}()"
