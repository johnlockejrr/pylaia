import re
from typing import Dict, List, Union

from bidi.algorithm import get_display

import laia.common.logging as log
from laia.utils.symbols_table import SymbolsTable

_logger = log.get_logger(__name__)


def untokenize(
    sentence: str, space_token: str = "<space>", space_display: str = " "
) -> str:
    return "".join(sentence.split()).replace(space_token, space_display)


def tokenize(
    sentence: str,
    space_token: str = "<space>",
    space_display: str = " ",
    symbols: set = {},
) -> str:
    pattern = re.compile(rf"{'|'.join(f'({re.escape(word)})' for word in symbols)}")
    return " ".join(
        [
            space_token if token == space_display else token
            for token in list(filter(None, pattern.split(sentence)))
        ]
    )


class ToTensor:
    def __init__(
        self,
        syms: Union[Dict, SymbolsTable],
        reading_order: str = "LTR",
        space_token: str = "<space>",
        space_display: str = " ",
    ) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms
        self.reading_order = reading_order
        self.space_token = space_token
        self.space_display = space_display

    def __call__(self, x: str) -> List[int]:
        if self.reading_order == "RTL":
            x = untokenize(
                x, space_token=self.space_token, space_display=self.space_display
            )
            x = get_display(x)
            x = tokenize(
                x,
                space_token=self.space_token,
                space_display=self.space_display,
                symbols=set(self._syms._sym2val.keys()),
            )
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
