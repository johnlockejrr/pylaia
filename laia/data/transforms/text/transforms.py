import re
from typing import Dict, List, Union

from bidi.algorithm import get_display

import laia.common.logging as log
from laia.utils.symbols_table import SymbolsTable

_logger = log.get_logger(__name__)


def untokenize(
    tokenized_sentence: str, space_token: str = "<space>", space_display: str = " "
) -> str:
    """
    Untokenize the input text from PyLaia format.


    Args:
        tokenized_sentence (str): The tokenized text to untokenize.
        space_token (str, optional): The token to use for spaces in the tokenized string. Defaults to "<space>".
        space_display (str, optional): The token used to represent spaces in the input string. Defaults to " ".

    Returns:
        str: The untokenized text.

    Example:
        untokenize("T h i s <space> i s <space> a n <space> e x a m p l e")
        >>> "This is an example"
        untokenize("T h e <space> O l y m p i c s <space> G a m e s <space> a r e <space> o r g a n i z e d <space> i n <space> <location>P a r i s")
        >>> "The Olympics Games are organized in <location>Paris"
    """
    return "".join(tokenized_sentence.split()).replace(space_token, space_display)


def tokenize(
    sentence: str,
    space_token: str = "<space>",
    space_display: str = " ",
    symbols: set = {},
) -> str:
    """
    Tokenize the input text in PyLaia format.

    Args:
        sentence (str): The input text.
        space_token (str, optional): The token to use for spaces in the tokenized string. Defaults to "<space>".
        space_display (str, optional): The token used to represent spaces in the input string. Defaults to " ".
        symbols (set, optional): The set of symbols from PyLaia vocabulary. Defaults to {}.

    Returns:
        str: The tokenized text with spaces replaced by `space_display`.

    Example:
        untokenize("This is an example")
        >>> "T h i s <space> i s <space> a n <space> e x a m p l e"
        tokenize("The Olympics Games are organized in <location>Paris")
        >>> "T h e <space> O l y m p i c s <space> G a m e s <space> a r e <space> o r g a n i z e d <space> i n <space> <location>P a r i s"
    """
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
