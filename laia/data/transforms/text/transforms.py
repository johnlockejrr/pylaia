from typing import Dict, List, Union

from bidi.algorithm import get_display

import laia.common.logging as log
from laia.utils.symbols_table import SymbolsTable

_logger = log.get_logger(__name__)


class ToTensor:
    def __init__(
        self, syms: Union[Dict, SymbolsTable], reading_order: str = "LTR"
    ) -> None:
        assert isinstance(syms, (Dict, SymbolsTable))
        self._syms = syms
        self.reading_order = reading_order

    def __call__(self, x: str) -> List[int]:
        if self.reading_order == "RTL":
            x = get_display(x)
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
