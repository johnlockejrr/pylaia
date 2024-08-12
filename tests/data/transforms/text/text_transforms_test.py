import pytest

from laia.data.transforms.text import ToTensor, tokenize, untokenize
from laia.utils import SymbolsTable


@pytest.mark.parametrize(
    "sentence,tokenization,space_token,space_display",
    [
        (
            "salut à toi l'endimanché",
            "s a l u t <space> à <space> t o i <space> l ' e n d i m a n c h é",
            "<space>",
            " ",
        ),
        (
            "salut à tous les paysans",
            "s a l u t <ESPACE> à <ESPACE> t o u s <ESPACE> l e s <ESPACE> p a y s a n s",
            "<ESPACE>",
            " ",
        ),
        (
            "salut-aussi-à-Rantanplan",
            "s a l u t <dash> a u s s i <dash> à <dash> R a n t a n p l a n",
            "<dash>",
            "-",
        ),
        (
            "أكلت اثنتي عشرة (12) قطعة بسكويت في 1/08/2024",
            "أ ك ل ت <space> ا ث ن ت ي <space> ع ش ر ة <space> ( 1 2 ) <space> ق ط ع ة <space> ب س ك و ي ت <space> ف ي <space> 1 / 0 8 / 2 0 2 4",
            "<space>",
            " ",
        ),
        (
            "أكلت اثنتي عشرة (<number>12) كعكة في <date>1/08/2024",
            "أ ك ل ت <space> ا ث ن ت ي <space> ع ش ر ة <space> ( <number> 1 2 ) <space> ك ع ك ة <space> ف ي <space> <date> 1 / 0 8 / 2 0 2 4",
            "<space>",
            " ",
        ),
    ],
)
def test_tokenize_untokenize(sentence, tokenization, space_token, space_display):
    SYMS = {
        "ط",
        "ك",
        "ل",
        "/",
        "ع",
        "<date>",
        "ش",
        "و",
        "0",
        "ا",
        "ث",
        "<space>",
        "4",
        "ي",
        "أ",
        "ب",
        "ف",
        "ت",
        "8",
        "<number>",
        "ر",
        ")",
        "1",
        "2",
        "(",
        "س",
        "ة",
        "ق",
        "ن",
        "a",
        "u",
        "s",
        "à",
        "'",
        "l",
        "e",
        "n",
        "d",
        "i",
        "m",
        "o",
        "h",
        "é",
        "c",
        "t",
        "R",
        "t",
        "p",
        "-",
        "o",
        "u",
        "p",
        "y",
        "<space>",
    }
    assert (
        tokenize(
            sentence, space_token=space_token, space_display=space_display, symbols=SYMS
        )
        == tokenization
    )
    assert (
        untokenize(tokenization, space_token=space_token, space_display=space_display)
        == sentence
    )


def test_call_with_dict(caplog):
    t = ToTensor({"a": 0, "b": 1, "<space>": 2, "<": 3})
    x = "a < b <space> a <sp"
    y = t(x)
    assert y == [0, 3, 1, 2, 0, None]
    assert caplog.messages.count('Could not find "<sp" in the symbols table') == 1


def test_call_with_symbols_table(caplog):
    st = SymbolsTable()
    for k, v in {"a": 0, "b": 1, "<space>": 2, "<": 3}.items():
        st.add(k, v)
    t = ToTensor(st)
    x = "a < b <space> a ö"
    y = t(x)
    assert y == [0, 3, 1, 2, 0, None]
    assert caplog.messages.count('Could not find "ö" in the symbols table') == 1
