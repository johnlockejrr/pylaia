import re
from pathlib import Path

import pytest

from laia.utils.symbols_table import SymbolsTable


def test_empty():
    st = SymbolsTable()
    assert len(st) == 0


def test_add():
    st = SymbolsTable()
    st.add("<eps>", 0)
    assert len(st) == 1
    st.add("a", 1)
    assert len(st) == 2


def test_add_valid_repeated():
    st = SymbolsTable()
    st.add("<eps>", 0)
    st.add("<eps>", 0)
    assert len(st) == 1


def test_add_repeated_symbol():
    st = SymbolsTable()
    st.add("a", 0)
    with pytest.raises(
        KeyError,
        match=re.escape(
            'Symbol "a" was already present in the table (assigned to value "0")'
        ),
    ):
        st.add("a", 1)


def test_add_repeated_value():
    st = SymbolsTable()
    st.add("a", 0)
    with pytest.raises(
        KeyError,
        match=re.escape(
            'Value "0" was already present in the table (assigned to symbol "a")'
        ),
    ):
        st.add("b", 0)


def test_getitem():
    st = SymbolsTable()
    st.add("a", 1)
    st.add("b", 2)
    assert st["a"] == 1
    assert st["b"] == 2
    assert st[1] == "a"
    assert st[2] == "b"
    assert st[-9] is None
    assert st["c"] is None


def test_iterator():
    st = SymbolsTable()
    st.add("a", 1)
    st.add("b", 2)
    it = iter(st)
    assert next(it) == ("a", 1)
    assert next(it) == ("b", 2)
    with pytest.raises(StopIteration):
        next(it)


def test_contains():
    st = SymbolsTable()
    st.add("a", 1)
    assert "a" in st
    assert 1 in st
    assert 2 not in st
    with pytest.raises(ValueError, match="SymbolsTable contains pairs"):
        assert None in st  # noqa: expected type


@pytest.mark.parametrize("as_type", [str, Path])
def test_load(tmpdir, as_type):
    file = tmpdir / "f"
    file.write_text("\n\na   1\nb     2\n", "utf-8")
    st = SymbolsTable(as_type(file))
    assert len(st) == 2
    assert st["a"] == 1
    assert st["b"] == 2
    assert st[1] == "a"
    assert st[2] == "b"


def test_load_value_error(tmpdir):
    file = tmpdir / "f"
    file.write_text("\n\na   1\nb     c\n", "utf-8")
    with pytest.raises(ValueError):
        SymbolsTable(file)


def test_save(tmpdir):
    st = SymbolsTable()
    st.add("a", 1)
    st.add("b", 2)
    st_file = tmpdir / "syms"
    st.save(str(st_file))
    assert st_file.read_text("utf-8") == "a 1\nb 2\n"
