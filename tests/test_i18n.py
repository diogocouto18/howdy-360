import re

from howdy360 import i18n


def test_english_default(monkeypatch):
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    assert i18n.language() == "en"
    assert i18n._("Stored models: {n}", n=3) == "Stored models: 3"


def test_portuguese(monkeypatch):
    monkeypatch.setenv("LANGUAGE", "pt_PT")
    assert i18n._("Stored models: {n}", n=3) == "Registos guardados: 3"
    assert i18n._("untranslated text") == "untranslated text"


def test_translations_keep_placeholders():
    field = re.compile(r"{(\w+)")
    for en, pt in i18n.PT.items():
        assert set(field.findall(en)) == set(field.findall(pt)), en
