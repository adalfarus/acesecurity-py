from __future__ import annotations

import random
import string

import pytest

from ..passwords import (
    PasswordFilter,
    PasswordGenerator,
    SecurePasswordGenerator,
    SimplePasswordGenerator,
)
from ..rand import Random


def test_password_filter_classifies_and_applies_extra_chars() -> None:
    filt = PasswordFilter(exclude_chars="a1!", extra_chars="Z9!ö")

    assert filt.extra_chars_dict == {
        "letters": "Z",
        "numbers": "9",
        "punctuations": "!",
        "unicode": "ö",
    }

    # excludes then adds set-specific extra chars
    assert filt.apply("ab1!", "letters") == "bZ"
    assert filt.apply("12", "numbers") == "29"
    assert filt.apply("!?", "punctuations") == "?!"


def test_password_filter_exclude_similar_and_will_filter() -> None:
    filt = PasswordFilter(exclude_chars="x", exclude_similar=True)

    assert filt.apply("il1Lo0OxY", "letters") == "Y"
    assert filt.filter_word("lilox") == ""
    assert filt.will_filter("l") is True
    assert filt.will_filter("A") is False


def test_simple_password_generator_pattern_and_complex_password() -> None:
    original_rng = SimplePasswordGenerator._rng
    try:
        SimplePasswordGenerator._rng = random.Random(123)
        pat = SimplePasswordGenerator.generate_pattern_password("Xx9-9")
        complex_pw = SimplePasswordGenerator.generate_complex_password(length=14)
    finally:
        SimplePasswordGenerator._rng = original_rng

    assert len(pat) == 5
    assert pat[0] in string.ascii_uppercase
    assert pat[1] in string.ascii_lowercase
    assert pat[2] in string.digits
    assert pat[3] == "-"
    assert pat[4] in string.digits

    assert len(complex_pw) == 14
    assert any(c.isupper() for c in complex_pw)
    assert any(c.islower() for c in complex_pw)
    assert any(c.isdigit() for c in complex_pw)
    assert any(c in string.punctuation for c in complex_pw)


def test_password_generator_complex_password_requires_min_length() -> None:
    gen = PasswordGenerator(Random(0))
    with pytest.raises(ValueError, match="at least 4"):
        gen.generate_complex_password(length=3)


def test_password_generator_random_and_pattern_lengths() -> None:
    gen = PasswordGenerator(Random(1))

    random_pw = gen.generate_random_password(length=17)
    pattern_pw = gen.generate_pattern_password("Xx9-9")

    assert len(random_pw) == 17
    assert len(pattern_pw) == 5
    assert pattern_pw[0] in string.ascii_uppercase
    assert pattern_pw[1] in string.ascii_lowercase
    assert pattern_pw[2] in string.digits
    assert pattern_pw[3] == "-"
    assert pattern_pw[4] in string.digits


def test_password_generator_sentence_invalid_char_position() -> None:
    gen = PasswordGenerator(Random(2))
    result = gen.generate_sentence_based_password_v3(
        sentence="alpha beta",
        char_position="invalid",  # type: ignore[arg-type]
    )
    assert result == "Invalid char_position."


def test_password_generator_words_and_sentence_return_info() -> None:
    gen = PasswordGenerator(Random(3))

    words_pw, source_sentence = gen.generate_words_based_password_v3(
        sentence="alpha beta gamma",
        shuffle_words=False,
        shuffle_characters=False,
        _return_info=True,
    )
    sentence_pw, sentence_source = gen.generate_sentence_based_password_v3(
        sentence="alpha beta",
        char_position="keep",
        random_case=False,
        extra_char="/",
        num_length=2,
        special_chars_length=1,
        password_format="{words}{extra}{numbers}{special}",
        _return_info=True,
    )

    assert source_sentence == "alpha beta gamma"
    assert words_pw == "alphabetagamma"
    assert sentence_source == "alpha beta"
    assert sentence_pw.startswith("ab/")
    assert len(sentence_pw) == len("ab/") + 2 + 1


def test_password_generator_ratio_v3_respects_length_and_filter() -> None:
    gen = PasswordGenerator(Random(4))
    filt = PasswordFilter(exclude_chars=string.ascii_letters + string.punctuation)

    pw = gen.generate_ratio_based_password_v3(
        length=16,
        letters_ratio=0.6,
        numbers_ratio=0.4,
        punctuations_ratio=0.0,
        unicode_ratio=0.0,
        filter_=filt,
    )

    assert len(pw) == 16
    assert all(c in string.digits for c in pw)


def test_password_generator_reduce_password_length_contract() -> None:
    gen = PasswordGenerator(Random(5))

    assert len(gen.reduce_password("abcdef", by=0)) == 6
    assert len(gen.reduce_password("abcdef", by=2)) == 2
    assert len(gen.reduce_password("abcdef", by="all")) == 1


def test_secure_password_generator_methods_return_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _load_words(self: PasswordGenerator) -> None:
        self._add_unique_words(["alpha", "beta", "gamma", "delta"])

    monkeypatch.setattr(PasswordGenerator, "load_def_dict", _load_words)
    monkeypatch.setattr(PasswordGenerator, "load_google_10000_dict", _load_words)
    monkeypatch.setattr(PasswordGenerator, "load_scowl_dict", lambda self, size="50": _load_words(self))
    monkeypatch.setattr(PasswordGenerator, "load_12_dicts", _load_words)

    sec = SecurePasswordGenerator("weak")

    outputs = [
        sec.random(),
        sec.passphrase(),
        sec.pattern(),
        sec.complex(),
        sec.mnemonic(),
        sec.ratio(),
        sec.words(),
        sec.sentence(),
        sec.complex_pattern(),
    ]

    for result in outputs:
        assert set(result.keys()) == {"extra_info", "password"}
        assert isinstance(result["password"], str)
        assert result["password"]
        assert isinstance(result["extra_info"], str)
        assert result["extra_info"]


def test_secure_password_generator_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load_words(self: PasswordGenerator) -> None:
        self._add_unique_words(["alpha", "beta", "gamma"])

    monkeypatch.setattr(PasswordGenerator, "load_def_dict", _load_words)
    monkeypatch.setattr(PasswordGenerator, "load_google_10000_dict", _load_words)
    monkeypatch.setattr(PasswordGenerator, "load_scowl_dict", lambda self, size="50": _load_words(self))
    monkeypatch.setattr(PasswordGenerator, "load_12_dicts", _load_words)

    sec = SecurePasswordGenerator("average")
    monkeypatch.setattr(SecurePasswordGenerator, "_exponential", lambda self, *args, **kwargs: 0)

    result = sec.generate_secure_password()

    assert set(result.keys()) == {"extra_info", "password"}
    assert result["password"]
