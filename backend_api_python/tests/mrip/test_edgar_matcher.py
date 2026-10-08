from __future__ import annotations

import pytest

from app.mrip.edgar.matcher import CompanyIndex, normalize

UNIVERSE = {
    "NVDA": "NVIDIA CORP",
    "MSFT": "MICROSOFT CORP",
    "AMZN": "AMAZON COM INC",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "AVGO": "Broadcom Inc.",
    "ANET": "Arista Networks, Inc.",
}


@pytest.fixture()
def index() -> CompanyIndex:
    return CompanyIndex(UNIVERSE)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Microsoft Corporation", "microsoft"),
        ("Amazon.com, Inc.", "amazon com"),
        ("Alphabet Inc. Class A", "alphabet"),
        ("The Boeing Company", "boeing"),
        ("NVIDIA Corp.", "nvidia"),
    ],
)
def test_normalize_strips_legal_suffixes_and_punctuation(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("mention", "ticker"),
    [
        ("Microsoft", "MSFT"),
        ("Microsoft Corporation", "MSFT"),
        ("NVIDIA", "NVDA"),
        ("Broadcom", "AVGO"),
        ("Arista Networks", "ANET"),
        ("Alphabet", "GOOGL"),
        ("Google", "GOOGL"),
        ("Facebook", "META"),
        ("Meta", "META"),
        ("Meta Platforms", "META"),
        ("Amazon", "AMZN"),
        ("Amazon Web Services", "AMZN"),
    ],
)
def test_known_names_resolve_to_universe_tickers(index, mention, ticker):
    match = index.resolve(mention)
    assert match.ticker == ticker
    assert match.generic is False


@pytest.mark.parametrize("mention", ["Apple", "TSMC", "Dell", "Microsoft Azure Stack Hub Limited Edition"])
def test_names_outside_the_universe_stay_unmatched(index, mention):
    match = index.resolve(mention)
    assert match.ticker is None
    assert match.generic is False


@pytest.mark.parametrize("mention", ["pineapple", "Pineapple", "metadata", "Metadata", "Microsofts", "Brodcom"])
def test_no_substring_or_fuzzy_matches(index, mention):
    assert index.resolve(mention).ticker is None


@pytest.mark.parametrize("mention", ["customers", "Customers", "the U.S. government", "distributors", "end users"])
def test_generic_terms_are_flagged_generic_not_matched(index, mention):
    match = index.resolve(mention)
    assert match.generic is True
    assert match.ticker is None


def test_ambiguous_normalised_titles_resolve_to_nothing():
    index = CompanyIndex({"AAA": "Widget Corp", "BBB": "Widget Inc."})
    assert index.resolve("Widget").ticker is None


def test_alias_is_ignored_when_target_not_in_universe():
    index = CompanyIndex({"NVDA": "NVIDIA CORP"})
    assert index.resolve("Alphabet").ticker is None
    assert index.resolve("Google").ticker is None
