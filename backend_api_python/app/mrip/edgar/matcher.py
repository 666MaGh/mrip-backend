"""Resolve names mentioned in filings to universe tickers.

Only normalised exact matches count: names are compared token-by-token after
case folding, accent stripping, punctuation removal and removal of legal-entity
suffixes (Inc, Corp, Corporation, Co, Ltd, Limited, PLC, LLC, NV, SA, Class A).
There is no fuzzy matching. A small curated alias map covers well-known
brand names. Ambiguous normalised names resolve to nothing.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Mapping

from app.mrip.edgar.types import NameMatch

# Curated aliases (normalised form of the alias -> ticker). Targets must be in the universe.
ALIASES: Mapping[str, str] = {
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta platforms": "META",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
    "amazon com": "AMZN",
    "amazon web services": "AMZN",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "broadcom": "AVGO",
    "apple": "AAPL",
    "vertiv": "VRT",
    "eaton": "ETN",
    "ge vernova": "GEV",
    "vistra": "VST",
    "constellation energy": "CEG",
    "freeport mcmoran": "FCX",
}

# Phrases that describe a class of counterparties, never a specific listed company.
GENERIC_NAMES = frozenset(
    {
        "customer", "customers", "client", "clients", "government", "u s government", "us government",
        "federal government", "the u s government", "distributor", "distributors", "end user", "end users",
        "oem", "oems", "retailer", "retailers", "reseller", "resellers", "supplier", "suppliers", "vendor",
        "vendors", "third party", "third parties", "others", "other",
    }
)

_LEGAL_SUFFIXES = frozenset({"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "nv", "sa", "ag", "se"})
_CLASS_RE = re.compile(r"\bcl(?:ass)? [a-z]$")


def normalize(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    cleaned = _CLASS_RE.sub("", cleaned).strip()
    tokens = cleaned.split()
    while tokens and tokens[0] == "the":
        tokens.pop(0)
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


class CompanyIndex:
    """Normalised company name -> ticker, built from the universe titles and ALIASES."""

    def __init__(self, universe_titles: Mapping[str, str]) -> None:
        candidates: dict[str, set[str]] = {}
        for ticker, title in universe_titles.items():
            key = normalize(title)
            if key:
                candidates.setdefault(key, set()).add(ticker)
        for alias, ticker in ALIASES.items():
            if ticker in universe_titles:
                candidates.setdefault(normalize(alias), set()).add(ticker)
        self._names = {key: next(iter(tickers)) for key, tickers in candidates.items() if len(tickers) == 1}

    def __len__(self) -> int:
        return len(self._names)

    def resolve(self, name: str) -> NameMatch:
        key = normalize(name)
        if not key or key in GENERIC_NAMES:
            return NameMatch(name=name, ticker=None, generic=True)
        return NameMatch(name=name, ticker=self._names.get(key))
