"""Universe data sources: fetch and parse constituent lists.

Provides parsers for publicly available trading universe catalogues
(e.g. S&P 500) and a fetch wrapper that chains HTTP errors to UniverseError
for consistent error handling.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Sequence

import requests

from app.mrip.universe.models import UniverseMember, UniverseError

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

# Minimum members required to detect an HTML error page parsed as CSV.
MIN_MEMBERS = 100

# Minimum members required for OMX Stockholm Large Cap list.
MIN_OMX_MEMBERS = 50


def parse_sp500_csv(text: str) -> list[UniverseMember]:
    """Parse S&P 500 constituent CSV.

    The source CSV (CC0 / Open Data Commons PDDL) has columns:
        Symbol, Security, GICS Sector, GICS Sub-Industry, Headquarters Location,
        Date added, CIK, Founded

    Raises UniverseError if the header is invalid or fewer than MIN_MEMBERS
    members are parsed (guards against HTML error pages being parsed as data).
    Returns members in order, de-duplicated by symbol (keeps first occurrence).
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None or "Symbol" not in reader.fieldnames or "Security" not in reader.fieldnames:
        raise UniverseError("CSV header missing Symbol or Security column")

    members: list[UniverseMember] = []
    seen_symbols: set[str] = set()

    for row in reader:
        symbol = (row.get("Symbol") or "").strip()
        name = (row.get("Security") or "").strip()

        # Skip rows with blank symbol or name.
        if not symbol or not name:
            continue

        # De-duplicate by symbol, keeping the first occurrence.
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)

        sector = (row.get("GICS Sector") or "").strip() or None
        sub_industry = (row.get("GICS Sub-Industry") or "").strip() or None

        members.append(
            UniverseMember(
                universe="SP500",
                symbol=symbol,
                name=name,
                exchange="US",
                currency="USD",
                sector=sector,
                sub_industry=sub_industry,
                data_symbol=symbol,  # CBOE accepts the dotted form (BRK.B, BF.B).
                data_provider="cboe",
            )
        )

    if len(members) < MIN_MEMBERS:
        raise UniverseError(
            f"parsed only {len(members)} members, expected at least {MIN_MEMBERS}; "
            "likely an HTML error page"
        )

    return members


def fetch_sp500(
    session: requests.Session | None = None, url: str = SP500_CSV_URL, timeout: float = 30.0
) -> list[UniverseMember]:
    """Fetch and parse S&P 500 constituents.

    Raises UniverseError if the HTTP request fails or the response cannot be parsed.
    A session object may be injected for testing (must have a .get(url, timeout=...)
    method that raises appropriate exceptions on network/HTTP failure).
    """
    if session is None:
        session = requests.Session()

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return parse_sp500_csv(response.text)
    except requests.RequestException as e:
        raise UniverseError(f"failed to fetch {url}: {e}") from e
    except UniverseError:
        raise
    except Exception as e:
        raise UniverseError(f"failed to parse S&P 500 CSV: {e}") from e


def parse_omx_csv(text: str) -> list[UniverseMember]:
    """Parse OMX Stockholm Large Cap constituent CSV.

    The source CSV has columns:
        Symbol, Name, Sector, Currency

    Lines starting with '#' and blank lines are skipped.
    Symbols are converted from Nasdaq short form to hyphenated form:
    "VOLV B" -> "VOLV-B", "ALIV SDB" -> "ALIV-SDB", "ABB" -> "ABB".

    Returns members in order, de-duplicated by symbol (keeps first occurrence).
    Raises UniverseError if the header is invalid or fewer than MIN_OMX_MEMBERS
    members are parsed.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        # Skip comment lines (start with '#') and blank lines.
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)

    # Reconstruct text without comments/blanks for CSV reader.
    filtered_text = "\n".join(lines)
    reader = csv.DictReader(StringIO(filtered_text))

    if reader.fieldnames is None or "Symbol" not in reader.fieldnames or "Name" not in reader.fieldnames:
        raise UniverseError("CSV header missing Symbol or Name column")

    members: list[UniverseMember] = []
    seen_symbols: set[str] = set()

    for row in reader:
        symbol_raw = (row.get("Symbol") or "").strip()
        name = (row.get("Name") or "").strip()

        # Skip rows with blank symbol or name.
        if not symbol_raw or not name:
            continue

        # Convert symbol from Nasdaq short form to hyphenated form.
        # "VOLV B" -> "VOLV-B", "ALIV SDB" -> "ALIV-SDB", "ABB" -> "ABB"
        symbol = symbol_raw.upper().replace(" ", "-")

        # De-duplicate by symbol, keeping the first occurrence.
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)

        sector = (row.get("Sector") or "").strip() or None
        currency = (row.get("Currency") or "").strip() or "SEK"

        members.append(
            UniverseMember(
                universe="OMXSLC",
                symbol=symbol,
                name=name,
                exchange="XSTO",
                currency=currency,
                sector=sector,
                sub_industry=None,
                data_symbol=f"{symbol}.ST",
                data_provider="yahoo",
            )
        )

    if len(members) < MIN_OMX_MEMBERS:
        raise UniverseError(
            f"parsed only {len(members)} members, expected at least {MIN_OMX_MEMBERS}"
        )

    return members


def load_omxslc() -> list[UniverseMember]:
    """Load OMX Stockholm Large Cap constituents from the packaged data file.

    The data file is located at app/mrip/universe/data/omxslc.csv relative to
    the module directory.

    Raises UniverseError if the file cannot be read or parsed.
    """
    try:
        data_file = Path(__file__).parent / "data" / "omxslc.csv"
        text = data_file.read_text(encoding="utf-8")
        return parse_omx_csv(text)
    except FileNotFoundError as e:
        raise UniverseError(f"OMX Stockholm data file not found: {e}") from e
    except UniverseError:
        raise
    except Exception as e:
        raise UniverseError(f"failed to parse OMX Stockholm CSV: {e}") from e
