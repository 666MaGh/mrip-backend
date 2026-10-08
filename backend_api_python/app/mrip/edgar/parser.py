"""Deterministic extraction of customer/supplier statements from a 10-K.

Pipeline: HTML -> visible text (stdlib html.parser, hidden inline-XBRL header
skipped) -> lines with section headings -> sentences -> rule matches. No LLM,
no numeric computation beyond reading the percentage printed in the sentence.
Rules are intentionally conservative: a sentence must carry an explicit
customer/supplier cue, and names are only extracted as capitalised runs.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

from app.mrip.edgar.types import CandidateStatement, StatementKind

MAX_SENTENCE_CHARS = 1000
MIN_SENTENCE_CHARS = 25

_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "tr", "li", "ul", "ol", "table", "section", "article", "title", "hr",
     "h1", "h2", "h3", "h4", "h5", "h6"}
)
_SKIP_TAGS = frozenset({"script", "style", "head", "ix:header"})

_HEADING_KEYWORDS = (
    "customer", "concentration", "major customer", "significant customer", "segment",
    "supplier", "supply", "sources and availability", "manufactur", "sources of",
    "principal products", "geographic", "revenue recognition",
)

_ABBREVIATIONS = ("U.S.", "Inc.", "Corp.", "Co.", "Ltd.", "No.", "approx.", "e.g.", "i.e.", "Mr.", "Dr.")
_PLACEHOLDER = "\x00"

_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s?(?:%|percent\b)", re.IGNORECASE)
_CUSTOMER_WORD_RE = re.compile(r"\bcustomers?\b", re.IGNORECASE)
_REVENUE_WORD_RE = re.compile(r"\b(?:revenues?|net sales|sales|receivables?)\b", re.IGNORECASE)
_ACCOUNTED_RE = re.compile(r"\baccounted for\b", re.IGNORECASE)
_CUSTOMER_LIST_RE = re.compile(
    r"\b(?:customers?\s+(?:include|includes|including|were|was|are|is)|"
    r"(?:major|largest|significant|principal|key|direct)\s+customers?\s+(?:include|includes|including))\b",
    re.IGNORECASE,
)
_SUPPLIER_RE = re.compile(
    r"\b(?:depend(?:s|ed|ent|ence)?\s+(?:up)?on|rel(?:y|ies|ied)\s+(?:up)?on|"
    r"sole[- ]source[sd]?|single[- ]source[sd]?|sole supplier|single supplier|"
    r"purchases? (?:\w+ ){0,3}from|manufactured by|"
    r"(?:contract )?manufacturers? such as|foundr(?:y|ies) such as|supplied by|"
    r"from (?:a |one )?(?:single|sole|limited number of) (?:third[- ]party )?(?:supplier|vendor|source|manufacturer)s?)\b",
    re.IGNORECASE,
)
_NAME_RUN_RE = re.compile(r"[A-Z][\w&'\-.]*(?:[ \t]+(?:&[ \t]*)?[A-Z][\w&'\-.]*)*")

# Direction rules. Each pattern below encodes who is the grammatical subject:
#  * supplier cues ("we depend on X", "our chips are supplied by X") need the FILER as subject;
#  * "X supplies us" has a named third party as subject and the filer as object (X is the supplier);
#  * "X will purchase ... / buys our products / purchases from us" has a named third party as
#    subject buying from the filer, so X is the CUSTOMER;
#  * "our products are sold to X" and "we sell ... to X" have the filer as seller, so X is the CUSTOMER.
# Casing is significant where a capitalised run is the name, so these patterns are not IGNORECASE.
_FILER_MARKER = r"(?:We|we|Our|our|us|The Company|the Company|The Registrant|the Registrant)"
_FILER_MARKER_RE = re.compile(rf"\b{_FILER_MARKER}\b")
_SUBJECT_NAME = r"[A-Z][\w&'\-.]*(?:[ \t]+(?:&[ \t]*)?[A-Z][\w&'\-.]*)*"
_BUY_OBJECT = (
    r"(?:(?:the|our|certain|all|substantially all|a|an|some|additional|significant|such)\s+)*"
    r"(?:(?:[a-z][\w-]*\s+){0,3}?(?:output|outputs|products?|energy|power|capacity|electricity|services|goods|"
    r"semiconductors?|chips?|systems?|equipment|components?|solutions?|attributes?)\b"
    # "products from third parties" is a supply relation to someone else; emit nothing.
    r"(?!\s+(?:from|of|by|made|manufactured|sourced)\s+(?!(?:us|our)\b))"
    r"|from\s+(?:us|our\s+(?:company|business)|the\s+(?:Company|Registrant))\b)"
)
_COUNTERPARTY_BUYS_RE = re.compile(
    rf"(?P<subject>{_SUBJECT_NAME})\s+(?:(?:will|would|may|can|could|has|have|had|also|currently)\s+)*"
    rf"(?:purchase[sd]?|buy|buys|bought|procure[sd]?|licen[cs]e[sd]?)\s+{_BUY_OBJECT}"
)
# Leading list of named counterparties ("Amazon, Alphabet and Meta"), nothing after the list.
_NAME_LIST = rf"{_SUBJECT_NAME}(?:(?:,\s*|\s+(?:and|or)\s+){_SUBJECT_NAME})*"
_COUNTERPARTY_PASSIVE_RE = re.compile(
    rf"\b(?:We|we|Our|our|The Company's|the Company's|The Registrant's|the Registrant's)\b"
    r"(?:(?! [A-Z])[^.;]){0,80}?\b(?:is|are|was|were|has been|have been)\s+(?:also\s+)?"
    rf"(?:sold|licensed|purchased|bought|shipped)\s+(?:to|by)\s+(?P<target>{_NAME_LIST})"
)
_FILER_SELLS_TO_RE = re.compile(
    rf"\b{_FILER_MARKER}\s+(?:also\s+)?(?:sell|sells|sold|supply|supplies|supplied|ship|ships|"
    r"license|licenses|licensed|provide|provides|provided)\s+(?:(?:[a-z][\w-]*|,)\s+){0,6}?"
    rf"to\s+(?P<target>{_NAME_LIST})"
)
_THIRD_PARTY_SUPPLIES_US_RE = re.compile(
    rf"(?P<subject>{_SUBJECT_NAME})\s+(?:(?:will|has|have|had|also|currently)\s+)*"
    r"(?:supply|supplies|supplied|sell|sells|sold|provide|provides|provided|license|licenses|licensed)\s+"
    r"(?:(?:[a-z][\w-]*)\s+){0,4}?(?:us|our (?:company|business))\b"
)
_PURCHASE_CONTEXT_RE = re.compile(
    r"\b(?:purchases?|purchased|buys?|bought|procurement|costs?|suppliers?|vendors?)\b", re.IGNORECASE
)

# Leading/standalone words that are never a counterparty name on their own.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "we", "our", "us", "its", "their", "this", "these", "those", "such", "any",
        "each", "one", "two", "three", "four", "five", "some", "most", "no", "in", "for", "during",
        "fiscal", "year", "item", "note", "section", "form", "asc", "customer", "customers", "client",
        "clients", "revenue", "revenues", "sales", "net", "total", "company", "companies", "risk",
        "factors", "products", "product", "accounts", "receivable", "receivables", "direct", "indirect",
        "major", "largest", "significant", "principal", "key", "single", "sole", "third", "party",
        "parties", "government", "federal", "department", "defense", "united", "states", "u.s.", "u.s",
        "it", "they", "many", "certain", "also", "however", "as",
        "all", "other", "others", "during", "when", "if", "to", "on", "at", "from", "by", "with",
        "and", "or", "but", "our", "your", "his", "her",
        "january", "february", "march", "april", "june", "july", "august", "september", "october",
        "november", "december", "may",
    }
)

# Legal-entity words are kept inside names (the matcher strips them) but never form a name alone.
_LEGAL_ONLY = frozenset(
    {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "ltd", "ltd.", "limited", "llc", "plc", "n.v.", "group", "holdings"}
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in ("td", "th"):
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    """Visible text of an HTML document, one block per line, whitespace normalised."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    lines: list[str] = []
    for raw in extractor.text().replace("\xa0", " ").replace("​", "").splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def is_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if re.match(r"(?i)^item\s+\d+[a-c]?[.:\s]", line):
        return True
    lowered = line.lower().rstrip(":. ")
    return len(lowered) <= 60 and not line.endswith(".") and any(k in lowered for k in _HEADING_KEYWORDS)


def split_sentences(paragraph: str) -> list[str]:
    protected = paragraph
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    pieces = re.split(r"(?<=[.!?;])\s+(?=[\"'(\[]?[A-Z0-9])", protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in pieces if p.strip()]


def _is_stop(word: str) -> bool:
    return word.lower().strip(",.") in _STOP_WORDS


def extract_names(sentence: str) -> tuple[str, ...]:
    """Capitalised runs with leading stop words removed; order preserved, duplicates dropped."""
    names: list[str] = []
    for match in _NAME_RUN_RE.finditer(sentence):
        words = [w.rstrip(".,") for w in match.group(0).split()]
        while words and _is_stop(words[0]):
            words.pop(0)
        while words and _is_stop(words[-1]):
            words.pop()
        name = " ".join(words).strip(" ,&-'")
        if not name or _is_stop(name) or name.lower() in _LEGAL_ONLY or len(name) < 2:
            continue
        if name not in names:
            names.append(name)
    return tuple(names)


def _names_before_accounted(sentence: str) -> str:
    idx = _ACCOUNTED_RE.search(sentence)
    return sentence[: idx.start()] if idx else ""


def classify(sentence: str) -> list[tuple[StatementKind, float | None, tuple[str, ...]]]:
    """Return (kind, percent, names) for each rule the sentence satisfies."""
    found: list[tuple[StatementKind, float | None, tuple[str, ...]]] = []
    percent_match = _PERCENT_RE.search(sentence)
    percent = float(percent_match.group(1)) if percent_match else None
    if percent is not None and not 0.0 < percent <= 100.0:
        percent = None

    customer_cue = bool(_CUSTOMER_WORD_RE.search(sentence) or _CUSTOMER_LIST_RE.search(sentence))
    if customer_cue and percent is not None and _REVENUE_WORD_RE.search(sentence):
        found.append(("customer", percent, extract_names(sentence)))
    elif _CUSTOMER_LIST_RE.search(sentence):
        found.append(("customer", percent, extract_names(sentence)))
    elif percent is not None and _ACCOUNTED_RE.search(sentence) and _REVENUE_WORD_RE.search(sentence):
        prefix = _names_before_accounted(sentence)
        # "Purchases from Micron accounted for 20% of cost of revenue": Micron is a supplier, not a customer.
        names = () if _PURCHASE_CONTEXT_RE.search(prefix) else extract_names(prefix)
        if names:
            found.append(("customer", percent, names))

    if not any(kind == "customer" for kind, _, _ in found):
        counterparty_names = _counterparty_customer_names(sentence)
        if counterparty_names:
            found.append(("customer", percent, counterparty_names))

    supplier_names = _supplier_names(sentence)
    if supplier_names:
        found.append(("supplier", percent, supplier_names))
    return found


def _filer_is_subject_before(prefix: str) -> bool:
    """True when the last filer marker before a cue is its subject: no third-party name or clause break between."""
    markers = list(_FILER_MARKER_RE.finditer(prefix))
    if not markers:
        return False
    tail = prefix[markers[-1].end():]
    return not (_NAME_RUN_RE.search(tail) or "," in tail or ";" in tail)


def _supplier_names(sentence: str) -> tuple[str, ...]:
    """Supplier names. Filer depends on X / X supplies the filer; the cue must have the filer as subject."""
    for match in _SUPPLIER_RE.finditer(sentence):
        if _filer_is_subject_before(sentence[: match.start()]):
            names = extract_names(sentence[match.end():])
            if names:
                return names
    for match in _THIRD_PARTY_SUPPLIES_US_RE.finditer(sentence):
        names = extract_names(match.group("subject"))
        if names:
            return names
    return ()


def _counterparty_customer_names(sentence: str) -> tuple[str, ...]:
    """Customer names: a named counterparty buys from the filer, or the filer sells to a named counterparty."""
    for match in _COUNTERPARTY_BUYS_RE.finditer(sentence):
        names = extract_names(match.group("subject"))
        if names:
            return names
    for pattern in (_COUNTERPARTY_PASSIVE_RE, _FILER_SELLS_TO_RE):
        for match in pattern.finditer(sentence):
            names = extract_names(match.group("target"))
            if names:
                return names
    return ()


def parse_text(text: str) -> list[CandidateStatement]:
    """Candidate statements from already-extracted filing text."""
    results: list[CandidateStatement] = []
    seen: set[tuple[str, str]] = set()
    section = "(none)"
    for line in text.splitlines():
        if is_heading(line):
            section = line[:120]
            continue
        for sentence in split_sentences(line):
            if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
                continue
            for kind, percent, names in classify(sentence):
                key = (kind, sentence)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    CandidateStatement(sentence=sentence, section=section, percent=percent, kind=kind, names=names)
                )
    return results


def parse_10k_html(html: str) -> list[CandidateStatement]:
    return parse_text(html_to_text(html))
