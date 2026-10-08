from __future__ import annotations

from app.mrip.edgar.parser import MAX_SENTENCE_CHARS, html_to_text, parse_10k_html

FIXTURE = """<html><head><title>nvda-10k</title><style>p {color: red}</style></head><body>
<div style="display:none"><ix:header>HIDDEN Microsoft 99% of revenue customers</ix:header></div>
<p>Item 1. Business</p>
<p>Our products are sold through distributors and original equipment manufacturers.</p>
<h3>Concentration of Credit Risk</h3>
<p>Sales to Microsoft Corporation accounted for 16% of our total revenue in fiscal 2026.</p>
<p>Two direct customers accounted for 39% of total accounts receivable as of January 25, 2026.</p>
<p>Our customers include Amazon, Alphabet and Meta, which together buy our data center systems.</p>
<p>The U.S. government accounted for 10% of total revenue, and customers in China were 11% of sales.</p>
<p>Sales to Dell and Lenovo are not material to our results of operations this year.</p>
<p>We depend on TSMC to manufacture substantially all of our semiconductor products.</p>
<p>Our networking products rely on a single supplier, Broadcom, for certain components.</p>
<p>We purchase memory from Micron and SK hynix under long-term agreements.</p>
<table><tr><td>Customer A</td><td>12%</td></tr></table>
</body></html>"""


def _by_kind(statements, kind):
    return [s for s in statements if s.kind == kind]


def test_html_to_text_skips_scripts_and_hidden_inline_xbrl_header():
    text = html_to_text(FIXTURE)
    assert "HIDDEN" not in text
    assert "color: red" not in text
    assert "Item 1. Business" in text.splitlines()
    assert "Sales to Microsoft Corporation accounted for 16% of our total revenue in fiscal 2026." in text


def test_percentage_customer_sentence_extracts_percent_names_and_section():
    statements = parse_10k_html(FIXTURE)
    msft = [s for s in statements if "Microsoft Corporation" in s.sentence]
    assert len(msft) == 1
    assert msft[0].kind == "customer"
    assert msft[0].percent == 16.0
    assert msft[0].names == ("Microsoft Corporation",)
    assert msft[0].section == "Concentration of Credit Risk"


def test_receivables_percentage_counts_as_customer_statement_without_names():
    statements = parse_10k_html(FIXTURE)
    ar = [s for s in statements if "accounts receivable" in s.sentence]
    assert len(ar) == 1
    assert ar[0].kind == "customer"
    assert ar[0].percent == 39.0
    assert ar[0].names == ()


def test_explicit_customer_list_names_are_extracted():
    statements = parse_10k_html(FIXTURE)
    listed = [s for s in statements if s.sentence.startswith("Our customers include")]
    assert listed and listed[0].kind == "customer"
    assert listed[0].names == ("Amazon", "Alphabet", "Meta")
    assert listed[0].percent is None


def test_government_percentage_yields_no_named_counterparty():
    statements = parse_10k_html(FIXTURE)
    for statement in statements:
        assert not any("U.S" in name or "government" in name.lower() for name in statement.names)


def test_customer_without_revenue_or_percent_is_ignored():
    statements = parse_10k_html(FIXTURE)
    assert not [s for s in statements if "Dell and Lenovo" in s.sentence]


def test_supplier_statements_are_extracted_with_names():
    suppliers = _by_kind(parse_10k_html(FIXTURE), "supplier")
    by_name = {name: s for s in suppliers for name in s.names}
    assert by_name["TSMC"].percent is None
    assert "Broadcom" in by_name
    assert "Micron" in by_name
    assert "SK" in by_name["Micron"].names


def test_sentences_over_cap_are_skipped_and_duplicates_collapsed():
    long_sentence = "Our largest customer accounted for 20% of revenue, " + ("and more detail " * 80) + "end."
    assert len(long_sentence) > MAX_SENTENCE_CHARS
    repeated = "<p>We depend on TSMC to manufacture our chips.</p><p>We depend on TSMC to manufacture our chips.</p>"
    html = f"<body><p>{long_sentence}</p>{repeated}</body>"
    statements = parse_10k_html(html)
    assert all(len(s.sentence) <= MAX_SENTENCE_CHARS for s in statements)
    assert len([s for s in statements if "TSMC" in s.names]) == 1


def test_abbreviations_do_not_split_sentences():
    statements = parse_10k_html("<p>The U.S. government accounted for 10% of our revenue. We depend on TSMC to build chips.</p>")
    assert [s.sentence for s in statements if s.kind == "supplier"] == [
        "We depend on TSMC to build chips."
    ]
