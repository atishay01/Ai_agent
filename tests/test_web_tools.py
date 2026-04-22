"""Tests for the custom web tools.

The HTTP-bound tools (`get_usd_brl_rate`, `lookup_brazilian_state`) are
mocked with ``respx`` so tests are offline and deterministic. The
``calculate`` tool is pure arithmetic and can be tested directly.
"""

from unittest.mock import patch

import pytest

from web_tools import calculate, get_usd_brl_rate, lookup_brazilian_state


# ---------------------------------------------------------------------
# calculate — pure, no network.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expr", "expected_fragment"),
    [
        ("2 + 2", "= 4"),
        ("5921678.12 * 0.2009", "= 1,189,665.13"),
        ("100 - 30", "= 70"),
        ("2 ** 10", "= 1024"),
        ("-(5 + 3)", "= -8"),
        ("10 / 4", "= 2.50"),
    ],
)
def test_calculate_valid(expr: str, expected_fragment: str) -> None:
    out = calculate.invoke({"expression": expr})
    assert expected_fragment in out


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('rm -rf /')",  # arbitrary code
        "open('secret').read()",
        "1 // 0",  # floor-div not in _OPS
        "a + 1",  # name
        "print(1)",
        "",
    ],
)
def test_calculate_rejects_unsafe(expr: str) -> None:
    out = calculate.invoke({"expression": expr})
    assert out.lower().startswith("could not evaluate")


# ---------------------------------------------------------------------
# get_usd_brl_rate — mocked HTTP.
# ---------------------------------------------------------------------
def test_get_usd_brl_rate_ok() -> None:
    fake_json = {"date": "2025-01-15", "rates": {"USD": 0.1995}}
    with patch("web_tools.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_json
        mock_get.return_value.raise_for_status.return_value = None
        out = get_usd_brl_rate.invoke({})
    assert "2025-01-15" in out
    assert "0.1995" in out
    assert "BRL" in out and "USD" in out


# ---------------------------------------------------------------------
# lookup_brazilian_state — mocked HTML.
# ---------------------------------------------------------------------
_WIKIPEDIA_SNIPPET = """
<html><body>
<table class="wikitable">
  <tr><th>Name</th><th>Code</th><th>Capital</th></tr>
  <tr><td>S&atilde;o Paulo</td><td>SP</td><td>S&atilde;o Paulo</td></tr>
  <tr><td>Rio de Janeiro</td><td>RJ</td><td>Rio de Janeiro</td></tr>
</table>
</body></html>
"""


def test_lookup_state_found() -> None:
    with patch("web_tools.requests.get") as mock_get:
        mock_get.return_value.content = _WIKIPEDIA_SNIPPET.encode("utf-8")
        mock_get.return_value.raise_for_status.return_value = None
        out = lookup_brazilian_state.invoke({"state_code": "RJ"})
    assert "RJ" in out
    assert "Rio de Janeiro" in out


def test_lookup_state_not_found() -> None:
    with patch("web_tools.requests.get") as mock_get:
        mock_get.return_value.content = _WIKIPEDIA_SNIPPET.encode("utf-8")
        mock_get.return_value.raise_for_status.return_value = None
        out = lookup_brazilian_state.invoke({"state_code": "ZZ"})
    assert "No Brazilian state" in out
