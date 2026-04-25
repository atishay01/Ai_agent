"""
Extra tools the agent can call besides SQL:
  - get_usd_brl_rate     : live BRL->USD rate (requests, with TTL cache + last-known fallback)
  - lookup_brazilian_state: state name + capital (Wikipedia, with hardcoded fallback)
  - calculate            : exact arithmetic (avoids LLM math drift)

Both web tools degrade gracefully:
  * ``get_usd_brl_rate`` caches successful responses for ``settings.exchange_rate_ttl_seconds``
    and falls back to the last good rate (or a hardcoded baseline) if the API is down.
  * ``lookup_brazilian_state`` falls back to a hardcoded table of all 26 states
    + Federal District if Wikipedia is unreachable or the page layout changes.
"""

import ast
import operator as op
import threading
import time

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from config import settings
from logging_setup import logger

_log = logger.bind(component="web_tools")


# ---------- Tool 1: live exchange rate -------------------------------
# Last successful (rate, date, fetched_at_monotonic) tuple; persists for the
# life of the process so a brief Frankfurter outage doesn't break currency
# answers. Hardcoded baseline is used only on the very first call if the
# API is unreachable — picked as a recent plausible BRL->USD value.
_RATE_FALLBACK = (0.20, "fallback", 0.0)
_rate_state: tuple[float, str, float] = _RATE_FALLBACK
# Single-flight lock: prevents two concurrent misses from both hitting
# Frankfurter when the cache expires.
_rate_lock = threading.Lock()


def _format_rate(rate: float, date: str) -> str:
    return f"Exchange rate on {date}: 1 BRL = {rate:.4f} USD"


@tool
def get_usd_brl_rate() -> str:
    """Return the current Brazilian Real (BRL) to US Dollar (USD) rate.
    All prices in the database are in BRL; use this when the user asks
    for anything in USD."""
    global _rate_state
    rate, date, fetched_at = _rate_state
    now = time.monotonic()

    # Serve from in-process TTL cache so we don't hammer Frankfurter.
    if fetched_at and (now - fetched_at) < settings.exchange_rate_ttl_seconds:
        return _format_rate(rate, date)

    # Single-flight: only one thread fetches; others wait and re-check
    # the freshly-populated cache. Avoids a thundering herd on TTL expiry.
    with _rate_lock:
        rate, date, fetched_at = _rate_state
        now = time.monotonic()
        if fetched_at and (now - fetched_at) < settings.exchange_rate_ttl_seconds:
            return _format_rate(rate, date)

        try:
            r = requests.get(
                "https://api.frankfurter.app/latest",
                params={"from": "BRL", "to": "USD"},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            rate = float(data["rates"]["USD"])
            date = str(data.get("date", "unknown"))
            _rate_state = (rate, date, now)
            return _format_rate(rate, date)
        except Exception as exc:
            _log.warning("Frankfurter unavailable, using last-known rate: {}", exc)
            # Note in the response that this is a fallback so the agent can
            # mention it in the answer if appropriate.
            return _format_rate(rate, date) + " (cached fallback — live API unavailable)"


# ---------- Tool 2: scrape Wikipedia for state info ------------------
# Hardcoded fallback table used when Wikipedia is unreachable or the
# scraping logic fails to find a match. Source: official IBGE list.
_BR_STATES: dict[str, tuple[str, str]] = {
    "AC": ("Acre", "Rio Branco"),
    "AL": ("Alagoas", "Maceió"),
    "AP": ("Amapá", "Macapá"),
    "AM": ("Amazonas", "Manaus"),
    "BA": ("Bahia", "Salvador"),
    "CE": ("Ceará", "Fortaleza"),
    "DF": ("Distrito Federal", "Brasília"),
    "ES": ("Espírito Santo", "Vitória"),
    "GO": ("Goiás", "Goiânia"),
    "MA": ("Maranhão", "São Luís"),
    "MT": ("Mato Grosso", "Cuiabá"),
    "MS": ("Mato Grosso do Sul", "Campo Grande"),
    "MG": ("Minas Gerais", "Belo Horizonte"),
    "PA": ("Pará", "Belém"),
    "PB": ("Paraíba", "João Pessoa"),
    "PR": ("Paraná", "Curitiba"),
    "PE": ("Pernambuco", "Recife"),
    "PI": ("Piauí", "Teresina"),
    "RJ": ("Rio de Janeiro", "Rio de Janeiro"),
    "RN": ("Rio Grande do Norte", "Natal"),
    "RS": ("Rio Grande do Sul", "Porto Alegre"),
    "RO": ("Rondônia", "Porto Velho"),
    "RR": ("Roraima", "Boa Vista"),
    "SC": ("Santa Catarina", "Florianópolis"),
    "SP": ("São Paulo", "São Paulo"),
    "SE": ("Sergipe", "Aracaju"),
    "TO": ("Tocantins", "Palmas"),
}


def _format_state(code: str, name: str, capital: str) -> str:
    return f"{code} = {name} (capital: {capital})"


def _state_from_table(code: str) -> str | None:
    entry = _BR_STATES.get(code)
    if entry is None:
        return None
    name, capital = entry
    return _format_state(code, name, capital)


@tool
def lookup_brazilian_state(state_code: str) -> str:
    """Given a 2-letter Brazilian state code (e.g. 'SP'), return the
    state full name and capital city.

    Answers from the bundled IBGE table first (canonical, 27 entries,
    O(1)). Wikipedia is only consulted for codes that aren't in the
    local table — i.e. essentially never. This avoids per-call HTTP
    latency, scraping fragility, and rate-limit risk for the common path.
    Use when the user asks for a state's name or capital.
    """
    code = state_code.strip().upper()[:2]
    local = _state_from_table(code)
    if local is not None:
        return local

    # Code is not in the canonical IBGE table (likely garbage, e.g. "ZZ").
    # Try Wikipedia as a last-resort scrape in case the user typed an
    # archaic or alternate code we don't carry.
    url = "https://en.wikipedia.org/wiki/States_of_Brazil"
    try:
        r = requests.get(url, headers={"User-Agent": "Olist/1.0"}, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "lxml")
        for row in soup.select("table.wikitable tr"):
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if code in cells:
                i = cells.index(code)
                name = cells[i - 1] if i - 1 >= 0 else "?"
                capital = cells[i + 1] if i + 1 < len(cells) else "?"
                return _format_state(code, name, capital)
    except Exception as exc:
        _log.warning("Wikipedia state lookup failed for {}: {}", code, exc)

    return f"No Brazilian state found for code '{code}'."


# ---------- Tool 3: exact calculator ---------------------------------
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("only arithmetic allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly.
    Use this for multiplication involving big numbers and
    currency conversions — LLM math can drift on large values.
    Example input: "5921678.12 * 0.2009"
    """
    try:
        val = _eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {val:,.2f}" if isinstance(val, float) else f"{expression} = {val}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


WEB_TOOLS = [get_usd_brl_rate, lookup_brazilian_state, calculate]


if __name__ == "__main__":
    print(get_usd_brl_rate.invoke({}))
    print(lookup_brazilian_state.invoke({"state_code": "SP"}))
    print(calculate.invoke({"expression": "5921678.12 * 0.2009"}))
