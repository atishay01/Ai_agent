"""
Extra tools the agent can call besides SQL:
  - get_usd_brl_rate     : live BRL->USD rate (requests)
  - lookup_brazilian_state: scrape Wikipedia for state full name + capital
  - calculate            : exact arithmetic (avoids LLM math drift)
"""

import ast
import operator as op

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# ---------- Tool 1: live exchange rate -------------------------------
@tool
def get_usd_brl_rate() -> str:
    """Return the current Brazilian Real (BRL) to US Dollar (USD) rate.
    All prices in the database are in BRL; use this when the user asks
    for anything in USD."""
    r = requests.get("https://api.frankfurter.app/latest",
                     params={"from": "BRL", "to": "USD"}, timeout=5)
    r.raise_for_status()
    data = r.json()
    rate = data["rates"]["USD"]
    return f"Exchange rate on {data['date']}: 1 BRL = {rate:.4f} USD"


# ---------- Tool 2: scrape Wikipedia for state info ------------------
@tool
def lookup_brazilian_state(state_code: str) -> str:
    """Given a 2-letter Brazilian state code (e.g. 'SP'), return the
    state full name and capital city by scraping Wikipedia.
    Use when the user asks for a state's name or capital."""
    code = state_code.strip().upper()[:2]
    url = "https://en.wikipedia.org/wiki/States_of_Brazil"
    r = requests.get(url, headers={"User-Agent": "Olist/1.0"}, timeout=8)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    # Each row in the main wikitable lists: flag, coat of arms, name,
    # code, capital, ... — we find the row whose code matches.
    for row in soup.select("table.wikitable tr"):
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if code in cells:
            # The cell immediately before the code cell is the state name,
            # the one immediately after is usually the capital.
            i = cells.index(code)
            name    = cells[i - 1] if i - 1 >= 0 else "?"
            capital = cells[i + 1] if i + 1 < len(cells) else "?"
            return f"{code} = {name} (capital: {capital})"
    return f"No Brazilian state found for code '{code}'."


# ---------- Tool 3: exact calculator ---------------------------------
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
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
        return f"{expression} = {val:,.2f}" if isinstance(val, float) \
               else f"{expression} = {val}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


WEB_TOOLS = [get_usd_brl_rate, lookup_brazilian_state, calculate]


if __name__ == "__main__":
    print(get_usd_brl_rate.invoke({}))
    print(lookup_brazilian_state.invoke({"state_code": "SP"}))
    print(calculate.invoke({"expression": "5921678.12 * 0.2009"}))
