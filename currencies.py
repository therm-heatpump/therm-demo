"""
Currencies supported by THERM for tariff rates and cost displays.

Profiles store both a display symbol ("currency": "€") and an ISO code
("currency_code": "EUR"). Legacy profiles only have the symbol.
No Streamlit import.
"""
from __future__ import annotations

import re
from typing import Any

# EUR, GBP, USD first; then alphabetical by currency name.
CURRENCIES: list[tuple[str, str, str]] = [
    ("EUR", "€", "Euro"),
    ("GBP", "£", "British Pound"),
    ("USD", "$", "US Dollar"),
    ("AUD", "A$", "Australian Dollar"),
    ("BGN", "лв", "Bulgarian Lev"),
    ("CAD", "C$", "Canadian Dollar"),
    ("CZK", "Kč", "Czech Koruna"),
    ("DKK", "kr", "Danish Krone"),
    ("HUF", "Ft", "Hungarian Forint"),
    ("ISK", "kr", "Icelandic Króna"),
    ("JPY", "¥", "Japanese Yen"),
    ("NZD", "NZ$", "New Zealand Dollar"),
    ("NOK", "kr", "Norwegian Krone"),
    ("PLN", "zł", "Polish Złoty"),
    ("RON", "lei", "Romanian Leu"),
    ("SEK", "kr", "Swedish Krona"),
    ("CHF", "CHF", "Swiss Franc"),
]

_SYMBOLS: dict[str, str] = {code: symbol for code, symbol, _ in CURRENCIES}
_ISO3_RE = re.compile(r"^[A-Za-z]{3}$")

_LEGACY_SYMBOLS: dict[str, tuple[str, str]] = {
    "€": ("EUR", "€"),
    "£": ("GBP", "£"),
    "$": ("USD", "$"),
}


def symbol_for(code: str) -> str:
    """The display symbol for a known ISO code, otherwise the code itself."""
    if not code or not isinstance(code, str):
        return ""
    code_clean = code.strip().upper()
    return _SYMBOLS.get(code_clean, code)


def resolve(profile: dict[str, Any] | None) -> tuple[str, str]:
    """
    (code, symbol) for a profile dictionary.

    Uses currency_code if present; otherwise maps legacy symbols (€, £, $).
    Defaults to ("EUR", "€") when unknown or empty.
    """
    if not isinstance(profile, dict):
        return ("EUR", "€")

    code = profile.get("currency_code")
    if isinstance(code, str) and code.strip():
        code_upper = code.strip().upper()
        symbol = profile.get("currency")
        if not isinstance(symbol, str) or not symbol.strip():
            symbol = symbol_for(code_upper)
        return (code_upper, symbol)

    legacy = profile.get("currency")
    if isinstance(legacy, str) and legacy in _LEGACY_SYMBOLS:
        return _LEGACY_SYMBOLS[legacy]

    return ("EUR", "€")


def from_home_assistant(code: str | None) -> tuple[str, str] | None:
    """
    (code, symbol) if code is a valid 3-letter ISO code (known or unknown), else None.
    """
    if not isinstance(code, str):
        return None
    code_clean = code.strip().upper()
    if not _ISO3_RE.fullmatch(code_clean):
        return None
    return (code_clean, symbol_for(code_clean))
