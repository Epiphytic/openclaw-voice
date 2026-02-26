"""TTS text normalizer — clean up text for natural speech synthesis.

Handles:
- Markdown stripping (bold, italic, headers, code blocks, links)
- Code symbol expansion (* → star, -> → arrow, >= → greater than or equal)
- snake_case / camelCase humanization
- Time unit expansion (1s → one second, 250ms → 250 milliseconds)
- Number-to-words for small numbers
- File path simplification
- Emoji removal
- Discord mention cleanup
"""

import re

# ---------------------------------------------------------------------------
# Number-to-words (0-20 + tens, keeps larger numbers as digits)
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _num_to_words(n: int) -> str:
    """Convert small integers to words. Returns digit string for large numbers."""
    if 0 <= n <= 20:
        return _ONES[n]
    if 21 <= n <= 99:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    return str(n)


# ---------------------------------------------------------------------------
# Symbol expansion
# ---------------------------------------------------------------------------

_SYMBOL_MAP = {
    "->": " arrow ",
    "=>": " arrow ",
    ">=": " greater than or equal to ",
    "<=": " less than or equal to ",
    "!=": " not equal to ",
    "==": " equals ",
    "&&": " and ",
    "||": " or ",
    "...": " ",
    "``": "",
    "`": "",
}

# ---------------------------------------------------------------------------
# Time unit expansion
# ---------------------------------------------------------------------------

_TIME_UNITS = {
    "ms": "millisecond",
    "s": "second",
    "m": "minute",
    "h": "hour",
    "d": "day",
}


def _expand_time_unit(match: re.Match) -> str:
    num = match.group(1)
    unit = match.group(2)
    word = _TIME_UNITS.get(unit, unit)
    try:
        n = int(num)
        num_word = _num_to_words(n)
        plural = "s" if n != 1 else ""
        return f"{num_word} {word}{plural}"
    except ValueError:
        return f"{num} {word}s"


# ---------------------------------------------------------------------------
# Identifier humanization
# ---------------------------------------------------------------------------

def _humanize_identifier(s: str) -> str:
    """Convert snake_case or camelCase to spoken words."""
    # snake_case
    s = s.replace("_", " ").replace("-", " ")
    # camelCase → camel Case
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

def normalize_for_tts(text: str) -> str:
    """Normalize text for TTS output. Returns cleaned text suitable for speech."""
    if not text:
        return text

    # Strip code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", " ", text)

    # Strip inline code (`...`)
    text = re.sub(r"`([^`]+)`", lambda m: _humanize_identifier(m.group(1)), text)

    # Strip markdown headers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Strip bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)  # strikethrough

    # Strip markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Strip bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Discord mentions <@123456> → ""
    text = re.sub(r"<@!?\d+>", "", text)
    text = re.sub(r"<#\d+>", "", text)
    text = re.sub(r"<@&\d+>", "", text)

    # Emoji :name: → ""
    text = re.sub(r"<a?:\w+:\d+>", "", text)  # custom Discord emoji
    text = re.sub(r":\w+:", "", text)  # text emoji shortcodes

    # Expand time units (1s, 250ms, 2h, etc.) — must come before symbol expansion
    text = re.sub(r"\b(\d+)(ms|s|m|h|d)\b", _expand_time_unit, text)

    # Expand symbols
    for sym, replacement in _SYMBOL_MAP.items():
        text = text.replace(sym, replacement)

    # Hash references: #5 → "number 5", PR #12 → "PR 12"
    text = re.sub(r"(?<!\w)#(\d+)", r"number \1", text)

    # File paths: /home/user/foo/bar.py → "bar.py"
    text = re.sub(r"(?:/[\w.-]+){2,}", lambda m: m.group(0).rsplit("/", 1)[-1], text)

    # Expand small standalone numbers to words (1-20)
    def _expand_small_num(m: re.Match) -> str:
        n = int(m.group(0))
        if 0 <= n <= 20:
            return _num_to_words(n)
        return m.group(0)

    text = re.sub(r"\b(\d{1,2})\b", _expand_small_num, text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing punctuation artifacts
    text = re.sub(r"^\s*[,;:]\s*", "", text)
    text = re.sub(r"\s*[,;:]\s*$", "", text)

    return text
