"""TTS text normalizer for technical/code content.

Pre-processes text before it is sent to the Kokoro TTS engine so that code,
markdown, symbols, and other technical artefacts are rendered naturally when
spoken aloud.

Transformations are applied in a fixed pipeline order:

1.  Strip markdown (code blocks, inline code, bold/italic, headers, links, URLs)
2.  Expand time/duration units (``1s`` → "1 second", ``250ms`` → "250 milliseconds")
3.  Expand code symbols (``->`` → "arrow", ``!=`` → "not equal to", …)
4.  Simplify file paths for speech (before identifier humanisation)
5.  Humanise snake_case and camelCase identifiers
6.  Expand PR / GitHub issue references (``#5`` → "number 5")
7.  Expand version numbers (``v1.3.2`` → "version 1.3.2")
8.  Strip emoji characters
9.  Collapse whitespace

Usage::

    from openclaw_voice.tts_normalizer import normalize_for_speech

    clean = normalize_for_speech(raw_text)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns (compiled once at module load for efficiency)
# ---------------------------------------------------------------------------

# ── Markdown ──────────────────────────────────────────────────────────────────

# Fenced code blocks (``` or ~~~, optional language tag)
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)

# Inline code (`...`)
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")

# Bold+italic (***text***)
_RE_BOLD_ITALIC = re.compile(r"\*{3}([^*]+)\*{3}")

# Bold (**text**)
_RE_BOLD = re.compile(r"\*{2}([^*]+)\*{2}")

# Italic (*text* or _text_)  — we only handle *text* here; _text_ is covered by
# the snake_case step if inside an identifier, but bare _italic_ → italic.
_RE_ITALIC_STAR = re.compile(r"\*([^*\n]+)\*")
_RE_ITALIC_UNDER = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")

# ATX headers (# Header, ## Header, …)
_RE_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Markdown links: [text](url)
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Bare URLs inside angle brackets: <https://…>
_RE_ANGLE_URL = re.compile(r"<https?://[^>]+>")

# Bare URLs (http/https)
_RE_BARE_URL = re.compile(r"https?://\S+")

# Blockquote lines (> text)
_RE_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)

# Horizontal rules (--- or *** or ___ on their own line)
_RE_HR = re.compile(r"^(?:\*{3,}|-{3,}|_{3,})\s*$", re.MULTILINE)

# ── Time / duration units ─────────────────────────────────────────────────────

# Order matters: longest unit suffix first so "ms" beats "m" and "s".
_TIME_UNITS: list[tuple[str, str, str]] = [
    # (suffix_pattern, singular, plural)
    ("ms", "millisecond", "milliseconds"),
    ("us", "microsecond", "microseconds"),
    ("ns", "nanosecond",  "nanoseconds"),
    ("hr", "hour",        "hours"),
    ("h",  "hour",        "hours"),
    ("m",  "minute",      "minutes"),
    ("s",  "second",      "seconds"),
]

# We build a single alternation so that longer suffixes are tried first.
_TIME_SUFFIX_ALT = "|".join(re.escape(s) for s, _, _ in _TIME_UNITS)
_RE_TIME_UNIT = re.compile(
    rf"(?<!\w)(\d+(?:\.\d+)?)({_TIME_SUFFIX_ALT})\b",
    re.IGNORECASE,
)

# Map suffix → (singular, plural) for O(1) lookup
_UNIT_MAP: dict[str, tuple[str, str]] = {
    s.lower(): (sg, pl) for s, sg, pl in _TIME_UNITS
}

# ── Code symbols ──────────────────────────────────────────────────────────────

# Order matters: longer/more-specific patterns before shorter/ambiguous ones.
_CODE_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\*\*"),          "double star"),
    (re.compile(r"->"),            "arrow"),
    (re.compile(r"=>"),            "arrow"),
    (re.compile(r">="),            "greater than or equal to"),
    (re.compile(r"<="),            "less than or equal to"),
    (re.compile(r"!="),            "not equal to"),
    (re.compile(r"=="),            "equals"),
    (re.compile(r"&&"),            "and"),
    (re.compile(r"\|\|"),          "or"),
    # Standalone pipe (not part of a word or table)
    (re.compile(r"(?<!\w)\|(?!\w)"), "pipe"),
    # Standalone star (not part of a word; e.g. bullet "* item" already stripped
    # by markdown pass, but catch any remaining bare *)
    (re.compile(r"(?<!\w)\*(?!\w)"), "star"),
]

# ── File paths ────────────────────────────────────────────────────────────────

# Absolute or relative paths: starts with / or ./ or ../ or looks like
# path/with/slashes.ext
_RE_FILE_PATH = re.compile(
    r"(?<!\w)"                        # not preceded by word char
    r"(?:"
    r"(?:/[\w.\-]+)+"                 # absolute: /foo/bar/baz.py
    r"|"
    r"(?:\.\.?/)?[\w.\-]+(?:/[\w.\-]+)+(?:\.\w+)?"  # relative: ./foo/bar.py
    r")"
)

# Extension → spoken form
_EXT_MAP: dict[str, str] = {
    "py":    "dot py",
    "js":    "dot js",
    "ts":    "dot ts",
    "jsx":   "dot jsx",
    "tsx":   "dot tsx",
    "json":  "dot json",
    "toml":  "dot toml",
    "yaml":  "dot yaml",
    "yml":   "dot yaml",
    "md":    "dot md",
    "txt":   "dot text",
    "sh":    "dot sh",
    "rs":    "dot rs",
    "go":    "dot go",
    "rb":    "dot rb",
    "java":  "dot java",
    "c":     "dot c",
    "h":     "dot h",
    "cpp":   "dot cpp",
    "cs":    "dot c sharp",
    "html":  "dot html",
    "css":   "dot css",
}

# ── Version numbers ───────────────────────────────────────────────────────────

_RE_VERSION = re.compile(r"\bv(\d+(?:\.\d+)+)\b", re.IGNORECASE)

# ── PR / Issue refs ───────────────────────────────────────────────────────────

# "PR #7" → "PR 7"
_RE_PR_REF = re.compile(r"\bPR\s+#(\d+)\b", re.IGNORECASE)
# Remaining "#5" → "number 5"
_RE_ISSUE_REF = re.compile(r"(?<!\w)#(\d+)\b")

# ── Identifiers (snake_case / camelCase) ─────────────────────────────────────

# snake_case: word chars with at least one underscore (not all digits)
_RE_SNAKE = re.compile(
    r"(?<!\w)"           # not preceded by word char
    r"_*"                # optional leading underscores
    r"([a-zA-Z]\w*?)"   # first word (must start with a letter)
    r"(?:_+\w+)+"        # one or more _word segments
    r"_*"                # optional trailing underscores
    r"(?!\w)"            # not followed by word char
)

# camelCase: at least one interior uppercase letter
_RE_CAMEL = re.compile(
    r"(?<![A-Z])\b"      # word boundary not preceded by uppercase
    r"([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)"  # camelCase body
    r"\b"
)

# ── Emoji ─────────────────────────────────────────────────────────────────────

# Unicode ranges covering the major emoji blocks.  We use a broad range to
# catch skin-tone modifiers, ZWJ sequences, etc. without a dependency on the
# `emoji` library.
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "\U0000200D"             # ZWJ
    "\U0000FE0F"             # variation selector-16
    "\U000020E3"             # combining enclosing keycap
    "]",
    flags=re.UNICODE,
)

# ── Whitespace ────────────────────────────────────────────────────────────────

_RE_WHITESPACE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _expand_time_unit(m: re.Match[str]) -> str:
    """Replace a ``<number><unit>`` match with its spoken form.

    Args:
        m: Regex match with groups (number, unit_suffix).

    Returns:
        Spoken form string, e.g. "1 second" or "250 milliseconds".
    """
    number_str = m.group(1)
    suffix = m.group(2).lower()
    singular, plural = _UNIT_MAP[suffix]
    # Try to determine singular vs plural.
    try:
        value = float(number_str)
        unit = singular if value == 1.0 else plural
    except ValueError:
        unit = plural
    return f"{number_str} {unit}"


def _humanize_snake(m: re.Match[str]) -> str:
    """Convert a snake_case identifier match to a space-separated phrase.

    Args:
        m: Regex match for the whole snake_case token.

    Returns:
        Human-readable phrase with underscores replaced by spaces and
        leading/trailing underscores stripped.
    """
    token = m.group(0).strip("_")
    return token.replace("_", " ")


def _humanize_camel(m: re.Match[str]) -> str:
    """Convert a camelCase identifier match to a space-separated phrase.

    Args:
        m: Regex match for the camelCase token.

    Returns:
        Human-readable phrase with spaces inserted before each capital letter.
    """
    token = m.group(0)
    # Insert a space before each uppercase letter that follows a lowercase letter.
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
    return spaced.lower()


def _path_to_speech(path: str) -> str:
    """Convert a file-system path to a speakable phrase.

    Args:
        path: A filesystem path string (absolute or relative).

    Returns:
        A comma-separated, speech-friendly phrase.

    Examples:
        ``/src/openclaw_voice/foo.py`` → ``src, openclaw voice, foo dot py``
    """
    # Strip leading slash/dot-slash
    path = path.lstrip("/").lstrip("./").lstrip("/")

    parts = path.split("/")
    spoken_parts: list[str] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        is_last = i == len(parts) - 1
        if is_last and "." in part:
            name, _, ext = part.rpartition(".")
            ext_spoken = _EXT_MAP.get(ext.lower(), f"dot {ext}")
            # Humanise the name portion (may be snake_case)
            name_clean = name.replace("_", " ").replace("-", " ")
            spoken_parts.append(f"{name_clean} {ext_spoken}")
        else:
            clean = part.replace("_", " ").replace("-", " ")
            spoken_parts.append(clean)

    return ", ".join(spoken_parts) if spoken_parts else path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_for_speech(text: str) -> str:
    """Normalize text for natural TTS output of technical/code content.

    Applies a fixed transformation pipeline to make markdown, code symbols,
    identifiers, file paths, and other technical artefacts sound natural
    when spoken by a TTS engine.

    Args:
        text: Raw input text, potentially containing markdown or code.

    Returns:
        Cleaned, speech-friendly string.

    Examples:
        >>> normalize_for_speech("**flush_delay_s** is `250ms`")
        'flush delay s is 250 milliseconds'
        >>> normalize_for_speech("See PR #7 in v1.3.2")
        'See PR 7 in version 1.3.2'
    """
    if not text:
        return text

    # 1. Strip markdown ---------------------------------------------------

    # Fenced code blocks (replace with just the code text, stripped of fences)
    def _unfence(m: re.Match[str]) -> str:
        inner = m.group(0)
        # Drop the opening/closing fence lines
        lines = inner.splitlines()
        # First line is the fence (``` or ~~~ + optional lang), last is closing fence
        body_lines = lines[1:-1] if len(lines) > 2 else lines
        return " ".join(line.strip() for line in body_lines if line.strip())

    text = _RE_FENCED_CODE.sub(_unfence, text)

    # Inline code — keep the content, drop the backticks
    text = _RE_INLINE_CODE.sub(r"\1", text)

    # Bold+italic, bold, italic
    text = _RE_BOLD_ITALIC.sub(r"\1", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)

    # ATX headers — strip the leading #+ 
    text = _RE_HEADER.sub("", text)

    # Markdown links → link text only
    text = _RE_MD_LINK.sub(r"\1", text)

    # Bare URLs in angle brackets — drop entirely
    text = _RE_ANGLE_URL.sub("", text)

    # Remaining bare URLs — drop entirely
    text = _RE_BARE_URL.sub("", text)

    # Blockquotes
    text = _RE_BLOCKQUOTE.sub("", text)

    # Horizontal rules
    text = _RE_HR.sub("", text)

    # 2. Expand time/duration units ----------------------------------------
    text = _RE_TIME_UNIT.sub(_expand_time_unit, text)

    # 3. Expand code symbols -----------------------------------------------
    for pattern, replacement in _CODE_SYMBOLS:
        text = pattern.sub(replacement, text)

    # 4. File paths (before identifier humanisation so underscores in paths
    #    are handled by _path_to_speech, not the snake_case regex) ----------
    text = _RE_FILE_PATH.sub(lambda m: _path_to_speech(m.group(0)), text)

    # 5. Humanise identifiers (snake_case before camelCase) ----------------
    text = _RE_SNAKE.sub(_humanize_snake, text)
    text = _RE_CAMEL.sub(_humanize_camel, text)

    # 6. PR / issue refs ---------------------------------------------------
    text = _RE_PR_REF.sub(r"PR \1", text)   # "PR #7" → "PR 7"
    text = _RE_ISSUE_REF.sub(r"number \1", text)  # "#5" → "number 5"

    # 7. Version numbers ---------------------------------------------------
    text = _RE_VERSION.sub(r"version \1", text)

    # 8. Strip emoji -------------------------------------------------------
    text = _RE_EMOJI.sub("", text)

    # 9. Collapse whitespace -----------------------------------------------
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text
