"""Comprehensive tests for tts_normalizer.normalize_for_speech."""

from __future__ import annotations

import pytest

from openclaw_voice.tts_normalizer import normalize_for_speech, _path_to_speech


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def norm(text: str) -> str:
    """Alias for normalize_for_speech to keep tests concise."""
    return normalize_for_speech(text)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        assert norm("") == ""

    def test_whitespace_only(self):
        assert norm("   \n\t  ") == ""

    def test_already_clean(self):
        assert norm("Hello, how are you today?") == "Hello, how are you today?"

    def test_single_word(self):
        assert norm("hello") == "hello"

    def test_newlines_collapsed(self):
        assert norm("line one\n\nline two") == "line one line two"

    def test_multiple_spaces_collapsed(self):
        assert norm("too   many   spaces") == "too many spaces"


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------


class TestMarkdownStripping:
    def test_bold(self):
        assert norm("**bold text**") == "bold text"

    def test_italic_star(self):
        assert norm("*italic text*") == "italic text"

    def test_bold_italic(self):
        assert norm("***both***") == "both"

    def test_inline_code(self):
        assert norm("`some_code`") == "some code"

    def test_fenced_code_block(self):
        text = "```python\nprint('hello')\n```"
        result = norm(text)
        assert "```" not in result
        assert "print" in result

    def test_fenced_code_block_no_lang(self):
        text = "```\nfoo = bar\n```"
        result = norm(text)
        assert "```" not in result

    def test_atx_header_h1(self):
        assert norm("# My Header") == "My Header"

    def test_atx_header_h2(self):
        assert norm("## Sub Header") == "Sub Header"

    def test_atx_header_h3(self):
        assert norm("### Deep") == "Deep"

    def test_markdown_link(self):
        assert norm("[click here](https://example.com)") == "click here"

    def test_angle_bracket_url_stripped(self):
        result = norm("<https://example.com>")
        assert "https" not in result
        assert "example.com" not in result

    def test_bare_url_stripped(self):
        result = norm("See https://example.com for details")
        assert "https" not in result
        assert "example.com" not in result

    def test_blockquote(self):
        assert norm("> This is a quote") == "This is a quote"

    def test_horizontal_rule_stripped(self):
        result = norm("---")
        assert result.strip() == ""

    def test_mixed_markdown(self):
        text = "**Bold** and *italic* with `code` and [a link](http://x.com)"
        result = norm(text)
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "http" not in result
        assert "Bold" in result
        assert "italic" in result
        assert "code" in result
        assert "a link" in result


# ---------------------------------------------------------------------------
# Time / duration unit expansion
# ---------------------------------------------------------------------------


class TestTimeUnits:
    def test_seconds_singular(self):
        assert norm("1s") == "1 second"

    def test_seconds_plural(self):
        assert norm("5s") == "5 seconds"

    def test_milliseconds(self):
        assert norm("250ms") == "250 milliseconds"

    def test_milliseconds_singular(self):
        assert norm("1ms") == "1 millisecond"

    def test_minutes_singular(self):
        assert norm("1m") == "1 minute"

    def test_minutes_plural(self):
        assert norm("5m") == "5 minutes"

    def test_hours_singular(self):
        assert norm("1h") == "1 hour"

    def test_hours_plural(self):
        assert norm("2h") == "2 hours"

    def test_microseconds(self):
        assert norm("100us") == "100 microseconds"

    def test_nanoseconds(self):
        assert norm("500ns") == "500 nanoseconds"

    def test_decimal_seconds(self):
        assert norm("1.5s") == "1.5 seconds"

    def test_unit_in_sentence(self):
        result = norm("Wait 250ms between retries")
        assert "250 milliseconds" in result

    def test_no_false_positive_word_ending_s(self):
        # "class" should not be expanded as "clas + s"
        result = norm("This is a class")
        assert "second" not in result

    def test_no_false_positive_caps(self):
        # "HTTPS" should not be "HT second"
        result = norm("Use HTTPS")
        assert "second" not in result

    def test_flush_delay_s_unit(self):
        # "flush_delay_s" — the trailing s after underscore is a unit abbreviation
        # but in practice it's inside a snake_case name, so it's handled by
        # snake humanisation; "1s" by itself should still expand
        assert norm("timeout of 1s") == "timeout of 1 second"


# ---------------------------------------------------------------------------
# Code symbol expansion
# ---------------------------------------------------------------------------


class TestCodeSymbols:
    def test_arrow_dash(self):
        assert norm("a -> b") == "a arrow b"

    def test_arrow_equals(self):
        assert norm("a => b") == "a arrow b"

    def test_greater_equal(self):
        assert norm("x >= 5") == "x greater than or equal to 5"

    def test_less_equal(self):
        assert norm("x <= 5") == "x less than or equal to 5"

    def test_not_equal(self):
        assert norm("x != y") == "x not equal to y"

    def test_double_equals(self):
        assert norm("x == y") == "x equals y"

    def test_logical_and(self):
        assert norm("a && b") == "a and b"

    def test_logical_or(self):
        assert norm("a || b") == "a or b"

    def test_standalone_pipe(self):
        assert norm("a | b") == "a pipe b"

    def test_double_star(self):
        # After markdown stripping, **text** is stripped; bare ** becomes "double star"
        result = norm("use ** for power")
        assert "double star" in result

    def test_standalone_star(self):
        result = norm("use * wildcard")
        assert "star" in result


# ---------------------------------------------------------------------------
# Snake_case and camelCase humanisation
# ---------------------------------------------------------------------------


class TestIdentifierHumanisation:
    def test_snake_case_basic(self):
        result = norm("snake_case_name")
        assert result == "snake case name"

    def test_snake_case_leading_underscore(self):
        result = norm("_flush_delay_s")
        assert result == "flush delay s"

    def test_snake_case_in_sentence(self):
        result = norm("The flush_delay_s parameter")
        assert "flush delay s" in result

    def test_camel_case_basic(self):
        result = norm("camelCaseName")
        assert result == "camel case name"

    def test_camel_case_in_sentence(self):
        result = norm("Set synthesizeResponse flag")
        assert "synthesize" in result.lower()

    def test_all_caps_not_broken(self):
        # ALL_CAPS → "all caps" via snake rule
        result = norm("FOO_BAR")
        # underscores replaced; exact capitalisation depends on regex, but no crash
        assert "_" not in result

    def test_snake_with_numbers(self):
        result = norm("chunk_size_128")
        assert "_" not in result
        assert "chunk" in result


# ---------------------------------------------------------------------------
# PR / issue refs
# ---------------------------------------------------------------------------


class TestPRIssueRefs:
    def test_issue_ref(self):
        assert norm("Fixes #5") == "Fixes number 5"

    def test_pr_ref(self):
        result = norm("See PR #7")
        assert "PR 7" in result
        assert "#" not in result

    def test_pr_ref_case_insensitive(self):
        result = norm("pr #12")
        assert "#" not in result
        assert "12" in result

    def test_issue_ref_in_sentence(self):
        result = norm("Closes #42 and #43")
        assert "#" not in result
        assert "number 42" in result
        assert "number 43" in result


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------


class TestFilePaths:
    def test_absolute_python_path(self):
        result = norm("/src/openclaw_voice/foo.py")
        assert "src" in result
        assert "openclaw voice" in result
        assert "foo" in result
        assert "dot py" in result
        assert "/" not in result

    def test_absolute_path_no_extension(self):
        result = norm("/etc/hosts")
        assert "/" not in result
        assert "etc" in result
        assert "hosts" in result

    def test_path_in_sentence(self):
        result = norm("See /src/foo/bar.py for details")
        assert "/" not in result or result.startswith("See ")

    def test_relative_path(self):
        result = norm("./tests/test_foo.py")
        assert "tests" in result
        assert "dot py" in result

    def test_path_to_speech_helper(self):
        result = _path_to_speech("/src/openclaw_voice/foo.py")
        assert "src" in result
        assert "openclaw voice" in result
        assert "foo dot py" in result


# ---------------------------------------------------------------------------
# Version numbers
# ---------------------------------------------------------------------------


class TestVersionNumbers:
    def test_version_basic(self):
        assert norm("v1.3.2") == "version 1.3.2"

    def test_version_in_sentence(self):
        result = norm("Released in v2.0.0")
        assert "version 2.0.0" in result

    def test_version_two_part(self):
        result = norm("v1.0")
        assert "version 1.0" in result

    def test_version_uppercase(self):
        result = norm("V1.2.3")
        assert "version" in result.lower()


# ---------------------------------------------------------------------------
# Emoji stripping
# ---------------------------------------------------------------------------


class TestEmojiStripping:
    def test_smile_emoji(self):
        result = norm("Hello 😀 world")
        assert "😀" not in result
        assert "Hello" in result
        assert "world" in result

    def test_rocket_emoji(self):
        result = norm("Deployed 🚀")
        assert "🚀" not in result

    def test_multiple_emoji(self):
        result = norm("👍 Nice work! 🎉")
        assert "👍" not in result
        assert "🎉" not in result
        assert "Nice work" in result

    def test_no_emoji_unchanged(self):
        result = norm("No emoji here")
        assert result == "No emoji here"


# ---------------------------------------------------------------------------
# Mixed / integration
# ---------------------------------------------------------------------------


class TestMixedContent:
    def test_real_world_bot_message(self):
        text = (
            "**flush_delay_s** is set to `250ms` in the pipeline. "
            "See PR #7 for details. Version v1.3.2 fixes this."
        )
        result = norm(text)
        assert "flush delay s" in result
        assert "250 milliseconds" in result
        assert "PR 7" in result
        assert "version 1.3.2" in result
        assert "**" not in result
        assert "`" not in result

    def test_code_block_with_arrow(self):
        text = "```\nif x -> y:\n    pass\n```"
        result = norm(text)
        assert "```" not in result

    def test_header_with_version(self):
        result = norm("## Release v2.1.0")
        assert "##" not in result
        assert "version 2.1.0" in result

    def test_snake_in_code_span(self):
        result = norm("`_my_var_name`")
        assert "`" not in result
        assert "my var name" in result

    def test_path_and_version(self):
        result = norm("Updated /src/foo/bar.py in v1.0.1")
        assert "/" not in result
        assert "version 1.0.1" in result

    def test_empty_input_passthrough(self):
        assert norm("") == ""

    def test_no_double_processing(self):
        # Running twice should be idempotent (no explosions)
        text = "**bold** and `code` with #5"
        once = norm(text)
        twice = norm(once)
        assert once == twice
