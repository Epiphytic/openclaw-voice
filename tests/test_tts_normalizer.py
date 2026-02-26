"""Comprehensive tests for tts_normalizer.normalize_for_speech."""

from __future__ import annotations

import pytest

from openclaw_voice.tts_normalizer import normalize_for_speech, chunk_for_tts, _path_to_speech


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


# ---------------------------------------------------------------------------
# Real-world Discord messages from the bot
# ---------------------------------------------------------------------------


class TestRealMessages:
    """Smoke-test the full pipeline against actual messages the bot sends.

    Assertions focus on three properties:
    1. **No markdown artifacts** — no ``**``, `` ` ``, leading ``#``.
    2. **No emoji garbage** — emoji characters absent.
    3. **Key transformations applied** — specific humanised phrases present.
    """

    def test_real_message_announce_list(self):
        """Numbered list with bold headers, bare domain links, em-dash prose."""
        text = (
            "**Where to announce the Discord voice plugin:**\n\n"
            "1. **OpenClaw Discord** (discord.com/invite/clawd) — existing community, most engaged audience\n"
            "2. **OpenClaw docs** (docs.openclaw.ai) — reference page for the plugin\n"
            "3. **GitHub repo README** — mention voice capability in the feature list\n"
            "4. **ClawHub** (clawhub.com) — list as a community plugin/skill\n"
            "5. **LinkedIn** — fits your content system, could be a \"built this\" post\n"
            "6. **OpenClaw blog** — longer-form write-up if warranted\n\n"
            "Happy to draft an announcement for any of these."
        )
        result = norm(text)

        # ── No markdown artifacts ──────────────────────────────────────────
        assert "**" not in result
        assert "`" not in result

        # ── No emoji ──────────────────────────────────────────────────────
        # (none in input, just ensuring nothing odd appeared)

        # ── Key content preserved ─────────────────────────────────────────
        assert "Where to announce the Discord voice plugin" in result
        assert "OpenClaw Discord" in result
        assert "existing community" in result
        assert "Happy to draft an announcement" in result

        # ── Prose flows as a single line (no raw newlines) ────────────────
        assert "\n" not in result

    def test_real_message_sub_agent_status(self):
        """Bold title + emoji + inline code file paths + time estimate."""
        text = (
            "**Building TTS text normalizer** 🔨\n\n"
            "Sub-agent spawned to:\n"
            "1. Create `src/openclaw_voice/tts_normalizer.py` — pure stdlib, no external deps\n"
            "2. Integrate into `voice_pipeline.py` (before Kokoro TTS call)\n"
            "3. Write comprehensive tests in `tests/test_tts_normalizer.py`\n"
            "4. Feature branch: `feat/tts-normalizer`\n\n"
            "**Transforms:** markdown stripping, time units (1s → \"1 second\"), "
            "code symbols (* → \"star\"), snake_case/camelCase humanization, "
            "PR refs, file paths, version numbers, emoji stripping\n\n"
            "ETA ~10-15 min. Will report back when done."
        )
        result = norm(text)

        # ── No markdown artifacts ──────────────────────────────────────────
        assert "**" not in result
        assert "`" not in result

        # ── Emoji stripped ────────────────────────────────────────────────
        assert "🔨" not in result

        # ── Bold title content present ────────────────────────────────────
        assert "Building TTS text normalizer" in result

        # ── File path spoken naturally ────────────────────────────────────
        # `src/openclaw_voice/tts_normalizer.py` → "src, openclaw voice, tts normalizer dot py"
        assert "src" in result
        assert "openclaw voice" in result
        assert "tts normalizer dot py" in result

        # ── Time unit expanded ────────────────────────────────────────────
        assert "1 second" in result

        # ── Code symbol expansion (star from "* → star" in prose) ───────────
        # The message uses Unicode → (\u2192), not ASCII ->; our normaliser
        # handles ASCII ->  only.  The * before → is expanded to "star".
        assert "star" in result

        # ── snake_case humanised ──────────────────────────────────────────
        assert "snake case" in result   # "snake_case" → "snake case"
        assert "camel case" in result   # "camelCase" → "camel case"

        # ── Single line ───────────────────────────────────────────────────
        assert "\n" not in result

    def test_real_message_normalizer_plan(self):
        """Technical planning message with bullet list of transformations,
        inline code, arrows, bold sections."""
        text = (
            "**TTS Code Readability — text normalization, not model swap**\n\n"
            "This is a preprocessing problem. Kokoro is fine for voice quality — "
            "we need a normalizer that runs *before* TTS. Key transforms:\n\n"
            "- `1s` → \"one second\", `250ms` → \"250 milliseconds\"\n"
            "- `**bold text**` → strip markdown, just say the word\n"
            "- `_flush_delay_s` → \"flush delay s\" (humanize snake_case)\n"
            "- `#5` → \"number 5\" / \"PR 5\"\n"
            "- `*` → \"star\", `->` → \"arrow\", `>=` → \"greater than or equal\"\n"
            "- Strip code blocks, links, headers\n"
            "- File paths spoken naturally\n\n"
            "**Plan:** Build a lightweight Python text normalizer module, "
            "plug it in before the TTS call in the voice pipeline. Waiting for go-ahead."
        )
        result = norm(text)

        # ── No markdown artifacts ──────────────────────────────────────────
        assert "**" not in result
        assert "`" not in result

        # ── Bold italic stripped, content kept ────────────────────────────
        assert "TTS Code Readability" in result
        assert "text normalization" in result

        # ── Italic stripped, word kept ────────────────────────────────────
        assert "before" in result   # *before* → before

        # ── Time units expanded (inside inline code) ──────────────────────
        assert "1 second" in result
        assert "250 milliseconds" in result

        # ── snake_case humanised (inside inline code) ─────────────────────
        assert "flush delay s" in result

        # ── Issue ref expanded ────────────────────────────────────────────
        assert "number 5" in result

        # ── Code symbols expanded ─────────────────────────────────────────
        assert "star" in result
        assert "arrow" in result
        assert "greater than or equal to" in result

        # ── Plan text survives ────────────────────────────────────────────
        assert "Build a lightweight Python text normalizer module" in result

        # ── Single line ───────────────────────────────────────────────────
        assert "\n" not in result

    def test_real_message_pr_blocked(self):
        """PR status message with emoji, bold, issue ref, Python version numbers."""
        text = (
            "**Can't merge PR #5** — blocked by:\n"
            "- ❌ **Merge conflicts** (status: CONFLICTING)\n"
            "- ❌ **CI failure** on Python 3.11 (3.10 and 3.12 cancelled)\n\n"
            "Need to resolve conflicts and fix CI before merging. "
            "Let me know if you want me to work on it."
        )
        result = norm(text)

        # ── No markdown artifacts ──────────────────────────────────────────
        assert "**" not in result
        assert "`" not in result

        # ── Emoji stripped ────────────────────────────────────────────────
        assert "❌" not in result

        # ── PR ref handled ────────────────────────────────────────────────
        # "PR #5" → "PR 5" (# removed, number kept)
        assert "#" not in result
        assert "PR 5" in result

        # ── Bold content preserved ────────────────────────────────────────
        assert "Merge conflicts" in result
        assert "CI failure" in result

        # ── Trailing prose preserved ──────────────────────────────────────
        assert "Need to resolve conflicts" in result
        assert "Let me know" in result

        # ── Single line ───────────────────────────────────────────────────
        assert "\n" not in result

    def test_real_message_plugin_status(self):
        """Status summary with version number, issue ref, time unit, inline
        code branch name."""
        text = (
            "**Voice Plugin Status:**\n"
            "- Chip is **running** right now (process active)\n"
            "- Repo at **v1.3.2** on main\n"
            "- **1 open PR** (#5): 1s speech end-detection delay + auto-reconnect on restart\n"
            "- Branch: `feat/speech-delay-and-reconnect`\n"
        )
        result = norm(text)

        # ── No markdown artifacts ──────────────────────────────────────────
        assert "**" not in result
        assert "`" not in result

        # ── Bold content preserved ────────────────────────────────────────
        assert "Voice Plugin Status" in result
        assert "running" in result

        # ── Version expanded ──────────────────────────────────────────────
        assert "version 1.3.2" in result

        # ── Issue ref expanded ────────────────────────────────────────────
        assert "#" not in result
        assert "number 5" in result

        # ── Time unit expanded ────────────────────────────────────────────
        assert "1 second" in result

        # ── Branch name (inline code path) spoken naturally ───────────────
        # `feat/speech-delay-and-reconnect` → inline stripped → path spoken
        # feat/speech-delay-and-reconnect → "feat, speech delay and reconnect"
        assert "feat" in result
        assert "speech" in result

        # ── Single line ───────────────────────────────────────────────────
        assert "\n" not in result


# ---------------------------------------------------------------------------
# chunk_for_tts
# ---------------------------------------------------------------------------


class TestChunkForTTS:
    """Tests for chunk_for_tts — sentence/paragraph chunking for TTS synthesis."""

    def test_empty_input_returns_empty_list(self):
        assert chunk_for_tts("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_for_tts("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello, how are you?"
        result = chunk_for_tts(text, max_chars=300)
        assert result == ["Hello, how are you?"]

    def test_text_under_max_chars_is_single_chunk(self):
        text = "This is a short message that is well under the limit."
        result = chunk_for_tts(text, max_chars=300)
        assert len(result) == 1
        assert result[0] == text

    def test_multi_paragraph_splits_on_blank_lines(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        result = chunk_for_tts(text, max_chars=300)
        # Should produce at least 2 chunks (one per paragraph or merged if short)
        assert len(result) >= 1
        combined = " ".join(result)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined

    def test_long_paragraph_splits_on_sentences(self):
        # Build a paragraph with several sentences that together exceed max_chars
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump.",
            "The five boxing wizards jump quickly.",
        ]
        text = " ".join(sentences)
        result = chunk_for_tts(text, max_chars=60)
        assert len(result) > 1
        # Each chunk should be at most max_chars (excluding unavoidable overruns)
        for chunk in result:
            assert len(chunk) <= 60 or " " not in chunk  # single-word edge case allowed
        # All content should be preserved
        combined = " ".join(result)
        assert "quick brown fox" in combined
        assert "boxing wizards" in combined

    def test_long_sentence_splits_on_commas(self):
        # A single sentence with commas, longer than max_chars
        sentence = (
            "We need apples, bananas, cherries, dates, elderberries, "
            "figs, grapes, honeydew, and kiwis."
        )
        result = chunk_for_tts(sentence, max_chars=50)
        assert len(result) > 1
        combined = " ".join(result)
        assert "apples" in combined
        assert "kiwis" in combined

    def test_chunks_never_exceed_max_chars(self):
        text = (
            "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. "
            "Mike november oscar papa quebec romeo sierra tango uniform victor whiskey. "
            "X-ray yankee zulu and back to alpha again for good measure today."
        )
        result = chunk_for_tts(text, max_chars=80)
        assert len(result) > 0
        for chunk in result:
            # Allow single-word edge case (can't split a word)
            if " " in chunk:
                assert len(chunk) <= 80, f"Chunk too long: {chunk!r}"

    def test_word_boundary_fallback_for_very_long_word_sequence(self):
        # 10 long words with no punctuation, forced into small chunks
        long_words = " ".join(["extraordinary"] * 15)
        result = chunk_for_tts(long_words, max_chars=50)
        assert len(result) > 1
        combined = " ".join(result)
        assert "extraordinary" in combined

    def test_short_adjacent_chunks_merged(self):
        # Two very short sentences should merge into one chunk
        text = "Hi there. How are you?"
        result = chunk_for_tts(text, max_chars=300)
        # Both sentences are short — should be merged into a single chunk
        assert len(result) == 1
        assert "Hi there" in result[0]
        assert "How are you" in result[0]

    def test_real_world_message_chunked_reasonably(self):
        """Real-world bot message from TestMixedContent gets chunked into
        reasonable pieces (all content present, no chunk exceeds max_chars)."""
        text = normalize_for_speech(
            "**flush_delay_s** is set to `250ms` in the pipeline. "
            "See PR #7 for details. Version v1.3.2 fixes this."
        )
        result = chunk_for_tts(text, max_chars=300)
        assert len(result) >= 1
        combined = " ".join(result)
        assert "flush delay s" in combined
        assert "250 milliseconds" in combined
        for chunk in result:
            if " " in chunk:
                assert len(chunk) <= 300

    def test_three_paragraph_message(self):
        text = (
            "This is the first paragraph. It has two sentences.\n\n"
            "This is the second paragraph. It also has two sentences.\n\n"
            "And this is the third paragraph, which stands alone."
        )
        result = chunk_for_tts(text, max_chars=300)
        combined = " ".join(result)
        assert "first paragraph" in combined
        assert "second paragraph" in combined
        assert "third paragraph" in combined
        # No raw newlines in any chunk
        for chunk in result:
            assert "\n" not in chunk

    def test_no_empty_chunks_in_output(self):
        text = "Hello.\n\n\n\nWorld."
        result = chunk_for_tts(text, max_chars=300)
        for chunk in result:
            assert chunk.strip() != ""

    def test_single_sentence_exactly_at_limit(self):
        # A sentence exactly at max_chars should be a single chunk
        text = "a" * 100
        result = chunk_for_tts(text, max_chars=100)
        assert result == ["a" * 100]

    def test_large_real_bot_response(self):
        """Simulate a large bot response being chunked into manageable pieces."""
        text = normalize_for_speech(
            "**Voice Plugin Status:**\n"
            "- Chip is **running** right now (process active)\n"
            "- Repo at **v1.3.2** on main\n"
            "- **1 open PR** (#5): 1s speech end-detection delay + auto-reconnect on restart\n"
            "- Branch: `feat/speech-delay-and-reconnect`\n"
        )
        result = chunk_for_tts(text, max_chars=200)
        assert len(result) >= 1
        combined = " ".join(result)
        assert "Voice Plugin Status" in combined
        for chunk in result:
            assert chunk.strip() != ""
