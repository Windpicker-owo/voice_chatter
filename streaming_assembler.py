"""Streaming speech assembler with candidate-based sentence segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRONG_TERMINATORS = frozenset("。！？!?；;")
_CLOSERS = frozenset(
    "”"   # " RIGHT DOUBLE QUOTATION MARK
    "’"   # ' RIGHT SINGLE QUOTATION MARK
    "」"   # 」 RIGHT CORNER BRACKET
    "』"   # 』 RIGHT WHITE CORNER BRACKET
    "》"   # 》 RIGHT DOUBLE ANGLE BRACKET
    "）"   # ） FULLWIDTH RIGHT PARENTHESIS
    "】"   # 】 RIGHT BLACK LENTICULAR BRACKET
    ")"        # ) ASCII RIGHT PARENTHESIS
    "］"   # ］ FULLWIDTH RIGHT SQUARE BRACKET
    "}"        # } ASCII RIGHT CURLY BRACKET
    "〉"   # 〉 RIGHT ANGLE BRACKET
    "»"   # » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "＂"   # ＂ FULLWIDTH QUOTATION MARK
    "\""       # " ASCII QUOTATION MARK
    "'"        # ' ASCII APOSTROPHE
)
_CONTINUATION_STARTERS = frozenset("。！？!?；;：:，,、…")
_TAG_RE = re.compile(r"\[[^\[\]\r\n]+\]")
_TAG_ONLY_RE = re.compile(r"(?:\[[^\[\]\r\n]+\]\s*)+")


@dataclass(slots=True)
class SpeechAssemblerConfig:
    min_sentence_chars: int = 4
    merge_short_sentences: bool = True
    treat_single_newline_as_space: bool = True
    double_newline_as_boundary: bool = True


class StreamingSpeechAssembler:
    """Assemble streamed text deltas into TTS-ready sentences using a candidate model.

    Instead of immediately emitting text on every sentence-ending punctuation mark,
    this assembler forms a *pending candidate* and waits to see whether the next
    delta contains closing punctuation, tag-only suffixes, or genuinely new content.
    """

    def __init__(self, config: SpeechAssemblerConfig | None = None) -> None:
        self.config = config or SpeechAssemblerConfig()
        self.buffer = ""
        self.pending_candidate = ""
        self.pending_prefix = ""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def feed(self, text_delta: str) -> list[str]:
        """Feed a content delta and return sentences ready for TTS submission."""
        if not text_delta:
            return []

        # Only normalize line endings; keep \n in buffer so cross-delta
        # \n\n is recognised correctly.
        self.buffer += text_delta.replace("\r\n", "\n").replace("\r", "\n")
        return self._process_buffer()

    def flush_candidate(self, *, force_short: bool = False) -> list[str]:
        """Force-submit the current pending candidate (used by grace timeout).

        When *force_short* is True, the minimum-length gate is bypassed so
        that short replies (e.g. "嗯。") still play after the grace period.
        """
        ready: list[str] = []
        if self.pending_candidate:
            adsorbed = _adsorb_suffix(self.buffer)
            if adsorbed:
                self.pending_candidate += adsorbed
                self.buffer = self.buffer[len(adsorbed):]
            text = self._finalize_candidate(force_short=force_short)
            if text:
                ready.append(text)
        return ready

    def flush_all(self) -> list[str]:
        """Flush everything at stream end — candidate, buffer tail, and prefix."""
        ready: list[str] = []
        ready.extend(self.flush_candidate(force_short=True))

        tail = self.buffer.strip()
        self.buffer = ""
        if tail:
            tail = _clean_newlines(tail, self.config.treat_single_newline_as_space)
            if self.pending_prefix:
                tail = f"{self.pending_prefix}{tail}"
                self.pending_prefix = ""
            ready.append(tail)
        elif self.pending_prefix:
            ready.append(self.pending_prefix)
            self.pending_prefix = ""

        return ready

    # ------------------------------------------------------------------
    # internal processing
    # ------------------------------------------------------------------

    def _process_buffer(self) -> list[str]:
        ready: list[str] = []

        while True:
            if self.pending_candidate:
                if self._handle_pending_candidate(ready):
                    continue
                break

            idx = _find_terminator(self.buffer, self.config.double_newline_as_boundary)
            if idx < 0:
                break

            # Leading terminators aren't real sentence boundaries —
            # they're continuation punctuation (e.g. "!Again!" after "Hello?").
            # Preserve them as pending_prefix so they merge with the next candidate.
            if idx == 0:
                tt0 = _terminator_width(self.buffer, 0)
                self.pending_prefix += self.buffer[:tt0]
                self.buffer = self.buffer[tt0:]
                continue

            tt = _terminator_width(self.buffer, idx)
            # For double-newline terminators, exclude the newlines from the candidate.
            if tt > 1:
                self.pending_candidate = self.buffer[:idx]
                self.buffer = self.buffer[idx + tt:]
            else:
                end = idx + 1
                self.pending_candidate = self.buffer[:end]
                self.buffer = self.buffer[end:]

            # Adsorb consecutive terminators
            while self.buffer and self.buffer[0] in _STRONG_TERMINATORS:
                self.pending_candidate += self.buffer[0]
                self.buffer = self.buffer[1:]

        return ready

    def _handle_pending_candidate(self, ready: list[str]) -> bool:
        """Return True if the caller should continue the outer loop."""
        if not self.buffer:
            return False

        adsorbed = _adsorb_suffix(self.buffer)
        if adsorbed:
            self.pending_candidate += adsorbed
            self.buffer = self.buffer[len(adsorbed):]

        if not self.buffer:
            return False

        if _starts_new_sentence(self.buffer):
            text = self._finalize_candidate()
            if text:
                ready.append(text)
            return True

        return False

    def _finalize_candidate(self, *, force_short: bool = False) -> str:
        text = self.pending_candidate
        self.pending_candidate = ""

        if self.config.merge_short_sentences and not force_short:
            plain = _TAG_RE.sub("", text)
            plain = "".join(
                ch for ch in plain
                if ch not in _STRONG_TERMINATORS and ch not in _CLOSERS
            ).strip()
            if len(plain) < self.config.min_sentence_chars:
                self.pending_prefix += _clean_newlines(text, self.config.treat_single_newline_as_space)
                return ""

        if self.pending_prefix:
            text = f"{self.pending_prefix}{text}"
            self.pending_prefix = ""

        return _clean_newlines(text, self.config.treat_single_newline_as_space)


# ------------------------------------------------------------------
# module-private helpers
# ------------------------------------------------------------------

def _find_terminator(buffer: str, double_newline_boundary: bool) -> int:
    """Tag-aware scan for the first sentence terminator in *buffer*."""
    in_tag = False
    for i, ch in enumerate(buffer):
        if ch == "[":
            in_tag = True
        elif ch == "]" and in_tag:
            in_tag = False
        elif not in_tag:
            if ch in _STRONG_TERMINATORS:
                return i
            if double_newline_boundary and ch == "\n":
                if i + 1 < len(buffer) and buffer[i + 1] == "\n":
                    return i
    return -1


def _terminator_width(buffer: str, idx: int) -> int:
    """Return how many characters the terminator at *idx* consumes."""
    if buffer[idx] == "\n" and idx + 1 < len(buffer) and buffer[idx + 1] == "\n":
        return 2
    return 1


def _adsorb_suffix(text: str) -> str:
    """Extract leading closers, continuation phrases, and complete tag-only chunks."""
    adsorbed = ""
    rest = text

    while rest and rest[0] in _CLOSERS:
        adsorbed += rest[0]
        rest = rest[1:]

    # If the remaining text starts with continuation punctuation,
    # scan forward (past the first char) to find the next terminator
    # and adsorb the full continuation phrase.
    if rest and rest[0] in _CONTINUATION_STARTERS:
        idx = _find_terminator(rest[1:], False)
        if idx >= 0:
            adsorbed += rest[:idx + 2]  # +2 for the skipped first char + terminator
            rest = rest[idx + 2:]
        else:
            adsorbed += rest
            rest = ""

    while rest:
        m = _TAG_ONLY_RE.match(rest)
        if not m:
            break
        adsorbed += m.group()
        rest = rest[m.end():]

    return adsorbed


def _starts_new_sentence(text: str) -> bool:
    """Return True when *text* begins content that starts a new sentence."""
    if not text:
        return False
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0] in _CLOSERS or stripped[0] in _CONTINUATION_STARTERS or stripped[0] == "[":
        return False
    return True


def _clean_newlines(text: str, treat_single_as_space: bool) -> str:
    """Replace single newlines with spaces; preserve \\n\\n as paragraph breaks."""
    if not treat_single_as_space:
        return text
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\n":
            j = i
            while j < len(text) and text[j] == "\n":
                j += 1
            count = j - i
            if count >= 2:
                result.append("\n\n")
            else:
                if result and result[-1] != " ":
                    result.append(" ")
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result).strip()


__all__ = ["SpeechAssemblerConfig", "StreamingSpeechAssembler"]
