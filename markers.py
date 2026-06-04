"""Speech segment helpers for voice_chatter."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


_SENTENCE_END_RE = re.compile(r"(.+?(?:……|[。！？!?；;]|\n+))", re.DOTALL)
_TAG_RE = re.compile(r"\[[^\[\]\r\n]+\]")
_TAG_ONLY_RE = re.compile(r"(?:\[[^\[\]\r\n]+\]\s*)+")
_DISALLOWED_PLAIN_CHARS = frozenset({"_", "\\", "`", "^", "|"})
_CONTINUATION_PREFIX_CHARS = frozenset({"。", "！", "？", "!", "?", "；", ";", "：", ":", "，", ",", "、", "…"})


@dataclass
class SpeechSegment:
    """A single spoken segment to synthesize and emit."""

    text: str
    wait_before: float = 0.0
    emotion: str | None = None
    markers: dict[str, Any] = field(default_factory=dict)


def sanitize_tts_text(text: str) -> str:
    """Remove unsupported special characters while preserving complete tags."""

    if not text:
        return ""

    cleaned_parts: list[str] = []
    cursor = 0
    for match in _TAG_RE.finditer(text):
        cleaned_parts.append(_sanitize_plain_tts_text(text[cursor : match.start()]))
        cleaned_parts.append(match.group(0))
        cursor = match.end()

    cleaned_parts.append(_sanitize_plain_tts_text(text[cursor:]))
    return "".join(cleaned_parts).strip()


def _sanitize_plain_tts_text(text: str) -> str:
    kept: list[str] = []
    for ch in text:
        if ch in _DISALLOWED_PLAIN_CHARS:
            continue
        if ch in "\r\n\t ":
            kept.append(ch)
            continue

        category = unicodedata.category(ch)
        if category.startswith(("L", "N", "P")):
            kept.append(ch)
            continue
        if category.startswith("Z"):
            kept.append(" ")

    return "".join(kept)


def starts_with_continuation_punctuation(text: str) -> bool:
    """Return True when the first non-tag char is continuation punctuation."""

    probe = text.lstrip()
    while probe:
        match = _TAG_RE.match(probe)
        if match is None:
            break
        probe = probe[match.end() :].lstrip()

    return bool(probe) and probe[0] in _CONTINUATION_PREFIX_CHARS


def is_tag_only_tts_text(text: str) -> bool:
    """Return True when the sanitized text contains only complete tags."""

    stripped = text.strip()
    return bool(stripped) and _TAG_ONLY_RE.fullmatch(stripped) is not None


def parse_speech_segments(
    content: str,
    *,
    split_sentences: bool = True,
    default_emotion: str | None = None,
) -> list[SpeechSegment]:
    """Split content into speech segments without parsing or stripping any tags."""

    normalized_default_emotion = (default_emotion or "").strip() or None
    chunks = split_complete_sentences(content) if split_sentences else [content]
    markers = {"emotion": normalized_default_emotion} if normalized_default_emotion else {}

    segments: list[SpeechSegment] = []
    pending_prefix_tags = ""
    for chunk in chunks:
        clean = chunk.strip()
        if not clean:
            continue
        if _TAG_ONLY_RE.fullmatch(clean):
            if segments:
                segments[-1].text = f"{segments[-1].text}{clean}"
            else:
                pending_prefix_tags = f"{pending_prefix_tags}{clean}"
            continue

        if pending_prefix_tags:
            clean = f"{pending_prefix_tags}{clean}"
            pending_prefix_tags = ""

        clean = sanitize_tts_text(clean)
        if not clean:
            continue
        if segments and starts_with_continuation_punctuation(clean):
            segments[-1].text = f"{segments[-1].text}{clean}"
            continue

        segments.append(
            SpeechSegment(
                text=clean,
                wait_before=0.0,
                emotion=normalized_default_emotion,
                markers=dict(markers),
            )
        )

    if pending_prefix_tags and segments:
        segments[-1].text = f"{segments[-1].text}{pending_prefix_tags}"
    return segments


def split_complete_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries while preserving punctuation."""

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(stripped):
        chunk = match.group(1).strip()
        if chunk:
            chunks.append(chunk)
        cursor = match.end()

    tail = stripped[cursor:].strip()
    if tail:
        chunks.append(tail)
    return chunks


__all__ = [
    "is_tag_only_tts_text",
    "SpeechSegment",
    "parse_speech_segments",
    "sanitize_tts_text",
    "split_complete_sentences",
    "starts_with_continuation_punctuation",
]
