"""Sentence segmentation for incremental streamed text."""

from __future__ import annotations


_TERMINATORS = {"。", "！", "？", "!", "?", "；", ";"}


class StreamingSentenceSegmenter:
    """Emit complete sentences only when sentence-ending punctuation arrives."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text_delta: str, *, flush: bool = False) -> list[str]:
        self.buffer += text_delta
        sentences: list[str] = []

        start = 0
        index = 0
        while index < len(self.buffer):
            ch = self.buffer[index]
            if ch in _TERMINATORS:
                sentence = self.buffer[start : index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1
            elif ch == "…" and self.buffer[index : index + 2] == "……":
                sentence = self.buffer[start : index + 2].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 2
                index += 1
            elif ch == "\n":
                sentence = self.buffer[start : index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1
            index += 1

        self.buffer = self.buffer[start:]
        if flush:
            tail = self.buffer.strip()
            self.buffer = ""
            if tail:
                sentences.append(tail)
        return sentences


__all__ = ["StreamingSentenceSegmenter"]
