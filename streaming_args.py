"""Incremental extraction of a JSON string field from partial tool args."""

from __future__ import annotations


class PartialJsonStringFieldExtractor:
    """Extract incremental content for a target string field from partial JSON."""

    def __init__(self, field_name: str) -> None:
        self._field_name = field_name
        self._state = "seek_key_start"
        self._key_buffer: list[str] = []
        self._key_escape = False
        self._key_unicode_remaining = 0
        self._target_key = False

        self._value_buffer: list[str] = []
        self._escape = False
        self._unicode_buffer = ""
        self._unicode_remaining = 0

        self._skip_mode: str | None = None
        self._skip_depth = 0
        self._skip_escape = False
        self._skip_unicode_remaining = 0
        self._skip_string_quote = '"'

    def feed(self, delta: str) -> str:
        """Feed a JSON delta and return the newly decoded field content."""
        if not delta or self._state == "done":
            return ""

        emitted: list[str] = []
        for ch in delta:
            out = self._consume(ch)
            if out:
                emitted.append(out)
        return "".join(emitted)

    def _consume(self, ch: str) -> str:
        if self._state == "seek_key_start":
            if ch == '"':
                self._key_buffer = []
                self._key_escape = False
                self._key_unicode_remaining = 0
                self._state = "in_key"
            return ""

        if self._state == "in_key":
            return self._consume_key_char(ch)

        if self._state == "seek_colon":
            if ch == ":":
                self._state = "seek_value_start"
            return ""

        if self._state == "seek_value_start":
            if ch in " \t\r\n":
                return ""
            if self._target_key and ch == '"':
                self._state = "in_target_string"
                self._escape = False
                self._unicode_buffer = ""
                self._unicode_remaining = 0
                return ""

            self._start_skip_value(ch)
            return ""

        if self._state == "in_target_string":
            return self._consume_target_char(ch)

        if self._state == "skip_value":
            self._consume_skip_char(ch)
            return ""

        return ""

    def _consume_key_char(self, ch: str) -> str:
        if self._key_unicode_remaining:
            if ch.lower() in "0123456789abcdef":
                self._key_buffer.append(ch)
                self._key_unicode_remaining -= 1
                if self._key_unicode_remaining == 0:
                    hex_digits = "".join(self._key_buffer[-4:])
                    self._key_buffer = self._key_buffer[:-4]
                    self._key_buffer.append(chr(int(hex_digits, 16)))
                    self._key_escape = False
            else:
                self._key_unicode_remaining = 0
                self._key_escape = False
                self._key_buffer.append(ch)
            return ""

        if self._key_escape:
            if ch == "u":
                self._key_unicode_remaining = 4
                return ""
            self._key_buffer.append(self._decode_escape(ch))
            self._key_escape = False
            return ""

        if ch == "\\":
            self._key_escape = True
            return ""

        if ch == '"':
            self._target_key = "".join(self._key_buffer) == self._field_name
            self._state = "seek_colon"
            return ""

        self._key_buffer.append(ch)
        return ""

    def _consume_target_char(self, ch: str) -> str:
        if self._unicode_remaining:
            if ch.lower() not in "0123456789abcdef":
                text = "\\u" + self._unicode_buffer + ch
                self._unicode_buffer = ""
                self._unicode_remaining = 0
                self._escape = False
                self._value_buffer.append(text)
                return text

            self._unicode_buffer += ch
            self._unicode_remaining -= 1
            if self._unicode_remaining == 0:
                decoded = chr(int(self._unicode_buffer, 16))
                self._unicode_buffer = ""
                self._escape = False
                self._value_buffer.append(decoded)
                return decoded
            return ""

        if self._escape:
            if ch == "u":
                self._unicode_buffer = ""
                self._unicode_remaining = 4
                return ""
            decoded = self._decode_escape(ch)
            self._escape = False
            self._value_buffer.append(decoded)
            return decoded

        if ch == "\\":
            self._escape = True
            return ""

        if ch == '"':
            self._state = "done"
            return ""

        self._value_buffer.append(ch)
        return ch

    def _start_skip_value(self, ch: str) -> None:
        self._state = "skip_value"
        self._skip_escape = False
        self._skip_unicode_remaining = 0
        if ch == '"':
            self._skip_mode = "string"
            self._skip_depth = 0
            self._skip_string_quote = '"'
            return
        if ch in "[{":
            self._skip_mode = "composite"
            self._skip_depth = 1
            return
        if ch == "}":
            self._reset_after_value()
            return
        self._skip_mode = "bare"
        self._skip_depth = 0
        self._consume_skip_char(ch)

    def _consume_skip_char(self, ch: str) -> None:
        if self._skip_mode == "string":
            if self._skip_unicode_remaining:
                if ch.lower() in "0123456789abcdef":
                    self._skip_unicode_remaining -= 1
                else:
                    self._skip_unicode_remaining = 0
                    self._skip_escape = False
                return
            if self._skip_escape:
                if ch == "u":
                    self._skip_unicode_remaining = 4
                else:
                    self._skip_escape = False
                return
            if ch == "\\":
                self._skip_escape = True
                return
            if ch == self._skip_string_quote:
                self._reset_after_value()
            return

        if self._skip_mode == "composite":
            if ch == '"':
                self._skip_mode = "composite_string"
                self._skip_string_quote = '"'
                return
            if ch in "[{":
                self._skip_depth += 1
                return
            if ch in "]}":
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._reset_after_value()
                return
            return

        if self._skip_mode == "composite_string":
            if self._skip_unicode_remaining:
                if ch.lower() in "0123456789abcdef":
                    self._skip_unicode_remaining -= 1
                else:
                    self._skip_unicode_remaining = 0
                    self._skip_escape = False
                return
            if self._skip_escape:
                if ch == "u":
                    self._skip_unicode_remaining = 4
                else:
                    self._skip_escape = False
                return
            if ch == "\\":
                self._skip_escape = True
                return
            if ch == self._skip_string_quote:
                self._skip_mode = "composite"
            return

        if self._skip_mode == "bare":
            if ch == "," or ch == "}":
                self._reset_after_value()

    def _reset_after_value(self) -> None:
        self._state = "seek_key_start"
        self._target_key = False
        self._skip_mode = None
        self._skip_depth = 0
        self._skip_escape = False
        self._skip_unicode_remaining = 0

    @staticmethod
    def _decode_escape(ch: str) -> str:
        return {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }.get(ch, ch)


__all__ = ["PartialJsonStringFieldExtractor"]
