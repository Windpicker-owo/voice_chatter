"""Bridge streamed tool-call args into low-latency sentence TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.kernel.llm import StreamEvent
from src.kernel.logger import Logger

from .streaming_args import PartialJsonStringFieldExtractor
from .streaming_segmenter import StreamingSentenceSegmenter
from .streaming_tts import StreamingTTSPipeline
from .tts import TTSBackend


@dataclass(slots=True)
class StreamingSayCallState:
    call_id: str
    tool_name: str | None = None
    content_extractor: PartialJsonStringFieldExtractor = field(
        default_factory=lambda: PartialJsonStringFieldExtractor("content")
    )
    segmenter: StreamingSentenceSegmenter = field(default_factory=StreamingSentenceSegmenter)
    pipeline: StreamingTTSPipeline | None = None
    preplayed: bool = False
    pending_prefix: str = ""


class VoiceSayStreamObserver:
    """Observe streamed tool args and preplay `action-say.content` sentence by sentence."""

    def __init__(
        self,
        *,
        backend: TTSBackend,
        chat_stream: Any,
        max_parallel_tts: int,
        min_sentence_chars: int,
        flush_tail_on_done: bool,
        empty_audio_retry_count: int,
        logger: Logger | None,
    ) -> None:
        self._backend = backend
        self._chat_stream = chat_stream
        self._max_parallel_tts = max_parallel_tts
        self._min_sentence_chars = max(1, int(min_sentence_chars or 1))
        self._flush_tail_on_done = flush_tail_on_done
        self._empty_audio_retry_count = max(0, int(empty_audio_retry_count or 0))
        self._logger = logger
        self._states: dict[str, StreamingSayCallState] = {}
        self._disabled = False
        self._warned_no_tool_delta = False
        self._seen_tool_args_delta = False
        self.preplayed_say_call_ids: set[str] = set()

    async def __call__(self, event: StreamEvent) -> None:
        if self._disabled:
            return
        try:
            await self._handle_event(event)
        except Exception as exc:  # noqa: BLE001
            self._disabled = True
            if self._logger is not None:
                self._logger.warning(
                    f"Streaming voice observer degraded to normal mode: {exc}"
                )

    async def _handle_event(self, event: StreamEvent) -> None:
        if event.tool_args_delta:
            self._seen_tool_args_delta = True

        call_id = event.tool_call_id
        if not call_id:
            return

        state = self._states.get(call_id)
        if state is None:
            state = StreamingSayCallState(call_id=call_id)
            self._states[call_id] = state

        if event.tool_name:
            state.tool_name = event.tool_name
        if state.tool_name != "action-say":
            return

        if state.pipeline is None:
            state.pipeline = StreamingTTSPipeline(
                backend=self._backend,
                chat_stream=self._chat_stream,
                max_parallel=self._max_parallel_tts,
                empty_audio_retry_count=self._empty_audio_retry_count,
                logger=self._logger,
            )

        if not event.tool_args_delta:
            return

        content_delta = state.content_extractor.feed(event.tool_args_delta)
        if not content_delta:
            return

        sentences = state.segmenter.feed(content_delta, flush=False)
        for sentence in sentences:
            await self._submit_sentence(state, sentence)

    async def finalize(self) -> None:
        if self._disabled:
            return

        if not self._seen_tool_args_delta and not self._warned_no_tool_delta:
            self._warned_no_tool_delta = True
            if self._logger is not None:
                self._logger.warning(
                    "Streaming voice observer received no tool_args_delta events; falling back to normal tool execution."
                )

        for state in self._states.values():
            if state.tool_name != "action-say" or state.pipeline is None:
                continue

            if self._flush_tail_on_done:
                for sentence in state.segmenter.feed("", flush=True):
                    await self._submit_sentence(state, sentence)

            if state.pending_prefix.strip():
                await state.pipeline.submit_text(state.pending_prefix.strip())
                state.pending_prefix = ""
                state.preplayed = True

            await state.pipeline.close()
            if state.preplayed:
                self.preplayed_say_call_ids.add(state.call_id)

    async def _submit_sentence(self, state: StreamingSayCallState, sentence: str) -> None:
        merged = f"{state.pending_prefix}{sentence}" if state.pending_prefix else sentence
        if len(merged.strip()) < self._min_sentence_chars:
            state.pending_prefix = merged
            return
        state.pending_prefix = ""
        assert state.pipeline is not None
        await state.pipeline.submit_text(merged)
        state.preplayed = True


__all__ = ["StreamingSayCallState", "VoiceSayStreamObserver"]
