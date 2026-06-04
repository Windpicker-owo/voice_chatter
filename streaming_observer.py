"""Bridge streamed tool-call args into low-latency sentence TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.kernel.llm import StreamEvent
from src.kernel.logger import Logger

from .markers import is_tag_only_tts_text, sanitize_tts_text
from .streaming_args import PartialJsonStringFieldExtractor
from .streaming_assembler import SpeechAssemblerConfig, StreamingSpeechAssembler
from .streaming_tts import StreamingTTSPipeline
from .tts import TTSBackend


@dataclass(slots=True)
class StreamingSayCallState:
    call_id: str
    tool_name: str | None = None
    content_extractor: PartialJsonStringFieldExtractor = field(
        default_factory=lambda: PartialJsonStringFieldExtractor("content")
    )
    emotion_extractor: PartialJsonStringFieldExtractor = field(
        default_factory=lambda: PartialJsonStringFieldExtractor("emotion")
    )
    provider_extractor: PartialJsonStringFieldExtractor = field(
        default_factory=lambda: PartialJsonStringFieldExtractor("provider")
    )
    assembler: StreamingSpeechAssembler = field(default_factory=StreamingSpeechAssembler)
    preplayed: bool = False
    submitted_count: int = 0
    emitted_count: int = 0
    candidate_flush_task: asyncio.Task[None] | None = None
    emotion: str | None = None
    provider: str | None = None


class VoiceSayStreamObserver:
    """Observe streamed tool args and preplay action-say.content sentence by sentence.

    Sentence segmentation is delegated to StreamingSpeechAssembler.
    TTS synthesis / ordered emission is handled by a single observer-level
    StreamingTTSPipeline, so multiple action-say calls share one global
    playback sequence.
    """

    def __init__(
        self,
        *,
        backend: TTSBackend,
        chat_stream: Any,
        max_parallel_tts: int,
        min_sentence_chars: int,
        continuation_grace_ms: int = 80,
        flush_tail_on_done: bool,
        empty_audio_retry_count: int,
        logger: Logger | None,
    ) -> None:
        self._backend = backend
        self._chat_stream = chat_stream
        self._max_parallel_tts = max_parallel_tts
        self._continuation_grace_seconds = max(0.0, float(continuation_grace_ms or 0) / 1000.0)
        self._flush_tail_on_done = flush_tail_on_done
        self._empty_audio_retry_count = max(0, int(empty_audio_retry_count or 0))
        self._logger = logger
        self._assembler_config = SpeechAssemblerConfig(
            min_sentence_chars=max(1, int(min_sentence_chars or 1)),
            merge_short_sentences=True,
            treat_single_newline_as_space=True,
            double_newline_as_boundary=True,
        )

        self._states: dict[str, StreamingSayCallState] = {}
        self._disabled = False
        self._warned_no_tool_delta = False
        self._seen_tool_args_delta = False
        self._pipeline: StreamingTTSPipeline | None = None
        self.preplayed_say_call_ids: set[str] = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

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

    async def finalize(self) -> None:
        try:
            for state in self._states.values():
                await self._cancel_candidate_flush_async(state)

                if not self._disabled and self._flush_tail_on_done:
                    ready_texts = state.assembler.flush_all()
                else:
                    ready_texts = state.assembler.flush_candidate()

                for text in ready_texts:
                    await self._submit_text(state, text)
        finally:
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is not None:
                await pipeline.close()

            for state in self._states.values():
                if state.emitted_count > 0 and not self._disabled:
                    self.preplayed_say_call_ids.add(state.call_id)

            self._states.clear()

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: StreamEvent) -> None:
        if event.tool_args_delta:
            self._seen_tool_args_delta = True

        call_id = event.tool_call_id
        if not call_id:
            return

        state = self._states.get(call_id)
        if state is None:
            state = StreamingSayCallState(
                call_id=call_id,
                assembler=StreamingSpeechAssembler(config=self._assembler_config),
            )
            self._states[call_id] = state

        if event.tool_name:
            state.tool_name = event.tool_name
        if state.tool_name != "action-say":
            return

        if not event.tool_args_delta:
            return

        state.emotion = self._update_optional_field(
            current=state.emotion,
            delta=state.emotion_extractor.feed(event.tool_args_delta),
        )
        state.provider = self._update_optional_field(
            current=state.provider,
            delta=state.provider_extractor.feed(event.tool_args_delta),
        )

        content_delta = state.content_extractor.feed(event.tool_args_delta)
        if not content_delta:
            return

        ready_texts = state.assembler.feed(content_delta)
        for text in ready_texts:
            await self._submit_text(state, text)

        self._schedule_candidate_flush(state)

    # ------------------------------------------------------------------
    # TTS submission
    # ------------------------------------------------------------------

    async def _submit_text(self, state: StreamingSayCallState, text: str) -> None:
        text = sanitize_tts_text(text)
        if not text:
            return
        if not state.preplayed and state.submitted_count == 0 and is_tag_only_tts_text(text):
            return

        state.submitted_count += 1

        def _on_emit() -> None:
            state.emitted_count += 1

        pipeline = self._ensure_pipeline()
        await pipeline.submit_text(
            text,
            emotion=state.emotion,
            provider=state.provider,
            on_emit=_on_emit,
        )
        state.preplayed = True

    # ------------------------------------------------------------------
    # candidate flush timer
    # ------------------------------------------------------------------

    def _schedule_candidate_flush(self, state: StreamingSayCallState) -> None:
        self._cancel_candidate_flush(state)
        if self._continuation_grace_seconds <= 0:
            return
        state.candidate_flush_task = asyncio.create_task(
            self._delayed_flush_candidate(state)
        )

    def _cancel_candidate_flush(self, state: StreamingSayCallState) -> None:
        task = state.candidate_flush_task
        if task is None:
            return
        task.cancel()
        state.candidate_flush_task = None

    async def _cancel_candidate_flush_async(self, state: StreamingSayCallState) -> None:
        task = state.candidate_flush_task
        if task is None:
            return
        task.cancel()
        state.candidate_flush_task = None
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _delayed_flush_candidate(self, state: StreamingSayCallState) -> None:
        try:
            await asyncio.sleep(self._continuation_grace_seconds)
            ready = state.assembler.flush_candidate(force_short=True)
            for text in ready:
                await self._submit_text(state, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self._logger is not None:
                self._logger.warning(
                    f"streaming sentence candidate flush failed: {exc}",
                    exc_info=True,
                )
        finally:
            if state.candidate_flush_task is asyncio.current_task():
                state.candidate_flush_task = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> StreamingTTSPipeline:
        if self._pipeline is None:
            self._pipeline = StreamingTTSPipeline(
                backend=self._backend,
                chat_stream=self._chat_stream,
                max_parallel=self._max_parallel_tts,
                empty_audio_retry_count=self._empty_audio_retry_count,
                logger=self._logger,
            )
        return self._pipeline

    @staticmethod
    def _update_optional_field(current: str | None, delta: str) -> str | None:
        if not delta:
            return current
        merged = f"{current or ''}{delta}".strip()
        return merged or None


__all__ = ["StreamingSayCallState", "VoiceSayStreamObserver"]
